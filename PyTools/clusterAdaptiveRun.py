from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from io import open
import os
import glob
import shutil
import numpy as np
import argparse
from AdaptivePELE.utilities import utilities
from AdaptivePELE.freeEnergies import cluster, extractCoords
try:
    basestring
except NameError:
    basestring = str


def parseArgs():
    parser = argparse.ArgumentParser(description="Script that reclusters the Adaptive clusters")
    parser.add_argument('nClusters', type=int)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("-atomId", nargs="*", default="", help="Atoms to use for the coordinates of the conformation, if not specified use the center of mass")
    parser.add_argument('-o', type=str, help="Output folder")
    parser.add_argument('-f', type=str, default=".", help="Trajectory folder")
    parser.add_argument('-top', type=str, default=None, help="Topology file for non-pdb trajectories")
    args = parser.parse_args()
    return args.nClusters, args.ligand_resname, args.atomId, args.o, args.f, args.top


def writePDB(pmf_xyzg, title="clusters.pdb"):
    templateLine = "HETATM%s  H%sCLT L 502    %s%s%s  0.75%s           H\n"

    content = ""
    for j, line in enumerate(pmf_xyzg):
        number = str(j).rjust(5)
        number3 = str(j).ljust(3)
        x = ("%.3f" % line[0]).rjust(8)
        y = ("%.3f" % line[1]).rjust(8)
        z = ("%.3f" % line[2]).rjust(8)
        g = 0
        content += templateLine % (number, number3, x, y, z, g)

    with open(title, 'w') as f:
        f.write(content)


def writeInitialStructures(centers_info, filename_template, topology=None):
    if topology is not None:
        topology_contents = utilities.getTopologyFile(topology)
    for cluster_num in centers_info:
        epoch_num, traj_num, snap_num = map(int, centers_info[cluster_num]['structure'])
        trajectory = glob.glob("%d/trajectory_%d*" % (epoch_num, traj_num))[0]
        snapshots = utilities.getSnapshots(trajectory, topology=topology)
        if isinstance(snapshots[0], basestring):
            with open(filename_template % cluster_num, "w") as fw:
                fw.write(snapshots[snap_num])
        else:
            utilities.write_mdtraj_object_PDB(snapshots[snap_num], filename_template % cluster_num, topology_contents)


def get_centers_info(trajectoryFolder, trajectoryBasename, num_clusters, clusterCenters):
    centersInfo = {x: {"structure": None, "minDist": 1e6, "center": None} for x in range(num_clusters)}

    trajFiles = glob.glob(os.path.join(trajectoryFolder, trajectoryBasename))
    for traj in trajFiles:
        _, epoch, iTraj = os.path.splitext(traj)[0].split("_", 3)
        trajCoords = np.loadtxt(traj)
        if len(trajCoords.shape) < 2:
            trajCoords = [trajCoords]
        for snapshot in trajCoords:
            nSnap = snapshot[0]
            snapshotCoords = snapshot[1:]
            dist = np.sqrt(np.sum((clusterCenters-snapshotCoords)**2, axis=1))
            for clusterInd in range(num_clusters):
                if dist[clusterInd] < centersInfo[clusterInd]['minDist']:
                    centersInfo[clusterInd]['minDist'] = dist[clusterInd]
                    centersInfo[clusterInd]['structure'] = (epoch, int(iTraj), nSnap)
                    centersInfo[clusterInd]['center'] = snapshotCoords
    return centersInfo


def main(num_clusters, output_folder, ligand_resname, atom_ids, folder_name=".", topology=None):
    extractCoords.main(folder_name, lig_resname=ligand_resname, non_Repeat=True, atom_Ids=atom_ids)
    trajectoryFolder = "allTrajs"
    trajectoryBasename = "traj*"
    stride = 1
    clusterCountsThreshold = 0

    folders = utilities.get_epoch_folders(folder_name)
    folders.sort(key=int)

    if os.path.exists("discretized"):
        # If there is a previous clustering, remove to cluster again
        shutil.rmtree("discretized")
    clusteringObject = cluster.Cluster(num_clusters, trajectoryFolder,
                                       trajectoryBasename, alwaysCluster=False,
                                       stride=stride)
    clusteringObject.clusterTrajectories()
    clusteringObject.eliminateLowPopulatedClusters(clusterCountsThreshold)
    clusterCenters = clusteringObject.clusterCenters

    centersInfo = get_centers_info(trajectoryFolder, trajectoryBasename, num_clusters, clusterCenters)
    COMArray = [centersInfo[i]['center'][:3] for i in range(num_clusters)]
    if output_folder is not None:
        outputFolder = os.path.join(output_folder, "")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    else:
        outputFolder = ""
    writePDB(COMArray, outputFolder+"clusters_%d_KMeans_allSnapshots.pdb" % num_clusters)
    writeInitialStructures(centersInfo, outputFolder+"initial_%d.pdb", topology=topology)

if __name__ == "__main__":
    n_clusters, lig_name, atom_id, output, traj_folder, top = parseArgs()
    main(n_clusters, output, lig_name, atom_id, folder_name=traj_folder, topology=top)
