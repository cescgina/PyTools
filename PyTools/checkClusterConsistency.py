from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from io import open
import numpy as np
import os
import glob
import argparse
import shutil
from AdaptivePELE.freeEnergies import cluster
from AdaptivePELE.utilities import utilities
try:
    basestring
except NameError:
    basestring = str


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Extract conformations belonging to clusters"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("-conf", default=None, help="Path of the folders where the conformations are stored")
    parser.add_argument("-c", default=None, help="Path of the cluster centers")
    parser.add_argument("-top", type=str, default=None, help="Topology file for non-pdb trajectories")
    args = parser.parse_args()
    return args.ligand_resname, args.conf, args.c, args.top


def main(ligand, clusters_file, conf_folder, topology=None):
    trajFolder = "allTrajs_nonRepeat"
    cluster_centers = np.loadtxt(clusters_file)
    if not os.path.exists("discretized"):
        os.makedirs("discretized")
    if not os.path.exists(trajFolder):
        os.makedirs(trajFolder)
    stride = 1
    clusterCountsThreshold = 0
    trajBasename = "coord*"
    if topology is not None:
        topology_contents = utilities.getTopologyFile(topology)
    else:
        topology_contents = None
    epoch_folders = utilities.get_epoch_folders(conf_folder)
    numClusters = cluster_centers.shape[0]
    coordinates = [[] for cl in range(numClusters)]
    for it in epoch_folders:
        files = glob.glob(conf_folder+"%s/extractedCoordinates/coord*" % it)
        for f in files:
            traj = os.path.splitext(f)[0].split("_")[-1]
            shutil.copy(f, trajFolder+"/coord_%s_%s.dat" % (it, traj))
    clusteringObject = cluster.Cluster(numClusters, trajFolder, trajBasename,
                                       alwaysCluster=False, stride=stride)
    clusteringObject.clusterTrajectories()
    clusteringObject.eliminateLowPopulatedClusters(clusterCountsThreshold)
    for i in range(numClusters):
        if not os.path.exists("cluster_%d" % i):
            os.makedirs("cluster_%d/allStructures" % i)
    dtrajs_files = glob.glob("discretized/*.disctraj")
    for dtraj in dtrajs_files:
        print(dtraj)
        traj = np.loadtxt(dtraj)
        epoch, traj_num = map(int, os.path.splitext(dtraj)[0].split("_", 3)[1:])
        trajPositions = np.loadtxt(trajFolder+"/coord_%d_%d.dat" % (epoch, traj_num))
        trajFile = glob.glob(os.path.join(conf_folder+"%d/trajectory_%d*" % (epoch, traj_num)))[0]
        snapshots = utilities.getSnapshots(trajFile, topology=topology)
        for nSnap, cluster_num in enumerate(traj):
            coordinates[int(cluster_num)].append(trajPositions[nSnap])
            filename = "cluster_%d/allStructures/conf_%d_%d_%d.pdb" % (cluster_num, epoch, traj_num, nSnap)
            if isinstance(snapshots[nSnap], basestring):
                with open(filename, "w") as fw:
                    fw.write(snapshots[nSnap])
            else:
                utilities.write_mdtraj_object_PDB(snapshots[nSnap], filename, topology_contents)
    for cl in range(numClusters):
        np.savetxt("cluster_%d/positions.dat" % cl, coordinates[cl])

if __name__ == "__main__":
    lig, conformation_folder, clusters, top = parse_arguments()
    main(lig, clusters, conformation_folder, topology=top)
