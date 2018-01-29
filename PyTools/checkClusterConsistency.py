import numpy as np
import os
import glob
import argparse
import shutil
from AdaptivePELE.freeEnergies import cluster
from AdaptivePELE.utilities import utilities


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Extract conformations belonging to clusters"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("-conf", default=None, help="Path of the folders where the conformations are stored")
    parser.add_argument("-c", default=None, help="Path of the cluster centers")
    args = parser.parse_args()
    return args.ligand_resname, args.conf, args.c


def main(ligand, clusters_file, conf_folder):
    trajFolder = "allTrajs_nonRepeat"
    cluster_centers = np.loadtxt(clusters_file)
    if not os.path.exists("discretized"):
        os.makedirs("discretized")
    if not os.path.exists(trajFolder):
        os.makedirs(trajFolder)
    stride = 1
    clusterCountsThreshold = 0
    trajBasename = "coord*"
    epoch_folders = utilities.get_epoch_folders(conf_folder)
    numClusters = cluster_centers.shape[0]
    coordinates = [[] for cl in xrange(numClusters)]
    for it in epoch_folders:
        files = glob.glob(conf_folder+"%s/extractedCoordinates/coord*" % it)
        for f in files:
            traj = os.path.splitext(f)[0].split("_")[-1]
            shutil.copy(f, trajFolder+"/coord_%s_%s.dat" % (it, traj))
    clusteringObject = cluster.Cluster(numClusters, trajFolder, trajBasename,
                                       alwaysCluster=False, stride=stride)
    clusteringObject.clusterTrajectories()
    clusteringObject.eliminateLowPopulatedClusters(clusterCountsThreshold)
    for i in xrange(numClusters):
        if not os.path.exists("cluster_%d" % i):
            os.makedirs("cluster_%d/allStructures" % i)
    dtrajs_files = glob.glob("discretized/*.disctraj")
    for dtraj in dtrajs_files:
        print dtraj
        traj = np.loadtxt(dtraj)
        epoch, traj_num = map(int, os.path.splitext(dtraj)[0].split("_", 3)[1:])
        trajPositions = np.loadtxt(trajFolder+"/coord_%d_%d.dat" % (epoch, traj_num))
        snapshots = utilities.getSnapshots(conf_folder+"/%d/trajectory_%d.pdb" % (epoch, traj_num))
        for nSnap, cluster_num in enumerate(traj):
            coordinates[int(cluster_num)].append(trajPositions[nSnap])
            filename = "cluster_%d/allStructures/conf_%d_%d_%d.pdb" % (cluster_num, epoch, traj_num, nSnap)
            with open(filename, "w") as fw:
                fw.write(snapshots[nSnap])
    for cl in xrange(numClusters):
        np.savetxt("cluster_%d/positions.dat" % cl, coordinates[cl])

if __name__ == "__main__":
    lig, conformation_folder, clusters = parse_arguments()
    main(lig, clusters, conformation_folder)
