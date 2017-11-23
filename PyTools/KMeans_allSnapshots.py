import os
import glob
import numpy as np
import argparse
from AdaptivePELE.utilities import utilities
from AdaptivePELE.testing import cluster


def parseArgs():
    parser = argparse.ArgumentParser(description="Script that reclusters the Adaptive clusters")
    parser.add_argument('nClusters', type=int)
    parser.add_argument('-o', type=str, help="Output folder")
    args = parser.parse_args()
    return args.nClusters, args.o


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


def writeInitialStructures(centers_info, filename_template):
    for cluster_num in centersInfo:
        epoch_num, traj_num, snap_num = map(int, centersInfo[cluster_num]['structure'])
        trajectory = "%d/trajectory_%d.pdb" % (epoch_num, traj_num)
        snapshots = utilities.getSnapshots(trajectory)
        with open(filename_template % cluster_num, "w") as fw:
            fw.write(snapshots[snap_num])

n_clusters, output = parseArgs()
trajectoryFolder = "allTrajs"
trajectoryBasename = "traj*"
stride = 1
clusterCountsThreshold = 0

folders = utilities.get_epoch_folders(".")
folders.sort(key=int)


clusteringObject = cluster.Cluster(n_clusters, trajectoryFolder,
                                   trajectoryBasename, alwaysCluster=False,
                                   stride=stride)
clusteringObject.clusterTrajectories()
clusteringObject.eliminateLowPopulatedClusters(clusterCountsThreshold)
clusterCenters = clusteringObject.clusterCenters
dtrajs = clusteringObject.dtrajs

centersInfo = {x: {"structure": None, "minDist": 1e6, "center": None} for x in xrange(n_clusters)}

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
        for clusterInd in xrange(n_clusters):
            if dist[clusterInd] < centersInfo[clusterInd]['minDist']:
                centersInfo[clusterInd]['minDist'] = dist[clusterInd]
                centersInfo[clusterInd]['structure'] = (epoch, int(iTraj), nSnap)
                centersInfo[clusterInd]['center'] = snapshotCoords

COMArray = [centersInfo[i]['center'] for i in xrange(n_clusters)]
IndexesSet = set([centersInfo[i]['structure'] for i in xrange(n_clusters)])
if output is not None:
    outputFolder = os.path.join(output, "")
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
else:
    outputFolder = ""
writePDB(COMArray, outputFolder+"clusters_%d_KMeans_allSnapshots.pdb" % n_clusters)
writeInitialStructures(centersInfo, outputFolder+"initial_%d.pdb")
