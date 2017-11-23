import os
import glob
import pickle
import numpy as np
import argparse
# from AdaptivePELE.testing import estimateDG
# from AdaptivePELE.testing import computeDeltaG
# from AdaptivePELE.testing import estimate
from AdaptivePELE.testing import cluster
from AdaptivePELE.utilities import utilities
from AdaptivePELE.atomset import atomset
import pyemma.coordinates as coor
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Estimate the time-structure based Independent Components (TICA) from a simulation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("nTICs", type=int, help="Number of Indepent Components to consider")
    parser.add_argument("numClusters", type=int, help="Number of clusters to create")
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("lag", type=int, help="Lagtime to use in the TICA model")
    parser.add_argument("nTraj", type=int, help="Number of trajectories per epoch")
    parser.add_argument("-o", default=None, help="Path of the folders")
    args = parser.parse_args()
    return args.nTICs, args.numClusters, args.ligand_resname, args.lag, args.nTraj, args.o


def extractCOM(PDB_snapshot):
    pdb_obj = atomset.PDB()
    pdb_obj.initialise(snapshot, resname=ligand_resname)
    return pdb_obj.getCOM()


nTICs, numClusters, ligand_resname, lag, nTraj, out_path = parse_arguments()
if out_path is None:
    folderPath = ""
else:
    folderPath = out_path
trajectoryFolder = "tica_projected_trajs"
trajectoryBasename = "tica_traj*"
stride = 1
clusterCountsThreshold = 0

folders = utilities.get_epoch_folders(".")
folders.sort(key=int)
if not os.path.exists("tica.pkl"):
    trajs = []
    for epoch in folders:
        trajFiles = glob.glob(os.path.join(folderPath, "%s/repeatedExtractedCoordinates/coord*" % epoch))
        trajFiles.sort(key=lambda x: int(x[x.rfind("_")+1:-4]))
        for traj in trajFiles:
            trajs.append(np.loadtxt(traj))

    tica = coor.tica(data=trajs, lag=lag)
    with open("tica.pkl", "w") as f:
        pickle.dump(tica, f)
else:
    with open("tica.pkl") as f:
        tica = pickle.load(f)

projected = tica.get_output(dimensions=range(nTICs))
if not os.path.exists(trajectoryFolder):
    os.makedirs(trajectoryFolder)

    for i, epoch in enumerate(folders):
        for iTraj, traj in enumerate(projected[i*nTraj:(i+1)*nTraj]):
            auxArr = np.zeros_like(traj[:, 0])
            # Add a first column of indexes because it is the format that the
            # cluster module of the testing package reads
            np.savetxt(os.path.join(trajectoryFolder, "%s_%d_%d.dat" % (trajectoryBasename[:-1], int(epoch), iTraj+1)), np.hstack((auxArr.reshape(-1, 1), traj)))

clusteringObject = cluster.Cluster(numClusters, trajectoryFolder,
                                   trajectoryBasename, alwaysCluster=False,
                                   stride=stride)
clusteringObject.clusterTrajectories()
clusteringObject.eliminateLowPopulatedClusters(clusterCountsThreshold)

utilities.makeFolder("tica_COM")
trajsUniq = []
projectedUniq = []
for epoch in folders:
    trajFiles = glob.glob(os.path.join(folderPath, "%s/extractedCoordinates/coord*" % epoch))
    trajFiles.sort(key=lambda x: int(x[x.rfind("_")+1:-4]))
    for trajName in trajFiles:
        trajNum = int(trajName[trajName.rfind("_")+1:-4])
        snapshotsPDB = utilities.getSnapshots(os.path.join(folderPath, "%s/trajectory_%d.pdb" % (epoch, trajNum)))
        trajCOM = [extractCOM(snapshot) for snapshot in snapshotsPDB]
        trajsUniq.append(trajCOM)
        trajLoad = np.loadtxt(trajName)
        if len(trajLoad.shape) == 1:
            trajLoad = trajLoad[np.newaxis, :]
        projectedTraj = tica.transform(trajLoad)[:, :nTICs]
        projectedUniq.append(projectedTraj)
        np.savetxt("tica_COM/traj_%s_%d.dat" % (epoch, trajNum), np.hstack((np.array(trajCOM), projectedTraj)))

clusterCenters = clusteringObject.clusterCenters
dtrajs = clusteringObject.assignNewTrajectories(projectedUniq)
centersInfo = {x: {"structure": None, "minDist": 1e6} for x in xrange(numClusters)}
for i, epoch in enumerate(folders):
    for iTraj, traj in enumerate(projectedUniq[i*nTraj:(i+1)*nTraj]):
        for nSnap, snapshot in enumerate(traj):
            clusterInd = dtrajs[i*nTraj+iTraj][nSnap]
            dist = np.sqrt(np.sum((clusterCenters[clusterInd]-snapshot)**2))
            if dist < centersInfo[clusterInd]['minDist']:
                centersInfo[clusterInd]['minDist'] = dist
                centersInfo[clusterInd]['structure'] = (epoch, iTraj+1, nSnap)

if not os.path.exists("clusterCenters"):
    os.makedirs("clusterCenters")
COM_list = []
for clusterNum in centersInfo:
    epoch, trajNum, snap = centersInfo[clusterNum]['structure']
    COM_list.append(trajsUniq[int(epoch)*nTraj+(trajNum-1)][snap])
    snapshots = utilities.getSnapshots(os.path.join(folderPath, "%s/trajectory_%d.pdb" % (epoch, trajNum)))
    pdb_object = atomset.PDB()
    pdb_object.initialise(snapshots[snap], resname=ligand_resname)
    pdb_object.writePDB("clusterCenters/cluster_%d.pdb" % clusterNum)

distances = [[nC, centersInfo[nC]['minDist']] for nC in xrange(numClusters)]
np.savetxt("clusterCenters/clusterDistances_%dcl_%dTICs.dat" % (numClusters, nTICs), distances)
utilities.write_PDB_clusters(COM_list, "clusterCenters/clustersCenters_%dcl_%dTICs.pdb" % (numClusters, nTICs))
plotTICA = True
if plotTICA:
    plt.rcParams.update({'legend.markerscale': 10})
    # coords = np.array(projected)
    states = [4, 5, 6, 7, 8]
    states = range(3, 9)
    states = range(10)
    for state in states:
        plt.figure()
        # plt.plot(coords[:,:,state].flatten(), 'x', markersize=0.5, label="Tica %d" % (state+1))
        plotNum = 0
        for traj in projected:
            try:
                plt.plot(range(plotNum, plotNum+traj.shape[0]), traj[:, state], 'x', markersize=0.5, color="r")
                plotNum += traj.shape[0]
                # plt.plot(traj[:, 2], traj[:, state], 'x', markersize=0.5, color="r")
            except IndexError:
                plt.plot([plotNum], traj[state], 'x', markersize=0.5, color="r")
                plotNum += 1
                # plt.plot(traj[2], traj[state], 'x', markersize=0.5, color="r")
        plt.title("Tica %d" % (state+1))
        # plt.title("Comparing different Tica")
        # plt.xlabel("Tica 3")
        # plt.ylabel("Tica %d" % (state+1))
        # plt.savefig("tica_3_%d_IC.png" % (state + 1))
        plt.savefig("tica_%d_IC.png" % (state + 1))
    # plt.show()
    # import sys
    # sys.exit()
