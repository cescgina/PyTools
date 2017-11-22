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
    args = parser.parse_args()
    return args.nTICs, args.numClusters, args.ligand_resname, args.lag, args.nTraj

nTICs, numClusters, ligand_resname, lag, nTraj = parse_arguments()
trajectoryFolder = "tica_projected_trajs"
trajectoryBasename = "tica_traj*"
numClusters = 40
stride = 1
clusterCountsThreshold = 0
# lagtime = 100
# lagtimes = [10, 50, 100, 200, 300, 500, 600, 700, 800, 1000]
# numberOfITS = -1

folders = utilities.get_epoch_folders(".")
folders.sort(key=int)
if not os.path.exists("tica.pkl"):
    trajs = []
    for epoch in folders:
        trajFiles = glob.glob("%s/repeatedExtractedCoordinates/coord*" % epoch)
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

trajsUniq = []
for epoch in folders:
    trajFiles = glob.glob("%s/extractedCoordinates/coord*" % epoch)
    trajFiles.sort(key=lambda x: int(x[x.rfind("_")+1:-4]))
    print trajFiles
    for traj in trajFiles:
        trajLoad = np.loadtxt(traj)
        if len(trajLoad.shape) == 1:
            trajLoad = trajLoad[np.newaxis, :]
        trajsUniq.append(trajLoad)

projectedUniq = tica.transform(trajsUniq)
projectedUniq_filtered = [traj[:, :nTICs] for traj in projectedUniq]

utilities.makeFolder("tica_COM")
for i, epoch in enumerate(folders):
    for iTraj, traj in enumerate(projectedUniq_filtered[i*nTraj:(i+1)*nTraj]):
        # TODO:remove hardcoded path
        snapshotsPDB = utilities.getSnapshots("/home/jgilaber/urokinases_free_energy/1o3f_adaptive_expl_sameR/%s/trajectory_%d.pdb" % (epoch, iTraj+1))
        COM_array = []
        for snapshot in snapshotsPDB:
            pdb_object = atomset.PDB()
            pdb_object.initialise(snapshot, resname=ligand_resname)
            COM_array.append(pdb_object.getCOM())
        try:
            np.savetxt("tica_COM/traj_%d_%d.dat" % (i, iTraj+1), np.hstack((np.array(COM_array), traj)))
        except:
            import pdb
            pdb.set_trace()

clusterCenters = clusteringObject.clusterCenters
dtrajs = clusteringObject.assignNewTrajectories(projectedUniq_filtered)
centersInfo = {x: {"structure": None, "minDist": 1e6} for x in xrange(numClusters)}
for i, epoch in enumerate(folders):
    for iTraj, traj in enumerate(projectedUniq_filtered[i*nTraj:(i+1)*nTraj]):
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
    epoch, traj, snap = centersInfo[clusterNum]['structure']
    # TODO:remove hardcoded path
    snapshots = utilities.getSnapshots("/home/jgilaber/urokinases_free_energy/1o3f_adaptive_expl_sameR/%s/trajectory_%d.pdb" % (epoch, traj))
    pdb_object = atomset.PDB()
    pdb_object.initialise(snapshots[snap], resname=ligand_resname)
    COM_list.append(pdb_object.getCOM())
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

# # plt.legend()
# # plt.figure()
# # for state in range(3, 6):
# #     # plt.figure()
# #     plt.plot(coords[:,:,state].flatten(), 'x', markersize=0.5, label="Tica %d" % (state+1))
# #     # plt.title("Tica %d" % (state+1))
# # plt.legend()
# plt.show()

# conformations = {k: [] for k in range(3)}
# IC = 8
# frames = 0
# for i, epoch in enumerate(folders):
#     for iTraj, traj in enumerate(projected[i*nTraj:(i+1)*nTraj]):
#         # np.savetxt(os.path.join(trajectoryFolder, "%s_%d_%d.dat" % (trajectoryBasename[:-1], int(epoch), iTraj+1)), traj)
#         if len(traj.shape) < 2:
#             traj = [traj]
#         for nSnap, snapshot in enumerate(traj):
#             if frames < 7000 and abs(snapshot[IC]+0.1) < 0.1 and len(conformations[0]) < 7000:
#                 conformations[0].append((epoch, iTraj+1, nSnap))
#             else:
#                 if abs(snapshot[IC]-3) < 2.0 and len(conformations[1]) < 7000:
#                     conformations[1].append((epoch, iTraj+1, nSnap))
#                 elif len(conformations[2]) < 3000 and abs(snapshot[IC]+3.0) < 2.0:
#                     conformations[2].append((epoch, iTraj+1, nSnap))
#             frames += 1
#
# for ind, confs in conformations.iteritems():
#     with open("conformations_%d.dat" % ind, "w") as f:
#         f.write("\n".join(map(str, confs))+"\n")
# plt.plot(tica.eigenvalues[:10])
# plt.figure()
# plt.plot(tica.eigenvectors[:,:10], 'x')
# plt.show()

# trajArr = np.array(trajs)
# eiv = []
# lagtimes = [10, 25, 50, 100, 200, 300, 500, 1000]
# for lag in lagtimes:
#     tica = coor.tica(data=trajs, lag=lag)
#     print tica
#     # projected = tica.get_output()
#     # ic = tica.eigenvectors
#     eiv.append(tica.eigenvalues)
#     # a1 = tica.get_output()
#     # a2 = tica.get_output(dimensions=range(3))
#
# plots = True
# if plots:
#     for lag, ev in zip(lagtimes, eiv):
#         # plt.plot(np.abs(ev), label="Lagtime %d" % lag)
#         plt.plot(ev, label="Lagtime %d" % lag)
#     plt.legend()
#     plt.show()
# if os.path.exists("MSM_object.pkl"):
#     with open("MSM_object.pkl") as f:
#         MSMObject = pickle.load(f)
# else:
#     calculateMSM = estimate.MSM(error=False, dtrajs=clusteringObject.dtrajs)
#     calculateMSM.estimate(lagtime=lagtime, lagtimes=lagtimes, numberOfITS=numberOfITS)
#     MSMObject = calculateMSM.MSM_object
#     with open("MSM_object.pkl", "w") as fw:
#         pickle.dump(MSMObject, fw)

# pi, clusters = computeDeltaG.ensure_connectivity(MSMObject, clusteringObject.clusterCenters)
# d = 0.75
# bins = computeDeltaG.create_box(clusters, projected, d)
# microstateVolume = computeDeltaG.calculate_microstate_volumes(clusters, projected, bins, d)
# np.savetxt("volumeOfClusters.dat", microstateVolume)
#
# gpmf, string = calculate_pmf(microstateVolume, pi)
#
# pmf_xyzg = np.hstack((clusters, np.expand_dims(gpmf,axis=1)))
# np.savetxt("pmf_xyzg.dat", pmf_xyzg)
#
# writePDB(pmf_xyzg)
