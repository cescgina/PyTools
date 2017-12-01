import os
import glob
import pickle
import numpy as np
import argparse
import itertools
from AdaptivePELE.testing import cluster
from AdaptivePELE.testing import extractCoords
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
    parser.add_argument("nTraj", type=int, help="Number of real trajectories per epoch (i.e number of processors-1)")
    parser.add_argument("totalSteps", type=int, default=0, help="Total number of steps in traj. Equivalent to epoch length in adaptive runs")
    parser.add_argument("-o", default=None, help="Path of the folders where the trajectories are stored")
    parser.add_argument("-stride", type=int, default=1, help="Stride, e.g. select one conformation out of every x, default 1, that is take all")
    parser.add_argument("-atomId", type=str, default="", help="Atoms to use for the coordinates of the conformation, if not specified use the center of mass")
    parser.add_argument("-r", "-repeat", action="store_true", help="Force the extraction and repeat the coordinates")
    args = parser.parse_args()
    return args.nTICs, args.numClusters, args.ligand_resname, args.lag, args.nTraj, args.totalSteps, args.o, args.stride, args.atomId, args.r


def get_coords(conformation, atom, lig_name):
    pdb_obj = atomset.PDB()
    if atom:
        atom_name = atom.split(":")[1]
        pdb_obj.initialise(conformation, atomname=atom_name)
        # getAtomCoords returns an array, while getCOM returns a list
        return pdb_obj.getAtom(atom).getAtomCoords().tolist()
    else:
        pdb_obj.initialise(conformation, resname=lig_name)
        return pdb_obj.getCOM()


if __name__ == "__main__":
    nTICs, numClusters, ligand_resname, lag, nTraj, n_steps, out_path, stride_conformations, atomId, repeat = parse_arguments()
    if out_path is None:
        folderPath = ""
        curr_folder = "."
    else:
        folderPath = out_path
        curr_folder = out_path

    folders = utilities.get_epoch_folders(curr_folder)
    if not os.path.exists(os.path.join(folderPath, "0/repeatedExtractedCoordinates/"))or repeat:
        # Extract ligand and alpha carbons coordinates
        extractCoords.main(folder_name=curr_folder, lig_resname=ligand_resname, numtotalSteps=n_steps, protein_CA=True, non_Repeat=False)
    trajectoryFolder = "tica_projected_trajs"
    trajectoryBasename = "tica_traj*"
    stride = 1
    clusterCountsThreshold = 0

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
            trajCOM = [get_coords(snapshot, atomId, ligand_resname) for snapshot in itertools.islice(snapshotsPDB, 0, None, stride_conformations)]
            trajsUniq.append(trajCOM)
            trajLoad = np.loadtxt(trajName)
            if len(trajLoad.shape) == 1:
                trajLoad = trajLoad[np.newaxis, :]
            projectedTraj = tica.transform(trajLoad[::stride_conformations])[:, :nTICs]
            projectedUniq.append(projectedTraj)
            np.savetxt("tica_COM/traj_%s_%d.dat" % (epoch, trajNum), np.hstack((np.array(trajCOM), projectedTraj)),
                       header="COM coordinates x\ty\tz\t TICA coordinates\t"+"\t".join(["TICA %d" % tic for tic in xrange(nTICs)]) + "\n")

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
        states = range(nTICs)
        for state in states:
            plt.figure()
            # plt.plot(coords[:,:,state].flatten(), 'x', markersize=0.5, label="Tica %d" % (state+1))
            plotNum = 0
            for traj in projected:
                try:
                    plt.plot(range(plotNum, plotNum+traj.shape[0]), traj[:, state], 'x', markersize=0.5, color="r")
                    plotNum += traj.shape[0]
                    # plt.plot(traj[:, 2], traj[:, state], 'x', markersize=0.5, color="r")
                except IndexError as e:
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
