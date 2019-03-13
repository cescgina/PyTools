from __future__ import print_function
import os
import argparse
import numpy as np
from scipy.linalg import lu, solve
from AdaptivePELE.atomset import atomset
from AdaptivePELE.freeEnergies import utils
from AdaptivePELE.utilities import utilities
from PyTools.plotMSMAdvancedInfo import getSASAvalues
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Plot information related to an MSM"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("path", type=str, help="Path to the folder with the MSM data")
    parser.add_argument("path_SASA", type=str, help="Path to the folder with the simulation reports")
    parser.add_argument("-n", "--nEigen", type=int, default=4, help="Number of eigenvector to plot")
    parser.add_argument("--resname", type=str, default=None, help="Resname of the ligand in the native pdb")
    parser.add_argument("--native", type=str, default=None, help="Path to the native structure to extract the minimum position")
    args = parser.parse_args()
    return args.nEigen, args.path, args.resname, args.native, args.path_SASA


def calculate_q(A_mat, P_mat, k, ek):
    perm, L, U = lu(A_mat.T)
    x = solve(L.T, ek)
    xa = solve(U[:-1, :-1], -U[:-1, -1])
    xa = np.array(xa.tolist()+[1.0])
    norm_factor = xa.dot(perm.dot(x))
    si = np.outer(xa, perm.dot(x))/norm_factor
    q_vec = []
    for i in range(k):
        q_vec.append(si[i].dot((np.diag(P_mat[i])-np.outer(P_mat[i], P_mat[i])).dot(si[i])))
    return np.array(q_vec)


def main(path, native, resname, nEigen, path_sasa):
    # Z = np.array([[4380, 153, 15, 2, 0, 0], [211, 4788, 1, 0, 0, 0], [169, 1, 4604, 226, 0, 0],
    #               [3, 13, 158, 4823, 3, 0], [0, 0, 0, 4, 4978, 18], [7, 5, 0, 0, 62, 4926]])
    MSM = utilities.readClusteringObject(os.path.join(path, "MSM_object_0.pkl"))
    Z = MSM.count_matrix_full
    np.set_printoptions(precision=4)
    m = 100
    k = Z.shape[0]
    alpha = 1/float(k)
    U_counts = Z + alpha
    w = U_counts.sum(axis=1)
    P_rev = utils.buildRevTransitionMatrix(U_counts)
    P = U_counts / w[:, np.newaxis]
    P = P_rev
    eigvalues, _ = np.linalg.eig(P)
    eigvalues.sort()
    eigvalues = eigvalues[::-1]
    eigvalues_rev, _ = np.linalg.eig(P_rev)
    eigvalues_rev.sort()
    eigvalues_rev = eigvalues_rev[::-1]
    ek = np.zeros(k)
    ek[k-1] = 1.0
    variance = []
    contribution = []
    nEigs = 10
    for index in range(1, nEigs):
        A = P - eigvalues[index]*np.eye(k)
        q = calculate_q(A, P, k, ek)
        score = (q/(w+1))-(q/(w+1+m))
        norm_q = q/q.sum()
        contribution.append(score/score.sum())
        variance.append(norm_q)
    sasa = getSASAvalues(os.path.join(path, "representative_structures", "representative_structures_0.dat"), 4, path_sasa)
    pdb_native = atomset.PDB()
    pdb_native.initialise(u"%s" % native, resname=resname)
    minim = pdb_native.getCOM()
    clusters = np.loadtxt(os.path.join(path, "clusterCenters_0.dat"))
    distance = np.linalg.norm(clusters-minim, axis=1)

    variance = np.array(variance[:nEigen])
    variance = variance.sum(axis=0)
    variance /= variance.sum()
    print(variance)
    # states = variance.argsort()[-1:-10:-1]
    # print(" ".join(["structures/cluster_%d.pdb" % st for st in states]))
    f, axarr = plt.subplots(1, 2)
    axarr[0].scatter(distance, variance)
    axarr[0].set_xlabel("Distance to minimum")
    axarr[0].set_ylabel("Variance")
    axarr[1].scatter(sasa, variance)
    axarr[1].set_xlabel("SASA")
    axarr[0].set_ylabel("Variance")
    f.suptitle("Variance for eigenvalues 2-%d" % (nEigen+1))
    # for ind, var in enumerate(variance[:5]):
    #     f, axarr = plt.subplots(1, 2)
    #     axarr[0].scatter(distance, var)
    #     axarr[0].set_xlabel("Distance to minimum")
    #     axarr[0].set_ylabel("Variance")
    #     axarr[1].scatter(sasa, var)
    #     axarr[1].set_xlabel("SASA")
    #     axarr[0].set_ylabel("Variance")
    #     f.suptitle("Variance for eigenvalue %d" % (ind+2))
    plt.show()


if __name__ == "__main__":

    n, path_files, lig_resname, native_path, path_SASA = parse_arguments()
    main(path_files, native_path, lig_resname, n, path_SASA)
