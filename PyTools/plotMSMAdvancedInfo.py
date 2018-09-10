from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
import os
import numpy as np
import argparse
import itertools
from AdaptivePELE.utilities import utilities
from AdaptivePELE.atomset import atomset
from AdaptivePELE.freeEnergies import computeDeltaG
from msmtools.analysis import rdl_decomposition
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Plot information related to an MSM"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-n", "--nEigen", type=int, default=4, help="Number of eigenvector to plot")
    parser.add_argument("-c", "--clusters", type=int, nargs="*", help="Number of clusters to analyse")
    parser.add_argument("-l", "--lagtimes", type=int, nargs="*", help="Lagtimes to analyse")
    parser.add_argument("--minima", type=float, nargs="*", default=None, help="Coordinates of the minima")
    parser.add_argument("--nRuns", type=int, default=1, help="Number of independent calculations to plot")
    parser.add_argument("-o", default=None, help="Path of the folder where to store the plots")
    parser.add_argument("-m", type=int, default=5, help="Number of eigenvalues to sum in the GMRQ")
    parser.add_argument("--plotEigenvectors", action="store_true", help="Plot the eigenvectors")
    parser.add_argument("--plotGMRQ", action="store_true", help="Plot the GMRQ")
    parser.add_argument("--plotPMF", action="store_true", help="Plot the PMF")
    parser.add_argument("--savePlots", action="store_true", help="Save the plots to disk")
    parser.add_argument("--showPlots", action="store_true", help="Show the plots to screen")
    parser.add_argument("--filter", type=int, nargs="*", default=None, help="Clusters to plot")
    parser.add_argument("--path", type=str, help="Path to the folder with the MSM data")
    parser.add_argument("--native", type=str, default=None, help="Path to the native structure to extract the minimum position")
    parser.add_argument("--resname", type=str, default=None, help="Resname of the ligand in the native pdb")
    args = parser.parse_args()
    return args.nEigen, args.clusters, args.lagtimes, args.nRuns, args.minima, args.o, args.m, args.plotEigenvectors, args.plotGMRQ, args.plotPMF, args.savePlots, args.showPlots, args.filter, args.path, args.native, args.resname


def main(nEigenvectors, nRuns, m, outputFolder, plotEigenvectors, plotGMRQ, plotPMF, clusters, lagtimes, minPos, save_plots, showPlots, filtered, destFolder):
    if save_plots and outputFolder is None:
        outputFolder = "plots_MSM"
    if save_plots:
        eigenPlots = os.path.join(outputFolder, "eigenvector_plots")
        GMRQPlots = os.path.join(outputFolder, "GMRQ_plots")
        PMFPlots = os.path.join(outputFolder, "PMF_plots")
    if save_plots and not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
    if filtered is not None:
        filter_str = "_filtered"
    else:
        filter_str = ""
    if plotEigenvectors and save_plots and not os.path.exists(eigenPlots):
        os.makedirs(eigenPlots)
    if plotGMRQ and save_plots and not os.path.exists(GMRQPlots):
        os.makedirs(GMRQPlots)
    if plotPMF and save_plots and not os.path.exists(PMFPlots):
        os.makedirs(PMFPlots)
    minPos = np.array(minPos)
    GMRQValues = {}
    print("Running from " + destFolder)
    if plotGMRQ:
        GMRQValues = []

    if not os.path.exists(os.path.join(destFolder, "eigenvectors")):
        os.makedirs(os.path.join(destFolder, "eigenvectors"))
    for i in range(nRuns):
        titleVar = "%s, run %d" % (destFolder, i)
        if plotGMRQ or plotEigenvectors:
            msm_object = utilities.readClusteringObject(os.path.join(destFolder, "MSM_object_%d.pkl" % i))
        if plotGMRQ:
            GMRQValues.append(np.sum(msm_object.eigenvalues()[:m]))
        if plotEigenvectors or plotPMF:
            clusters = np.loadtxt(os.path.join(destFolder, "clusterCenters_%d.dat" % i))
            distance = np.linalg.norm(clusters-minPos, axis=1)
            volume = np.loadtxt(os.path.join(destFolder, "volumeOfClusters_%d.dat" % i))
            print("Total volume for system %s , run %d" % (destFolder, i), volume.sum())
            if filtered is not None:
                volume = volume[filtered]
                clusters = clusters[filtered]
                distance = distance[filtered]
        if plotEigenvectors:
            if clusters.size != msm_object.stationary_distribution.size:
                mat = computeDeltaG.reestimate_transition_matrix(msm_object.count_matrix_full)
            else:
                mat = msm_object.transition_matrix
            _, _, L = rdl_decomposition(mat)
            figures = []
            axes = []
            for _ in range((nEigenvectors-1)//4+1):
                f, axarr = plt.subplots(2, 2, figsize=(12, 12))
                f.suptitle(titleVar)
                figures.append(f)
                axes.append(axarr)

            for j, row in enumerate(L[:nEigenvectors]):
                pdb_filename = os.path.join(destFolder, "eigenvectors", "eigen_%d_run_%d.pdb" % (j+1, i))
                if j:
                    atomnames = utilities.getAtomNames(utilities.sign(row, tol=1e-3))
                    utilities.write_PDB_clusters(clusters, use_beta=False, elements=atomnames, title=pdb_filename)
                else:
                    utilities.write_PDB_clusters(np.vstack((clusters.T, row)).T, use_beta=True, elements=None, title=pdb_filename)
                if filtered is not None:
                    row = row[filtered]
                np.savetxt(os.path.join(destFolder, "eigenvectors", "eigen_%d_run_%d%s.dat" % (j+1, i, filter_str)), row)
                axes[j//4][(j//2) % 2, j % 2].scatter(distance, row)
                axes[j//4][(j//2) % 2, j % 2].set_xlabel("Distance to minimum")
                axes[j//4][(j//2) % 2, j % 2].set_ylabel("Eigenvector %d" % (j+1))
            if save_plots:
                for j, fg in enumerate(figures):
                    fg.savefig(os.path.join(eigenPlots, "eigenvector_%d_run_%d%s.png" % (j+1, i, filter_str)))
        if plotPMF:
            data = np.loadtxt(os.path.join(destFolder, "pmf_xyzg_%d.dat" % i))
            g = data[:, -1]
            if filtered is not None:
                g = g[filtered]
            print("Clusters with less than 2 PMF:")
            print(" ".join(map(str, np.where(g < 2)[0])))
            print("")
            f, axarr = plt.subplots(2, 2, figsize=(12, 12))
            f.suptitle(titleVar)
            axarr[1, 0].scatter(distance, g)
            axarr[0, 1].scatter(distance, volume)
            axarr[0, 0].scatter(g, volume)
            axarr[1, 0].set_xlabel("Distance to minima")
            axarr[1, 0].set_ylabel("PMF")
            axarr[0, 1].set_xlabel("Distance to minima")
            axarr[0, 1].set_ylabel("Volume")
            axarr[0, 0].set_xlabel("PMF")
            axarr[0, 0].set_ylabel("Volume")
            if save_plots:
                f.savefig(os.path.join(PMFPlots, "pmf_run_%d%s.png" % (i, filter_str)))
    if plotGMRQ:
        for t in GMRQValues:
            plt.figure()
            plt.title("%s" % (destFolder))
            plt.xlabel("Number of states")
            plt.ylabel("GMRQ")
            plt.boxplot(GMRQValues)
            if save_plots:
                plt.savefig(os.path.join(GMRQPlots, "GMRQ.png" % t))
    if showPlots and (plotEigenvectors or plotGMRQ or plotPMF):
        plt.show()

if __name__ == "__main__":
    n_eigen, clusters_list, lagtime_list, runs, minim, output, size_m, plotEigen, plotGMRQs, plotPMFs, write_plots, show_plots, filter_clusters, path_MSM, native, resname = parse_arguments()
    if native is not None:
        if resname is None:
            raise ValueError("Resname not specified!!")
        pdb_native = atomset.PDB()
        pdb_native.initialise(native, resname=resname)
        minim = pdb_native.getCOM()
    if lagtime_list is not None and clusters_list is not None:
        root, leaf = os.path_MSM.split(path_MSM)
        for tau, k in itertools.product(lagtime_list, clusters_list):
            outPath = "".join([root, os.path_MSM.join("%dlag" % tau, "%dcl" % k), leaf])
            main(n_eigen, runs, size_m, output, plotEigen, plotGMRQs, plotPMFs, clusters_list, lagtime_list, minim, write_plots, show_plots, filter_clusters, outPath)
    else:
        main(n_eigen, runs, size_m, output, plotEigen, plotGMRQs, plotPMFs, clusters_list, lagtime_list, minim, write_plots, show_plots, filter_clusters, path_MSM)
