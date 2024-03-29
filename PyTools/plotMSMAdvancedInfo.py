from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
import os
import glob
import argparse
import itertools
import numpy as np
from AdaptivePELE.utilities import utilities
from AdaptivePELE.atomset import atomset
from AdaptivePELE.freeEnergies import computeDeltaG, getRepresentativeStructures as getR
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
    parser.add_argument("--atomIds", nargs='*', default=None, help="serial:atomName:resname, e.g. 2048:C1:AIN. May contain more than one atomId")
    parser.add_argument("--SASA_col", type=int, default=None, help="Column of the SASA in the reports (starting to count from 1)")
    parser.add_argument("--path_report", type=str, help="Path to the folder with the reports")
    args = parser.parse_args()
    if args.SASA_col is not None:
        args.SASA_col -= 1
    return args.nEigen, args.clusters, args.lagtimes, args.nRuns, args.minima, args.o, args.m, args.plotEigenvectors, args.plotGMRQ, args.plotPMF, args.savePlots, args.showPlots, args.filter, args.path, args.native, args.resname, args.SASA_col, args.path_report, args.atomIds


def getSASAvalues(representative_file, sasa_col, path_to_report):
    clusters_info = np.loadtxt(representative_file, skiprows=1, dtype=int)
    extract_info = getR.getExtractInfo(clusters_info)
    sasa = [0 for _ in clusters_info]
    for trajFile, extraInfo in extract_info.items():
        report_filename = glob.glob(os.path.join(path_to_report, "%d", "report*_%d") % trajFile)[0]
        report = utilities.loadtxtfile(report_filename)
        for pair in extraInfo:
            sasa[pair[0]] = report[pair[1], sasa_col]
    return sasa


def main(nEigenvectors, nRuns, m, outputFolder, plotEigenvectors, plotGMRQ, plotPMF, clusters, lagtimes, minPos, save_plots, showPlots, filtered, destFolder, sasa_col, path_to_report):
    if save_plots and outputFolder is None:
        outputFolder = "plots_MSM"
    if outputFolder is not None:
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
    print("Running from", destFolder)
    if plotGMRQ:
        GMRQValues = []

    if not os.path.exists(os.path.join(destFolder, "eigenvectors")):
        os.makedirs(os.path.join(destFolder, "eigenvectors"))
    for i in range(nRuns):
        if sasa_col is not None:
            representatives_files = os.path.join(destFolder, "representative_structures/representative_structures_%d.dat" % i)
            sasa = getSASAvalues(representatives_files, sasa_col, path_to_report)

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
                if sasa_col is not None:
                    sasa = sasa[filtered]
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
            Q = msm_object.count_matrix_full.diagonal()/msm_object.count_matrix_full.sum()
            plt.figure()
            plt.scatter(distance, Q)
            plt.xlabel("Distance to minimum")
            plt.ylabel("Metastability")
            if save_plots:
                plt.savefig(os.path.join(eigenPlots, "Q_run_%d%s.png" % (i, filter_str)))
            if save_plots:
                for j, fg in enumerate(figures):
                    fg.savefig(os.path.join(eigenPlots, "eigenvector_%d_run_%d%s.png" % (j+1, i, filter_str)))
                plt.figure()
                plt.scatter(distance, L[0])
                plt.xlabel("Distance to minimum")
                plt.ylabel("Eigenvector 1")
                plt.savefig(os.path.join(eigenPlots, "eigenvector_1_alone_run_%d%s.png" % (i, filter_str)))
        if plotPMF:
            data = np.loadtxt(os.path.join(destFolder, "pmf_xyzg_%d.dat" % i))
            g = data[:, -1]
            annotations = ["Cluster %d" % i for i in range(g.size)]
            if filtered is not None:
                g = g[filtered]
                annotations = np.array(annotations)[filtered].tolist()
            print("Clusters with less than 2 PMF:")
            print(" ".join(map(str, np.where(g < 2)[0])))
            print("")
            fig_pmf, axarr = plt.subplots(2, 2, figsize=(12, 12))
            fig_pmf.suptitle(titleVar)
            sc1 = axarr[1, 0].scatter(distance, g)
            sc2 = axarr[0, 1].scatter(distance, volume)
            sc3 = axarr[0, 0].scatter(g, volume)
            axes = [axarr[0, 1], axarr[1, 0], axarr[0, 0]]
            scs = [sc2, sc1, sc3]
            if sasa_col is not None:
                axarr[1, 1].scatter(sasa, g)
            axarr[1, 0].set_xlabel("Distance to minima")
            axarr[1, 0].set_ylabel("PMF")
            axarr[0, 1].set_xlabel("Distance to minima")
            axarr[0, 1].set_ylabel("Volume")
            axarr[0, 0].set_xlabel("PMF")
            axarr[0, 0].set_ylabel("Volume")
            annot1 = axarr[1, 0].annotate("", xy=(0, 0), xytext=(20, 20),
                                          textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="w"),
                                          arrowprops=dict(arrowstyle="->"))
            annot1.set_visible(False)
            annot2 = axarr[0, 1].annotate("", xy=(0, 0), xytext=(20, 20),
                                          textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="w"),
                                          arrowprops=dict(arrowstyle="->"))
            annot2.set_visible(False)
            annot3 = axarr[0, 0].annotate("", xy=(0, 0), xytext=(20, 20),
                                          textcoords="offset points",
                                          bbox=dict(boxstyle="round", fc="w"),
                                          arrowprops=dict(arrowstyle="->"))
            annot3.set_visible(False)
            annot_list = [annot2, annot1, annot3]
            if sasa_col is not None:
                axarr[1, 1].set_xlabel("SASA")
                axarr[1, 1].set_ylabel("PMF")
            if save_plots:
                fig_pmf.savefig(os.path.join(PMFPlots, "pmf_run_%d%s.png" % (i, filter_str)))
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
        if plotPMFs:

            def update_annot(ind, sc, annot):
                """Update the information box of the selected point"""
                pos = sc.get_offsets()[ind["ind"][0]]
                annot.xy = pos
                annot.set_text(annotations[int(ind["ind"][0])])
                # annot.get_bbox_patch().set_facecolor(cmap(norm( z_values[ind["ind"][0]])))

            def hover(event):
                """Action to perform when hovering the mouse on a point"""
                # vis = any([annot.get_visible() for annot in annot_list])
                for i, ax_comp in enumerate(axes):
                    vis = annot_list[i].get_visible()
                    if event.inaxes == ax_comp:
                        for j in range(len(axes)):
                            if j != i:
                                annot_list[j].set_visible(False)
                        cont, ind = scs[i].contains(event)
                        if cont:
                            update_annot(ind, scs[i], annot_list[i])
                            annot_list[i].set_visible(True)
                            fig_pmf.canvas.draw_idle()
                        else:
                            if vis:
                                annot_list[i].set_visible(False)
                                fig_pmf.canvas.draw_idle()
            fig_pmf.canvas.mpl_connect("motion_notify_event", hover)
        plt.show()

if __name__ == "__main__":
    n_eigen, clusters_list, lagtime_list, runs, minim, output, size_m, plotEigen, plotGMRQs, plotPMFs, write_plots, show_plots, filter_clusters, path_MSM, native, resname, SASA_col, path_report, atomIds = parse_arguments()
    if native is not None:
        if resname is None:
            raise ValueError("Resname not specified!!")
        pdb_native = atomset.PDB()
        pdb_native.initialise(u"%s" % native, resname=resname)
        if atomIds is not None:
            minim = []
            for atomId in atomIds:
                minim.extend(pdb_native.getAtom(atomId).getAtomCoords())
        else:
            minim = pdb_native.getCOM()
    if lagtime_list is not None and clusters_list is not None:
        root, leaf = os.path.split(path_MSM)
        for tau, k in itertools.product(lagtime_list, clusters_list):
            outPath = os.path.join(root, "%dlag" % tau, "%dcl" % k, leaf)
            main(n_eigen, runs, size_m, output, plotEigen, plotGMRQs, plotPMFs, clusters_list, lagtime_list, minim, write_plots, show_plots, filter_clusters, outPath, SASA_col, path_report)
    else:
        main(n_eigen, runs, size_m, output, plotEigen, plotGMRQs, plotPMFs, clusters_list, lagtime_list, minim, write_plots, show_plots, filter_clusters, path_MSM, SASA_col, path_report)
