from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from builtins import range
from io import open
import numpy as np
import os
import itertools
import pyemma.plots as mplt
import matplotlib.pyplot as plt
from AdaptivePELE.freeEnergies import estimate
from AdaptivePELE.utilities import utilities


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Plot information related to an MSM"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-n", "--nStates", type=int, default=4, help="Number of states to use in CK test")
    parser.add_argument("-c", "--clusters", type=int, nargs="*", default=None, help="Number of clusters to analyse")
    parser.add_argument("-l", "--lagtimes", type=int, nargs="*", default=None, help="Lagtimes to analyse")
    parser.add_argument("-l_ITS", "--lagtimes_ITS", type=int, nargs="*", default=None, help="Lagtimes to analyse in the ITS")
    parser.add_argument("--nRuns", type=int, default=1, help="Number of independent calculations to plot")
    parser.add_argument("--path", type=str, help="Path to the folder with the MSM data")
    parser.add_argument("--plotITS", action="store_true", help="Plot the ITS")
    parser.add_argument("--plotCK", action="store_true", help="Plot the CK tests")
    parser.add_argument("--savePlots", action="store_true", help="Save the plots to disk")
    parser.add_argument("--showPlots", action="store_true", help="Show the plots to screen")
    args = parser.parse_args()
    return args.nStates, args.clusters, args.lagtimes, args.nRuns, args.plotITS, args.plotCK, args.savePlots, args.showPlots, args.lagtimes_ITS, args.path


def getMinCluster(pdbFile):
    minVal = 100
    minCl = None
    minCoords = None
    with open(pdbFile) as f:
        for line in f:
            lineContents = line.rstrip().split()
            try:
                beta = float(lineContents[-2])
            except:
                print(lineContents)
            cl = int(lineContents[1])
            coords = map(float, [lineContents[-6], lineContents[-5], lineContents[-4]])
            if beta < minVal:
                minVal = beta
                minCl = cl
                minCoords = np.array(coords)
    return minCl, minCoords


def main(plots, lagtimes_ITS, nRuns, nStates, plot_ITS, plot_CK, save_plot, path):
    if lagtimes_ITS is None:
        lagtimes_ITS = [1, 50, 100, 200, 400, 600, 800, 1000]
    MSM_object = estimate.MSM()
    MSM_object.lagtimes = lagtimes_ITS
    if plot_ITS and save_plot and not os.path.exists(os.path.join(path, "ck_plots")):
        os.makedirs(os.path.join(path, "ck_plots"))
    print("Analysing folder ", path)
    for i in range(nRuns):
        print("Plotting validity checks for run %d" % i)
        MSM = utilities.readClusteringObject(os.path.join(path, "MSM_object_%d.pkl" % i))
        assert len(MSM.dtrajs_full) == len(MSM.dtrajs_active)
        MSM_object.MSM_object = MSM
        MSM_object.dtrajs = MSM.dtrajs_full
        if plot_ITS and (save_plot or plots):
            MSM_object._calculateITS()
            if save_plot:
                plt.savefig(os.path.join(path, "its_%d.png" % i))
        if plot_CK and (save_plot or plots):
            # setting mlags to None chooses automatically the number of lagtimes
            # according to the longest trajectory available
            CK_test = MSM_object.MSM_object.cktest(nStates, mlags=None)
            mplt.plot_cktest(CK_test)
            if save_plot:
                plt.savefig(os.path.join(path, "ck_plots/", "ck_%d.png" % i))
        if plots:
            plt.show()
        plt.close('all')


if __name__ == "__main__":
    states, cluster, lagtime, runs, ITS, CK, save_plots, show_plots, lagtime_ITS, path_MSM = parse_arguments()
    if path_MSM.endswith("/"):
        path_MSM = path_MSM[:-1]
    if lagtime is not None and cluster is not None:
        root, leaf = os.path_MSM.split(path_MSM)
        for tau, k in itertools.product(lagtime, cluster):
            outPath = "".join([root, os.path_MSM.join("%dlag" % tau, "%dcl" % k), leaf])
            main(show_plots, lagtime_ITS, runs, states, ITS, CK, save_plots, outPath)
    else:
        main(show_plots, lagtime_ITS, runs, states, ITS, CK, save_plots, path_MSM)
