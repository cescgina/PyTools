from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
import os
import glob
import argparse
import numpy as np
import pyemma.msm as msm
from AdaptivePELE.freeEnergies import cluster
from AdaptivePELE.utilities import utilities
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Calculate the autocorrelation function of a MSM discretization"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-l", "--lagtimes", type=int, nargs="*", help="Lagtimes to analyse")
    parser.add_argument("-n", "--n_clusters", type=int, help="Number of clusters")
    parser.add_argument("-o", default=None, help="Path of the folder where to store the plots")
    parser.add_argument("--savePlots", action="store_true", help="Save the plots to disk")
    parser.add_argument("--showPlots", action="store_true", help="Show the plots to screen")
    parser.add_argument("--dtrajs", type=str, help="Path to the folder with the discretized trajectories")
    parser.add_argument("--clusters", type=str, default=None, help="Path to the clustering file")
    parser.add_argument("--trajs", type=str, default=None, help="Path to the trajectories files")
    args = parser.parse_args()
    return args.clusters, args.lagtimes, args.o, args.savePlots, args.showPlots, args.dtrajs, args.trajs, args.n_clusters


def __rm(filename):
    try:
        os.remove(filename)
    except OSError:
        pass


def __rmFiles(trajWildcard):
    allfiles = glob.glob(trajWildcard)
    for f in allfiles:
        __rm(f)


def __cleanupFiles(trajWildcard, cleanupClusterCenters=True):
    __rmFiles("clustering_object.pkl")
    # __rmFiles("MSM_object.pkl")
    # __rmFiles("discretized/traj_*")
    __rmFiles(trajWildcard)
    if cleanupClusterCenters:
        __rmFiles("discretized/clusterCenter*")


def create_plots(autoCorr, plots_path, save_plot, show_plot, nclusters, lagtimes, threshold=2):
    if threshold < 1:
        fig_filename = "autoCorr_thres_%s.png" % str(threshold).replace(".", "_")
        filtered = np.where(autoCorr[:, -1] > threshold)[0]
        if len(filtered) == 0:
            raise ValueError("The threshold specified is too strict, no states found above it")
    else:
        fig_filename = "autoCorr_no_thres.png"
        filtered = list(range(nclusters))
    axes = plt.plot(lagtimes, autoCorr[:, filtered])
    plt.xlabel("Lagtime")
    plt.title("Metastability Q of discretization")
    if len(filtered) < 20:
        for i, ax in zip(filtered, axes):
            ax.set_label("Cluster %d" % i)
        plt.legend()
    if save_plot:
        plt.savefig(os.path.join(plots_path, fig_filename))
    if show_plot:
        plt.show()


def main(lagtimes, clusters_file, disctraj, trajs, n_clusters, plots_path, save_plot, show_plot, lagtime_resolution=20):
    if disctraj is not None:
        dtraj_files = glob.glob(os.path.join(disctraj, "*traj*.disctraj"))
        dtrajs = [np.loadtxt(f, dtype=int) for f  in dtraj_files]
        clusterCenters = np.loadtxt(clusters_file)
    else:
        clusteringObject = cluster.Cluster(n_clusters, trajs, "traj*", alwaysCluster=False, discretizedPath=disctraj)
        if clusters_file is not None:
            # only assign
            clusteringObject.clusterCentersFile = clusters_file
        clusteringObject.clusterTrajectories()
        clusterCenters = clusteringObject.clusterCenters
        dtrajs = clusteringObject.dtrajs
    Q = []
    for lag in lagtimes:
        msm_obj = msm.estimate_markov_model(dtrajs, lag)
        counts = msm_obj.count_matrix_full
        Q.append(counts.diagonal()/counts.sum())
    Q = np.array(Q)

    print("Clusters over 0.01 metastability")
    correlation_limit = 0.01
    states2 = np.where(Q[-1] > correlation_limit)[0]
    size2 = states2.size
    if len(states2):
        print(" ".join(map(str, states2)))
    print("Number of clusters:", size2, ", %.2f%% of the total" % (100*size2 / float(n_clusters)))
    utilities.write_PDB_clusters(np.hstack((clusterCenters, Q[:-1].T)), use_beta=True, title="cluster_Q.pdb")
    if plots_path is None:
        plots_path = ""
    else:
        utilities.makeFolder(plots_path)
    create_plots(Q, plots_path, save_plot, show_plot, n_clusters, lagtimes, threshold=2.0)

if __name__ == "__main__":
    clusters, lagtime, output_plots_path, save_plots, show_plots, dtraj_path, trajs_path, num_clusters = parse_arguments()
    main(lagtime, clusters, dtraj_path, trajs_path, num_clusters, output_plots_path, save_plots, show_plots)
