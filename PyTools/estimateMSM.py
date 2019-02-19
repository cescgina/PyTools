from __future__ import print_function
import os
import argparse
import numpy as np
import pyemma.coordinates as coor
import pyemma.msm as msm
import pyemma.plots as mplt
import warnings
import matplotlib
import matplotlib.pyplot as plt
from AdaptivePELE.utilities import utilities
from AdaptivePELE.atomset import atomset, RMSDCalculator
from AdaptivePELE.freeEnergies import cluster
from AdaptivePELE.freeEnergies.estimateDG import getCentersInfo
from AdaptivePELE.freeEnergies import getRepresentativeStructures as getRepr


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Plot information related to an MSM"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-n", "--nSets", type=int, default=4, help="Number of sets for PCCA")
    parser.add_argument("-c", "--clusters", type=int, help="Number of clusters")
    parser.add_argument("-l", "--lagtime", type=int, help="Lagtime for the MSM")
    parser.add_argument("--use_TICA", action="store_true", help="Wether to use TICA")
    parser.add_argument("-l_TICA", "--lagtime_TICA", type=int, default=30, help="Lagtime for the TICA")
    parser.add_argument("--peptide", type=str, help="Name of the peptide")
    args = parser.parse_args()
    return args.nSets, args.clusters, args.lagtime, args.lagtime_TICA, args.peptide, args.use_TICA


def plot_sampled_function(xall, yall, zall, ax=None, nbins=100, nlevels=20, cmap=plt.cm.bwr, cbar=True, cbar_label=None, title=""):
    # histogram data
    xmin = np.min(xall)
    xmax = np.max(xall)
    dx = (xmax - xmin) / float(nbins)
    ymin = np.min(yall)
    ymax = np.max(yall)
    dy = (ymax - ymin) / float(nbins)
    xbins = np.linspace(xmin - 0.5*dx, xmax + 0.5*dx, num=nbins)
    ybins = np.linspace(ymin - 0.5*dy, ymax + 0.5*dy, num=nbins)
    xI = np.digitize(xall, xbins)
    yI = np.digitize(yall, ybins)
    # result
    z = np.zeros((nbins, nbins))
    N = np.zeros((nbins, nbins))
    # average over bins
    for t in range(len(xall)):
        z[xI[t], yI[t]] += zall[t]
        N[xI[t], yI[t]] += 1.0

    with warnings.catch_warnings() as cm:
        warnings.simplefilter('ignore')
        z /= N
    # do a contour plot
    extent = [xmin, xmax, ymin, ymax]
    if ax is None:
        ax = plt.gca()
    ax.contourf(z.T, 100, extent=extent, cmap=cmap)
    if cbar:
        cbar = plt.colorbar()
        if cbar_label is not None:
            cbar.ax.set_ylabel(cbar_label)
    if title:
        ax.set_title(title)
    return ax


def plot_sampled_density(xall, yall, zall, ax=None, nbins=100, cmap=plt.cm.Blues, cbar=True, cbar_label=None, title=""):
    return plot_sampled_function(xall, yall, zall, ax=ax, nbins=nbins, cmap=cmap, cbar=cbar, cbar_label=cbar_label, title=title)


n_sets, numClusters, msm_lag, lag_TICA, peptide, use_tica = parse_arguments()
trajectoryFolder = "allTrajs"
trajectoryBasename = "traj_*_*.dat"
stride = 1
clusterCountsThreshold = 0
numberOfITS = -1
trajs, trajFilenames = cluster.loadTrajFiles(trajectoryFolder, trajectoryBasename)
if use_tica:
    tica_obj = coor.tica(trajs, lag=lag_TICA, var_cutoff=0.9, kinetic_map=True)
    print('TICA dimension ', tica_obj.dimension())

    Y = tica_obj.get_output()  # get tica coordinates
    print('Number of trajectories = ', np.shape(Y)[0])
    print('Number of frames = ', np.shape(Y)[1])
    print('Number of dimensions = ', np.shape(Y)[2])
    print('Number of clusters = ', numClusters)
    print('Number of states = ', n_sets)
    print('Value of lag-time = ', msm_lag)
    matplotlib.rcParams.update({'font.size': 14})
    dt = 1
    for ij in range(np.shape(Y)[0]):
        plt.figure(figsize=(8, 5))
        ax1 = plt.subplot(311)
        x = dt*np.arange(Y[ij].shape[0])
        plt.plot(x, Y[ij][:, 0])
        plt.ylabel('IC 1')
        plt.xticks([])

        ax1 = plt.subplot(312)
        plt.plot(x, Y[ij][:, 1])
        plt.ylabel('IC 2')
        plt.xticks([])

        ax1 = plt.subplot(313)
        plt.plot(x, Y[ij][:, 2])
        plt.ylabel('IC 3')
        plt.xlabel('time (frames)')
        plt.xticks([])
        plt.savefig("traj_%d_ICs.png" % (ij+1))
        # if we have many trajectories having them all open might consume a lot of
        # memory
        plt.close()
else:
    Y = trajs
clustering = coor.cluster_kmeans(Y, k=numClusters, max_iter=100)

dtrajs = clustering.dtrajs
cc_x = clustering.clustercenters[:, 0]
cc_y = clustering.clustercenters[:, 1]
cc_z = clustering.clustercenters[:, 2]
xall = np.vstack(Y)[:, 0]
yall = np.vstack(Y)[:, 1]
plt.figure(figsize=(8, 5))
mplt.plot_free_energy(xall, yall, cmap="Spectral")
plt.plot(cc_x, cc_y, linewidth=0, marker='o', markersize=5, color='black')
plt.xlabel("IC 1")
plt.ylabel("IC 2")
plt.title("FES IC1-2")
plt.savefig("fes_IC1-2.png")

plt.figure(figsize=(8, 5))
mplt.plot_free_energy(xall, np.vstack(Y)[:, 2], cmap="Spectral")
plt.plot(cc_x, cc_z, linewidth=0, marker='o', markersize=5, color='black')
plt.xlabel("IC 1")
plt.ylabel("IC 3")
plt.title("FES IC1-3")
plt.savefig("fes_IC1-3.png")

lags = None
plt.figure(figsize=(8, 5))
its = msm.timescales_msm(dtrajs, lags=lags, nits=10)
mplt.plot_implied_timescales(its, ylog=True, units='steps', linewidth=2)
plt.savefig("its.png")


# its = msm.timescales_msm(dtrajs, lags=lags, nits=10, errors='bayes', n_jobs=-1)
# plt.figure(figsize=(8, 5))
# mplt.plot_implied_timescales(its, show_mean=False, ylog=False, units='steps', linewidth=2)
# plt.savefig("its_errors.png")

M = msm.estimate_markov_model(dtrajs, msm_lag)
print('fraction of states used = ', M.active_state_fraction)
print('fraction of counts used = ', M.active_count_fraction)

f = plt.figure(figsize=(8, 5))
pi = M.stationary_distribution
ax = mplt.scatter_contour(cc_x[M.active_set], cc_y[M.active_set], pi, fig=f)
plt.xlabel("IC 1")
plt.ylabel("IC 2")
f.suptitle("Stationary distribution")
plt.savefig("stationary_distribution.png")

labels = ["second", "third", "fourth", "fifth", "sixth"]
eigenvectors_right = M.eigenvectors_right(k=6)
for eig_ind in range(1, 6):
    f = plt.figure(figsize=(8, 5))
    label = labels[eig_ind-1]
    ax = mplt.scatter_contour(cc_x[M.active_set], cc_y[M.active_set], eigenvectors_right[:, eig_ind], fig=f)
    plt.xlabel("IC 1")
    plt.ylabel("IC %d" % (eig_ind+1))
    f.suptitle("%s eigenvector" % label.capitalize())
    plt.savefig("%s_eig.png" % label)

# M = msm.bayesian_markov_model(dtrajs, msm_lag)
# M = msm.estimate_markov_model(dtrajs, msm_lag)
ck = M.cktest(n_sets, mlags=None, err_est=False, show_progress=False)
# mplt.plot_cktest(ck, diag=True, figsize=(7, 7), layout=(2, 2), padding_top=0.1, y01=False, padding_between=0.3, dt=0.1, units='ns')
mplt.plot_cktest(ck, diag=False, figsize=(7, 7), y01=False, padding_between=0.3)
plt.savefig("ck_test.png")

M.pcca(n_sets)
pcca_dist = M.metastable_distributions
membership = M.metastable_memberships  # get PCCA memberships
# memberships over trajectory
# dist_all = [np.hstack([pcca_dist[i, :][dtraj] for dtraj in M.discrete_trajectories_full]) for i in range(n_sets)]
# mem_all = [np.hstack([membership[:, i][dtraj] for dtraj in M.discrete_trajectories_full]) for i in range(n_sets)]

# fig, axes = plt.subplots(1, n_sets, figsize=(16, 3))
# matplotlib.rcParams.update({'font.size': 12})
# axes = axes.flatten()
# np.seterr(invalid='warn')
# for k in range(n_sets):
#     plot_sampled_density(xall, yall, dist_all[k], ax=axes[k], cmap=plt.cm.Blues, cbar=False, title="Set %d" % (k+1))
# plt.xlabel("IC 1")
# plt.ylabel("IC 2")
# plt.savefig("set_membership.png")

plt.figure(figsize=(8, 5))
pcca_sets = M.metastable_sets
mplt.plot_free_energy(xall, yall, cmap="Spectral")
size = 50
cols = ['orange', 'magenta', 'red', 'blue', 'green', 'black']
for i in range(n_sets):
    plt.scatter(cc_x[pcca_sets[i]], cc_y[pcca_sets[i]], color=cols[i], s=size, label="Set %d" % (i+1))
plt.xlabel("IC 1")
plt.ylabel("IC 2")
plt.legend(loc='best')
plt.savefig("fes_pcca.png")
# if not os.path.exists("data"):
#     os.makedirs("data")
# else:
#     for f in os.listdir("data"):
#         os.remove(os.path.join("data", f))
# centersInfo = getCentersInfo(clustering.clustercenters, Y, trajFilenames, dtrajs)
# centersInfo_processed = []
# for cl in centersInfo:
#     epoch_num, traj_num, snapshot_num = centersInfo[cl]["structure"]
#     centersInfo_processed.append([cl, int(epoch_num), int(traj_num), int(snapshot_num)])
# extractInfo = getRepr.getExtractInfo(centersInfo_processed)
# # extractInfo is a dictionary organized as {[epoch, traj]: [cluster, snapshot]}
# cl_sets = {}
# for i_set, cl_list in enumerate(pcca_sets):
#     for cl in cl_list:
#         cl_sets[cl] = i_set + 1
#
# top_path = "/home/jgilaber/peptides_md/%s/simulation_peptides_%s_shorter/topologies/topology_0.pdb" % (peptide, peptide)
# top = utilities.getTopologyFile(top_path)
# rmsd_values = np.zeros(numClusters)
# rmsd_calc = RMSDCalculator.RMSDCalculator()
# PDB_initial = atomset.PDB()
# PDB_initial.initialise(top_path, type="PROTEIN", heavyAtoms=True)
# for pair in extractInfo:
#     snapshots = utilities.getSnapshots("/home/jgilaber/peptides_md/%s/simulation_peptides_%s_shorter/%d/trajectories_fixed_%d.xtc" % (peptide, peptide, pair[0], pair[1]))
#     for cl, n_snap in extractInfo[pair]:
#         if cl not in cl_sets:
#             filename = "data/cluster_%d.pdb" % cl
#         else:
#             filename = "data/cluster_%d_set_%d.pdb" % (cl, cl_sets[cl])
#         PDB = atomset.PDB()
#         PDB.initialise(snapshots[n_snap], heavyAtoms=True, type="PROTEIN", topology=top)
#         rmsd_val = rmsd_calc.computeRMSD(PDB_initial, PDB)
#         rmsd_values[cl] = rmsd_val
#         PDB.writePDB(filename)
# with open("rmsd_states.txt", "w") as fw:
#     fw.write("Cluster\tRSMD(A)\tProbability(%)\n")
#     for cl in range(numClusters):
#         if cl not in M.active_set:
#             prob = 0.00
#         else:
#             prob = 100*pi[np.where(M.active_set == cl)[0][0]]
#         fw.write("%d\t%.3f\t%.5f\n" % (cl, rmsd_values[cl], prob))
# prob = pi[np.argmin(rmsd_values[M.active_set])]
# kb = 0.0019872041
# T = 300
# dG = -kb*T*(np.log(np.max(pi))-np.log(prob))
# print("Estimation of the dG penalty for the solvated peptide", dG)
# f = plt.figure(figsize=(8, 5))
# ax = mplt.scatter_contour(cc_x, cc_y, rmsd_values, fig=f)
# plt.xlabel("IC 1")
# plt.ylabel("IC 2")
# f.suptitle("RMSD to the initial position")
# plt.savefig("rmsd_initial.png")
