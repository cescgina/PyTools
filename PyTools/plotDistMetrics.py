from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from io import open
import numpy as np
import os
import scipy.optimize as optim
from AdaptivePELE.utilities import utilities
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def reward_new(x, rews):
    return -(x*rews).sum()


def reward(x, rews):
    return -(x[:, np.newaxis]*rews).sum()


folders = utilities.get_epoch_folders(".")
for folder in folders[::-1]:
    if os.path.exists(folder+"/clustering/object.pkl"):
        cl_object = utilities.readClusteringObject(folder+"/clustering/object.pkl")
        break
# first_cluster = 0
trajToDivide = 144*2
rewardsEvol = []
weightsEvol = []
weightsEvol_new = []
weights = None
weights_new = None
metricInd = 4
labels = ["TE", "RMSD", "BE", "SASA"]
plots = True
for folder in folders[10:]:
    print("")
    print("Epoch", folder)
    summary = np.loadtxt(folder+"/clustering/summary.txt")
    end_cluster = int(summary[-1, 0])+1
    # metrics = [cl.metrics[3:] for cl in cl_object.clusters.clusters[first_cluster:end_cluster]]
    # first_cluster = end_cluster+1
    metrics = [cl.metrics[3:] for cl in cl_object.clusters.clusters[:end_cluster]]
    metrics = np.array(metrics).T

    mean = np.mean(metrics, axis=1)
    median = np.median(metrics, axis=1)
    std = np.std(metrics, axis=1)
    print("Mean", mean)
    print("Median", median)
    print("Standard deviation", std)
    # Identify least populated clusters
    # population = [cl.elements for cl in cl_object.clusters.clusters[:end_cluster]]
    population = summary[:, 1]
    # Divide by density
    population /= summary[:, -2]
    argweights = np.argsort(population)
    if plots:
        f, axarr = plt.subplots(2, 2, figsize=(12, 12))
        for i, metric in enumerate(labels):

            axarr[i/2, i % 2].hist(metrics[i, :], histtype="step", label="All")
            axarr[i/2, i % 2].hist(metrics[i, argweights[:trajToDivide]], histtype="step", label="Least populated")
            axarr[i/2, i % 2].axvline(mean[i], color='k', label="Mean")
            axarr[i/2, i % 2].axvline(median[i], color='b', label="Median")
            axarr[i/2, i % 2].set_title(metric + ", Epoch %s" % folder)
            if i == 0:
                axarr[i/2, i % 2].legend()
    metrics = metrics[:, argweights[:trajToDivide]]
    rewProv = np.abs(metrics-mean[:, np.newaxis])/std[:, np.newaxis]

    metrics_dispersion = np.abs(metrics-mean[:, np.newaxis]).sum(axis=1)
    print("Accumulated dispersion", metrics_dispersion)
    metrics_scale = metrics_dispersion/std
    print("Unweighted reward", metrics_scale)
    rewardsEvol.append(metrics_scale)
    # constraints so the weights have values between 0 and 1
    cons = ({'type': 'eq', 'fun': lambda x: np.array(x.sum()-1)})
    bounds = [(0, 1)]*metricInd

    if weights is None:
        weights = np.ones(metricInd)/metricInd
        weights_new = np.ones(metricInd)/metricInd
    optimResult_new = optim.minimize(reward_new, weights_new, args=(metrics_scale,), method="SLSQP", constraints=cons, bounds=bounds)
    weights_new = optimResult_new.x
    weightsEvol_new.append(weights_new)

    optimResult = optim.minimize(reward, weights, args=(rewProv,), method="SLSQP", constraints=cons, bounds=bounds)
    weights = optimResult.x
    print("Weights", weights)
    weightsEvol.append(weights)
    if plots:
        plt.show()
    # labels = ["Total energy", "RMSD", "Binding energy", "SASA"]
    # labels = ["Total energy", "RMSD", "Binding energy", "SASA"]
    # for i, dist in enumerate(metrics):
    #     plt.figure()
    #     plt.hist(dist)
    #     plt.axvline(mean[i], color='k', label="Mean")
    #     plt.axvline(median[i], color='b', label="Median")
    #     plt.title(labels[i] + ", Epoch %s" % folder)
    #     plt.legend()
# rewardsEvol = np.array(rewardsEvol)
# weightsEvol = np.array(weightsEvol)
# colors = ['r', 'b', 'c', 'g']
# # plt.gca().set_color_cycle(colors)
# plt.figure()
# lines = plt.plot(rewardsEvol)
# plt.legend(lines, labels)
# plt.figure()
# lines2 = plt.plot(weightsEvol)
# plt.title("Weights spawning")
# plt.legend(lines2, labels)
# plt.figure()
# lines2 = plt.plot(weightsEvol_new)
# plt.title("Weights new spawning")
# plt.legend(lines2, labels)
plt.show()
