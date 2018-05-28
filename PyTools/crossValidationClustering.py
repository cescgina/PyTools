from __future__ import absolute_import, division, print_function, unicode_literals
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
from AdaptivePELE.freeEnergies import cluster
from AdaptivePELE.freeEnergies import estimate
plt.switch_backend("pdf")
plt.style.use("ggplot")


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Plot information related to an MSM"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-l", "--lagtimes", type=int, nargs="*", help="Lagtimes to analyse")
    parser.add_argument("-m", type=int, default=6, help="Number of eigenvalues to sum in the GMRQ")
    args = parser.parse_args()
    return args.lagtimes, args.m


def main(lagtimes, m):
    clusters = list(range(10, 500, 20))
    trajectoryFolder = "allTrajs"
    trajectoryBasename = "traj*"
    stride = 1

    if not os.path.exists("scores"):
        os.makedirs("scores")

    for tau in lagtimes:
        scores = []
        scores_cv = []
        for k in clusters:
            print("Calculating scores with %d clusters" % k)
            clusteringObject = cluster.Cluster(k, trajectoryFolder, trajectoryBasename, alwaysCluster=True, stride=stride)
            clusteringObject.clusterTrajectories()
            # clusteringObject.eliminateLowPopulatedClusters(clusterCountsThreshold)
            try:
                calculateMSM = estimate.MSM(error=False, dtrajs=clusteringObject.dtrajs)
                calculateMSM.estimate(lagtime=tau, lagtimes=None)
                MSM = calculateMSM.MSM_object
                print("MSM estimated on %d states" % MSM.nstates)
            except Exception:
                print("Estimation error in %d clusters, %d lagtime" % (k, tau))
                scores.append(0)
                scores_cv.append(np.array([0, 0, 0, 0, 0]))
                continue
            try:
                scores.append(MSM.score(MSM.dtrajs_full, score_k=m))
            except Exception:
                print("Estimation error in %d clusters, %d lagtime" % (k, tau))
                scores.append(0)
                scores_cv.append(np.array([0, 0, 0, 0, 0]))
                continue
            try:
                scores_cv.append(MSM.score_cv(MSM.dtrajs_full, score_k=m, n=5))
            except Exception:
                print("Estimation error in %d clusters, %d lagtime" % (k, tau))
                scores_cv.append(np.array([0, 0, 0, 0, 0]))
        np.save(os.path.join("scores", "scores_lag_%d.npy" % tau), scores)
        np.save(os.path.join("scores", "scores_cv_lag_%d.npy" % tau), scores_cv)
        mean_scores = [sc.mean() for sc in scores_cv]
        std_scores = [sc.std() for sc in scores_cv]
        plt.figure()
        plt.plot(clusters, scores, label="Training")
        plt.errorbar(clusters, mean_scores, yerr=std_scores, fmt='k', label="Testing")
        plt.xlabel("Number of states")
        plt.ylabel("Score")
        plt.legend()
        plt.savefig("scores_cv_lag_%d.png" % tau)

if __name__ == "__main__":
    lags, GMRQ = parse_arguments()
    main(lags, GMRQ)
