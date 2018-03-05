import os
import numpy as np
from AdaptivePELE.utilities import utilities
from AdaptivePELE.freeEnergies import computeDeltaG
from msmtools.analysis import rdl_decomposition
import matplotlib.pyplot as plt
plt.style.use("ggplot")


nEigenvectors = 4
nRuns = 10
m = 15
outputFolder = "figures_eigenvectors_compare_1o3f"
plotEigenvectors = False
plotPMF = False
plotGMRQ = True
iterations = [(25, 100), (50, 100), (100, 100), (200, 100), (400, 100),
              (25, 200), (50, 200), (100, 200), (200, 200), (400, 200),
              (25, 400), (50, 400), (100, 400), (200, 400), (400, 400)]
# iterations = [(400, 400), (400, 200), (400, 100)]
# systems = ["1o3f_PELE_sampl_40_waters", "1o3f_PELE_sampl_40_alt_box", "1sqa_PELE_sampl_40"]
# minima = np.array([[44.348, -3.041, 26.375]]*2+[[31.657, 5.141, 28.070]])
# systems = ["1sqa_PELE_sampl_40"]
# minima = np.array([[31.657, 5.141, 28.070]])
systems = ["1o3f_PELE_sampl_40_waters"]
minima = np.array([[44.348, -3.041, 26.375]])
if not os.path.exists(outputFolder):
    os.makedirs(outputFolder)

GMRQfigures = {}
for runFolder, minPos in zip(systems, minima):
    print "Running from " + runFolder
    for tau, k in iterations:
        if plotGMRQ:
            if tau not in GMRQfigures:
                fig = plt.figure()
                ax = fig.add_subplot(111)
                GMRQfigures[tau] = (fig, ax)
                GMRQfigures[tau][0].suptitle("%s, %dlag" % (runFolder, tau))
                GMRQfigures[tau][1].set_xlabel("Number of states")
                GMRQfigures[tau][1].set_ylabel("GMRQ")
        destFolder = os.path.join(runFolder, "%dlag/%dcl" % (tau, k))
        print "Lagtime %d, clusters %d" % (tau, k)
        if not os.path.exists(destFolder+"/MSM_0/eigenvectors"):
            os.makedirs(destFolder+"/MSM_0/eigenvectors")
        for i in xrange(nRuns):
            titleVar = "%s, %dcl, %dlag, run %d" % (runFolder, k, tau, i)
            if plotGMRQ or plotEigenvectors:
                msm_object = utilities.readClusteringObject(destFolder+"/MSM_0/MSM_object_%d.pkl" % i)
            if plotGMRQ:
                GMRQfigures[tau][1].plot(k, np.sum(msm_object.eigenvalues()[:m]), 'x')
            if plotEigenvectors or plotPMF:
                clusters = np.loadtxt(destFolder+"/MSM_0/clusterCenters_%d.dat" % i)
                distance = np.linalg.norm(clusters-minPos, axis=1)
                volume = np.loadtxt(os.path.join(runFolder, "%dlag" % tau, "%dcl" % k, "MSM_0", "volumeOfClusters_%d.dat" % i))
                print "Total volume for system %s" % runFolder, volume.sum()
            if plotEigenvectors:
                if clusters.size != msm_object.stationary_distribution.size:
                    mat = computeDeltaG.reestimate_transition_matrix(msm_object.count_matrix_full)
                else:
                    mat = msm_object.transition_matrix
                R, D, L = rdl_decomposition(mat)
                figures = []
                axes = []
                for i in xrange((nEigenvectors-1)/4+1):
                    f, axarr = plt.subplots(2, 2, figsize=(12, 12))
                    f.suptitle(titleVar)
                    figures.append(f)
                    axes.append(axarr)

                for j, row in enumerate(L[:nEigenvectors]):
                    axes[j/4][(j/2) % 2, j % 2].scatter(distance, row)
                    axes[j/4][(j/2) % 2, j % 2].set_xlabel("Distance to minimum")
                    axes[j/4][(j/2) % 2, j % 2].set_ylabel("Eigenvector %d" % (j+1))
                    plt.savefig(os.path.join(outputFolder, "%s_eigenvector_%d.png" % (runFolder, j+1)))

            if plotPMF:
                data = np.loadtxt(os.path.join(runFolder, "%dlag" % tau, "%dcl" % k, "MSM_0", "pmf_xyzg_0.dat"))
                g = data[:, -1]
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
if plotEigenvectors or plotGMRQ or plotPMF:
    plt.show()
