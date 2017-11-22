import os
import glob
import numpy as np
from AdaptivePELE.testing import estimateDG
from AdaptivePELE.testing import cluster
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def writePDB(clusterCenters, autoCorrValue, title="clusters.pdb"):
    templateLine = "HETATM%s  H%sCLT L 502    %s%s%s  0.75%s           H\n"
    with open(title, 'w') as f:
        for i, line in enumerate(clusterCenters):
            number = str(i).rjust(5)
            number3 = str(i).ljust(3)
            x = ("%.3f" % line[0]).rjust(8)
            y = ("%.3f" % line[1]).rjust(8)
            z = ("%.3f" % line[2]).rjust(8)
            g = ("%.3f" % autoCorrValue[i]).rjust(8)

            f.write(templateLine % (number, number3, x, y, z, g))


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

lagtimes = range(1, 200, 20)
nLags = len(lagtimes)
nclusters = 100

parameters = estimateDG.Parameters(ntrajs=None,
                                   length=None,
                                   lagtime=1,
                                   nclusters=nclusters,
                                   nruns=1,
                                   skipFirstSteps=0,
                                   useAllTrajInFirstRun=True,
                                   computeDetailedBalance=True,
                                   trajWildcard="traj_*",
                                   folderWithTraj="allTrajs",
                                   lagtimes=[1, 10, 25, 50],
                                   clusterCountsThreshold=0)

if not os.path.exists("autoCorr.npy"):
    workingControlFile = "control_MSM.conf"
    nWorkingTrajs = None
    bootstrap = False
    origFilesWildcard = os.path.join(parameters.folderWithTraj, parameters.trajWildcard)
    estimateDG.__prepareWorkingControlFile(parameters.lagtime, parameters.nclusters, parameters.folderWithTraj, parameters.trajWildcard, workingControlFile, parameters.lagtimes, parameters.clusterCountsThreshold)
    copiedFiles = estimateDG.copyWorkingTrajectories(origFilesWildcard, parameters.length, nWorkingTrajs, bootstrap, parameters.skipFirstSteps)
    clusteringObject = cluster.Cluster(parameters.nclusters, parameters.folderWithTraj, parameters.trajWildcard, alwaysCluster=False)
    clusteringObject.clusterTrajectories()
    C = np.zeros((nclusters, nLags))
    Ci = np.zeros((nclusters, nLags))
    Cf = np.zeros((nclusters, nLags))
    autoCorr = np.zeros((nclusters, nLags))
    N = 0
    M = np.zeros(nLags)
    dtrajs = glob.glob("discretized/traj*")
    for trajectory in dtrajs:
        traj = np.loadtxt(trajectory, dtype=int)
        Nt = traj.size
        N += Nt
        for il, lagtime in enumerate(lagtimes):
            M[il] += Nt-lagtime
            for i in xrange(Nt-lagtime):
                autoCorr[traj[i], il] += (traj[i] == traj[i+lagtime])
                C[traj[i], il] += 1
                Ci[traj[i], il] += 1
                if i > lagtime:
                    Cf[traj[i], il] += 1
            for j in xrange(Nt-lagtime, Nt):
                C[traj[j], il] += 1
                Cf[traj[j], il] += 1

    mean = C/float(N)
    var = (N*C-(C**2))/float(N*(N-1))
    autoCorr += M*mean**2-(Ci+Cf)*mean
    autoCorr /= N
    autoCorr /= var
    np.save("autoCorr.npy", autoCorr)
    __cleanupFiles(parameters.trajWildcard, False)
else:
    autoCorr = np.load("autoCorr.npy")

clusterCenters = np.loadtxt("clusterCenters_2.dat")
writePDB(clusterCenters, autoCorr[:,-1])
# plt.imshow(autoCorr, extent=[0, lagtimes[-1], 0, nclusters])
# plt.colorbar()
print "Clusters with more than 0.2 autocorrelation"
size2 = np.where(autoCorr[:,-1] > 0.2)[0].size
print size2, size2 / float(nclusters)
print "Clusters with more than 0.1 autocorrelation"
size1 = np.where(autoCorr[:,-1] > 0.1)[0].size
print size1, size1 / float(nclusters)
threshold = 0.2

if threshold < 1:
    filtered = np.where(autoCorr[:,-1] > threshold)[0]
    print filtered
    print autoCorr[filtered,-1]
else:
    filtered = range(nclusters)
axes = plt.plot(lagtimes, autoCorr.T[:,filtered])
[ax.set_label("Cluster %d" % i) for i, ax in zip(filtered, axes)]
plt.legend()
plt.savefig("autoCorr_thres0-2.png")
plt.show()
