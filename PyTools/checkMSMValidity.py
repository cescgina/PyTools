import numpy as np
import os
from AdaptivePELE.freeEnergies import estimate
from AdaptivePELE.utilities import utilities
import pyemma.plots as mplt
import matplotlib.pyplot as plt


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
                print lineContents
            cl = int(lineContents[1])
            coords = map(float, [lineContents[-6], lineContents[-5], lineContents[-4]])
            if beta < minVal:
                minVal = beta
                minCl = cl
                minCoords = np.array(coords)
    return minCl, minCoords


params = [(25, 200), (50, 200), (100, 200), (200, 200)]
plots = False
MSM_object = estimate.MSM()
MSM_object.lagtimes = [1, 50, 100, 200, 400, 600, 800, 1000]
for tau, k in params:
    path = "%dlag/%dcl/MSM_0/"
    print "********"
    print path % (tau, k)
    for i in xrange(10):
        temp_path = path % (tau, k)
        print "Plotting validity checks for run %d" % i
        MSM = utilities.readClusteringObject(temp_path + "MSM_object_%d.pkl" % i)
        assert len(MSM.dtrajs_full) == len(MSM.dtrajs_active)
        MSM_object.MSM_object = MSM
        MSM_object.dtrajs = MSM.dtrajs_full
        if not os.path.exists(temp_path + "ck_plots/"):
            os.makedirs(temp_path + "ck_plots/")
        if True or not os.path.exists(temp_path + "its_%d.png" % i):
            MSM_object._calculateITS()
            plt.savefig(temp_path + "its_%d.png" % i)
        # setting mlags to None chooses automatically the number of lagtimes
        # according to the longest trajectory available
        CK_test = MSM_object.MSM_object.cktest(5, mlags=None)
        figCK, foo = mplt.plot_cktest(CK_test)
        plt.savefig(temp_path + "ck_plots/" + "ck_%d.png" % i)
        if plots:
            plt.show()
        plt.close('all')
