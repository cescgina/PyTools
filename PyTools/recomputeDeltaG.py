from __future__ import absolute_import, division, print_function, unicode_literals
import os
import shutil
import argparse
import numpy as np
from AdaptivePELE.utilities import utilities
from AdaptivePELE.freeEnergies import computeDeltaG as DG


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Recompute DG"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("path", type=str, help="Path to the folder with the MSM data")
    parser.add_argument("output", default=None, help="Path of the folder where to store the new results")
    parser.add_argument("-n", "--nRuns", type=int, default=1, help="Number of independent calculations to plot")
    parser.add_argument("--divide_volume", action="store_true", help="Wether to divide the probability by the volumes")
    args = parser.parse_args()
    return args.nRuns, args.output, args.path, args.divide_volume


def main(nRuns, output, path, divide_volume):
    utilities.makeFolder(output)
    MSM_template = os.path.join(path, "MSM_object_%d.pkl")
    volumes_template = os.path.join(path, "volumeOfClusters_%d.dat")
    clusters_template = os.path.join(path, "clusterCenters_%d.dat")
    output_template = os.path.join(output, "pmf_xyzg_%d.dat")
    output_MSM_template = os.path.join(output, "MSM_object_%d.pkl")
    output_volumes_template = os.path.join(output, "volumeOfClusters_%d.dat")
    output_clusters_template = os.path.join(output, "clusterCenters_%d.dat")
    deltaGs = []

    for i in range(nRuns):
        print("Running iterations %d" % i)
        MSM = MSM_template % i
        volumes = volumes_template % i
        clusters = clusters_template % i
        allClusters = np.loadtxt(clusters)
        microstateVolume = np.loadtxt(volumes)
        MSMObject = DG.loadMSM(MSM)
        pi, cluster_centers = DG.ensure_connectivity(MSMObject, allClusters)
        gpmf, string = DG.calculate_pmf(microstateVolume, pi, divide_volume=divide_volume)
        print(string)
        deltaGs.append(string)

        pmf_xyzg = np.hstack((cluster_centers, np.expand_dims(gpmf, axis=1)))
        np.savetxt(output_template % i, pmf_xyzg)
        shutil.copy(MSM, output_MSM_template % i)
        shutil.copy(volumes, output_volumes_template % i)
        shutil.copy(clusters, output_clusters_template % i)

    arr_dG = [float(line.split()[1]) for line in deltaGs]
    with open(os.path.join(output, "results_summary.txt"), "w") as fw:
        fw.write("\n")
        fw.write("=====\n")
        fw.write("dG\n")
        fw.write("bound    Delta G     Delta W     Binding Volume:     Binding Volume contribution\n")
        for el in deltaGs:
            fw.write("%s\n" % el)
        fw.write("=====\n")
        fw.write("dG = %f +- %f\n" % (np.mean(arr_dG), np.std(arr_dG)))

if __name__ == "__main__":
    n, out, folder, divide = parse_arguments()
    main(n, out, folder, divide)
