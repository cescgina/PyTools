import numpy as np
import glob
import argparse
import shutil
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Bootstrap over the conformations to obtain an approximate idea of the intra-cluster diversity"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("n_conf", type=int, help="Number of conformations to select")
    parser.add_argument("-c", default=None, help="Path of the cluster centers")
    args = parser.parse_args()
    return args.ligand_resname, args.c, args.n_conf


def main(ligand, clusters_file, n_conformations):
    cluster_centers = np.loadtxt(clusters_file)
    numClusters = cluster_centers.shape[0]
    sizes = []
    distances = []
    distance_mean = []
    cl_ind = []
    for cl in xrange(numClusters):
        conf_files = glob.glob("cluster_%d/allStructures/conf_*.pdb" % cl)
        sizes.append(len(conf_files))
        positions = np.loadtxt("cluster_%d/positions.dat" % cl)[:, 1:]
        assert positions.shape[0] == len(conf_files), (cl, positions.shape[0], len(conf_files))
        dist = np.sqrt(np.sum((cluster_centers[cl]-positions)**2, axis=1))
        distance_mean.append(np.mean(dist))
        distances.extend(dist)
        cl_ind.extend([cl]*len(conf_files))
        selection = np.random.choice(conf_files, size=n_conformations)
        for f in selection:
            shutil.copy(f, "cluster_%d/" % cl)

    plt.plot(sizes, distance_mean, 'x')
    plt.ylabel("Distances (A)")
    plt.xlabel("Cluster size")
    plt.figure()
    plt.plot(cl_ind, distances, 'x', markersize=2)
    plt.ylabel("Distances (A)")
    plt.xlabel("Cluster number")
    plt.show()

if __name__ == "__main__":
    lig_name, clusters, num_conf = parse_arguments()
    main(lig_name, clusters, num_conf)
