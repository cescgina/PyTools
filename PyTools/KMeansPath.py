from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from io import open
import numpy as np
import os
from sklearn.cluster import KMeans
from AdaptivePELE.utilities import utilities as clu
from AdaptivePELE.atomset import RMSDCalculator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse


def parseArgs():
    parser = argparse.ArgumentParser(description="Script that reclusters the Adaptive clusters")
    parser.add_argument('nClusters', type=int)
    parser.add_argument('clustering', type=str)
    parser.add_argument('-o', type=str, help="Output folder")
    args = parser.parse_args()
    return args.nClusters, args.clustering, args.o


def plot_new_clusters(COMArray, model):
    fig = plt.figure()
    ax = Axes3D(fig)
    ccx = COMArray[:, 0]
    ccy = COMArray[:, 1]
    ccz = COMArray[:, 2]
    scatter1 = ax.scatter(ccx, ccy, zs=ccz, c=model.labels_)
    fig.colorbar(scatter1)
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlabel('x')
    plt.show()


def writePDB(pmf_xyzg, title="clusters.pdb"):
    templateLine = "HETATM%s  H%sCLT L 502    %s%s%s  0.75%s           H\n"

    content = ""
    for i, line in enumerate(pmf_xyzg):
        number = str(i).rjust(5)
        number3 = str(i).ljust(3)
        x = ("%.3f" % line[0]).rjust(8)
        y = ("%.3f" % line[1]).rjust(8)
        z = ("%.3f" % line[2]).rjust(8)
        g = 0
        content += templateLine % (number, number3, x, y, z, g)

    f = open(title, 'w')
    f.write(content)
    f.close()


def first_cluster_as_representative(model):
    final_clusters = {}
    for i, cluster in enumerate(model.labels_):
        if cluster in final_clusters:
            continue
        else:
            final_clusters[cluster] = i
    return [final_clusters[j] for j in range(len(final_clusters))]


def most_populated_cluster_as_representative(model, clusters_population, n_clusters):
    final_clusters = []
    for cl_num in range(n_clusters):
        cluster_elements = np.where(model.labels_ == cl_num)
        index = np.argmax(clusters_population[cluster_elements])
        final_clusters.append(cluster_elements[0][index])
    return final_clusters


def closest_cluster_as_representative(model, n_clusters, COM):
    final_clusters = []
    for cl_num in range(n_clusters):
        cluster_elements = np.where(model.labels_ == cl_num)
        index = np.argmin(np.sqrt((model.cluster_centers_[cl_num]-COM[cluster_elements, :])**2).sum(axis=2))
        final_clusters.append(cluster_elements[0][index])
    return final_clusters


def calculate_intercluster_distance(clustering, model, cluster_centers, RMSDCalculator):
    RMSD_array = []
    for i, cluster in enumerate(cluster_centers):
        cluster_center = clustering.getCluster(cluster)
        RMSD_cluster = 0
        n = 0
        for cl in np.where(model.labels_ == i)[0]:
            if cluster == cl:
                continue
            else:
                RMSD_cluster += RMSDCalc.computeRMSD(clustering.getCluster(cl).pdb, cluster_center.pdb)
                n += 1
        RMSD_array.append(RMSD_cluster/float(n))
    return RMSD_array


n_clusters, clustering_object, output = parseArgs()
print("Reading clustering object")
cluster_object = clu.readClusteringObject(clustering_object)
clusters = [cl.pdb.getCOM() for cl in cluster_object.clusterIterator()]
clusters_pop = np.array([cl.elements for cl in cluster_object.clusterIterator()])
clusters_contacts = np.array([cl.contacts for cl in cluster_object.clusterIterator()])
COMArray = np.array(clusters)
print("Number of adaptive clusters", len(clusters))
model = KMeans(n_clusters=n_clusters)
print("Reclustering")
model.fit(clusters)

# RMSDCalc = RMSDCalculator.RMSDCalculator()
# clusters_first = first_cluster_as_representative(model)
# clusters_pop = most_populated_cluster_as_representative(model, clusters_pop, n_clusters)
# distances_first = calculate_intercluster_distance(cluster_object, model, clusters_first, RMSDCalc)
# distances_pop = calculate_intercluster_distance(cluster_object, model, clusters_pop, RMSDCalc)
# print("Total intercluster distance first", sum(distances_first))
# print("Total intercluster distance population", sum(distances_pop))
#
# plt.plot(distances_first, marker='x', label="First clusters")
# plt.plot(distances_pop, marker='o', label="Most populated clusters")
# plt.legend()
# plt.show()

# final_clusters = first_cluster_as_representative(model)
# final_clusters = most_populated_cluster_as_representative(model, clusters_pop, n_clusters)
final_clusters = closest_cluster_as_representative(model, n_clusters, COMArray)
# contacts = [clusters_contacts[cl] for cl in final_clusters]
# plt.hist(contacts)
# plt.show()
if output is not None:
    outputFolder = os.path.join(output, "")
    if not os.path.exists(outputFolder):
        os.makedirs(outputFolder)
else:
    outputFolder = ""
if not os.path.exists(outputFolder+"clusters_all_KMeans.pdb"):
    writePDB(COMArray, outputFolder+"clusters_all_KMeans.pdb")
writePDB(COMArray[final_clusters], outputFolder+"clusters_%d_KMeans.pdb" % n_clusters)
for i, cl_ind in enumerate(final_clusters):
    cluster_object.getCluster(cl_ind).writePDB(outputFolder+"initial_%d.pdb" % i)
