import numpy as np
import glob
import os
from AdaptivePELE.utilities import utilities
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use("ggplot")

energyColumn = 3
SASAColumn = 6
norm_energy = True
plots = True
num_bins = 10
percentile = 25.0
n_clusters = 40

epochFolders = utilities.get_epoch_folders(".")
points = []
for epoch in epochFolders:
    report_files = glob.glob(os.path.join(epoch, "*report*"))
    report_files.sort(key=lambda x: int(x[x.rfind("_")+1:]))
    for report_name in report_files:
        traj_num = int(report_name[report_name.rfind("_")+1:])
        report = np.loadtxt(report_name)
        if len(report.shape) < 2:
            points.append([report[energyColumn], report[SASAColumn], int(epoch), traj_num, 0])
        else:
            epoch_line = np.array([int(epoch)] * report.shape[0])
            traj_line = np.array([traj_num] * report.shape[0])
            snapshot_line = np.array(range(report.shape[0]))
            points.extend(np.hstack((report[:, (energyColumn, SASAColumn)], epoch_line[:, np.newaxis], traj_line[:, np.newaxis], snapshot_line[:, np.newaxis])))
points = np.array(points)

points = points[points[:, 1].argsort()]
minSASA = points[0, 1]
maxSASA = points[-1, 1]
left_bins, step = np.linspace(minSASA, maxSASA, num=num_bins, endpoint=False, retstep=True)
indices = np.searchsorted(points[:, 1], left_bins)
thresholds = np.array([np.percentile(points[i:j, 0], percentile) for i, j in zip(indices[:-1], indices[1:])])
print indices
print left_bins
print thresholds
new_points = []
occupation = []
for ij, (i, j) in enumerate(zip(indices[:-1], indices[1:])):
    found = np.where(points[i:j, 0] < thresholds[ij])[0]
    occupation.append(len(found))
    if len(found) == 1:
        new_points.append(points[found+i])
    elif len(found) > 1:
        new_points.extend(points[found+i])

print occupation
points = np.array(new_points)
if norm_energy:
    energyMin = points.min(axis=0)[0]
    points[:, 0] -= energyMin
    energyMax = points.max(axis=0)[0]
    points[:, 0] /= energyMax

kmeans = KMeans(n_clusters=n_clusters).fit(points[:, :2])
if not os.path.exists("clusters_energy"):
    os.makedirs("clusters_energy")
for i, center in enumerate(kmeans.cluster_centers_):
    dist = np.linalg.norm((points[:, :2]-center), axis=1)
    epoch, traj, snapshot = points[dist.argmin(), 2:]
    with open("clusters_energy/initial_%d.pdb" % i, "w") as fw:
        fw.write(utilities.getSnapshots("%d/trajectory_%d.pdb" % (epoch, traj))[int(snapshot)])
if plots:
    plt.scatter(points[:, 1], points[:, 0], c=kmeans.labels_, alpha=0.5)
    plt.scatter(kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 0], c=range(n_clusters), marker='x', s=56, zorder=1)
    plt.xlabel("SASA")
    if norm_energy:
        plt.ylabel("Energy (normalized)")
        plt.savefig("clusters_energy_normalized.png")
    else:
        plt.ylabel("Energy (kcal/mol)")
        plt.savefig("clusters_no_normalized.png")
    plt.show()
