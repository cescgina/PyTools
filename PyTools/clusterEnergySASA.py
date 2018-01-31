import numpy as np
import glob
import os
from AdaptivePELE.utilities import utilities
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use("ggplot")

energyColumn = 3
SASAColumn = 6
norm_energy = False

n_clusters = 50
epochFolders = utilities.get_epoch_folders(".")
points = []
for epoch in epochFolders:
    report_files = glob.glob(os.path.join(epoch, "*report*"))
    report_files.sort(key=lambda x: int(x[x.rfind("_")+1:]))
    for report_name in report_files:
        report = np.loadtxt(report_name)
        if len(report.shape) < 2:
            points.append([report[energyColumn], report[SASAColumn]])
        else:
            points.extend(report[:, (energyColumn, SASAColumn)].tolist())
points = np.array(points)
if norm_energy:
    energyMin = points.min(axis=0)[0]
    points[:, 0] -= energyMin
    energyMax = points.max(axis=0)[0]
    points[:, 0] /= energyMax

kmeans = KMeans(n_clusters=n_clusters).fit(points)
plt.scatter(points[:, 1], points[:, 0], c=kmeans.labels_)
if norm_energy:
    plt.savefig("clusters_energy_normalized.png")
else:
    plt.savefig("clusters_no_normalized.png")
plt.show()
