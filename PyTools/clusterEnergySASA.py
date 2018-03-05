import numpy as np
import glob
import os
import argparse
from AdaptivePELE.utilities import utilities
from AdaptivePELE.freeEnergies import extractCoords
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def parseArgs():
    parser = argparse.ArgumentParser(description="Script that clusters the conformation from an Adaptive run")
    parser.add_argument('nClusters', type=int)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("SASA", type=int, help="Column of the SASA in report file")
    parser.add_argument("-atomId", nargs="*", default="", help="Atoms to use for the coordinates of the conformation, if not specified use the center of mass")
    parser.add_argument('-o', type=str, help="Output folder")
    parser.add_argument('-f', type=str, default=".", help="Trajectory folder")
    parser.add_argument("-norm_energy", action="store_false", help="Wether to normalize energy values in clustering (default True)")
    parser.add_argument("-bins", type=int, default=10, help="Number of bins to discretize the SASA values")
    parser.add_argument("-p", "--percentile", type=float, default=25.0, help="Percentile of the energy values per bin to filter")
    parser.add_argument("--plot", action="store_true", help="Wether to plot clusters with respect to SASA and energy values")
    parser.add_argument("-trajname", type=str, default="trajectory", help="Basename of the trajctory file, i.e for run_traj_1.pdb pass run_traj, default is trajectory")
    parser.add_argument("-energycluster", action="store_true", help="Wether to use the energy to get the clusters, default is false, uses the coordiantes")
    args = parser.parse_args()
    return args.nClusters, args.ligand_resname, args.atomId, args.o, args.f, args.SASA, args.norm_energy, args.bins, args.percentile, args.plot, args.trajname, args.energycluster


def writePDB(pmf_xyzg, title="clusters.pdb"):
    templateLine = "HETATM%s  H%sCLT L 502    %s%s%s  0.75%s           H\n"

    content = ""
    for j, line in enumerate(pmf_xyzg):
        number = str(j).rjust(5)
        number3 = str(j).ljust(3)
        x = ("%.3f" % line[0]).rjust(8)
        y = ("%.3f" % line[1]).rjust(8)
        z = ("%.3f" % line[2]).rjust(8)
        g = 0
        content += templateLine % (number, number3, x, y, z, g)

    with open(title, 'w') as f:
        f.write(content)


def main(n_clusters, output_folder, SASAColumn, norm_energy, num_bins,
         percentile, plots, atom_Ids, folder_name, traj_basename,
         cluster_energy):
    energyColumn = 3

    if output_folder is not None:
        outputFolder = os.path.join(output_folder, "")
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    else:
        outputFolder = ""

    extractCoords.main(folder_name, lig_resname=ligand_resname, non_Repeat=True, atom_Ids=atom_Ids)

    epochFolders = utilities.get_epoch_folders(folder_name)
    points = []
    for epoch in epochFolders:
        report_files = glob.glob(os.path.join(epoch, "*report*"))
        report_files.sort(key=lambda x: int(x[x.rfind("_")+1:]))
        for report_name in report_files:
            traj_num = int(report_name[report_name.rfind("_")+1:])
            coordinates = np.loadtxt(os.path.join(folder_name, "%s/extractedCoordinates/coord_%d.dat" % (epoch, traj_num)))
            report = np.loadtxt(report_name)
            if len(report.shape) < 2:
                points.append([report[energyColumn], report[SASAColumn], int(epoch), traj_num, 0]+coordinates[1:].tolist())
            else:
                epoch_line = np.array([int(epoch)] * report.shape[0])
                traj_line = np.array([traj_num] * report.shape[0])
                snapshot_line = np.array(range(report.shape[0]))
                points.extend(np.hstack((report[:, (energyColumn, SASAColumn)], epoch_line[:, np.newaxis], traj_line[:, np.newaxis], snapshot_line[:, np.newaxis], coordinates[:, 1:])))
    points = np.array(points)
    points = points[points[:, 1].argsort()]
    minSASA = points[0, 1]
    maxSASA = points[-1, 1]
    left_bins = np.linspace(minSASA, maxSASA, num=num_bins, endpoint=False)
    indices = np.searchsorted(points[:, 1], left_bins)
    thresholds = np.array([np.percentile(points[i:j, 0], percentile) for i, j in zip(indices[:-1], indices[1:])])

    new_points = []
    occupation = []
    for ij, (i, j) in enumerate(zip(indices[:-1], indices[1:])):
        found = np.where(points[i:j, 0] < thresholds[ij])[0]
        occupation.append(len(found))
        if len(found) == 1:
            new_points.append(points[found+i])
        elif len(found) > 1:
            new_points.extend(points[found+i])

    points = np.array(new_points)
    if norm_energy:
        energyMin = points.min(axis=0)[0]
        points[:, 0] -= energyMin
        energyMax = points.max(axis=0)[0]
        points[:, 0] /= energyMax

    if cluster_energy:
        print "Clustering using energy and SASA"
        kmeans = KMeans(n_clusters=n_clusters).fit(points[:, :2])
        title = "clusters_%d_energy_SASA.pdb"
    else:
        print "Clustering using ligand coordinates"
        kmeans = KMeans(n_clusters=n_clusters).fit(points[:, 5:8])
        title = "clusters_%d_energy_SASA_coords.pdb"
    centers_energy = []
    centers_coords = []
    for i, center in enumerate(kmeans.cluster_centers_):
        if cluster_energy:
            dist = np.linalg.norm((points[:, :2]-center), axis=1)
        else:
            dist = np.linalg.norm((points[:, 5:8]-center), axis=1)
        epoch, traj, snapshot = points[dist.argmin(), 2:5]
        centers_energy.append(points[dist.argmin(), :2])
        centers_coords.append(points[dist.argmin(), 5:8])
        with open(os.path.join(outputFolder, "initial_%d.pdb" % i), "w") as fw:
            fw.write(utilities.getSnapshots("%d/%s_%d.pdb" % (epoch, traj_basename, traj))[int(snapshot)])
    centers_energy = np.array(centers_energy)
    centers_coords = np.array(centers_coords)
    writePDB(centers_coords, os.path.join(outputFolder, title % n_clusters))
    if plots:
        plt.scatter(points[:, 1], points[:, 0], c=kmeans.labels_, alpha=0.5)
        plt.scatter(centers_energy[:, 1], centers_energy[:, 0], c=range(n_clusters), marker='x', s=56, zorder=1)
        plt.xlabel("SASA")
        if norm_energy:
            plt.ylabel("Energy (normalized)")
            plt.savefig(os.path.join(outputFolder, "clusters_energy_normalized.png"))
        else:
            plt.ylabel("Energy (kcal/mol)")
            plt.savefig(os.path.join(outputFolder, "clusters_no_normalized.png"))
        plt.show()

if __name__ == "__main__":
    nClusters, ligand_resname, atomId, output, traj_folder, SASA, normal_energy, n_bins, percent, plot, traj_Basename, cluster_energy = parseArgs()
    main(nClusters, output, SASA, normal_energy, n_bins, percent, plot, atomId, traj_folder, traj_Basename, cluster_energy)
