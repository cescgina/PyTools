import numpy as np
import glob
import matplotlib.pyplot as plt
import AdaptivePELE.atomset.atomset as atomset
from mpl_toolkits.mplot3d import Axes3D
plt.style.use("ggplot")


def sort_split_by_numbers(traj_name):
    trajNum, snapNum = traj_name.split("_")[-2:]
    return (int(trajNum), int(snapNum[:-4]))


native = "../../1o3p/1o3p_native.pdb"
cluster_centers = "/home/jgilaber/urokinases_free_energy/1o3p_PELE_sampl_40_newPrep/clustering_1atom/100lag/100cl/MSM_0/clusterCenters_0.dat"

PDB = atomset.PDB()
PDB.initialise(native, atomname="CA", type="PROTEIN")
CA_coords = [PDB.atoms[atom].getAtomCoords() for atom in PDB.atomList]
CA_coords = np.array(CA_coords)

clusters = np.loadtxt(cluster_centers)

nTraj = 20
files = glob.glob("allTrajs/traj*")
files.sort(key=sort_split_by_numbers)
n_trajs = len(files)
n = n_trajs/nTraj + bool(n_trajs % nTraj)
for i in xrange(n):
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlabel('x')
    ax.set_title("Trajs from %d to %d" % (i*nTraj, (i+1)*nTraj))
    for trajFile in files[i*nTraj:(i+1)*nTraj]:
        traj = np.loadtxt(trajFile)
        ax.plot(traj[:, 1], traj[:, 2], zs=traj[:, 3], linewidth=0.25)
    ax.scatter(clusters[:, 0], clusters[:, 1], clusters[:, 2])
    ax.plot(CA_coords[:, 0], CA_coords[:, 1], zs=CA_coords[:, 2], marker='o',
            linewidth=0.25, markersize=0.25)
plt.show()
