import glob
import numpy as np
from AdaptivePELE.utilities import utilities
import matplotlib.pyplot as plt
plt.style.use("ggplot")

files = glob.glob("dihedrals/allTrajs/traj_0_*")
colors = ["r", "k", "b", "g", "y", "c", "m", "olive", "p"]
n = 9
for j, f in enumerate(files):
    plt.figure()
    data = np.rad2deg(utilities.loadtxtfile(f)[:, 1:])
    # terminal residues lack one of the dihedrals
    for i in range(n-2):
        plt.scatter(data[:, i+n-1], data[:, i+1], color=colors[i], marker=".", label="Residue %d" % (i+2))
    plt.xlim(-180, 180)
    plt.ylim(-180, 180)
    plt.xlabel(r'$\Phi$ Angle (degrees)')
    plt.ylabel(r'$\Psi$ Angle (degrees)')
    plt.title(f)
    plt.legend()
    plt.show()
