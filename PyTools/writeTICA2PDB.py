from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
import glob
import argparse
import numpy as np
from AdaptivePELE.utilities import utilities


def sort_split_by_numbers(traj_name):
    trajNum, snapNum = traj_name.split("_")[-2:]
    return (int(trajNum), int(snapNum[:-4]))


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Write the time-structure based Independent Components (TICA) from a simulation into a pdb file"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("-stride", type=int, default=1, help="Stride, e.g. select one conformation out of every x, default 1, that is take all")
    args = parser.parse_args()
    return args.stride

if __name__ == "__main__":
    stride = parse_arguments()
    data_folder = "tica_COM/"
    COM_tica = []
    files = glob.glob(data_folder+"traj_*")
    files.sort(key=sort_split_by_numbers)
    for traj in files:
        traj_data = np.loadtxt(traj)
        if len(traj_data.shape) < 2:
            COM_tica.append(traj_data.tolist())
        else:
            COM_tica.extend(traj_data[::stride].tolist())

    COM_tica = np.array(COM_tica)
    # process each TICA so the beta value visualization works (shift to 0)
    COM_tica[:, 3:] -= np.min(COM_tica[:, 3:], axis=0)

    utilities.makeFolder("tica_pdb")
    nConf, nTICS = COM_tica.shape
    ind = [0, 1, 2, 0]
    for i in range(3, nTICS):
        ind[-1] = i
        utilities.write_PDB_clusters(COM_tica[:, ind], title="tica_pdb/tica_%d.pdb" % (i-2), use_beta=True)
