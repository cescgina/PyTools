from __future__ import print_function
import os
import glob
import argparse
import numpy as np
import pyemma.plots as mplt
import matplotlib.pyplot as plt
from AdaptivePELE.utilities import utilities


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Plot 2-D PMF"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("path", type=str, help="Path to the folder with the MSM data")
    parser.add_argument("--path_save", type=str, default=None, help="Path to the folder with the MSM data")
    args = parser.parse_args()
    return args.path, args.path_save


def main(path_files, path_to_save):
    files = glob.glob(os.path.join(path_files, "traj_*.dat"))
    Y = []
    for f in files:
        Y.append(utilities.loadtxtfile(f))

    Y_stack = np.vstack(Y)
    xall = Y_stack[:, 1]
    yall = Y_stack[:, 2]
    zall = Y_stack[:, 3]

    plt.figure(figsize=(8, 5))
    mplt.plot_free_energy(xall, yall, cmap="Spectral")
    plt.xlabel("x")
    plt.ylabel("y")
    if path_to_save is not None:
        plt.savefig(os.path.join(path_to_save, "x-y_decomposition.png"))

    plt.figure(figsize=(8, 5))
    mplt.plot_free_energy(xall, zall, cmap="Spectral")
    plt.xlabel("x")
    plt.ylabel("z")
    if path_to_save is not None:
        plt.savefig(os.path.join(path_to_save, "x-z_decomposition.png"))

    plt.figure(figsize=(8, 5))
    mplt.plot_free_energy(yall, zall, cmap="Spectral")
    plt.xlabel("y")
    plt.ylabel("z")
    if path_to_save is not None:
        plt.savefig(os.path.join(path_to_save, "y-z_decomposition.png"))
    plt.show()


if __name__ == "__main__":
    path, save_path = parse_arguments()
    main(path, save_path)
