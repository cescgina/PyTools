from __future__ import print_function
import os
import argparse
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from AdaptivePELE.utilities import utilities
plt.style.use("ggplot")


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Calculate rmsd from pdbs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("path", type=str, help="Path where the files are stored")
    parser.add_argument("trajectories", type=int, default=4, help="Number of trajectories per epoch")
    parser.add_argument("--traj_name", type=str, default="trajectory.xtc", help="Name of the trajectory files, including the format extension, default is trajectory.xtc")
    parser.add_argument("--plot_name", type=str, default=None, help="Name of the output file to save")
    args = parser.parse_args()
    return args.path, args.trajectories, args.traj_name, args.plot_name


def merge_atom_sets(protein, water, cu, traj):
    atoms = protein.tolist()+cu.tolist()
    water_set = set()
    for w in water:
        for at in traj.top.atom(w).residue.atoms:
            water_set.add(at.index)
    return sorted(atoms+list(water_set))


def main(sim_path, n_trajs, trajectory_name, plot_name):
    ref = md.load(os.path.join(sim_path, "topologies", "topology_0.pdb"))
    ref.remove_solvent(inplace=True)
    labels = []
    selections = []
    for res in ref.top.residues:
        if res.is_protein:
            labels.append("%s%d" % (res.code, res.resSeq))
            selections.append(ref.top.select("protein and symbol != 'H' and residue %d" % res.resSeq))
    selection = ref.top.select("protein and symbol != 'H'")
    avg_xyz = None
    global_traj = None
    trajectory_name = "_%d".join(os.path.splitext(trajectory_name))

    f1, ax1 = plt.subplots(1, 1)
    epochs = utilities.get_epoch_folders(sim_path)
    n_epochs = len(epochs)
    for epoch in epochs:
        with open(os.path.join(sim_path, epoch, "topologyMapping.txt")) as f:
            top_map = f.read().rstrip().split(":")
        for i in range(1, n_trajs+1):
            print("Processing epoch", epoch, "trajectory", i)
            trajectory = md.load(os.path.join(epoch, trajectory_name % i), top=os.path.join(sim_path, "topologies", "topology_%s.pdb" % top_map[i-1]))
            if global_traj is None:
                avg_xyz = np.mean(trajectory.xyz[:, selection, :], axis=0)
                global_traj = trajectory.remove_solvent()
            else:
                avg_xyz += np.mean(trajectory.xyz[:, selection, :], axis=0)
                global_traj += trajectory.remove_solvent()
            print(avg_xyz.shape)
    print(global_traj.xyz.shape)
    print(avg_xyz.shape)
    avg_xyz /= (n_epochs*n_trajs)
    rmsfs = []
    for i, ind in enumerate(selections):
        temp = 10*np.sqrt(3*np.mean((global_traj.xyz[:, ind, :] - avg_xyz[ind, :])**2, axis=(1, 2)))
        rmsfs.append(np.mean(temp))

    ax1.plot(rmsfs, 'x-')
    ax1.set_ylabel(r"RMSF ($\AA$)")
    f1.legend()
    if plot_name is not None:
        f1.savefig(plot_name)
    plt.show()

if __name__ == "__main__":
    in_path, num_trajs, traj_name, draw_plots = parse_arguments()
    main(in_path, num_trajs, traj_name, draw_plots)
