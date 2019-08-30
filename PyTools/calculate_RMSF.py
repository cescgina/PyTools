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
    parser.add_argument("--residues", type=str, nargs="*", default=None, help="Number of the residues for which to calculate the RMSF")
    args = parser.parse_args()
    return args.path, args.trajectories, args.traj_name, args.plot_name, args.residues


def parse_residues(res_input):
    if res_input is None:
        return res_input
    residues = set()
    for res_str in res_input:
        if "-" in res_str:
            start, stop = [int(l) for l in res_str.split("-")]
            residues.update(range(start, stop+1))
        else:
            residues.add(int(res_str))
    return residues


def main(sim_path, n_trajs, trajectory_name, plot_name, residues_selected):
    # since we remove the water molecules, any topology file will be fine
    ref = md.load(os.path.join(sim_path, "topologies", "topology_0.pdb"))
    ref.remove_solvent(inplace=True)
    labels = []
    selections = []
    for res in ref.top.residues:
        if res.is_protein and (residues_selected is None or res.resSeq in residues_selected):
            if residues_selected is not None:
                residues_selected.remove(res.resSeq)
            labels.append("%s%d" % (res.code, res.resSeq))
            selections.append(ref.top.select("protein and symbol != 'H' and residue %d" % res.resSeq))
    if residues_selected is not None and len(residues_selected):
        raise ValueError("Residues %s not found in protein!" % ", ".join(sorted([str(x) for x in residues_selected])))
    if not os.path.exists("rmsf.npy"):
        avg_xyz = None
        global_traj = None
        trajectory_name = "_%d".join(os.path.splitext(trajectory_name))

        epochs = utilities.get_epoch_folders(sim_path)
        n_epochs = len(epochs)
        for epoch in epochs:
            with open(os.path.join(sim_path, epoch, "topologyMapping.txt")) as f:
                top_map = f.read().rstrip().split(":")
            for i in range(1, n_trajs+1):
                print("Processing epoch", epoch, "trajectory", i)
                trajectory = md.load(os.path.join(epoch, trajectory_name % i), top=os.path.join(sim_path, "topologies", "topology_%s.pdb" % top_map[i-1]))
                if global_traj is None:
                    avg_xyz = np.mean(trajectory.xyz, axis=0)
                    global_traj = trajectory.remove_solvent()
                else:
                    avg_xyz += np.mean(trajectory.xyz, axis=0)
                    global_traj += trajectory.remove_solvent()
        avg_xyz /= (n_epochs*n_trajs)
        rmsfs = []
        for i, ind in enumerate(selections):
            temp = 10*np.sqrt(3*np.mean((global_traj.xyz[:, ind, :] - avg_xyz[ind, :])**2, axis=(1, 2)))
            rmsfs.append(np.mean(temp))
        np.save("rmsf.npy", rmsfs)
    else:
        rmsfs = np.load("rmsf.npy")
    f1, ax1 = plt.subplots(1, 1)
    x_vals = np.array(range(len(labels)))
    ax1.plot(rmsfs, 'x-')
    ax1.set_xticks(x_vals)
    ax1.set_ylabel(r"RMSF ($\AA$)")
    ax1.set_xticklabels(labels)
    ax1.tick_params(axis='x', rotation=90, labelsize=10)
    if plot_name is not None:
        f1.savefig(plot_name)
    plt.show()

if __name__ == "__main__":
    in_path, num_trajs, traj_name, draw_plots, selected_res = parse_arguments()
    selected_res = parse_residues(selected_res)
    main(in_path, num_trajs, traj_name, draw_plots, selected_res)
