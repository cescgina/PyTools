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
    parser.add_argument("residues", type=str, default=None, help="Atom pair to calculate the distances in the format res_name1:res_number1:atomname1-res_name2:res_number2:atomname2")
    parser.add_argument("--traj_name", type=str, default="trajectory.xtc", help="Name of the trajectory files, including the format extension, default is trajectory.xtc")
    parser.add_argument("--plot_name", type=str, default=None, help="Name of the output file to save")
    args = parser.parse_args()
    return args.path, args.trajectories, args.traj_name, args.plot_name, args.residues


def parse_selection(res_input):
    residues = res_input.split("-")
    return tuple([tuple(res.split(":")) for res in residues])


def main(sim_path, n_trajs, trajectory_name, plot_name, residues_selected):
    # since we remove the water molecules, any topology file will be fine
    info1, info2 = parse_selection(selected_res)
    cache_file = "distances.npy"
    if not os.path.exists(cache_file):
        global_traj = None
        trajectory_name = "_%d".join(os.path.splitext(trajectory_name))

        epochs = utilities.get_epoch_folders(sim_path)
        for epoch in epochs:
            with open(os.path.join(sim_path, epoch, "topologyMapping.txt")) as f:
                top_map = f.read().rstrip().split(":")
            for i in range(1, n_trajs+1):
                print("Processing epoch", epoch, "trajectory", i)
                trajectory = md.load(os.path.join(epoch, trajectory_name % i), top=os.path.join(sim_path, "topologies", "topology_%s.pdb" % top_map[i-1]))
                if global_traj is None:
                    global_traj = trajectory.remove_solvent()
                    atom1 = global_traj.top.select("resname '%s' and residue %s and name %s" % info1)
                    atom2 = global_traj.top.select("resname '%s' and residue %s and name %s" % info2)
                    if atom1.size == 0 or atom2.size == 0:
                        raise ValueError("Nothing found under current selection")
                else:
                    global_traj += trajectory.remove_solvent()
        distance = 10*md.compute_distances(global_traj, [atom1.tolist()+atom2.tolist()])
        np.save(cache_file, distance)
    else:
        distance = np.load(cache_file)
    f1, ax1 = plt.subplots(1, 1)
    ax1.plot(distance, 'x-')
    ax1.set_ylabel(r"Distance %s ($\AA$)" % residues_selected)
    if plot_name is not None:
        f1.savefig(plot_name)
    plt.show()

if __name__ == "__main__":
    in_path, num_trajs, traj_name, draw_plots, selected_res = parse_arguments()
    main(in_path, num_trajs, traj_name, draw_plots, selected_res)
