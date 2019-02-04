from __future__ import print_function
import os
import glob
import argparse
import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Plot RMSD of a trajectory"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("path", type=str, help="Path to the trajectories to analyze")
    parser.add_argument("--top", type=str, help="Path to the topology to analyze")
    parser.add_argument("--out_path", type=str, default="", help="Path to write the output, default")
    parser.add_argument("--traj_name", type=str, default="trajectory", help="Name of the trajectory file (default trajectory)")
    parser.add_argument("--save_plots", action="store_true", help="Whether to store the generated plots")
    parser.add_argument("--show_plots", action="store_true", help="Whether to store the generated plots")
    args = parser.parse_args()
    return args.path, args.top, args.traj_name, args.save_plots, args.show_plots, args.out_path


def rmsd(traj, indices, ref_frame=0, ref_traj=None):
    if ref_traj is None:
        ref_traj = traj
    return 10*np.sqrt(3*np.mean(np.square(traj.xyz[:, indices]-ref_traj.xyz[ref_frame, indices]), axis=(1, 2)))


def main(traj_path, top_file, traj_name, output_path, save_plot, show_plot):
    trajectories = glob.glob(os.path.join(traj_path, "%s_*" % traj_name))
    t_ref = md.load(top_file)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for trajectory in trajectories:
        print("Processing trajectory %s" % trajectory)
        t = md.load(trajectory, top=top_file)
        indices_to_align = t.top.select("protein and element != H")
        rmsd_traj = rmsd(t, indices_to_align, ref_traj=t_ref)
        print("Max", np.max(rmsd_traj[1:]))
        print("Min", np.min(rmsd_traj[1:]))
        print("Mean", np.mean(rmsd_traj[1:]))
        print("Standard deviation", np.std(rmsd_traj[1:]))
        plt.figure()
        plt.plot(rmsd_traj)
        plt.xlabel("Frame")
        plt.ylabel("RMSD(A)")
        plt.title("Trajectory %s" % trajectory)
        if save_plot:
            plt.savefig(os.path.join(output_path, "rmsd_%s.png" % os.path.split(trajectory)[1]))
    if show_plot:
        plt.show()

if __name__ == "__main__":
    trajs, top, name, save, show, output = parse_arguments()
    main(trajs, top, name, output, save, show)
