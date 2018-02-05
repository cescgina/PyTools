""" Write a pdb with each conformation and the specific metric as beta value,
    also write a file with the name in the pdb, the conformation info and the
    COM"""
import numpy as np
import argparse
import itertools
from AdaptivePELE.utilities import utilities
from PyTools.tica import get_coords


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Extract metric and COM information for the conformations obtained from a simulation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("metricCol", type=int, help="Column of the metric to consider")
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("nTraj", type=int, help="Number of trajectories per epoch")
    parser.add_argument("-filter", type=float, default=None, help="Filter the maximum value of the metric for visualization")
    parser.add_argument("-stride", type=int, default=1, help="Stride, e.g. select one conformation out of every x, default 1, that is take all")
    parser.add_argument("-atomId", type=str, default="", help="Atoms to user for the coordinates of the conformation, if not specified use the center of mass")
    parser.add_argument("-s", "-savingFreq", type=int, default=1, help="Saving frequency of PELE simulation")
    parser.add_argument("-t", "-trajectoryName", type=str, default="trajectory", help="Name of the trajectory files, e.g for trajectory_1.pdb the name is trajectory, default is trajectory")
    parser.add_argument("-r", "-reportName", type=str, default="report", help="Name of the report files, e.g for report_1.pdb the name is report, default is report")
    args = parser.parse_args()
    return args.metricCol, args.ligand_resname, args.nTraj, args.filter, args.stride, args.atomId, args.s, args.t, args.r


def main(metricCol, lig_resname, nTrajs, filter_val, stride, atomId, saving_frequency, trajectory_name, report_name):
    folders = utilities.get_epoch_folders(".")
    data = []
    minMetric = 1e6
    confData = []
    for epoch in folders:
        print "Processing epoch %s" % epoch
        for iTraj in xrange(1, nTrajs):
            report = np.loadtxt("%s/%s_%d" % (epoch, report_name, iTraj))
            if len(report.shape) < 2:
                report = report[np.newaxis, :]
            snapshots = utilities.getSnapshots("%s/%s_%d.pdb" % (epoch, trajectory_name, iTraj))
            for i, snapshot in enumerate(itertools.islice(snapshots, 0, None, stride)):
                report_line = i * stride * saving_frequency
                data.append(get_coords(snapshot, atomId, lig_resname) + [report[report_line, metricCol]])
                confData.append((epoch, iTraj, report_line))

    data = np.array(data)
    minInd = np.argmin(data[:, -1])
    minMetric = data[minInd, -1]
    data[:, -1] -= minMetric
    if filter_val is not None:
        data_filter = data.copy()
        data_filter[data_filter > filter_val] = filter_val
        namesPDB = utilities.write_PDB_clusters(data_filter, title="cluster_metric.pdb", use_beta=True)
    else:
        namesPDB = utilities.write_PDB_clusters(data, title="cluster_metric.pdb", use_beta=True)
    print "Min value for metric", minMetric, namesPDB[minInd]

    with open("conformation_data.dat", "w") as fw:
        fw.write("PDB name      Epoch Trajectory   Snapshot   COM x       y       x     Metric\n")
        for j, name in enumerate(namesPDB):
            info = [name.rjust(8)]+[str(x).rjust(10) for x in confData[j]]+[str(np.round(d, 3)).rjust(7) for d in data[j, :-1]] + [str(np.round(data[j, -1], 2)).rjust(10)]
            fw.write("{:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s}\n".format(*tuple(info)))

if __name__ == "__main__":
    metric_col, ligand, n_trajs, filter_value, stride_val, atom_ids, save_freq, traj_name, rep_name = parse_arguments()
    main(metric_col, ligand, n_trajs, filter_value, stride_val, atom_ids, save_freq, traj_name, rep_name)
