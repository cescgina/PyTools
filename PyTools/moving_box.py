from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from io import open
import numpy as np
import argparse
import itertools
from AdaptivePELE.utilities import utilities
from PyTools.tica import get_coords


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Get the maximum SASA value and select the box center accordingly"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("metricCol", type=int, help="Column of the metric to consider")
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("nTraj", type=int, help="Number of trajectories per epoch")
    parser.add_argument("-stride", type=int, default=1, help="Stride, e.g. select one conformation out of every x, default 1, that is take all")
    parser.add_argument("-atomId", type=str, default="", help="Atoms to user for the coordinates of the conformation, if not specified use the center of mass")
    parser.add_argument("-s", "-savingFreq", type=int, default=1, help="Saving frequency of PELE simulation")
    args = parser.parse_args()
    return args.metricCol, args.ligand_resname, args.nTraj, args.stride, args.atomId, args.s


def main(metricCol, lig_resname, nTrajs, stride, atomId, saving_frequency):
    folders = utilities.get_epoch_folders(".")
    box_center = None
    templateLine = "HETATM%s    H BOX Z 501    %s%s%s  0.75%s            H  \n"
    for epoch in folders:
        print("Processing epoch %s" % epoch)
        data = []
        confData = []
        maxEpoch = -1
        maxEpochCoords = None
        for iTraj in range(1, nTrajs):
            report = np.loadtxt("%s/report_%d" % (epoch, iTraj))
            if len(report.shape) < 2:
                report = report[np.newaxis, :]
            maxTrajIndex = np.argmax(report[:, metricCol])
            snapshots = utilities.getSnapshots("%s/trajectory_%d.pdb" % (epoch, iTraj))
            for i, snapshot in enumerate(itertools.islice(snapshots, 0, None, stride)):
                report_line = i * stride * saving_frequency
                data.append(get_coords(snapshot, atomId, lig_resname) + [report[report_line, metricCol]])
                confData.append((epoch, iTraj, report_line))
            if report[maxTrajIndex, metricCol] > maxEpoch:
                maxEpoch = report[maxTrajIndex, metricCol]
                maxEpochCoords = get_coords(snapshots[maxTrajIndex], atomId, lig_resname)
            if box_center is None and iTraj == 1:
                box_center = data[0][:3]
        data = np.array(data)
        minInd = np.argmin(data[:, -1])
        minMetric = data[minInd, -1]
        data[:, -1] -= minMetric
        utilities.write_PDB_clusters(data, title="epoch_%s.pdb" % epoch, use_beta=True)
        print("Max value for metric", maxEpoch, maxEpochCoords)
        with open("epoch_%s.pdb" % epoch, "a") as fa:
            fa.write("TER\n")
            serial = ("%d" % data.shape[0]).rjust(5)
            x = ("%.3f" % box_center[0]).rjust(8)
            y = ("%.3f" % box_center[1]).rjust(8)
            z = ("%.3f" % box_center[2]).rjust(8)
            g = ("%.2f" % 0).rjust(6)
            fa.write(templateLine % (serial, x, y, z, g))
        box_center = maxEpochCoords

    # with open("conformation_data.dat", "w") as fw:
    #     fw.write("PDB name      Epoch Trajectory   Snapshot   COM x       y       x     Metric\n")
    #     for j, name in enumerate(namesPDB):
    #         info = [name.rjust(8)]+[str(x).rjust(10) for x in confData[j]]+[str(np.round(d, 3)).rjust(7) for d in data[j, :-1]] + [str(np.round(data[j, -1], 2)).rjust(10)]
    #         fw.write("{:s} {:s} {:s} {:s} {:s} {:s} {:s} {:s}\n".format(*tuple(info)))

if __name__ == "__main__":
    metric_col, ligand, n_trajs, stride_val, atom_ids, save_freq = parse_arguments()
    main(metric_col, ligand, n_trajs, stride_val, atom_ids, save_freq)
