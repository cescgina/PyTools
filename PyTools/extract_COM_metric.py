""" Write a pdb with each conformation and the specific metric as beta value,
    also write a file with the name in the pdb, the conformation info and the
    COM"""
import numpy as np
import argparse
from AdaptivePELE.utilities import utilities
from AdaptivePELE.atomset import atomset


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
    args = parser.parse_args()
    return args.metricCol, args.ligand_resname, args.nTraj, args.filter


metricCol, lig_resname, nTrajs, filter_val = parse_arguments()

folders = utilities.get_epoch_folders(".")
data = []
minMetric = 1e6
confData = []
for epoch in folders:
    print "Processing epoch %s" % epoch
    for iTraj in xrange(1, nTrajs):
        report = np.loadtxt("%s/report_%d" % (epoch, iTraj))
        if len(report.shape) < 2:
            report = report[np.newaxis, :]
        snapshots = utilities.getSnapshots("%s/trajectory_%d.pdb" % (epoch, iTraj))
        for i, snapshot in enumerate(snapshots):
            pdb_obj = atomset.PDB()
            pdb_obj.initialise(snapshot, resname=lig_resname)
            data.append(pdb_obj.getCOM() + [report[i, metricCol]])
            confData.append((epoch, iTraj, i))

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
