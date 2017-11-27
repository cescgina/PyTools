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
    args = parser.parse_args()
    return args.metricCol, args.ligand_resname, args.nTraj


metricCol, lig_resname, nTrajs = parse_arguments()

folders = utilities.get_epoch_folders(".")
data = []
minMetric = 1e6
confData = []
for epoch in folders:
    for iTraj in xrange(1, nTrajs):
        report = np.loadtxt("%s/report_%d" % (epoch, iTraj))
        minMetric = min(minMetric, report[:, metricCol].min())
        snapshots = utilities.getSnapshots("%s/trajectory_%d.pdb" % (epoch, iTraj))
        for i, snapshot in enumerate(snapshots):
            pdb_obj = atomset.PDB()
            pdb_obj.initialise(snapshot, resname=lig_resname)
            data.append(pdb_obj.getCOM() + [report[i, metricCol]])
            confData.append((epoch, iTraj, i))

data = np.array(data)
print "Min value for metric", minMetric
data[:, -1] -= minMetric
namesPDB = utilities.write_PDB_clusters(data, title="cluster_BE.pdb", use_beta=True)
with open("conformation_data.dat", "w") as fw:
    fw.write("PDB name\tEpoch\tTrajectory\tSnapshot\tCOM x\t y\t x\tMetric\n")
    for j, name in enumerate(namesPDB):
        info = [name]+[x for x in confData[j]]+[d for d in data[j]]
        fw.write("{:s}\t{:s}\t{:d}\t{:d}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\n".format(*tuple(info)))
