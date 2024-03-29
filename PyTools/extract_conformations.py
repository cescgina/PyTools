from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from io import open
import numpy as np
import os
import glob
import argparse
from AdaptivePELE.utilities import utilities
try:
    basestring
except NameError:
    basestring = str


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Write certain conformations specified from a extract_COM_metric.py pdb"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("metricCol", type=int, help="Column of the metric to consider")
    parser.add_argument("names", type=str, nargs='+', help="Names of the conformation to extract")
    parser.add_argument("-o", type=str, default="conformations", help="Output path to write the structures")
    parser.add_argument("-t", "-trajectoryName", type=str, default="trajectory", help="Name of the trajectory files, e.g for trajectory_1.pdb the name is trajectory")
    parser.add_argument("-top", type=str, default=None, help="Topology file for non-pdb trajectories")
    args = parser.parse_args()
    return args.ligand_resname, args.metricCol, set(args.names), args.o, args.t, args.top


lig_resname, metricCol, names, output_folder, traj_name, topology = parse_arguments()
if not output_folder.endswith("/"):
    output_folder += "/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if topology is not None:
    topology_contents = utilities.getTopologyFile(topology)
else:
    topology = None
filename = "conformation_data.dat"
if not os.path.exists(filename):
    raise IOError("File conformation_data.dat not found, please be sure to run extract_COM_metric.py before this script")
print("Selected names: ", ' '.join(names))
with open(filename) as f:
    f.readline()
    for line in f:
        line = line.strip().split()
        if line[0] not in names:
            continue
        epoch, iTraj, nSnap = line[1:4]
        report = np.loadtxt("%s/report_%s" % (epoch, iTraj))
        print(line[0], "=>", "epoch %s, trajectory %s, snapshot %s" % tuple(line[1:4]), "metric", report[int(nSnap), metricCol])
        traj_file = glob.glob("%s/%s_%s*" % (epoch, traj_name, iTraj))
        snapshots = utilities.getSnapshots(traj_file, topology=topology)
        if isinstance(snapshots[int(nSnap)], basestring):
            with open(output_folder+"conf_%s_%s_%s.pdb" % (epoch, iTraj, nSnap), "w") as fw:
                fw.write(snapshots[int(nSnap)])
        else:
            utilities.write_mdtraj_object_PDB(snapshots[int(nSnap)], output_folder+"conf_%s_%s_%s.pdb" % (epoch, iTraj, nSnap), topology_contents)
