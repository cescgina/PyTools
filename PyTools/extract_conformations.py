import numpy as np
import os
import argparse
from AdaptivePELE.utilities import utilities
from AdaptivePELE.atomset import atomset


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
    args = parser.parse_args()
    return args.ligand_resname, args.metricCol, set(args.names), args.o


lig_resname, metricCol, names, output_folder = parse_arguments()
if not output_folder.endswith("/"):
    output_folder += "/"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

filename = "conformation_data.dat"
if not os.path.exists(filename):
    raise IOError("File conformation_data.dat not found, please be sure to run COM_BE.py before this script")
print names
with open(filename) as f:
    f.readline()
    for line in f:
        line = line.strip().split()
        if line[0] not in names:
            continue
        epoch, iTraj, nSnap = line[1:4]
        report = np.loadtxt("%s/report_%s" % (epoch, iTraj))
        print line[0], "=>", "epoch %s, trajectory %s, snapshot %s" % tuple(line[1:4]), "metric", report[int(nSnap), metricCol]
        snapshots = utilities.getSnapshots("%s/trajectory_%s.pdb" % (epoch, iTraj))
        pdb_obj = atomset.PDB()
        pdb_obj.initialise(snapshots[int(nSnap)], resname=lig_resname)
        pdb_obj.writePDB(output_folder+"conf_%s_%s_%s.pdb" % (epoch, iTraj, nSnap))
