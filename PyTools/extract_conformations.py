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
    parser.add_argument("names", type=str, nargs='+', help="Names of the conformation to extract")
    args = parser.parse_args()
    return args.ligand_resname, set(args.names)


lig_resname, names = parse_arguments()
output_folder = "conformations/"
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
        print line[0], "=>", "epoch %s, trajectory %s, snapshot %s" % tuple(line[1:4])
        epoch, iTraj, nSnap = line[1:4]
        report = np.loadtxt("%s/report_%s" % (epoch, iTraj))
        snapshots = utilities.getSnapshots("%s/trajectory_%s.pdb" % (epoch, iTraj))
        pdb_obj = atomset.PDB()
        pdb_obj.initialise(snapshots[int(nSnap)], resname=lig_resname)
        pdb_obj.writePDB(output_folder+"conf_%s_%s_%s.pdb" % (epoch, iTraj, nSnap))
