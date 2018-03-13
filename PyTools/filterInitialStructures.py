from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import range
from io import open
import glob
import argparse
from AdaptivePELE.atomset import RMSDCalculator, atomset


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Filter the initial structures"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("resname", help="Ligand resname in the pdb")
    parser.add_argument("native", help="Path to native structure")
    parser.add_argument("initial", help="Path to the initial structures")
    parser.add_argument("-ul", "-upperLim", type=int, default=10, help="Upper RMSD limit to native structure, default 10")
    parser.add_argument("-ll", "-lowerLim", type=int, default=0, help="Lower RMSD limit to native structure, default 0")
    args = parser.parse_args()
    return args.resname, args.native, args.initial, args.ul, args.ll


def main(lig_resname, native_path, initial_path, up_lim, low_lim):
    RMSDCalc = RMSDCalculator.RMSDCalculator()
    nativePDB = atomset.PDB()
    nativePDB.initialise(native_path, resname=lig_resname)
    filtered = []
    for conf in glob.glob(initial_path+"/initial*.pdb"):
        initialPDB = atomset.PDB()
        initialPDB.initialise(conf, resname=lig_resname)
        if low_lim < RMSDCalc.computeRMSD(nativePDB, initialPDB) < up_lim:
            filtered.append(conf)
    print(" ".join(filtered))


if __name__ == "__main__":
    resname, native, initial, upper, lower = parse_arguments()
    main(resname, native, initial, upper, lower)
