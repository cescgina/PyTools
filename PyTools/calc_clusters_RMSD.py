import glob
import argparse
from AdaptivePELE.atomset import atomset, RMSDCalculator


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Calculate rmsd from pdbs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("files", type=str, help="Glob string to find the input files")
    parser.add_argument("native", type=str, help="Glob string to find the input files")
    args = parser.parse_args()
    return args.ligand_resname, args.files, args.native


def main(resname, files_glob, native):
    input_files = glob.glob(files_glob)
    nativePDB = atomset.PDB()
    nativePDB.initialise(native, resname=resname)
    RMSDCalc = RMSDCalculator.RMSDCalculator()
    results = {}
    for f in input_files:
        p = atomset.PDB()
        p.initialise(f, resname=resname)
        results[f] = RMSDCalc.computeRMSD(nativePDB, p)
    with open("rmsd_file.dat", "w") as fw:
        fw.write("File\tRMSD(A)\n")
        for f in results:
            fw.write("%s\t%.4f\n" % (f, results[f]))


if __name__ == "__main__":
    res, file_glob, native_path = parse_arguments()
    main(res, file_glob, native_path)
