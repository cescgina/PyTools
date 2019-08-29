import os
import glob
import argparse
import multiprocessing as mp
from AdaptivePELE.atomset import atomset
from AdaptivePELE.utilities import utilities


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Calculate rmsd from pdbs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("path", type=str, help="Path where the files are stored")
    parser.add_argument("-n", "--processors", type=int, default=4, help="Number of processors to use")
    args = parser.parse_args()
    return args.path, args.ligand_resname, args.processors


def get_com(name, resname):
    PDB = atomset.PDB()
    PDB.initialise(name, resname=resname)
    return PDB.getCOM()


def main(path, resname, nProc):
    pdbs = glob.glob(os.path.join(path, "cluster_*.pdb"))
    pool = mp.Pool(nProc)
    workers = []
    for pdb_num in range(len(pdbs)):
        workers.append(pool.apply_async(get_com, args=(os.path.join(path, "cluster_%d.pdb" % pdb_num), resname)))
    pool.close()
    com = [worker.get() for worker in workers]
    utilities.write_PDB_clusters(com, title=os.path.join(path, "clusters.pdb"))


if __name__ == "__main__":
    folder, ligand_name, processors = parse_arguments()
    main(folder, ligand_name, processors)
