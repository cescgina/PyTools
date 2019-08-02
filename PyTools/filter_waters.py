import os
import glob
import argparse
import mdtraj as md


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Remove waters from pdbs"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("files", type=str, help="Glob string to find the input files")
    args = parser.parse_args()
    return args.ligand_resname, args.files


def merge_atom_sets(protein, water, hetero, traj):
    atoms = protein.tolist()+hetero.tolist()
    water_set = set()
    for w in water:
        for at in traj.top.atom(w).residue.atoms:
            water_set.add(at.index)
    return sorted(atoms+list(water_set))


def main(resname, files_glob):
    input_files = glob.glob(files_glob)
    for f in input_files:
        t = md.load(f)
        protein_set = t.top.select("protein")
        protein_noh = t.top.select("(protein or resname '%s') and symbol != H" % resname)
        waters_set = t.top.select("resname HOH")
        hetero = t.top.select("resname '%s'" % resname)
        waters = md.compute_neighbors(t, 0.4, protein_noh, haystack_indices=waters_set, periodic=False)
        write_selection = merge_atom_sets(protein_set, waters[0], hetero, t)
        t.atom_slice(write_selection, inplace=True)
        folder, file_name = os.path.split(f)
        t.save_pdb(os.path.join(folder, "filtered_%s" % file_name))

if __name__ == "__main__":
    res, file_glob = parse_arguments()
    main(res, file_glob)
