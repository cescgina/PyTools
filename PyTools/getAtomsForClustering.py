""" Automatically choose four atoms to improve the clustering"""
import argparse
import numpy as np
from AdaptivePELE.atomset import atomset
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.style.use("ggplot")


def parse_arguments():
    """
        Create command-line interface
    """
    desc = "Choose four atoms to improve the clustering"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("conformation", help="Path to a pdb file with the system to analyse")
    parser.add_argument("ligand_resname", type=str, help="Name of the ligand in the PDB")
    parser.add_argument("-n", "--nAtoms", type=int, default=3, help="Number of atoms to select in first filtering (default 3)")
    parser.add_argument("-d", "--debug", action="store_true", help="Wether to run in debug mode (more output)")
    parser.add_argument("-p", "--preprocess", action="store_false", help="Wether to turn of preprocessing")
    args = parser.parse_args()
    return args.conformation, args.ligand_resname, args.nAtoms, args.debug, args.preprocess


def rotate_x(rot_angle, coordinates):
    R = np.array([[1, 0, 0], [0, np.cos(rot_angle), -np.sin(rot_angle)], [0, np.sin(rot_angle), np.cos(rot_angle)]])
    return np.dot(R, coordinates.T).T


def rotate_z(rot_angle, coordinates):
    R = np.array([[np.cos(rot_angle), -np.sin(rot_angle), 0], [np.sin(rot_angle), np.cos(rot_angle), 0], [0, 0, 1]])
    return np.dot(R, coordinates.T).T


def rotate_y(rot_angle, coordinates):
    R = np.array([[np.cos(rot_angle), 0, np.sin(rot_angle)], [0, 1, 0], [-np.sin(rot_angle), 0, np.cos(rot_angle)]])
    return np.dot(R, coordinates.T).T


def get_vector_direction(coords, com):
    sort_distance = np.argsort(np.linalg.norm(coords, axis=1))
    max_point = coords[sort_distance[-1], :]
    min_point = coords[sort_distance[0], :]
    return max_point-min_point


def get_direction_x(coordinates):
    sort_distance = np.argsort(coordinates[:, 0])
    max_point = coordinates[sort_distance[-1], :]
    min_point = coordinates[sort_distance[0], :]
    return max_point-min_point


def preprocess_coords(coords, com, debug):
    print "Preprocessing ligand coordinates"
    if debug:
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter([0], [0], zs=[0], c="blue")

    # Get a vector representation of the molecule
    direction = get_vector_direction(coords, com)
    if debug:
        ax.scatter(direction[0], direction[1], zs=direction[2], c="blue")
        coords_plot = coords
        ax.scatter(coords_plot[:, 0], coords_plot[:, 1], zs=coords_plot[:, 2], c="blue")
    # Project vector onto the yz plane
    azimutal = np.arctan2(direction[1], direction[0])
    if azimutal < 0:
        azimutal += 2*np.pi
        azimutal %= 2*np.pi
    if azimutal < np.pi/2:
        azimutal -= np.pi/2
    else:
        azimutal = np.pi/2-azimutal
    # Rotate along the z axis
    direction = rotate_z(azimutal, direction)
    coords = rotate_z(azimutal, coords)
    # Get the angle of the vector wrt the y axis
    angle = np.arccos(np.dot(direction, np.array([0, 1, 0]))/np.linalg.norm(direction))
    # if z component is below zero rotate counter-clockwise
    angle *= np.sign(direction[2])
    # Rotate along the x axis
    if debug:
        coords_plot = coords
        ax.scatter(coords_plot[:, 0], coords_plot[:, 1], zs=coords_plot[:, 2], c="red")
        ax.scatter(direction[0], direction[1], zs=direction[2], c="red")
    coords = rotate_x(-angle, coords)
    direction = rotate_x(-angle, direction)
    if debug:
        coords_plot = coords
        ax.scatter(coords_plot[:, 0], coords_plot[:, 1], zs=coords_plot[:, 2], c="black")
        ax.scatter(direction[0], direction[1], zs=direction[2], c="black")
        plt.show()
    return coords


def main(snapshot, lig_resname, nAtoms_selection, debug, preprocess):
    PDB = atomset.PDB()
    PDB.initialise(snapshot, resname=lig_resname)
    com = np.array(PDB.getCOM())
    coords = np.array([PDB.getAtom(atom).getAtomCoords() for atom in PDB.atomList])
    atoms = np.array(PDB.atomList)
    if preprocess:
        coords = preprocess_coords(coords, com, debug)
        com = np.mean(coords, axis=0)

    axis_variation = np.std(coords, axis=0)
    atomsSel = []
    for ind in np.argsort(axis_variation)[-1:-3:-1]:
        comp_ind = np.array([i for i in xrange(3) if i != ind])
        order = np.argsort(coords[:, ind])
        imin = np.argmin(np.abs(coords[order[:nAtoms_selection]][:, comp_ind] - com[comp_ind]).sum(axis=1))
        imax = np.argmin(np.abs(coords[order[-nAtoms_selection:]][:, comp_ind] - com[comp_ind]).sum(axis=1))
        atomsSel.append(atoms[order[:nAtoms_selection]][imin])
        atomsSel.append(atoms[order[-nAtoms_selection:]][imax])
        if debug:
            print "min", ['X', 'Y', 'Z'][ind]
            print atoms[order[:nAtoms_selection]]
            print "max", ['X', 'Y', 'Z'][ind]
            print atoms[order[-nAtoms_selection:]]
    print "Atoms to select:"
    print " ".join(set(atomsSel))

if __name__ == "__main__":
    conf, ligand, n_atoms, set_debug, set_preprocess = parse_arguments()
    main(conf, ligand, n_atoms, set_debug, set_preprocess)
