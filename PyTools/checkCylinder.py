from __future__ import print_function
import numpy as np
import mdtraj as md

with open("0/topologyMapping.txt") as f:
    top_contents = f.read().rstrip().split(":")

for i in range(1, 5):
    # t = md.load("0/trajectory_%d.dcd" % i, top="topologies/topology_%s.pdb" % top_contents[i-1])
    t = md.load("trajectory_aligned_%d.xtc" % (i-1), top="top7UP.pdb")
    center = t.top.select("name DUM")
    base = t.top.select("name DUMB")
    ligand = t.top.select("name C12")
    d = np.linalg.norm((t.xyz[:, center, :]-t.xyz[:, ligand, :]).reshape((-1, 3)), axis=1)
    l = np.linalg.norm((t.xyz[:, center, :]-t.xyz[:, base, :]).reshape((-1, 3)), axis=1)
    alpha = (t.xyz[:, center, :]-t.xyz[:, base, :]).reshape((-1, 3))
    alpha /= np.linalg.norm(alpha, axis=1)[:, np.newaxis]
    lateral = (t.xyz[:, ligand, :]-t.xyz[:, base, :]).reshape((-1, 3))
    r = np.linalg.norm(np.cross(alpha, lateral), axis=1)
    r_vert = np.diag(np.dot(alpha, lateral.T)).copy()
    l *= 10
    r *= 10
    d *= 10
    r_vert *= 10
    # print(i, (r < 1.0) & (d < 1.23))
    print(i, max(2*l), "Lateral", "Base distance")
    for j, dj in enumerate(d):
        print((r[j] < 10 and 0 < r_vert[j] < 2*l[j]), r[j], r_vert[j])
