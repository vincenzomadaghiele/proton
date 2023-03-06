import os
import MDAnalysis as mda
from MDAnalysis.analysis import rms, dihedrals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# load tranjectory
xyz_path = "data/zundel_trajectory.xyz"
out_path = "data/zundel_log.out"
pto = mda.Universe(xyz_path)

# print info
print("Loaded " + str(pto))
print("Trajectory length: " + str(len(pto.trajectory))) 
print("Atom names: ", pto.atoms.names)

# extract potential from out file
f = open(out_path, "r")
energies = []
for i, line in enumerate(f):
    # first table line and remove errors
    if i > 259 and i not in [270, 3361, 3363]: 
        energy = float(line[22:37])
        energies.append(energy)
f.close() 


#%% Extract features

print()
print("Calculating distances...")
# Calculate r0 and k for distances
r0_matrix = []
kdist_matrix = []
for atom1 in range(pto.atoms.names.shape[0]):
    r0_line = []
    k_line = []
    for atom2 in range(pto.atoms.names.shape[0]):
        distances = []
        for ts in pto.trajectory:
            dist = rms.rmsd(pto.atoms[[atom1]].positions, pto.atoms[[atom2]].positions)
            distances.append(dist)
        # introduce check with bins
        r0 = np.array(distances).mean()
        k = np.array(distances).std()
        r0_line.append(r0)
        k_line.append(k)
    r0_matrix.append(r0_line)
    kdist_matrix.append(k_line)

r0_matrix = np.array(r0_matrix)
kdist_matrix = np.array(kdist_matrix)

print()
print("Calculating angles...")
# Calculate theta0 and k for angles
theta0_3Dmatrix = []
ktheta_3Dmatrix = []
for atom1 in range(pto.atoms.names.shape[0]):
    theta0_matrix = []
    ktheta_matrix = []
    for atom2 in range(pto.atoms.names.shape[0]):
        theta0_line = []
        k_line = []
        for atom3 in range(pto.atoms.names.shape[0]):
            angles = []
            for ts in pto.trajectory:                
                angle = pto.atoms[[atom1,atom2,atom3]].angle.value() # group three atoms
                angles.append(angle) # get angle value
            theta0 = np.array(angles).mean()
            k = np.array(angles).std()
            theta0_line.append(theta0)
            k_line.append(k)
        theta0_matrix.append(theta0_line)
        ktheta_matrix.append(k_line)
    theta0_3Dmatrix.append(theta0_matrix)
    ktheta_3Dmatrix.append(ktheta_matrix)

theta0_3Dmatrix = np.array(theta0_3Dmatrix)
ktheta_3Dmatrix = np.array(ktheta_3Dmatrix)
