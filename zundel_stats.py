'''
This file contains analysis of single attributes such as distances, angles, dihedrals, charges.
It shows examples of how to calculate those for the zelder species and how to plot them.
'''

import MDAnalysis as mda
from MDAnalysis.analysis import rms, dihedrals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# load tranjectory
pto = mda.Universe("data/zundel_trajectory.xyz")

# print info
print("Loaded " + str(pto))
print("Trajectory length: " + str(len(pto.trajectory))) 

# get elements
atoms = pto.atoms
residues = pto.residues
print("Atom names: ", atoms.names)
print("Atom masses: ", atoms.masses)
print("Positions: ")
print(atoms.positions)

# array of trajectory positions
trajectory_positions = []
for ts in pto.trajectory:
    # compile trajectory position vector
    time = pto.trajectory.time
    trajectory_positions.append(pto.atoms.positions)
trajectory_positions = np.asarray(trajectory_positions)


#%% Distance values

# array of distances from middle proton
oh1_dist = []
oh2_dist = []
for ts in pto.trajectory:    
    # select O atoms
    o1 = pto.atoms[[0]].positions
    o2 = pto.atoms[[1]].positions 
    h = pto.atoms[[3]].positions # select middle H
    # calculate distances
    oh1 = rms.rmsd(o1, h)
    oh2 = rms.rmsd(o2, h)
    oh1_dist.append(oh1)
    oh2_dist.append(oh2)
    
# distances over time
plt.title("Distance from middle H")
plt.xlabel("Trajectory time [fs]")
plt.ylabel("distance [A]")
plt.plot(oh1_dist[300:450], label="O1-H")
plt.plot(oh2_dist[300:450], label="O2-H")
plt.axhline(np.array(oh1_dist).mean(), color='r', linestyle='dashed', label="r0")
plt.legend()
plt.show()

# histogram of distances
plt.title("Histogram of distance from middle H")
plt.xlabel("distance [A]")
plt.ylabel("Density")
sns.kdeplot(data=oh1_dist, label="O1-H")
sns.kdeplot(data=oh2_dist, label="O2-H")
plt.axvline(np.array(oh2_dist).mean(), color='r', linestyle='dashed', label="r0")
plt.legend()
plt.show()

# histogram of difference
plt.title("Histogram of (O1-H) - (O2-H)")
plt.xlabel("distance [A]")
plt.ylabel("Density")
sns.histplot(data=np.array(oh1_dist)-np.array(oh2_dist), label="r1-r2", kde=True)
plt.show()

# bin data
BIN_THR = 0.15
oh1_dist = np.array(oh1_dist)
oh1_r0 = oh1_dist.mean()
oh1_dist_bin = oh1_dist[oh1_dist < oh1_r0 + BIN_THR]
oh1_dist_bin = oh1_dist_bin[oh1_dist_bin > oh1_r0 - BIN_THR]
oh2_dist = np.array(oh2_dist)
oh2_r0 = oh2_dist.mean()
oh2_dist_bin = oh2_dist[oh2_dist < oh2_r0 + BIN_THR]
oh2_dist_bin = oh2_dist_bin[oh2_dist_bin > oh2_r0 - BIN_THR]

# plot binned histogram
plt.title(f"Histogram of distance from middle H [binned between -{BIN_THR} and {BIN_THR}]")
plt.xlabel("distance [A]")
plt.ylabel("Density")
sns.histplot(data=oh1_dist_bin, label="O1-H", kde=True, fill=False, alpha=0.2)
sns.histplot(data=oh2_dist_bin, label="O2-H", kde=True, fill=False, alpha=0.2)
#sns.kdeplot(data=oh1_dist_bin, label="O1-H")
#sns.kdeplot(data=oh2_dist_bin, label="O2-H")
plt.axvline(np.array(oh1_dist_bin).mean(), color='b', linestyle='dashed', label="O1-H r0")
plt.axvline(np.array(oh2_dist_bin).mean(), color='r', linestyle='dashed', label="O2-H r0")
plt.legend()
plt.show()


#%% Angle values

# calculate angle
h2o = atoms[[0,2,6]]
h2o_angle = h2o.angle


# MIDDLE PROTON

angles = []
for ts in pto.trajectory:
    # calculate angle
    h2o_angle = atoms[[0,3,1]].angle.value() # group three atoms
    angles.append(h2o_angle) # get angle value

print("Proton angle O1-H3-O0")
print(f"Mean: {np.array(angles).mean()}")
print(f"Std: {np.array(angles).std()}")
print()

# angle values over time
plt.title("Middle proton angle O1-H3-O0")
plt.xlabel("Trajectory time [fs]")
plt.ylabel("Angle [degrees]")
plt.plot(angles[300:600])
plt.axhline(np.array(angles).mean(), color='r', linestyle='dashed')
plt.show()

# histogram of angles
plt.title("Histogram of middle proton angle O1-H3-O0")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=angles, kde=True, fill=False, alpha=0.5)
plt.axvline(np.array(angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()


# WATER MOLECULES

# H4-O1-H5 proton angle
angles = []
for ts in pto.trajectory:
    # calculate angle
    h2o_angle = atoms[[4,1,5]].angle.value() # group three atoms
    angles.append(h2o_angle) # get angle value

print("Proton angle H4-O1-H5")
print(f"Mean: {np.array(angles).mean()}")
print(f"Std: {np.array(angles).std()}")
print()

# histogram of angles
plt.title("Histogram of water molecule angle H4-O1-H5")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=angles, kde=True, fill=False, alpha=0.5)
plt.axvline(np.array(angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()

# H2-O0-H6 proton angle
angles = []
for ts in pto.trajectory:
    # calculate angle
    h2o_angle = atoms[[2,0,6]].angle.value() # group three atoms
    angles.append(h2o_angle) # get angle value
    
print("Proton angle H2-O0-H6")
print(f"Mean: {np.array(angles).mean()}")
print(f"Std: {np.array(angles).std()}")
print()

# histogram of angles
plt.title("Histogram of water molecule angle H2-O0-H6")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=angles, kde=True, fill=False, alpha=0.5)
plt.axvline(np.array(angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()

# HYDROGEN OXIGEN MIDDLE PROTON

#% H3-O1-H4 proton angle
angles = []
for ts in pto.trajectory:
    # calculate angle
    h2o_angle = atoms[[3,1,4]].angle.value() # group three atoms
    angles.append(h2o_angle) # get angle value

print("Proton angle H3-O1-H4")
print(f"Mean: {np.array(angles).mean()}")
print(f"Std: {np.array(angles).std()}")
print()

# histogram of angles
plt.title("Histogram of water molecule angle H3-O1-H4")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=angles, kde=True, fill=False, alpha=0.5)
plt.axvline(np.array(angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()

# H3-O0-H2 proton angle
angles = []
for ts in pto.trajectory:
    # calculate angle
    h2o_angle = atoms[[3,0,2]].angle.value() # group three atoms
    angles.append(h2o_angle) # get angle value

print("Proton angle H3-O0-H2")
print(f"Mean: {np.array(angles).mean()}")
print(f"Std: {np.array(angles).std()}")
print()

# histogram of angles
plt.title("Histogram of water molecule angle H3-O0-H2")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=angles, kde=True, fill=False, alpha=0.5)
plt.axvline(np.array(angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()


#%% Dihedral analysis

dih = dihedrals.Dihedral([pto.atoms[[4,1,0,2]]]).run()
dihedral_angles = dih.results.angles

# histogram of angles
plt.title("Dihedral angle of H4-O1-O0-H2")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=dihedral_angles, kde=True, fill=False, alpha=0.5)
#plt.axvline(np.array(dihedral_angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()

dih = dihedrals.Dihedral([pto.atoms[[4,1,0,6]]]).run()
dihedral_angles = dih.results.angles

# histogram of angles
plt.title("Dihedral angle of H4-O1-O0-H6")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=dihedral_angles, kde=True, fill=False, alpha=0.5)
#plt.axvline(np.array(dihedral_angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()


#%% Impropers dihedrals

impropers = []
for ts in pto.trajectory:
    # calculate angle
    improper = atoms[[3,2,6,0]].improper.value() # group three atoms
    impropers.append(improper) # get angle value

impropers00 = np.array(impropers)

print("Improper dihedral for O0")
print(f"Mean: {np.abs(np.array(impropers)).mean()}")
print(f"Std: {np.abs(np.array(impropers)).std()}")
print()

# histogram of angles
plt.title("Histogram of improper dihedrals O0")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=np.abs(impropers00), kde=True, fill=False, alpha=0.5)
plt.axvline(np.array(np.abs(impropers00)).mean(), color='r', linestyle='dashed', label="r0")
plt.show()


impropers = []
for ts in pto.trajectory:
    # calculate angle
    improper = atoms[[3,4,5,1]].improper.value() # group three atoms
    impropers.append(improper) # get angle value

impropers01 = np.array(impropers)

print("Improper dihedral for O1")
print(f"Mean: {np.abs(np.array(impropers)).mean()}")
print(f"Std: {np.abs(np.array(impropers)).std()}")
print()

# histogram of angles
plt.title("Histogram of improper dihedrals O1")
plt.xlabel("Angle [degrees]")
plt.ylabel("Density")
sns.histplot(data=np.abs(impropers01), kde=True, fill=False, alpha=0.5)
plt.axvline(np.array(np.abs(impropers01)).mean(), color='r', linestyle='dashed', label="r0")
plt.show()

# distances over time
plt.title("Improper dihedral angles")
plt.xlabel("Trajectory time [fs]")
plt.ylabel("angle [degrees]")
plt.plot(impropers00[300:450], label="O0")
plt.plot(impropers01[300:450], label="O1")
plt.axhline(np.array(impropers00).mean(), color='r', linestyle='dashed', label="r0")
plt.legend()
plt.show()

# Potential curve
sorted_impropers = np.sort(np.abs(impropers00))
V = np.abs(np.array(impropers00)).std() * ((np.abs(sorted_impropers) - np.abs(impropers00).mean()) ** 2)
plt.title("Improper dihedral potential")
plt.xlabel("Dihedral angle [degrees]")
plt.ylabel("Potential")
plt.plot(sorted_impropers, V)
plt.axvline(sorted_impropers[V.argmin()], color='r', linestyle='dashed', label="r0")
plt.show()
