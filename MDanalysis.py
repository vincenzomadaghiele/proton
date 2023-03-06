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

# calculate angle
h2o = atoms[[0,2,6]]
h2o_angle = h2o.angle

# iterate over trajectory timesteps
trajectory_positions = []
angles = []
oh1_dist = []
oh2_dist = []
for ts in pto.trajectory:
    # compile trajectory position vector
    time = pto.trajectory.time
    trajectory_positions.append(pto.atoms.positions)
    
    # calculate angle
    h2o_angle = atoms[[0,1,3]].angle.value() # group three atoms
    angles.append(h2o_angle) # get angle value
    
    # select O atoms
    o1 = pto.atoms[[0]].positions
    o2 = pto.atoms[[1]].positions 
    h = pto.atoms[[3]].positions # select middle H
    # calculate distances
    oh1 = rms.rmsd(o1, h)
    oh2 = rms.rmsd(o2, h)
    oh1_dist.append(oh1)
    oh2_dist.append(oh2)
    
    # rgyr = pto.atoms.center_of_mass()
    # print(f"Frame: {ts.frame:3d}, Time: {time:4.0f} ps, Rgyr: {rgyr} A")

#Dihedral analysis
dih = dihedrals.Dihedral([pto.atoms[[0,1,2,3]]]).run()
dihedral_angles = dih.results.angles


# array of trajectory positions
trajectory_positions = np.asarray(trajectory_positions)

# plot angle values over time
plt.title("Middle proton angle")
plt.xlabel("Trajectory time [fs]")
plt.ylabel("Angle [degrees]")
plt.plot(angles[300:600])
plt.axhline(np.array(angles).mean(), color='r', linestyle='dashed')
plt.show()

# plot angle values over time
plt.title("Distance from middle H")
plt.xlabel("Trajectory time [fs]")
plt.ylabel("distance [A]")
plt.plot(oh1_dist[300:450], label="O1-H")
plt.plot(oh2_dist[300:450], label="O2-H")
plt.axhline(np.array(oh1_dist).mean(), color='r', linestyle='dashed', label="r0")
plt.legend()
plt.show()

plt.title("Histogram of distance from middle H")
plt.xlabel("distance [A]")
plt.ylabel("Density")
sns.kdeplot(data=oh1_dist, label="O1-H")
sns.kdeplot(data=oh2_dist, label="O2-H")
plt.axvline(np.array(oh2_dist).mean(), color='r', linestyle='dashed', label="r0")
plt.legend()
plt.show()

plt.title("Histogram of (O1-H) - (O2-H)")
plt.xlabel("distance [A]")
plt.ylabel("Density")
sns.histplot(data=np.array(oh1_dist)-np.array(oh2_dist), label="r1-r2", kde=True)
plt.show()


#%% Plot histogram after binning

BIN_THR = 0.01
oh1_dist = np.array(oh1_dist)
oh1_r0 = oh1_dist.mean()
oh1_dist_bin = oh1_dist[oh1_dist < oh1_r0 + BIN_THR]
oh1_dist_bin = oh1_dist_bin[oh1_dist_bin > oh1_r0 - BIN_THR]
oh2_dist = np.array(oh2_dist)
oh2_r0 = oh2_dist.mean()
oh2_dist_bin = oh2_dist[oh2_dist < oh2_r0 + BIN_THR]
oh2_dist_bin = oh2_dist_bin[oh2_dist_bin > oh2_r0 - BIN_THR]

plt.title("Histogram of distance from middle H [binned between -0.1 and 0.1")
plt.xlabel("distance [A]")
plt.ylabel("Density")
sns.kdeplot(data=oh1_dist_bin, label="O1-H")
sns.kdeplot(data=oh2_dist, label="O2-H")
plt.axvline(np.array(oh1_dist_bin).mean(), color='b', linestyle='dashed', label="O1-H r0")
plt.axvline(np.array(oh2_dist_bin).mean(), color='r', linestyle='dashed', label="O2-H r0")
plt.legend()
plt.show()

