import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt

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
for ts in pto.trajectory:
    time = pto.trajectory.time
    trajectory_positions.append(pto.atoms.positions)
    
    # calculate angle
    h2o_angle = atoms[[0,2,6]].angle.value() # group three atoms
    angles.append(h2o_angle) # get angle value
    
    # rgyr = pto.atoms.center_of_mass()
    # print(f"Frame: {ts.frame:3d}, Time: {time:4.0f} ps, Rgyr: {rgyr} A")

# array of trajectory positions
trajectory_positions = np.asarray(trajectory_positions)

# plot angle values over time
plt.title("H2O angle")
plt.xlabel("Trajectory time")
plt.ylabel("Angle [degrees]")
plt.plot(angles)
plt.show()