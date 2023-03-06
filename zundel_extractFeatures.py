import os
import MDAnalysis as mda
from MDAnalysis.analysis import rms, dihedrals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def returnPositions(MDAuniverse):
    # iterate over trajectory timesteps
    trajectory_positions = []
    for ts in MDAuniverse.trajectory:
        # compile trajectory position vector
        time = MDAuniverse.trajectory.time
        trajectory_positions.append(MDAuniverse.atoms.positions)
    # array of trajectory positions
    trajectory_positions = np.asarray(trajectory_positions[:-1])
    return trajectory_positions

def calculateDistances(MDAuniverse):
    print()
    print("Calculating distances...")
    # Calculate r0 and k for distances
    r0_matrix = []
    kdist_matrix = []
    for atom1 in range(MDAuniverse.atoms.names.shape[0]):
        r0_line = []
        k_line = []
        for atom2 in range(MDAuniverse.atoms.names.shape[0]):
            distances = []
            for ts in MDAuniverse.trajectory:
                dist = rms.rmsd(MDAuniverse.atoms[[atom1]].positions, MDAuniverse.atoms[[atom2]].positions)
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
    return r0_matrix, kdist_matrix

def calculateAngles(MDAuniverse):
    print()
    print("Calculating angles...")
    # Calculate theta0 and k for angles
    theta0_3Dmatrix = []
    ktheta_3Dmatrix = []
    for atom1 in range(MDAuniverse.atoms.names.shape[0]):
        theta0_matrix = []
        ktheta_matrix = []
        for atom2 in range(MDAuniverse.atoms.names.shape[0]):
            theta0_line = []
            k_line = []
            for atom3 in range(MDAuniverse.atoms.names.shape[0]):
                angles = []
                for ts in MDAuniverse.trajectory:                
                    angle = MDAuniverse.atoms[[atom1,atom2,atom3]].angle.value() # group three atoms
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
    return theta0_3Dmatrix, ktheta_3Dmatrix

if __name__ == '__main__':

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
    
    # convert from ps to fs by averaging
    energies = np.array(energies)
    energies = np.mean(energies[:-1].reshape(-1, 10), axis=1)

    # get trajectory array
    trajectory_positions = returnPositions(pto)

    
    #%% Extract features
    
    # get distances and angles
    r0_matrix, kdist_matrix = calculateDistances(pto)
    theta0_3Dmatrix, ktheta_3Dmatrix = calculateAngles(pto)
    
