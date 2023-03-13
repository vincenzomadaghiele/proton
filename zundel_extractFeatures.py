import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import MDAnalysis as mda
from MDAnalysis.analysis import rms, dihedrals


class PotentialModel:
    def __init__(self, MDAuniverse):
        self.universe = MDAuniverse


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

def calculateDistance(MDAuniverse, atom_index1, atom_index2):
    distances = []
    for ts in MDAuniverse.trajectory:
        dist = rms.rmsd(MDAuniverse.atoms[[atom_index1]].positions, MDAuniverse.atoms[[atom_index2]].positions)
        distances.append(dist)
    r0 = np.array(distances).mean()
    k = np.array(distances).std()
    return r0, k, np.array(distances)

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
            # binning before saving params
            r0 = np.array(distances).mean()
            k = np.array(distances).std()
            r0_line.append(r0)
            k_line.append(k)
        r0_matrix.append(r0_line)
        kdist_matrix.append(k_line)

    r0_matrix = np.array(r0_matrix)
    kdist_matrix = np.array(kdist_matrix)
    return r0_matrix, kdist_matrix

def calculateAngle(MDAuniverse, atom1, atom2, atom3):
    angles = []
    for ts in MDAuniverse.trajectory:                
        angle = MDAuniverse.atoms[[atom1,atom2,atom3]].angle.value() # group three atoms
        angles.append(angle) # get angle value
    theta0 = np.array(angles).mean()
    k = np.array(angles).std()
    return theta0, k, np.array(angles)

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
                # binning before saving params
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

def extractFeaturesFromStructure(MDAuniverse, FRAME):
    i = 0
    # select correct frame
    for ts in MDAuniverse.trajectory:
        if i == FRAME:
            # extract positions 
            p = MDAuniverse.atoms.positions
            
            # extract distances
            r_matrix = []
            for atom1 in range(MDAuniverse.atoms.names.shape[0]):
                r_line = []
                for atom2 in range(MDAuniverse.atoms.names.shape[0]):
                    r_ij = rms.rmsd(MDAuniverse.atoms[[atom1]].positions, MDAuniverse.atoms[[atom2]].positions)
                    r_line.append(r_ij)
                r_matrix.append(r_line)
            r_matrix = np.array(r_matrix)

            # extract angles
            theta_3Dmatrix = []
            for atom1 in range(MDAuniverse.atoms.names.shape[0]):
                theta_matrix = []
                for atom2 in range(MDAuniverse.atoms.names.shape[0]):
                    theta_line = []
                    for atom3 in range(MDAuniverse.atoms.names.shape[0]):
                        theta = MDAuniverse.atoms[[atom1,atom2,atom3]].angle.value() 
                        theta_line.append(theta)
                    theta_matrix.append(theta_line)
                theta_3Dmatrix.append(theta_matrix)
            theta_3Dmatrix = np.array(theta_3Dmatrix)

            # extract dihedrals
            phi_4Dmatrix = []
            for atom1 in range(MDAuniverse.atoms.names.shape[0]):
                phi_3Dmatrix = []
                for atom2 in range(MDAuniverse.atoms.names.shape[0]):
                    phi_matrix = []
                    for atom3 in range(MDAuniverse.atoms.names.shape[0]):
                        phi_line = []
                        for atom4 in range(MDAuniverse.atoms.names.shape[0]):
                            phi = MDAuniverse.atoms[[atom1,atom2,atom3,atom4]].dihedral.value() 
                            phi_line.append(phi)
                        phi_matrix.append(phi_line)
                    phi_3Dmatrix.append(phi_matrix)
                phi_4Dmatrix.append(phi_3Dmatrix)
            phi_4Dmatrix = np.array(phi_4Dmatrix)
        i += 1
    return p, r_matrix, theta_3Dmatrix, phi_4Dmatrix


if __name__ == '__main__':

    # load tranjectory
    xyz_path = "data/zundel_trajectory_150ps.xyz"
    out_path = "data/zundel_log_150ps.out"
    charges_path = 'data/zundel_resp_charges.npy'
    pto = mda.Universe(xyz_path)
    charges = np.load(charges_path)

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
    energies = np.mean(energies.reshape(-1, 10), axis=1)

    # get trajectory array
    trajectory_positions = returnPositions(pto)
    q_mean = charges.mean(axis=0)

    
    #%% Extract features
    
    # get distances and angles
    r0_matrix, kdist_matrix = calculateDistances(pto)
    theta0_3Dmatrix, ktheta_3Dmatrix = calculateAngles(pto)

    
    #%% Calculate potential

    # extract structure data from trajectory
    FRAME = 0
    p, r_matrix, theta_3Dmatrix, phi_4Dmatrix = extractFeaturesFromStructure(pto, FRAME)
    q = charges[FRAME]
    
    # Bond stretching potential
    V_frame_bond = kdist_matrix * (r_matrix - r0_matrix)**2 
    V_frame_bond = np.sum(np.triu(V_frame_bond))

    # Angle potential
    #V_frame_angle = ktheta_3Dmatrix * (theta_3Dmatrix - theta0_3Dmatrix)**2 
    #V_frame_angle = np.sum(np.triu(V_frame_angle))    

    # Coulomb potential
    diag = np.diag(np.diag(np.ones(r_matrix.shape)))
    q_prod_charges = np.triu((q.reshape(-1, 1).T * q.reshape(-1, 1) ) / (r_matrix + diag))
    q_prod_charges = q_prod_charges - np.diag(np.diag(q_prod_charges))
    V_frame_charges = np.sum(q_prod_charges) / (4 * np.pi)
    
    