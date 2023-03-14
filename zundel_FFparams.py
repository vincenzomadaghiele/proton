'''
This file contains code to extract Force Field parameters
It extracts parameters given lists of bonds, angles, dihedrals and impropers.
'''

import MDAnalysis as mda
import numpy as np
import MD_extract_FFparams as MDpar


if __name__ == '__main__':
    
    charges = np.load('data/zundel_resp_charges.npy')
    pto = mda.Universe("data/zundel_trajectory_150ps.xyz")
    
    # print info
    print("Loaded " + str(pto))
    print("Trajectory length: " + str(len(pto.trajectory))) 
    
    q_mean = charges.mean(axis=0)
    
    # get elements
    atoms = pto.atoms
    residues = pto.residues
    print("Atom names: ", atoms.names)
    print("Atom masses: ", atoms.masses)
    print('Atom mean charges: ', q_mean)
    
    # array of trajectory positions
    trajectory_positions = []
    for ts in pto.trajectory:
        # compile trajectory position vector
        time = pto.trajectory.time
        trajectory_positions.append(pto.atoms.positions)
    trajectory_positions = np.asarray(trajectory_positions)
    
    
    #%% Distances
    
    print()
    print('Bond distance parameters')
    print('-'*20)
    
    # specify list of bonds
    bonds = [[2, 0], [6, 0], [3, 0], [3, 1], [4, 1], [5, 1]]
    r_b = []
    k_b = []
    for bond in bonds:
        atom_names = f'{atoms.names[bond[0]]}{bond[0]}-{atoms.names[bond[1]]}{bond[1]}'
        r, k, _ = MDpar.calculateDistance(pto, bond[0], bond[1])
        r_b.append(r)
        k_b.append(k)
        print(atom_names)
        print(f'k_b: {k}')
        print(f'r_b: {r}')
        print()
    
    
    #%% Angles
    
    print()
    print('Angle parameters')
    print('-'*20)
    
    # specify list of angles
    angles = [[2, 0, 6], [6, 0, 3], [2, 0, 3], 
              [4, 1, 3], [4, 1, 5], [5, 1, 3], [0, 3, 1]]
    theta0 = []
    k_theta = []
    for angle in angles:
        atom_names = f'{atoms.names[angle[0]]}{angle[0]}-{atoms.names[angle[1]]}{angle[1]}-{atoms.names[angle[2]]}{angle[2]}'
        theta, k, _ = MDpar.calculateAngle(pto, angle[0], angle[1], angle[2])
        theta0.append(theta)
        k_theta.append(k)
        print(atom_names)
        print(f'k_theta: {k}')
        print(f'theta0: {theta}')
        print()


    #%% Dihedral angles
    
    print()
    print('Dihedral angle parameters')
    print('-'*20)
    
    # specify list of improper dihedrals
    dihedrals = [[2, 0, 1, 4], [2, 0, 1, 5]]
    k_phis = []
    phi0s = []
    for dihedral in dihedrals:
        atom_names = f'{atoms.names[dihedral[0]]}{dihedral[0]}-{atoms.names[dihedral[1]]}{dihedral[1]}-{atoms.names[dihedral[2]]}{dihedral[2]}-{atoms.names[dihedral[3]]}{dihedral[3]}'
        k_phi, phi0, _ = MDpar.calculateDihedral(pto, dihedral[0], dihedral[1], dihedral[2], dihedral[3])
        k_phis.append(k_phi)
        phi0s.append(phi0)
        print(atom_names)
        print(f'k_phi: {k_phi}')
        print(f'phi0: {phi0}')
        print()


    #%% Impropers
    
    print()
    print('Impropers parameters')
    print('-'*20)
    
    # specify list of improper dihedrals
    impropers = [[2, 6, 3, 0], [4, 5, 3, 1]]
    xi0 = []
    k_xi = []
    for improper in impropers:
        atom_names = f'{atoms.names[improper[0]]}{improper[0]}-{atoms.names[improper[1]]}{improper[1]}-{atoms.names[improper[2]]}{improper[2]}-{atoms.names[improper[3]]}{improper[3]}'
        xi, k, _ = MDpar.calculateImproper(pto, improper[0], improper[1], improper[2], improper[3])
        xi0.append(xi)
        k_xi.append(k)
        print(atom_names)
        print(f'k_xi: {k}')
        print(f'xi0: {xi}')
        print()


