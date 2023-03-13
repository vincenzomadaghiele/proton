import MDAnalysis as mda
from MDAnalysis.analysis import rms, dihedrals
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculateDistance(MDAuniverse, atom_index1, atom_index2,
                      printExample=False, printHist=False, printPotential=False, atom_names=""):
    distances = []
    for ts in MDAuniverse.trajectory:
        dist = rms.rmsd(MDAuniverse.atoms[[atom_index1]].positions, MDAuniverse.atoms[[atom_index2]].positions)
        distances.append(dist)
    distances = np.array(distances)
    r0 = distances.mean()
    k = distances.std()
    
    if printExample:
        # distances over time
        plt.title(f"Bond Distance {atom_names}")
        plt.xlabel("Trajectory time [fs]")
        plt.ylabel("distance [A]")
        plt.plot(distances[300:600], label="distance")
        plt.axhline(r0, color='r', linestyle='dashed', label="r0")
        plt.legend()
        plt.show()
    
    if printHist:
        # histogram of distances
        plt.title(f"Histogram of distance {atom_names}")
        plt.xlabel("distance [A]")
        plt.ylabel("Density")
        sns.kdeplot(data=distances, label="distance")
        plt.axvline(r0, color='r', linestyle='dashed', label="r0")
        plt.legend()
        plt.show()
    
    if printPotential:
        # Potential curve
        sorted_dists = np.sort(np.abs(distances))
        V = k * ((np.abs(sorted_dists) - r0) ** 2)
        plt.title(f"Potential of bond distance {atom_names}")
        plt.xlabel("distance [A]")
        plt.ylabel("Potential")
        plt.plot(sorted_dists, V)
        plt.axvline(sorted_dists[V.argmin()], color='r', linestyle='dashed', label="r0")
        plt.show()

    return r0, k, distances

def calculateAngle(MDAuniverse, atom1, atom2, atom3,
                   printExample=False, printHist=False, printPotential=False, atom_names=""):
    angles = []
    for ts in MDAuniverse.trajectory:                
        angle = MDAuniverse.atoms[[atom1,atom2,atom3]].angle.value() # group three atoms
        angles.append(angle) # get angle value
    angles = np.array(angles) * (np.pi/180) # convert to radians
    theta0 = angles.mean()
    k = angles.std()
    
    if printExample:
        # distances over time
        plt.title(f"Angle {atom_names}")
        plt.xlabel("Trajectory time [fs]")
        plt.ylabel("Angle [rad]")
        plt.plot(angles[300:600], label="distance")
        plt.axhline(theta0, color='r', linestyle='dashed', label="r0")
        plt.legend()
        plt.show()
    
    if printHist:
        # histogram of distances
        plt.title(f"Histogram of angle {atom_names}")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Density")
        sns.kdeplot(data=angles, label="distance")
        plt.axvline(theta0, color='r', linestyle='dashed', label="r0")
        plt.legend()
        plt.show()
    
    if printPotential:
        # Potential curve
        sorted_dists = np.sort(np.abs(angles))
        V = k * ((np.abs(sorted_dists) - theta0) ** 2)
        plt.title(f"Potential of angle {atom_names}")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Potential")
        plt.plot(sorted_dists, V)
        plt.axvline(sorted_dists[V.argmin()], color='r', linestyle='dashed', label="theta0")
        plt.show()

    return theta0, k, angles

def calculateImproper(MDAuniverse, atom1, atom2, atom3, atom4,
                      printExample=False, printHist=False, printPotential=False, atom_names=""):
    impropers = []
    for ts in pto.trajectory:
        # calculate angle
        improper = atoms[[atom1, atom2, atom3, atom4]].improper.value() # group three atoms
        impropers.append(improper) # get angle value
    impropers = np.abs(np.array(impropers)) * (np.pi/180)
    xi0 = impropers.mean()
    k_xi = impropers.std()
    
    if printExample:
        # distances over time
        plt.title(f"Improper dihedral {atom_names}")
        plt.xlabel("Trajectory time [fs]")
        plt.ylabel("Angle [rad]")
        plt.plot(impropers[300:600], label="distance")
        plt.axhline(xi0, color='r', linestyle='dashed', label="r0")
        plt.legend()
        plt.show()
    
    if printHist:
        # histogram of distances
        plt.title(f"Histogram of improper dihedral {atom_names}")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Density")
        sns.kdeplot(data=impropers, label="distance")
        plt.axvline(xi0, color='r', linestyle='dashed', label="r0")
        plt.legend()
        plt.show()
    
    if printPotential:
        # Potential curve
        sorted_dists = np.sort(np.abs(impropers))
        V = k_xi * ((np.abs(sorted_dists) - xi0) ** 2)
        plt.title(f"Potential of improper dihedral {atom_names}")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Potential")
        plt.plot(sorted_dists, V)
        plt.axvline(sorted_dists[V.argmin()], color='r', linestyle='dashed', label="xi0")
        plt.show()

    return xi0, k_xi, impropers


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
    
    bonds = [[2, 0], [6, 0], [3, 0], [3, 1], [4, 1], [5, 1]]
    r_b = []
    k_b = []
    for bond in bonds:
        atom_names = f'{atoms.names[bond[0]]}{bond[0]}-{atoms.names[bond[1]]}{bond[1]}'
        r, k, _ = calculateDistance(pto, bond[0], bond[1])
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
    
    angles = [[2, 0, 6], [6, 0, 3], [2, 0, 3], 
              [4, 1, 3], [4, 1, 5], [5, 1, 3], [0, 3, 1]]
    theta0 = []
    k_theta = []
    for angle in angles:
        atom_names = f'{atoms.names[angle[0]]}{angle[0]}-{atoms.names[angle[1]]}{angle[1]}-{atoms.names[angle[2]]}{angle[2]}'
        theta, k, _ = calculateAngle(pto, angle[0], angle[1], angle[2])
        theta0.append(theta)
        k_theta.append(k)
        print(atom_names)
        print(f'k_theta: {k}')
        print(f'theta0: {theta}')
        print()


    #%% Impropers
    
    print()
    print('Impropers parameters')
    print('-'*20)
    
    impropers = [[2, 6, 3, 0], [4, 5, 3, 1]]
    xi0 = []
    k_xi = []
    for improper in impropers:
        atom_names = f'{atoms.names[improper[0]]}{improper[0]}-{atoms.names[improper[1]]}{improper[1]}-{atoms.names[improper[2]]}{improper[2]}-{atoms.names[improper[3]]}{improper[3]}'
        xi, k, _ = calculateImproper(pto, improper[0], improper[1], improper[2], improper[3])
        xi0.append(xi)
        k_xi.append(k)
        print(atom_names)
        print(f'k_xi: {k}')
        print(f'xi0: {xi}')
        print()
