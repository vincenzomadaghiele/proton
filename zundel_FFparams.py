import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import leastsq


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
        plt.axhline(theta0, color='r', linestyle='dashed', label="theta0")
        plt.legend()
        plt.show()
    
    if printHist:
        # histogram of distances
        plt.title(f"Histogram of angle {atom_names}")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Density")
        sns.kdeplot(data=angles, label="distance")
        plt.axvline(theta0, color='r', linestyle='dashed', label="theta0")
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
        plt.axhline(xi0, color='r', linestyle='dashed', label="xi0")
        plt.legend()
        plt.show()
    
    if printHist:
        # histogram of distances
        plt.title(f"Histogram of improper dihedral {atom_names}")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Density")
        sns.kdeplot(data=impropers, label="distance")
        plt.axvline(xi0, color='r', linestyle='dashed', label="xi0")
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

def calculateDihedral(MDAuniverse, atom1, atom2, atom3, atom4,
                      printExample=False, printHist=False, printPotential=False, atom_names=""):
    
    dihedrals = []
    for ts in pto.trajectory:
        dihedral = pto.atoms[[atom1, atom2, atom3, atom4]].dihedral.value() # group three atoms
        dihedrals.append(dihedral)
    dihedrals = np.array(dihedrals) * (np.pi/180) # convert to radians
    
    if printExample:
        # distances over time
        plt.title(f"Dihedral {atom_names}")
        plt.xlabel("Trajectory time [fs]")
        plt.ylabel("Angle [rad]")
        plt.plot(dihedrals[300:600], label="distance")
        plt.legend()
        plt.show()
    
    if printHist:
        # histogram of distances
        plt.title(f"Histogram of dihedral {atom_names}")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Density")
        sns.kdeplot(data=dihedrals, label="distance")
        plt.legend()
        plt.show()

    # calculate p(r)
    counts, bins = np.histogram(dihedrals, bins=100, density=True)    
    energies = - 0.616 * np.log(counts) # convert p(r) to energy E(i)
    N = counts.shape[0] # number of data points
    t = np.linspace(0, 2*np.pi, N)
    data = energies - np.min(energies)
            
    # fit cosine to energy 
    guess_phase = 1
    guess_amp = 1
    optimize_func = lambda x: x[0] * (1 + np.cos(2* t - x[1])) - data
    k_phi, phi0 = leastsq(optimize_func, [guess_amp, guess_phase])[0]
    
    # recreate the fitted curve using the optimized parameters
    data_fit = k_phi * (1 + np.cos(2 * t + phi0))
    
    # recreate the fitted curve using the optimized parameters
    fine_t = np.arange(0,max(t),0.1)
    data_fit = k_phi * (1 + np.cos(2 * fine_t + phi0))
    
    if printPotential:
    
        plt.title("Potential of dihedral angle of H4-O1-O0-H2")
        plt.xlabel("Angle [rad]")
        plt.ylabel("Density")
        plt.plot(t, data, '.')
        plt.plot(fine_t, data_fit, label='fitted curve')
        plt.legend()
        plt.show()

    return k_phi, phi0, dihedrals



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
    
    # specify list of angles
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
    
    # specify list of improper dihedrals
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
        k_phi, phi0, _ = calculateDihedral(pto, dihedral[0], dihedral[1], dihedral[2], dihedral[3], True, True, True, atom_names)
        k_phis.append(k_phi)
        phi0s.append(phi0)
        print(atom_names)
        print(f'k_phi: {k_phi}')
        print(f'phi0: {phi0}')
        print()
