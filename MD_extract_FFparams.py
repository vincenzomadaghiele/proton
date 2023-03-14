import MDAnalysis as mda
from MDAnalysis.analysis import rms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import leastsq


def calculateDistance(MDAuniverse, atom1, atom2,
                      printExample=False, printHist=False, printPotential=False, atom_names=""):
    
    # V(bond) = Kb(b - b0)**2
    # Kb: kcal/mole/A**2
    # b0: A
    
    distances = []
    for ts in MDAuniverse.trajectory:
        dist = rms.rmsd(MDAuniverse.atoms[[atom1]].positions, MDAuniverse.atoms[[atom2]].positions)
        distances.append(dist)
    distances = np.array(distances)
    r0 = distances.mean() # in A
    k = distances.std() 
    k = 0.616 / (2 * (k**2)) # kcal/mole/A**2

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
    
    # V(angle) = Ktheta(Theta - Theta0)**2
    # Ktheta: kcal/mole/rad**2
    # Theta0: degrees
    
    angles = []
    for ts in MDAuniverse.trajectory:                
        angle = MDAuniverse.atoms[[atom1,atom2,atom3]].angle.value() # group three atoms
        angles.append(angle) # get angle value
    angles = np.array(angles) * (np.pi/180) # convert to radians
    theta0 = angles.mean() * (180 / np.pi) # degrees
    k = angles.std() 
    k = 0.616 / (2 * (k**2)) # kcal/mole/rad**2
    
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


def calculateDihedral(MDAuniverse, atom1, atom2, atom3, atom4,
                      printExample=False, printHist=False, printPotential=False, atom_names=""):
    
    # V(dihedral) = Kchi(1 + cos(n(chi) - delta))
    # Kchi: kcal/mole
    # n: multiplicity
    # delta: degrees
    
    dihedrals = []
    for ts in MDAuniverse.trajectory:
        dihedral = MDAuniverse.atoms[[atom1, atom2, atom3, atom4]].dihedral.value() # group three atoms
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

    phi0 = phi0 * (180 / np.pi) # degrees
    return k_phi, phi0, dihedrals


def calculateImproper(MDAuniverse, atom1, atom2, atom3, atom4,
                      printExample=False, printHist=False, printPotential=False, atom_names=""):
    
    # V(improper) = Kpsi(psi - psi0)**2
    # Kpsi: kcal/mole/rad**2
    # psi0: degrees
    
    impropers = []
    for ts in MDAuniverse.trajectory:
        # calculate angle
        improper = MDAuniverse.atoms[[atom1, atom2, atom3, atom4]].improper.value() # group three atoms
        impropers.append(improper) # get angle value
    impropers = np.abs(np.array(impropers)) * (np.pi/180)
    xi0 = impropers.mean() * (180 / np.pi) # degrees
    k_xi = impropers.std()
    k_xi = 0.616 / (2 * (k_xi**2)) # kcal/mole/rad**2

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

