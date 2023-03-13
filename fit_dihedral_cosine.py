import numpy as np
from scipy.optimize import leastsq
import pylab as plt
import MDAnalysis as mda
import seaborn as sns


pto = mda.Universe("data/zundel_trajectory_150ps.xyz")

# print info
print("Loaded " + str(pto))
print("Trajectory length: " + str(len(pto.trajectory))) 

# get elements
atoms = pto.atoms
residues = pto.residues
print("Atom names: ", atoms.names)
print("Atom masses: ", atoms.masses)


dihedrals = []
for ts in pto.trajectory:
    dihedral = pto.atoms[[4,1,0,2]].dihedral.value() # group three atoms
    dihedrals.append(dihedral)
dihedrals = np.array(dihedrals) * (np.pi/180) # convert to radians

# Data curve
plt.title("Dihedral angle of H4-O1-O0-H2")
plt.xlabel("Timestep")
plt.ylabel("Angle [rad]")
plt.plot(dihedrals[300:600])
plt.show()

# histogram of angles
plt.title("Dihedral angle of H4-O1-O0-H2")
plt.xlabel("Angle [rad]")
plt.ylabel("Density")
sns.histplot(data=dihedrals, kde=True, fill=False, alpha=0.5)
#plt.axvline(np.array(dihedral_angles).mean(), color='r', linestyle='dashed', label="r0")
plt.show()


counts, bins = np.histogram(dihedrals, bins=100, density=True)
plt.plot(bins[:-1], - 0.616 * np.log(counts))
plt.show()


N = counts.shape[0] # number of data points
t = np.linspace(0, 2*np.pi, N)
data = counts - np.min(counts)

guess_mean = np.mean(data)
guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
guess_phase = 1
guess_amp = 1

# we'll use this to plot our first estimate. This might already be good enough for you
data_first_guess = guess_std*(1 + np.cos(2 * t - guess_phase))

# Define the function to optimize, in this case, we want to minimize the difference
# between the actual data and our "guessed" parameters
optimize_func = lambda x: x[0] * (1 + np.cos(2* t - x[1])) - data
est_amp, est_phase = leastsq(optimize_func, [guess_amp, guess_phase])[0]

# recreate the fitted curve using the optimized parameters
data_fit = est_amp * (1 + np.cos(2 * t + est_phase))

# recreate the fitted curve using the optimized parameters
fine_t = np.arange(0,max(t),0.1)
data_fit = est_amp * (1 + np.cos(2 * fine_t + est_phase))

plt.title("Potential of dihedral angle of H4-O1-O0-H2")
plt.xlabel("Angle [rad]")
plt.ylabel("Density")
plt.plot(t, data, '.')
plt.plot(fine_t, data_fit, label='fitted curve')
plt.legend()
plt.show()
