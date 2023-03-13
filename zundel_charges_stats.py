import MDAnalysis as mda
from MDAnalysis.analysis import rms, dihedrals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

charges = np.load('data/zundel_resp_charges.npy')
pto = mda.Universe("data/zundel_trajectory_150ps.xyz")

# print info
print("Loaded " + str(pto))
print("Trajectory length: " + str(len(pto.trajectory))) 

q_mean = charges.mean(axis=0)

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
    
# histogram of difference
plt.title("Histogram of (O1-H) - (O2-H)")
plt.xlabel("distance [A]")
plt.ylabel("Density")
sns.histplot(data=np.array(oh1_dist)-np.array(oh2_dist), label="r1-r2", kde=True)
plt.axvline(np.array(np.array(oh1_dist)-np.array(oh2_dist)).mean(), color='r', linestyle='dashed', label="r0")
plt.show()

middleH_charge = charges[:,3]
O0_charge = charges[:,0]
O1_charge = charges[:,1]

# histogram of difference
plt.title("Histogram of middle H charge")
plt.xlabel("Charge")
plt.ylabel("Density")
sns.histplot(data=middleH_charge, label="middle H charge", kde=True)
plt.axvline(np.array(middleH_charge).mean(), color='r', linestyle='dashed', label="r0")
plt.show()

# middle H charge vs distance from oxygen
df = pd.DataFrame(middleH_charge, columns=['middleH_charge'])
df = pd.concat([df, pd.DataFrame(np.array(oh1_dist)-np.array(oh2_dist))], axis=1)
df.columns = ['middle H charge', 'distance (O1-H) - (O2-H)']
plt.title('Histogram of middle H charge versus distance from Oxygens')
sns.histplot(data=df, x='middle H charge', y='distance (O1-H) - (O2-H)')
plt.show()

df = pd.DataFrame(O0_charge, columns=['O0 charge'])
df = pd.concat([df, pd.DataFrame(np.array(oh1_dist)-np.array(oh2_dist))], axis=1)
df.columns = ['O0 charge', 'distance (O1-H) - (O2-H)']
plt.title('Histogram of O0 charge versus distance from middle H')
sns.histplot(data=df, x='O0 charge', y='distance (O1-H) - (O2-H)')
plt.show()

df = pd.DataFrame(O1_charge, columns=['O1 charge'])
df = pd.concat([df, pd.DataFrame(np.array(oh1_dist)-np.array(oh2_dist))], axis=1)
df.columns = ['O1 charge', 'distance (O1-H) - (O2-H)']
plt.title('Histogram of O1 charge versus distance from middle H')
sns.histplot(data=df, x='O1 charge', y='distance (O1-H) - (O2-H)')
plt.show()

