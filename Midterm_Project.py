import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Reading the file directly from GitHub using pandas
df = pd.read_csv(r'https://raw.githubusercontent.com/dscolnic/Pantheon/master/lcparam_full_long.txt', sep='\s+',
                 header=None)

df.columns = ["name", "zcmb", "zhel", "dz", "mb", "dmb", "x1", "dx1", "color", "dcolor", "3rdvar", "d3rdvar",
              "cov_m_s", "cov_m_c", "cov_s_c", "set", "ra", "dec", "biascor"]

# Defining the columns used in the program
df['Mb'] = -19.3
df['mb'] = pd.to_numeric(df['mb'], errors='coerce')
df['mb - Mb'] = df['mb'] - df['Mb']
df['zcmb'] = pd.to_numeric(df['zcmb'], errors='coerce')
df['dmb'] = pd.to_numeric(df['dmb'], errors='coerce')
df['mb + dmb'] = df['mb'] + df['dmb']

# Plotting the Scatter Plot using the Supernovae data
x = df['zcmb']
y = df['mb - Mb']
yerr = df['dmb']
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.scatter(x, y, color="black", marker="o", s=5)
plt.xscale('log')
plt.xlabel('redshift z')
plt.ylabel('Distance Modulus')
plt.title('Distance Modulus vs. zcmb')
plt.show()

# Defining equations and variables for potential universes
C = 3e8
H_0_1 = 70000
Omega_M_1 = 0.3
Omega_L_1 = 0.7
H_0_2 = 70000
Omega_M_2 = 1.0
Omega_L_2 = 1.0
H_0_3 = 70000
Omega_M_3 = 0.3
Omega_L_3 = 0.0
H_0 = 71800
Omega_M = 0.285
Omega_L = 0.715

q_0_1 = (1/2) * Omega_M_1 - Omega_L_1
df['dL1'] = ((C / H_0_1) * df['zcmb']) * (1 + ((1 - q_0_1) / 2) * df['zcmb'])
df['dL1'] = pd.to_numeric(df['dL1'], errors='coerce')
df['muLCDM1'] = df['mb'] - 5 * np.log10(df['dL1']) - 25
df['new_muLCDM1'] = 5 * np.log10(df['dL1']) + 25

q_0_2 = (1/2) * Omega_M_2 - Omega_L_2
df['dL2'] = ((C / H_0_2) * df['zcmb']) * (1 + ((1 - q_0_2) / 2) * df['zcmb'])
df['dL2'] = pd.to_numeric(df['dL2'], errors='coerce')
df['muLCDM2'] = df['mb'] - 5 * np.log10(df['dL2']) - 25
df['new_muLCDM2'] = 5 * np.log10(df['dL2']) + 25

q_0_3 = (1/2) * Omega_M_3 - Omega_L_3
df['dL3'] = ((C / H_0_3) * df['zcmb']) * (1 + ((1 - q_0_3) / 2) * df['zcmb'])
df['dL3'] = pd.to_numeric(df['dL3'], errors='coerce')
df['muLCDM3'] = df['mb'] - 5 * np.log10(df['dL3']) - 25
df['new_muLCDM3'] = 5 * np.log10(df['dL3']) + 25

q_0 = (1/2) * Omega_M - Omega_L
df['dL'] = ((C / H_0) * df['zcmb']) * (1 + ((1 - q_0) / 2) * df['zcmb'])
df['dL'] = pd.to_numeric(df['dL'], errors='coerce')
df['muLCDM'] = df['mb'] - 5 * np.log10(df['dL']) - 25
df['new_muLCDM'] = 5 * np.log10(df['dL']) + 25

# Over plot the corresponding theoretical distance modulus mu_LCDM, assuming the following cosmological parameters
x = df['zcmb']
y = df['mb - Mb']
yerr = df['dmb']
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.scatter(x, y, color="black", marker="o", s=5)
y_1 = df['new_muLCDM1']
y_2 = df['new_muLCDM2']
y_3 = df['new_muLCDM3']
plt.plot(x, y_1)
plt.plot(x, y_2)
plt.plot(x, y_3)
plt.xscale('log')
plt.xlabel('redshift z')
plt.ylabel('Distance Modulus')
plt.title('Distance Modulus vs. zcmb')
plt.show()

# The diagram using the data that has been optimize
x = df['zcmb']
y = df['mb - Mb']
yerr = df['dmb']
plt.errorbar(x, y, yerr=yerr, fmt=".k", capsize=0)
plt.scatter(x, y, color="black", marker="o", s=5)
y_1 = df['new_muLCDM']
plt.plot(x, y_1)
plt.xscale('log')
plt.xlabel('redshift z')
plt.ylabel('Distance Modulus')
plt.title('Distance Modulus vs. zcmb')
plt.show()

# Diagrams of three theoretical distance modulus
x = df['zcmb']
y = df['new_muLCDM1']
yerr = df['dmb']
plt.scatter(x, y, color="black", marker="o", s=5)
plt.xscale('log')
plt.xlabel('redshift z')
plt.ylabel('Theoretical Distance Modulus')
plt.title('Theoretical Distance Modulus vs. zcmb ')
plt.show()

x = df['zcmb']
y = df['new_muLCDM2']
yerr = df['dmb']
plt.scatter(x, y, color="black", marker="o", s=5)
plt.xscale('log')
plt.xlabel('redshift z')
plt.ylabel('Theoretical Distance Modulus')
plt.title('Theoretical Distance Modulus vs. zcmb ')
plt.show()

x = df['zcmb']
y = df['new_muLCDM3']
yerr = df['dmb']
plt.scatter(x, y, color="black", marker="o", s=5)
plt.xscale('log')
plt.xlabel('redshift z')
plt.ylabel('Theoretical Distance Modulus')
plt.title('Theoretical Modulus vs. zcmb ')
plt.show()

# chi^2 of the supernovea data
df['chi^2'] = ((df['mb'] - df['mb + dmb']) ** 2) / (df['mb + dmb'])
a = df['chi^2']
chi = a.sum()
print('chi^2', chi)
df['chi^2_percentage'] = df['chi^2'] / df['dmb']