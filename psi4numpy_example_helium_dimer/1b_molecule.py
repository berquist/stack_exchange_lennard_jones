from __future__ import print_function

import psi4

import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


he_dimer = """
He
--
He 1 **R**
"""

distances = [2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0]
energies = []
for d in distances:
    # Build a new molecule at each separation
    mol = psi4.geometry(he_dimer.replace('**R**', str(d)))
    
    # Compute the Counterpoise-Corrected interaction energy
    en = psi4.energy('MP2/aug-cc-pVDZ', molecule=mol, bsse_type='cp')

    # Place in a reasonable unit, Wavenumbers in this case
    en *= 219474.6
    
    # Append the value to our list
    energies.append(en)

print("Finished computing the potential!")

# Fit data in least-squares way to a -12, -6 polynomial
powers = [-12, -6]
x = np.power(np.array(distances).reshape(-1, 1), powers)
coeffs = np.linalg.lstsq(x, energies)[0]

# Build list of points
fpoints = np.linspace(2, 7, 50).reshape(-1, 1)
fdata = np.power(fpoints, powers)

fit_energies = np.dot(fdata, coeffs)

fig, ax = plt.subplots()
ax.set_xlim((2, 7))  # X limits
ax.set_ylim((-7, 2))  # Y limits
ax.scatter(distances, energies)  # Scatter plot of the distances/energies
ax.plot(fpoints, fit_energies)  # Fit data
ax.plot([0,10], [0,0], 'k-')  # Make a line at 0
ax.set_xlabel(r'interatomic separation ($\AA{}$)')
ax.set_ylabel(r'interaction energy ($\mathrm{cm^{-1}}$)')
fig.savefig('1b_molecule.pdf', bbox_inches='tight')
