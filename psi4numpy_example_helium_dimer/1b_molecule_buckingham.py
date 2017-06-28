from __future__ import print_function

import psi4

import numpy as np
from scipy.optimize import curve_fit

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


he_dimer = """
He
--
He 1 **R**
"""

distances = [2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
energies = []
for d in distances:
    mol = psi4.geometry(he_dimer.replace('**R**', str(d)))
    en = psi4.energy('MP2/aug-cc-pVDZ', molecule=mol, bsse_type='cp')
    en *= 219474.6
    energies.append(en)

# Lennard-Jones fit
powers = [-12, -6]
x = np.power(np.array(distances).reshape(-1, 1), powers)
coeffs = np.linalg.lstsq(x, energies)[0]
fpoints = np.linspace(2, 10, 100).reshape(-1, 1)
fdata = np.power(fpoints, powers)
fit_energies_lj = np.dot(fdata, coeffs)

# Buckingham fit
def buckingham(r, a, b, c):
    return a * np.exp(-b * r) - (c * r ** (-6))
popt, pcov = curve_fit(buckingham, distances, energies, method='trf')
fit_energies_buckingham = buckingham(fpoints, *popt)

# Morse fits
# def morse_2(r, a, b):
#     return a * (1 - np.exp(-b * r)) ** 2
# def morse_3(r, a, b, c):
#     return a * (1 - np.exp(-b * (r - c))) ** 2
def morse_4(r, a, b, c, d):
    return (a * (1 - np.exp(-b * (r - c))) ** 2) + d
# tstart = [1.0e+3, 1]
# popt, pcov = curve_fit(morse_2, distances, energies, method='trf', p0=tstart, maxfev=40000000)
# fit_energies_morse_2 = morse_2(fpoints, *popt)
# tstart = [3.41838629,  1.7536397,  3.32438717]
# popt, pcov = curve_fit(morse_3, distances, energies, method='trf', p0=tstart, maxfev=40000000)
# fit_energies_morse_3 = morse_3(fpoints, *popt)
tstart = [1.0e+3, 1, 3, 0]
popt, pcov = curve_fit(morse_4, distances, energies, method='trf', p0=tstart, maxfev=40000000)
fit_energies_morse_4 = morse_4(fpoints, *popt)

fig, ax = plt.subplots()
ax.set_xlim((2, 10))
ax.set_ylim((-6, 2))
ax.scatter(distances, energies, color='black', label='MP2/aug-cc-pVDZ (CP)')
ax.plot(fpoints, fit_energies_lj, label='Lennard-Jones fit')
ax.plot(fpoints, fit_energies_buckingham, label='Buckingham fit')
# ax.plot(fpoints, fit_energies_morse_2, label='Morse fit (2 param)')
# ax.plot(fpoints, fit_energies_morse_3, label='Morse fit')
ax.plot(fpoints, fit_energies_morse_4, label='Morse fit (shifted)')
ax.plot([0,10], [0,0], 'k-')
ax.set_xlabel(r'interatomic separation ($\AA{}$)')
ax.set_ylabel(r'interaction energy ($\mathrm{cm^{-1}}$)')
ax.legend(loc='best', fancybox=True, framealpha=0.50)
fig.savefig('1b_molecule_buckingham.pdf', bbox_inches='tight')
