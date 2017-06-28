from __future__ import print_function

from collections import OrderedDict

import numpy as np

import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


# labels = OrderedDict([
#     ('int_nocp', 'No CP'),
#     ('int_cp', 'CP'),
# ])

df = pd.read_csv('e_ints_ccsd_t.csv', index_col=0)
print(df)

distances = np.asarray(df.index)
energies = np.asarray(df['int_cp'])
print(distances)
print(energies)

# Fit data in least-squares way to a -12, -6 polynomial
powers = [-12, -6]
x = np.power(np.array(distances).reshape(-1, 1), powers)
coeffs, residuals, rank, s = np.linalg.lstsq(x, energies)
print('coeffs')
print(coeffs)
print('residuals')
print(residuals)
print('s')
print(s)

# Build list of points
fpoints = np.linspace(min(distances), max(distances), 100).reshape(-1, 1)
fdata = np.power(fpoints, powers)
print('fdata.shape')
print(fdata.shape)

fit_energies = np.dot(fdata, coeffs)

##########

fig, ax = plt.subplots()

ax.plot(distances, energies * 627.503, label='calculation (CCSD(T)/def2-SVP, CP)', marker='o', linestyle='')
ax.plot(fpoints, fit_energies * 627.503, label='fit', marker='', linestyle='-')

ax.xaxis.grid(True)
ax.yaxis.grid(True)

ax.set_xlabel(r'interatomic distance ($\mathrm{\AA{}}$)')
ax.set_ylabel(r'interaction energy (kcal/mol)')

ax.legend(loc='best', fancybox=True, framealpha=0.50, fontsize='small')

fig.savefig('plot_separate.pdf', bbox_inches='tight')
