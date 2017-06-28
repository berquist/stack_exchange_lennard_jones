from __future__ import print_function

from collections import OrderedDict

import numpy as np

import pandas as pd

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)


labels = OrderedDict([
    ('elst10', 'electrostatics ($E_{\mathrm{elst}}^{(10)}$)'),
    ('exch10', 'exchange ($E_{\mathrm{exch}}^{(10)}$)'),
    # ('exch10_s2', 'exchange ($E_{\mathrm{exch}}^{(10)}(S^2)$)'),
    ('ind20', 'induction ($E_{\mathrm{ind,resp}}^{(20)}$)'),
    ('exch_ind20', 'exchange-induction ($E_{\mathrm{exch-ind,resp}}^{(20)}$)'),
    ('induction_delta_hf', 'delta HF ($\delta_{\mathrm{HF}}^{(2)}$)'),
    ('disp20', 'dispersion ($E_{\mathrm{disp}}^{(20)}$)'),
    ('exch_disp20', 'exchange-dispersion ($E_{\mathrm{exch-disp}}^{(20)}$)'),
    # ('total', 'total ($E_{SAPT0}$)'),
    # ('ct', 'charge transfer ($E_{\mathrm{ind}}^{(\mathrm{DCBS})} - E_{\mathrm{ind}}^{(\mathrm{MCBS})}$)'),
])

df = pd.read_csv('e_ints.csv', index_col=0)
print(df)

df['induction_delta_hf'] = df['hf'] - (df['elst10'] + df['exch10'] + df['ind20'] + df['exch_ind20'])
df['tot_elst'] = df['elst10']
df['tot_exch'] = df['exch10']
df['tot_ind'] = df['ind20'] + df['exch_ind20'] + df['induction_delta_hf']
df['tot_disp'] = df['disp20'] + df['exch_disp20']
df['tot'] = df['tot_elst'] + df['tot_exch'] + df['tot_ind'] + df['tot_disp']

df['tot_coulomb'] = df['tot_elst'] + df['tot_ind']
df['tot_vdw'] = df['tot_exch'] + df['tot_disp']

##########

conv = 627.503

fig, ax = plt.subplots()

for (etype, label) in labels.items():
    ax.plot(df.index, df[etype] * conv, label=label, marker='', linestyle='-')

ax.xaxis.grid(True)
ax.yaxis.grid(True)

ax.set_xlabel(r'interatomic distance ($\mathrm{\AA{}}$)')
ax.set_ylabel(r'energy (kcal/mol)')

ax.legend(loc='best', fancybox=True, framealpha=0.50, fontsize='small')

fig.savefig('plot_separate.pdf', bbox_inches='tight')

##########

fig, ax = plt.subplots()

ax.plot(df.index, conv * df['tot_elst'], label='total electrostatics ($E_{\mathrm{elst}}$)', marker='', color='black', linestyle='--')
ax.plot(df.index, conv * df['tot_exch'], label='total exchange ($E_{\mathrm{exch}}$)', marker='', color='black', linestyle='-.')
ax.plot(df.index, conv * df['tot_ind'], label='total induction ($E_{\mathrm{ind}}$)', marker='', color='black', linestyle=':')
ax.plot(df.index, conv * df['tot_disp'], label='total dispersion ($E_{\mathrm{disp}}$)', marker='', color='black', linestyle='-')
ax.plot(df.index, df['tot'], label='total interaction ($E_{\mathrm{SAPT0}}$)', marker='o', color='black', linestyle='')

ax.xaxis.grid(True)
ax.yaxis.grid(True)

ax.set_xlabel(r'interatomic distance ($\mathrm{\AA{}}$)')
ax.set_ylabel(r'energy (kcal/mol)')

ax.legend(loc='best', fancybox=True, framealpha=0.50, fontsize='small')

fig.savefig('plot_categories.pdf', bbox_inches='tight')

##########

distances = np.asarray(df.index)
conv = 219474.6
energies_vdw = np.asarray(df['tot_vdw']) * conv
energies_tot = np.asarray(df['tot']) * conv
energies_mp2 = np.asarray(df['_mp2']) * conv

# Lennard-Jones fit
powers = [-12, -6]
x = np.power(np.array(distances).reshape(-1, 1), powers)
coeffs_vdw = np.linalg.lstsq(x, energies_vdw)[0]
coeffs_tot = np.linalg.lstsq(x, energies_tot)[0]
coeffs_mp2 = np.linalg.lstsq(x, energies_mp2)[0]
fpoints = np.linspace(distances.min(), distances.max(), 100).reshape(-1, 1)
fdata = np.power(fpoints, powers)
fit_energies_vdw = np.dot(fdata, coeffs_vdw)
fit_energies_tot = np.dot(fdata, coeffs_tot)
fit_energies_mp2 = np.dot(fdata, coeffs_mp2)

# Check the quality of the fits.
# fit_energies_distances_vdw = np.dot(x, coeffs_vdw)
# fit_energies_distances_tot = np.dot(x, coeffs_tot)
# fit_energies_distances_mp2 = np.dot(x, coeffs_mp2)
# from scipy import stats
# slope_vdw, intercept_vdw, r_value_vdw, p_value_vdw, std_err_vdw = stats.linregress(energies_vdw, fit_energies_distances_vdw)
# slope_tot, intercept_tot, r_value_tot, p_value_tot, std_err_tot = stats.linregress(energies_tot, fit_energies_distances_tot)
# slope_mp2, intercept_mp2, r_value_mp2, p_value_mp2, std_err_mp2 = stats.linregress(energies_mp2, fit_energies_distances_mp2)
# rsq_vdw = r_value_vdw ** 2
# rsq_tot = r_value_tot ** 2
# rsq_mp2 = r_value_mp2 ** 2
# print(slope_vdw, intercept_vdw, r_value_vdw, p_value_vdw, std_err_vdw)
# print(slope_tot, intercept_tot, r_value_tot, p_value_tot, std_err_tot)
# print(slope_mp2, intercept_mp2, r_value_mp2, p_value_mp2, std_err_mp2)
# print(rsq_vdw)
# print(rsq_tot)
# print(rsq_mp2)

fig, ax = plt.subplots()

ax.plot(distances, df['tot_coulomb'] * conv, label='electrostatics + induction', marker='s', color='green', linestyle='')
ax.plot(distances, energies_vdw, label='exchange + dispersion', marker='*', color='blue', linestyle='')
ax.plot(distances, energies_tot, label=r'total interaction ($E_{\mathrm{SAPT0}}$)', marker='o', color='orange', linestyle='')
ax.plot(distances, energies_mp2, label=r'total interaction ($\Delta E_{\mathrm{MP2}}$)', marker='^', color='red', linestyle='')
ax.plot(fpoints, fit_energies_vdw, marker='', color='blue', linestyle='-', label='Lennard-Jones fit (exchange + dispersion)')
ax.plot(fpoints, fit_energies_tot, marker='', color='orange', linestyle='--', label='Lennard-Jones fit (total SAPT0 interaction)')
ax.plot(fpoints, fit_energies_mp2, marker='', color='red', linestyle='-.', label='Lennard-Jones fit (total MP2 interaction)')

ax.xaxis.grid(True)
ax.yaxis.grid(True)

ax.set_xlabel(r'interatomic distance ($\mathrm{\AA{}}$)')
ax.set_ylabel(r'energy ($\mathrm{cm}^{-1}$)')

ax.legend(loc='best', fancybox=True, framealpha=0.50, fontsize='small')

fig.savefig('plot_categories_coulomb_vdw.pdf', bbox_inches='tight')

plt.close('all')
