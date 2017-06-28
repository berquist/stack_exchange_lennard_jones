from __future__ import print_function

from collections import OrderedDict

import numpy as np
import pandas as pd

import psi4

psi4.set_memory('2 GB')
psi4.core.set_num_threads(4)

min_distance = 1.449701489552 + 0.953085394995
geometry_template = """
1 1
Na           0.000000000000     0.000000000000     0.000
--
-1 1
Cl           0.000000000000     0.000000000000     {}

symmetry c1
units angstrom
no_reorient
no_com
""".format

# Note: can't use Python bool type
# can't freeze core?
options = {
    'reference': 'rhf',
    'scf_type': 'direct',
    'df_scf_guess': 'false',
    'basis': 'def2-svp',
    'df_basis_sapt': 'def2-svp-ri',
}

step  = 0.05
start = min_distance - (10 * step)
stop  = min_distance + (25 * step)
interatomic_distances = np.arange(start, stop, step)

e_ints = OrderedDict([
    ('elst10', []),
    ('exch10', []),
    ('exch10_s2', []),
    ('ind20', []),
    ('exch_ind20', []),
    ('disp20', []),
    ('exch_disp20', []),
    ('hf', []),
    # ('dhf2', []),
    # ('ct', []),
    ('_mp2', []),
])

for idx, rint in enumerate(interatomic_distances):

    print('{}'.format(rint))

    psi4.core.set_output_file('output_{}.dat'.format(idx), False)

    dimer = psi4.geometry(geometry_template(rint))
    dimer.update_geometry()
    psi4.set_options(options)
    psi4.energy('sapt0')

    e_elst10 = psi4.get_variable("SAPT ELST10,R ENERGY")
    e_exch10 = psi4.get_variable("SAPT EXCH10 ENERGY")
    e_exch10_s2 = psi4.get_variable('SAPT EXCH10(S^2) ENERGY')
    e_ind20 = psi4.get_variable("SAPT IND20,R ENERGY")
    e_exch_ind20 = psi4.get_variable("SAPT EXCH-IND20,R ENERGY")
    e_disp20 = psi4.get_variable("SAPT DISP20 ENERGY")
    e_exch_disp20 = psi4.get_variable("SAPT EXCH-DISP20 ENERGY")
    e_hf = psi4.get_variable("SAPT HF TOTAL ENERGY")

    # e_dhf2 = e_hf - (e_elst10 + e_exch10 + e_ind20 + e_exch_ind20)

    # e_tot_elst = e_elst10
    # e_tot_exch = e_exch10
    # e_tot_ind = e_ind20 + e_exch_ind20 + e_dhf2
    # e_tot_disp = e_disp20 + e_exch_disp20

    # e_sapt0 = e_tot_elst + e_tot_exch + e_tot_ind + e_tot_disp

    # e_ct = psi4.get_variable("SAPT CT ENERGY")

    e_ints['elst10'].append(e_elst10)
    e_ints['exch10'].append(e_exch10)
    e_ints['exch10_s2'].append(e_exch10_s2)
    e_ints['ind20'].append(e_ind20)
    e_ints['exch_ind20'].append(e_exch_ind20)
    e_ints['disp20'].append(e_disp20)
    e_ints['exch_disp20'].append(e_exch_disp20)
    e_ints['hf'].append(e_hf)
    # e_ints['dhf2'].append(e_dhf2)
    # e_ints['ct'].append(e_ct)

    psi4.core.print_variables()
    psi4.core.clean()
    psi4.core.clean_variables()

    # Get the regular old MP2 binding energy.
    psi4.energy('mp2', bsse_type='nocp')
    e_int_mp2_nocp = psi4.get_variable("NON-COUNTERPOISE CORRECTED INTERACTION ENERGY")
    e_ints['_mp2'].append(e_int_mp2_nocp)

    psi4.core.print_variables()
    psi4.core.clean()
    psi4.core.clean_variables()

df = pd.DataFrame(e_ints, index=interatomic_distances)
print(df)
df.to_csv('e_ints.csv')
