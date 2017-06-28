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
}

step  = 0.05
start = min_distance - (10 * step)
stop  = min_distance + (25 * step)
interatomic_distances = np.arange(start, stop, step)

e_ints = OrderedDict([
    ('int_nocp', []),
    ('int_cp', []),
])

for idx, rint in enumerate(interatomic_distances):

    print('{}'.format(rint))

    psi4.core.set_output_file('output_ccsd_t_{}.dat'.format(idx), False)

    dimer = psi4.geometry(geometry_template(rint))
    dimer.update_geometry()
    psi4.set_options(options)
    psi4.energy('ccsd(t)', bsse_type=['nocp', 'cp'])

    e_int_nocp = psi4.get_variable("NON-COUNTERPOISE CORRECTED INTERACTION ENERGY")
    e_int_cp = psi4.get_variable("COUNTERPOISE CORRECTED INTERACTION ENERGY")

    e_ints['int_nocp'].append(e_int_nocp)
    e_ints['int_cp'].append(e_int_cp)

df = pd.DataFrame(e_ints, index=interatomic_distances)
print(df)
df.to_csv('e_ints_ccsd_t.csv')
