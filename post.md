

https://chemistry.stackexchange.com/questions/76708/how-to-calculate-lennard-jones-potential-by-computational-quantum-mechanical-too

# How to calculate Lennard-Jones potential with quantum mechanical methods

# Question

I want to know the procedure to calculate the Lennard-Jones potential for a metal-halogen pair (specifically vanadium-chlorine). Is it possible to calculate using any QM packages like Mopac, NWChem, or Gaussian?

PS: I am specifically looking for values of [A and B in the 12-6 potential][1].

  [1]: https://en.wikipedia.org/wiki/Lennard-Jones_potential#AB_form
  
  tags: quantum-chemistry computational-chemistry intermolecular-forces software

# Comments

https://chat.stackexchange.com/transcript/message/38346524#38346524

- "Whereas the functional form of the attractive term has a clear physical justification, the repulsive term has no theoretical justification." So if the very empiric equation contains even unjustified parts that make the model "just fit somehow", then I don’t think that there is something to debate whether it works here or not.

- The repulsive part is better described by the exponential in the Buckingham equation, though the $r^{-12}$ fits this reasonably well. I still argue that an equation that has the proper functional form of exchange and dispersion interactions can't be used to fit the Coulomb interaction. This is why there are both terms in force fields, and not just the LJ term.

- If this was argon-argon, then fine. There's no Coulomb interaction between them, their interaction is purely dispersive until their electron clouds penetrate each other and the exchange repulsion term dominates. But vanadium and chlorine, which are more "typical" atoms, will have a non-negligible electrostatic interaction.

- But the question was not "what potential to choose" but "how can I produce data to fit the LJ potential to" ... no?

- That's a good point, let me re-read it

- I see what you're saying. This is of course technically possible. It definitely isn't done this way, in part because of the physical limitations of the LJ model. But you can do it.

- Then you might want to write an answer and work out the part about choosing another potential.

- That is, you would want to neglect the Coulomb part of the interaction when fitting $A/B$.

# Answer

Yes, this is technically possible. A basic tutorial for this is in the excellent [Psi4Numpy project](https://github.com/psi4/psi4numpy/blob/master/Tutorials/01_Psi4NumPy-Basics/1b_molecule.ipynb), which I'll reproduce here with minor modifications. Their example fits the counterpoise-corrected MP2/aug-cc-pVDZ _total_ interaction energy of the helium dimer.

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
    ax.set_xlabel(r'intertomic separation ($\AA{}$)')
    ax.set_ylabel(r'interaction energy ($\mathrm{cm^{-1}}$)')
    fig.savefig('1b_molecule.pdf', bbox_inches='tight')

Once the set of energies is calculated, a [least-squares polynomial fit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html) is performed, giving

$$
f(x) = 6677721.45419193x^{-12} - 11394.79882998x^{-6}.
$$

This isn't a good (efficient or general) way of evaluating the fit function, but illustrates the placement of correct signs in the above code:

    def f(z):
        return coeffs[0]*z**powers[0] + coeffs[1]*z**powers[1]

From the above we can see that the first and second coefficients in the polynomial fit correspond directly to $A$ and $B$, respectively. In this example, the energies are in wavenumbers ($\pu{cm^{-1}}$). Most quantum packages print in atomic units (Hartrees) and dynamics packages print in kJ/mol, which is fine, but be aware that in the above fit the units are incorporated into $A$ and $B$. Here is the resulting fit compared against the individual interaction energy calculations:

<img src="https://i.stack.imgur.com/dFeig.png" />

---

This was not in the question, but I think it's important to ask if this is the correct approach and investigate further. Notice how the fit is qualitatively ok near the minimum, but the shape in the dissociation region is qualitatively wrong, and the potential simply isn't attractive enough in the limit of infinite separation. This could arise in three ways:

1. The curve fitting isn't working properly, or there's a bug in the code.
2. The functional form of the chosen interatomic potential is incorrect.
3. The (Boys-Bernardi) counterpoise correction, which is known to overcompensate for BSSE, is giving strange results at large separation.

Assume that point 1 isn't an issue (if there's a bug anywhere, please comment). Point 2 can be investigated by also fitting against the [Buckingham](https://en.wikipedia.org/wiki/Buckingham_potential) and shifted [Morse](https://en.wikipedia.org/wiki/Morse_potential) potentials, which should display better short-range and long-range behavior:

\begin{align}
V_{\text{Buckingham}}(r;A,B,C) &= A e^{-Br} - \frac{C}{r^{6}} \\
V_{\text{Morse}}(r;A,B,C,D) &= A \left( 1 - e^{-B(r-C)} \right)^{2} + D
\end{align}

<img src="https://i.stack.imgur.com/Z3aqf.png" />

Note that for these energies, an unshifted Morse potential (no $D$ parameter) fails miserably. See [here](https://stackoverflow.com/questions/36312303/morse-potential-fit-using-python-and-curve-fit-from-scipy) for more information about fitting Morse-type potentials. The Buckingham and Morse fits require a [more general (arbitrary) curve fitting routine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html), and the script that made this plot can be found as an HTML comment in the post source code below.

<!-- from __future__ import print_function -->

<!-- import psi4 -->

<!-- import numpy as np -->
<!-- from scipy.optimize import curve_fit -->

<!-- import matplotlib as mpl -->
<!-- mpl.use('Agg') -->
<!-- import matplotlib.pyplot as plt -->


<!-- he_dimer = """ -->
<!-- He -->
<!-- -- -->
<!-- He 1 **R** -->
<!-- """ -->

<!-- distances = [2.875, 3.0, 3.125, 3.25, 3.375, 3.5, 3.75, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] -->
<!-- energies = [] -->
<!-- for d in distances: -->
<!--     mol = psi4.geometry(he_dimer.replace('**R**', str(d))) -->
<!--     en = psi4.energy('MP2/aug-cc-pVDZ', molecule=mol, bsse_type='cp') -->
<!--     en *= 219474.6 -->
<!--     energies.append(en) -->

<!-- # Lennard-Jones fit -->
<!-- powers = [-12, -6] -->
<!-- x = np.power(np.array(distances).reshape(-1, 1), powers) -->
<!-- coeffs = np.linalg.lstsq(x, energies)[0] -->
<!-- fpoints = np.linspace(2, 10, 100).reshape(-1, 1) -->
<!-- fdata = np.power(fpoints, powers) -->
<!-- fit_energies_lj = np.dot(fdata, coeffs) -->

<!-- # Buckingham fit -->
<!-- def buckingham(r, a, b, c): -->
<!--     return a * np.exp(-b * r) - (c * r ** (-6)) -->
<!-- popt, pcov = curve_fit(buckingham, distances, energies, method='trf') -->
<!-- fit_energies_buckingham = buckingham(fpoints, *popt) -->

<!-- # Morse fits -->
<!-- # def morse_2(r, a, b): -->
<!-- #     return a * (1 - np.exp(-b * r)) ** 2 -->
<!-- # def morse_3(r, a, b, c): -->
<!-- #     return a * (1 - np.exp(-b * (r - c))) ** 2 -->
<!-- def morse_4(r, a, b, c, d): -->
<!--     return (a * (1 - np.exp(-b * (r - c))) ** 2) + d -->
<!-- # tstart = [1.0e+3, 1] -->
<!-- # popt, pcov = curve_fit(morse_2, distances, energies, method='trf', p0=tstart, maxfev=40000000) -->
<!-- # fit_energies_morse_2 = morse_2(fpoints, *popt) -->
<!-- # tstart = [3.41838629,  1.7536397,  3.32438717] -->
<!-- # popt, pcov = curve_fit(morse_3, distances, energies, method='trf', p0=tstart, maxfev=40000000) -->
<!-- # fit_energies_morse_3 = morse_3(fpoints, *popt) -->
<!-- tstart = [1.0e+3, 1, 3, 0] -->
<!-- popt, pcov = curve_fit(morse_4, distances, energies, method='trf', p0=tstart, maxfev=40000000) -->
<!-- fit_energies_morse_4 = morse_4(fpoints, *popt) -->

<!-- fig, ax = plt.subplots() -->
<!-- ax.set_xlim((2, 10)) -->
<!-- ax.set_ylim((-6, 2)) -->
<!-- ax.scatter(distances, energies, color='black', label='MP2/aug-cc-pVDZ (CP)') -->
<!-- ax.plot(fpoints, fit_energies_lj, label='Lennard-Jones fit') -->
<!-- ax.plot(fpoints, fit_energies_buckingham, label='Buckingham fit') -->
<!-- # ax.plot(fpoints, fit_energies_morse_2, label='Morse fit (2 param)') -->
<!-- # ax.plot(fpoints, fit_energies_morse_3, label='Morse fit') -->
<!-- ax.plot(fpoints, fit_energies_morse_4, label='Morse fit (shifted)') -->
<!-- ax.plot([0,10], [0,0], 'k-') -->
<!-- ax.set_xlabel(r'interatomic separation ($\AA{}$)') -->
<!-- ax.set_ylabel(r'interaction energy ($\mathrm{cm^{-1}}$)') -->
<!-- ax.legend(loc='best', fancybox=True, framealpha=0.50) -->
<!-- fig.savefig('1b_molecule_buckingham.pdf', bbox_inches='tight') -->

I find it interesting that the Buckingham potential gives a worse fit than the Lennard-Jones one, even though it is supposed to reproduce the repulsive wall better with $e^{-r}$ rather than $\frac{1}{r^{6}}$. The Morse fit is remarkably good, which makes sense considering there are 4 free parameters rather than 3 (Buckingham) or 2 (Lennard-Jones). Before going any further, a check on point 3 by setting `bsse_type=nocp`:

<img src="https://i.stack.imgur.com/q8CRZ.png" />

This reveals that the counterpoise correction is definitely interfering with the quality of the fit, and the functional forms of the Lennard-Jones and Buckingham potential appear to not describe dissociation properly, though extended to infinite separation, the Morse potential is the one that is qualitatively incorrect. This can be attributed to using a single functional form to describe the _total_ interaction or binding energy, and not the different components. I suspect that the apparent poor fit is due to a non-zero Coulomb interaction, rather than van der Waals-type interactions which at least the Lennard-Jones potential is meant to be used for.

To test this, here are interaction energy calculations at two levels of sophistication, both based on symmetry-adapted perturbation theory (SAPT). This is still the helium dimer, at 7 angstroms separation, with the aug-cc-pVDZ basis set and the monomer-centered basis approximation. All units are kcal/mol.

$$
\small
\begin{array}{lrr}
\hline
\text{Component}  & \text{SAPT0} & \text{SAPT2+3(CCD)}\delta_{\text{MP2}} \\
\hline
\text{Electrostatics} & -0.01796877 & -0.01777334 \\
\text{Exchange}       &  0.00000000 &  0.00000000 \\
\text{Induction}      &  0.01587019 &  0.01508578 \\
\text{Dispersion}     & -0.00013235 & -0.00016773 \\ \hline
\text{Electrostatics + Induction} & -0.00209858 & -0.00268756 \\ \hline
\text{Total}          & -0.00223093 & -0.00285529 \\
\hline
\end{array}
$$

At both levels of SAPT, the non-dispersive part of the interaction energy accounts for 94% of the total interaction! Is this true at 3 angstroms separation?

$$
\small
\begin{array}{lrr}
\hline
\text{Component}  & \text{SAPT0} & \text{SAPT2+3(CCD)}\delta_{\text{MP2}} \\
\hline
\text{Electrostatics} & -0.04852183 & -0.04836334 \\
\text{Exchange}       & 0.02045965 & 0.02219227 \\
\text{Induction}      & 0.04202349 & 0.04081412 \\
\text{Dispersion}     & -0.02228767 & -0.02843936 \\ \hline
\text{Electrostatics + Induction} & -0.00649834 & -0.00754922 \\ \hline
\text{Total}          & -0.00832637 & -0.01379630 \\
\hline
\end{array}
$$

The non-vdW interactions now account for 78% and 55% of the total interaction energy for each SAPT flavor, respectively. Although induction (also called polarization) is repulsive, I am including it in the Coulomb-type interaction since it is not purely quantum mechanical in nature like the exchange term. It is the exchange term, not induction, that is closer to the charge cloud penetration picture. Even so, the interaction between two helium atoms is not purely based on dispersion, and requires fitting another nonbonded term for electrostatics, the Coulomb term:

$$
V_{\text{Coulomb}}(\vec{r}_{i},\vec{r}_{j},q_{i},q_{j}) = \frac{1}{4\pi\epsilon_0} \frac{q_{i}q_{j}}{|\vec{r}_{i} - \vec{r}_{j}|}
$$

Our goal is to separate out the Coulomb-like terms from the interaction energy, so that $A$ will describe only the exchange contribution and $B$ will describe only the dispersion contribution; electrostatics and induction will be handled separately in the Coulomb term. This will be done by fitting only to the combination of exchange and dispersion, which are taken from SAPT calculations. MP2 binding energies are not CP-corrected.

<img src="https://i.stack.imgur.com/GHvfO.png" />

As you can see, using SAPT energies provides a much better fit than naively using the total interaction or binding energies. I'm not sure why the total SAPT energy works so well, but it may be due to the absolute magnitudes at each point being so small.

The importance of only fitting exchange and dispersion is obvious for $\ce{Na^+---Cl^-}$ (using def2-SVP for the basis).

<img src="https://i.stack.imgur.com/IsfQU.png" />

This is an unfair example, since the dispersion interaction is dwarfed by exchange, and it is an ionically-bound molecule. As a final example, consider $\ce{V(III)---Cl^-}$ (using def2-SV(P) for the basis, vanadium as a quintet).

<img src="https://i.stack.imgur.com/sa1O4.png" />

I had some trouble converging many of the calculations, presumably due to a non-optimal spin state at long range. For this reason (spin-state crossing), Lennard-Jones type models are poor for metal-x interactions. The problem appears to be similar to the sodium chloride, but the exchange plus dispersion fit looks good.

If you want to reproduce any of the work, rather than fish through the source for inputs and scripts, everything is in a [GitHub repository](). More references for SAPT can be found [here](https://chemistry.stackexchange.com/a/62962/194), along with the [Psi4 manual](http://psicode.org/psi4manual/master/sapt.html). Note that I didn't do any literature searching here, but I expect people have used energy decomposition approaches in the past as part of force field design. Hopefully this serves as a good starting point.
