# Comparing Luminosity-Distance Indicators with Strong-Lens Time Delays and BAO

The analysis combines DESI DR2 BAO with the TDCOSMO 2025 eight-system
hierarchical time-delay likelihood. It compares Pantheon+ Type Ia supernovae,
X-ray/UV quasars, and A118 gamma-ray bursts within a common distance model.

## Rebuild the results

Use Python 3.11 in an activated virtual environment or Conda environment.

```sh
python -m pip install -r requirements.txt
python reproduce.py
```

The script verifies the archive against `manifest.json`, checks the
recomputed numerical summaries against the archived reference summaries,
and writes the results to `results/`. No account or download is required
after the Python dependencies are installed. On Windows, activate the Conda
environment before running the command so that its DLLs are on the search path.

- `paper_summary.csv` contains the paper’s cosmological constraints. `posterior_summary.csv` also includes the archived baseline outputs, including the earlier prior-reweighted Pantheon+ check.
- `tables/` contains LaTeX table fragments for the nine results tables and
  numerical diagnostics for each ensemble. Table 1 of the manuscript lists
  the fixed lens priors defined in `inference.py`.
- `width_diagnostics.csv` records differences in the 68% interval widths
  for the node and matched-count comparisons. Chain-half and scramble
  columns are available for the additional fits. These differences are
  numerical checks and do not represent confidence intervals.
- `pantheon_selection_rows.csv` and `qso_grb_selections.csv` identify retained
  sources and, for Pantheon+, their zero-based covariance rows.
- The four figure PDFs cover redshift distributions, joint posteriors,
  low-redshift sensitivity, and the reconstructed expansion history.
  The triangle plot uses `data/cosmo_tools.py`.

The default command uses the supplied weighted posterior samples. It
does not rerun MCMC. To rerun an individual sensitivity fit, including
its two refined Sobol quadratures, use the same entry point:

```sh
python reproduce.py --fit z010_count --workers 8 --output reruns
```

The fit identifiers and settings are listed in
`lens_sn_grb_pilot/aj_checks/aj_opus_revision/run_plan.json`. The corresponding
directories contain the archived starting positions. A full fit takes longer
than regenerating the paper products. The historical flat-Lambda-CDM benchmark,
direct spline validation, and auxiliary Gaussian-distance comparison are
provided as saved validation records.

## Code and data

`reproduce.py` assembles results and provides the command-line entry point.
`inference.py` contains the likelihoods, sampler, and refinement.
`data/cosmo_tools.py` is the plotting helper. `vendor.zip` contains the exact
HierArc, lenstronomy, and threadpoolctl snapshots for the processed likelihood,
with their supplied licenses and metadata. The input files and weighted draws
are separate repository files, with checksums and original input sources in
`manifest.json`. Cite the originating publications when using their data.

The LaTeX submission archive and authors' contact details are distributed
with the manuscript. The research outputs in this repository are keyed
to the manuscript revision of 2026-11-23.

## License

The original analysis code and scripts in this repository are released under
the MIT License (see `LICENSE`). Third-party public datasets (`inputs/`,
`tdc_inputs/`, `data/J_A+A_642_A150.tar/`, and `lens_sn_grb_pilot/bao_dr2/`)
and vendored third-party libraries (`vendor.zip`) remain subject to their
respective upstream licenses and citation terms.

