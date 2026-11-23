"""Rebuild manuscript results from the archived posterior samples."""

import argparse
import csv
from hashlib import file_digest
from importlib.util import module_from_spec, spec_from_file_location
import json
from math import isclose
import os
from pathlib import Path
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import solve_triangular
from scipy.special import logsumexp

from inference import (
    PROJECT, AMATI, BOUNDS, GRB_PATH, NAMES, QSO_PATH, RATE_Z, REFERENCE_H, ROOT, SPLINE_NODES,
    CandleLikelihood, GeometryLikelihood, distance_geometry, pack_geometry, prepare_sn,
    read_qso, refine, sample, sn_subsample, uncapped_likelihood, weighted_quantile,
)


ORIGINAL = ROOT/'aj_checks'/'tdcosmo2025'


FOLLOWUP = ROOT/'aj_checks'/'aj_followup'


REPEATED = {'bao_lenses', 'qso_grb', 'pantheon_plus_z023'}


LABELS = {
    'bao_lenses': 'BAO + time delays',
    'pantheon_plus': 'Pantheon+',
    'qso': 'Quasars',
    'grb': 'GRBs',
    'qso_grb': 'Quasars + GRBs',
    'qso_grb_evolution': 'Quasars + GRBs, evolving intercepts',
    'qso_grb_groups': 'Quasars + GRBs, survey offsets',
    'pantheon_plus_z023': r'Pantheon+, $z_{\mathrm{HD}}\geq0.23$',
    'pantheon_plus_lowz_matched': 'Pantheon+, nearby fraction matched',
    'pantheon_plus_count_control': 'Pantheon+, matched count control',
    'bao_lenses_smooth3': 'BAO + time delays, smoothing width 3',
    'qso_grb_smooth3': 'Quasars + GRBs, smoothing width 3',
    'pantheon_plus_smooth3': 'Pantheon+, smoothing width 3',
}


CORE = ('H0', 'r_d', 'Omega_k', 'H0_r_d', 'k70')


COLORS = ('#304967', '#ad492d', '#237e75')


def load_run(directory, case):
    path = directory/f'{case}_refined.json'
    metadata = json.loads(path.read_text(encoding='utf8'))
    chain_meta = json.loads((directory/f"{metadata['source']}.json").read_text(encoding='utf8'))
    with np.load(directory/f'{case}_refined_draws.npz') as f:
        names = f['parameter_names'].tolist()
        point = f['points']
        logw = f['log_relative_weight']
    w = np.exp(logw-logsumexp(logw))
    for k in ('H0', 'r_d', 'Omega_k'):
        if not np.isfinite(metadata['combined']['parameters'][k]).all():
            raise ValueError(f'Invalid posterior summary in {path}')
    return names, point, w, metadata, chain_meta


def derived(names, point):
    values = dict(zip(names, point.T))
    h = REFERENCE_H*np.exp(values['log_H0_over_70'])
    product = REFERENCE_H*values['H0rd_over_70']
    values.update(H0=h, r_d=product/h, H0_r_d=product,
                  k70=values['Omega_k']*(h/REFERENCE_H)**2,
                  E015=np.exp(values['ln_E_z0.15']))
    return values


def summarize(names, point, w):
    v = derived(names, point)
    stats = {k: weighted_quantile(val, w) for k, val in v.items()}
    centered = np.column_stack([v['H0'], v['E015']])
    centered -= w @ centered
    cov = centered.T @ (centered*w[:, None])
    return dict(quantiles=stats, quadrature_ess=float(1/(w@w)), max_weight=float(w.max()),
                corr_h0_e015=float(cov[0, 1]/np.sqrt(cov[0, 0]*cov[1, 1])))


def assemble(case):
    input_case = case.removesuffix('_smooth3')
    if case.endswith('_smooth3') and input_case != 'pantheon_plus':
        directories = [FOLLOWUP/'smooth3']
    elif input_case == 'pantheon_plus':
        directories = [ORIGINAL]
    elif input_case in REPEATED:
        directories = [ORIGINAL, FOLLOWUP]
    else:
        directories = [FOLLOWUP]
    runs = [load_run(d, input_case) for d in directories]
    return combine(case, runs)


def combine(case, runs):
    names = runs[0][0]
    if any(r[0] != names for r in runs):
        raise ValueError(f'Incompatible parameter names for {case}')
    reference = runs[0][-1]
    defaults = dict(h0_max=105., qso_z_max=1.3, smooth=1.5, sn_rows_selected=None, sn_rows=None,
                    qso_rows=0, grb_rows=0)
    for run in runs[1:]:
        changed = [k for k, default in defaults.items()
                   if run[-1].get(k, default) != reference.get(k, default)]
        if changed:
            raise ValueError(f"Incompatible metadata in {case}, {run[3]['source']}: {changed}")
    point = np.concatenate([r[1] for r in runs])
    w = np.concatenate([r[2] for r in runs])/len(runs)
    if case == 'pantheon_plus_smooth3':
        z = np.log1p(RATE_Z)
        columns = [names.index(f'ln_E_z{t:g}') for t in RATE_Z[1:]]
        slopes = np.diff(np.c_[np.zeros(len(point)), point[:, columns]], axis=1)/np.diff(z)
        change = np.sum(np.diff(slopes, axis=1)**2, axis=1)
        logw = np.log(w)-.5*change*(3.**-2-1.5**-2)
        w = np.exp(logw-logsumexp(logw))
    summary = summarize(names, point, w)
    summary['fit_config'] = 'importance reweighting' if case == 'pantheon_plus_smooth3' else 'direct MCMC with refined lens quadrature'
    summary['ensembles'] = [
        {**{key: r[3][key] for key in ('source', 'physical_retained_over_tau', 'weighted_half_summaries')},
         **{key: r[3].get(key) for key in ('source_steps', 'refined_power', 'refined_rx_power', 'points')},
         'chain_diagnostics': r[-1].get('diagnostics', {}),
         'combined': {key: r[3]['combined'][key] for key in ('ess', 'max_weight', 'parameters')},
         'quadrature_medians': [{k: x['parameters'][k][1] for k in CORE} for x in r[3]['independent']]}
        for r in runs]
    meta = runs[-1][-1]
    summary['counts'] = {key: meta.get(key) for key in ('sn_rows', 'sn_unique', 'sn_rows_z_lt_015',
                                                        'sn_unique_z_lt_015', 'qso_rows', 'grb_rows')}
    summary['common_support'] = {key: meta.get(key, defaults[key]) for key in ('h0_max', 'qso_z_max', 'smooth')}
    if case.endswith('_smooth3'):
        summary['common_support']['smooth'] = 3.
    return summary, names, point, w


def interval(x, digits):
    lower, middle, upper = x
    return rf'\({middle:.{digits}f}^{{+{upper-middle:.{digits}f}}}_{{-{middle-lower:.{digits}f}}}\)'


def parameter_width(data, case, param):
    q = data[case][0]['quantiles'][param]
    return q[2]-q[0]


def table(data, cases, caption, label, label_map=None):
    labels = LABELS if label_map is None else label_map
    body = []
    for key in cases:
        q = data[key][0]['quantiles']
        body.append(' & '.join([labels[key], interval(q['H0'], 1), interval(q['r_d'], 1),
                                interval(q['Omega_k'], 3), interval(q['H0_r_d'], 0)]) + r' \\')
    return '\n'.join([
        r'\begin{deluxetable*}{lcccc}',
        rf'\tablecaption{{{caption}\label{{{label}}}}}',
        r'\tablehead{\colhead{Data or variant} & \colhead{\(H_0\) (\(\Hunit\))} &',
        r'\colhead{\(\rd\) (Mpc)} & \colhead{\(\OmK\)} & \colhead{\(H_0\rd\) (\(\mathrm{km\,s^{-1}}\))}}',
        r'\startdata', *body, r'\enddata',
        r'\tablecomments{Medians and central 68\% posterior intervals. Quoted interval widths are calculated before rounding. All rows include DESI DR2 BAO and the eight TDCOSMO 2025 time-delay systems.}',
        r'\end{deluxetable*}', ''
    ])


def plot_posteriors(data, folder, **plot_style):
    plotting_path = ROOT.parent/'data'/'cosmo_tools.py'
    spec = spec_from_file_location('manuscript_cosmo_tools', plotting_path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load {plotting_path}')
    cosmo_tools = module_from_spec(spec)
    spec.loader.exec_module(cosmo_tools)

    samples = []
    for case in ('bao_lenses', 'pantheon_plus', 'qso_grb'):
        _, names, point, w = data[case]
        idx = np.searchsorted(np.cumsum(w), (np.arange(24000)+.5)/24000)
        v = derived(names, point[idx])
        samples.append(np.column_stack([v[k] for k in CORE[:3]]))
    styles = ['-', '--', '-.']
    cosmo_tools.plot_getdist_comparison(
        samples,
        [r'H_0~[\mathrm{km\,s^{-1}\,Mpc^{-1}}]', r'r_{\rm d}~[\mathrm{Mpc}]', r'\Omega_{\rm k}'],
        ['BAO + time delays', '+ Pantheon+', '+ quasars and GRBs'],
        colors=COLORS, filled=False, contour_ls=styles,
        line_args=[dict(color=color, ls=style) for color, style in zip(COLORS, styles)],
        filename=str(folder/'posterior_comparison.pdf'),
        **plot_style,
    )
    plt.close('all')


def plot_redshift(folder, *, legend_size=8):
    q, _ = read_qso(QSO_PATH)
    q = q[q.z < 1.3].z.to_numpy()
    grb_raw = pd.read_csv(GRB_PATH)
    grb = grb_raw.query('0 < z <= 2.4').z.to_numpy()
    raw = pd.read_csv(AMATI/'review/Pantheon+SH0ES.dat', sep=r'\s+')
    eligible = raw.query('IS_CALIBRATOR == 0 and 0.023 <= zHD <= 2.26')
    z_sn = eligible.groupby('CID').zHD.median().to_numpy()
    fig, axs = plt.subplots(2, 1, figsize=(4.5, 4.6), sharex=True, layout='constrained')
    variants = ((True, 'Nearby fraction matched', '#5b4495'), (False, 'Matched count control', '#288779'))
    for ax in axs:
        ax.axvspan(0, .15, color='.89', label=None)
        ax.set_xlim(0, 2.43)
        ax.set_ylabel('Fraction per bin')
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='.91', linewidth=.6)
        ax.spines[['top', 'right']].set_visible(False)
    bins = np.linspace(0, 2.5, 26)
    for val, title, color in ((z_sn, 'Pantheon+', COLORS[1]), (q, 'Quasars', COLORS[2]), (grb, 'GRBs', COLORS[0])):
        axs[0].stairs(np.histogram(val, bins=bins)[0]/len(val), bins, label=f'{title} ({len(val)})',
                      color=color, linewidth=1.45)
    axs[1].stairs(np.histogram(z_sn, bins=bins)[0]/len(z_sn), bins, label=f'Pantheon+ ({len(z_sn)})',
                  color='.4', linewidth=1.2)
    for matched, title, color in variants:
        ix = sn_subsample(eligible, np.r_[q, grb], matched)
        z = raw.loc[ix].groupby('CID').zHD.median().to_numpy()
        axs[1].stairs(np.histogram(z, bins=bins)[0]/len(z), bins, label=f'{title} ({len(z)})',
                      color=color, linewidth=1.35)
    for ax in axs:
        ax.legend(frameon=False, fontsize=legend_size)
    axs[1].set_xlabel('Redshift')
    fig.savefig(folder/'redshift_coverage.pdf')
    plt.close(fig)


def plot_nearby(data, folder, *, row_size=8, axis_size=None):
    case_labels = {
        'pantheon_plus': 'Full Pantheon+',
        'pantheon_plus_count_control': 'Matched count control',
        'pantheon_plus_lowz_matched': 'Nearby fraction matched',
        'pantheon_plus_z023': r'$z_{\rm HD} \geq 0.23$',
        'qso_grb': 'Quasars + GRBs',
    }
    case_labels['bao_lenses'] = 'BAO + time delays'
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 4.3), layout='constrained')
    for ax, key, axis_label in zip(axes, ('H0','E015'),
                                   (r'$H_0\,[\mathrm{km\,s^{-1}\,Mpc^{-1}}]$', r'$E(0.15)\equiv H(0.15)/H_0$')):
        for i, (case, _) in enumerate(case_labels.items()):
            lo, med, hi = data[case][0]['quantiles'][key]
            color = {'bao_lenses': COLORS[0], 'qso_grb': COLORS[2]}.get(case, COLORS[1])
            y = len(case_labels)-1-i
            ax.errorbar(med, y, xerr=[[med-lo], [hi-med]], fmt='o', color=color,
                        linewidth=1.3, markersize=4, capsize=2)
        ax.set_xlabel(axis_label, fontsize=axis_size)
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='.91')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    for ax in axes:
        ax.set_yticks(np.arange(len(case_labels))[::-1], list(case_labels.values()), fontsize=row_size)
    fig.savefig(folder/'nearby_leverage.pdf')
    plt.close(fig)


def resample_nearby(data):
    _, names, point, w = data['pantheon_plus_lowz_matched']
    q, _ = read_qso(QSO_PATH)
    g = pd.read_csv(GRB_PATH)
    probe_z = np.r_[q[q.z < 1.3].z, g.query('0 < z <= 2.4').z]
    lens_table = pd.read_csv(ROOT/'lenses.csv').dropna(subset='ddt_mpc')
    p = pack_geometry(point[:, [names.index(key) for key in NAMES]])

    def log_sn(seed):
        sn = prepare_sn(AMATI, SPLINE_NODES, .023,
                        row_selector=lambda candidate: sn_subsample(candidate, probe_z, True, seed=seed))
        model = GeometryLikelihood(sn, lens_table, None, None, None, False, False, 1.5)
        terms = [model.terms(chunk)[0]['sn'] for chunk in np.array_split(p, 32)]
        return np.concatenate(terms), sn['row_indices']

    base, _ = log_sn(904135)
    variants = {}
    for seed in (904136, 904137):
        alternative, indices = log_sn(seed)
        logw = np.log(w) + alternative-base
        mixed = np.exp(logw-logsumexp(logw))
        variants[str(seed)] = summarize(names, point, mixed)
        variants[str(seed)]['selected_rows'] = indices.tolist()
    return variants


REVIEW = ROOT/'aj_checks'/'aj_review_v2'


MAIN = ('bao_lenses', 'pantheon_plus', 'qso', 'grb', 'qso_grb')


ROBUST = ('pantheon_plus_z023', 'pantheon_plus_lowz_matched', 'pantheon_plus_count_control',
          'qso_grb_evolution', 'qso_grb_groups', 'bao_lenses_smooth3',
          'pantheon_plus_smooth3', 'qso_grb_smooth3')


EXTRA = {
    'cut10': ('cut10_fine', 'qso_grb', r'Quasars $z<1.0$ + GRBs'),
    'cut15': ('cut15', 'qso_grb', r'Quasars $z<1.5$ + GRBs'),
    'wide_constant': ('wide', 'qso_grb', r'Constant intercepts, $H_0<140$'),
    'wide_evolution': ('wide', 'qso_grb_evolution', r'Evolving intercepts, $H_0<140$'),
}


def collect():
    data = {key: assemble(key) for key in LABELS}
    for key in ('bao_lenses', 'qso_grb'):
        latest = REVIEW/('long_fine' if key == 'bao_lenses' else 'long')
        data[key] = combine(key, [load_run(d, key) for d in (ORIGINAL, FOLLOWUP, latest)])
        runs = [load_run(d, key) for d in (FOLLOWUP/'smooth3', REVIEW/'smooth3')]
        data[key+'_smooth3'] = combine(key+'_smooth3', runs)
    for key in ('pantheon_plus', 'pantheon_plus_smooth3'):
        data[key] = combine(key, [load_run(REVIEW/'fine', 'pantheon_plus')])
    key = 'pantheon_plus_lowz_matched'
    data[key] = combine(key, [load_run(REVIEW/d, key) for d in ('fine', 'extra')])
    key = 'qso_grb_groups'
    data[key] = combine(key, [load_run(d, key) for d in (FOLLOWUP, REVIEW/'extra_fine')])
    for key, (directory, case, _) in EXTRA.items():
        data[key] = combine(key, [load_run(REVIEW/directory, case)])
    return data


def transform(data, case):
    _, names, points, weights = data[case]
    return derived(names, points), weights


def plot_expansion(data, folder, *, legend_size=8, axis_size=None):
    z = np.unique(np.r_[np.linspace(0, RATE_Z[-1], 121), RATE_Z])
    grid = np.log1p(RATE_Z)
    left = np.clip(np.searchsorted(grid, np.log1p(z), side='right')-1, 0, len(grid)-2)
    fraction = (np.log1p(z)-grid[left])/(grid[left+1]-grid[left])
    fig, panels = plt.subplots(3, 1, figsize=(4.5, 5.5), sharex=True, sharey=True, layout='constrained')
    for ax, case, caption, color in zip(
        panels, ('bao_lenses', 'pantheon_plus', 'qso_grb'),
        ('BAO + time delays', '+ Pantheon+', '+ quasars and GRBs'), COLORS,
    ):
        values, w = transform(data, case)
        nodes = np.column_stack([np.zeros(len(w)), *[values[f'ln_E_z{k:g}'] for k in RATE_Z[1:]]])
        ln_e = nodes[:, left]*(1-fraction) + nodes[:, left+1]*fraction
        lo, med, hi = np.array([weighted_quantile(np.exp(x), w) for x in ln_e.T]).T
        ax.fill_between(z, lo, hi, color=color, alpha=.22, linewidth=0)
        ax.plot(z, med, color=color, label=caption, lw=1.6)
        ax.plot(z, np.sqrt(.3*(1+z)**3+.7), '--', color='.25', lw=.9, label=r'Flat $\Lambda$CDM')
        ax.plot(RATE_Z, np.full(len(RATE_Z), .97), '|', color='.55', transform=ax.get_xaxis_transform())
        ax.set_xlim(RATE_Z[0], RATE_Z[-1])
        ax.legend(loc='upper left', frameon=False, fontsize=legend_size)
        ax.tick_params(axis='both', labelsize=axis_size)
        ax.spines[['top', 'right']].set_visible(False)
    panels[1].set_ylabel(r'$E(z)$', fontsize=axis_size)
    panels[-1].set_xlabel('Redshift', fontsize=axis_size)
    fig.savefig(folder/'expansion_history.pdf')
    plt.close(fig)


def residuals(data):
    """Return posterior draws of chi-square per measured component."""
    geometry, _ = uncapped_likelihood(1.5)
    bao_n = len(geometry.bao)
    grb = pd.read_csv(GRB_PATH, dtype={'grb_id': str}).query('0 < z <= 2.4')
    qso, _ = read_qso(QSO_PATH, max_redshift=1.3)
    sn_error, rows = {}, {}
    for case, (meta, names, point, w) in data.items():
        within_weights = np.searchsorted(np.cumsum(w), (np.arange(2048)+.5)/2048)
        values = derived(names, point[within_weights])
        fake_m70 = np.full(len(within_weights), -19.35)
        p = pack_geometry(np.column_stack([values.get(key, fake_m70) for key in NAMES]))
        term, valid = geometry.terms(p)
        if not valid.all():
            raise ValueError(f'Invalid BAO geometry in {case}')
        distances = {'bao': (-2*term['bao'], bao_n)}
        if case.startswith('pantheon_plus'):
            selector = None
            if case in ('pantheon_plus_lowz_matched', 'pantheon_plus_count_control'):
                def selector(candidate):
                    return sn_subsample(candidate, np.r_[qso.z, grb.z], case.endswith('lowz_matched'))
            sn = prepare_sn(AMATI, SPLINE_NODES, .23 if case.endswith('_z023') else .023,
                            row_selector=selector)
            nearby = sn['z'] < .15
            counts = dict(sn_rows=len(sn['z']), sn_unique=len(np.unique(sn['cid'])),
                          sn_rows_z_lt_015=int(nearby.sum()), sn_unique_z_lt_015=len(np.unique(sn['cid'][nearby])))
            for name, count in counts.items():
                if meta['counts'][name] not in (count, None):
                    raise ValueError(f'Supernova count mismatch in {case}: {name}')
            meta['counts'].update(counts)
            dm, _, _ = distance_geometry(p[:, 2:2+len(RATE_Z)], p[:, 0], RATE_Z, sn['z'])
            y = 5*np.log10(dm/sn['z']) + p[:, 1, None]
            white_model = solve_triangular(sn['whiten'], y.T, lower=True)
            chi2 = ((sn['white_data'][:, None]-white_model)**2).sum(axis=0)
            distances['sn'] = chi2, len(sn['z'])
            sn_error[case] = float(np.median(np.linalg.norm(sn['whiten'], axis=1)))
        elif meta['counts']['qso_rows'] or meta['counts']['grb_rows']:
            q = qso if case not in ('cut10', 'cut15') else read_qso(QSO_PATH, 1. if case == 'cut10' else 1.5)[0]
            q = q if meta['counts']['qso_rows'] else q.iloc[:0]
            g = grb if meta['counts']['grb_rows'] else grb.iloc[:0]
            if (len(q), len(g)) != (meta['counts']['qso_rows'], meta['counts']['grb_rows']):
                raise ValueError(f'Sample mismatch in {case}')
            candle = CandleLikelihood(geometry, q, g, case in ('qso_grb_evolution', 'wide_evolution'),
                                      case == 'qso_grb_groups', include_legacy_lenses=False)
            checks = {}
            candle.evaluate(np.column_stack([values[key] for key in candle.names]), residual_chi2=checks)
            distances.update({kind: (obs, count) for kind, obs, count in
                              (('qso', checks.get('qso'), len(q)), ('grb', checks.get('grb'), len(g))) if count})
        rows[case] = {kind: dict(size=n, quantiles=np.quantile(x/n, (.16, .50, .84)).tolist())
                      for kind, (x, n) in distances.items()}
    return rows, sn_error


def correlation(x, y, w):
    x = x-w@x
    y = y-w@y
    return float((w*x)@y/np.sqrt(((w*x)@x)*((w*y)@y)))


def sensitivity(data, sn_error, thinning):
    report = dict(linear_prior={}, wide_prior={})
    for key in ('bao_lenses', 'pantheon_plus', 'qso_grb', 'qso_grb_evolution'):
        _, names, points, w = data[key]
        h0 = derived(names, points)['H0']
        linear = w*h0
        report['linear_prior'][key] = summarize(names, points, linear/linear.sum())
    for key in ('wide_constant', 'wide_evolution'):
        v, w = transform(data, key)
        high = v['H0'] > 105.
        info = dict(tail_probability=float(w@high),
                    truncated_H0=weighted_quantile(v['H0'][~high], w[~high]/w[~high].sum()))
        if key == 'wide_evolution':
            info['corr'] = {c: correlation(v['H0'], v[c], w) for c in ('qso_evolution', 'grb_evolution')}
        report['wide_prior'][key] = info
    v, w = transform(data, 'qso_grb')
    report['scatter_mu'] = {
        'qso': weighted_quantile(2.5*v['qso_scatter']/(1-v['qso_gamma']), w),
        'grb': weighted_quantile(2.5*v['grb_scatter'], w),
    }
    report['sn_marginal_error_median'] = sn_error
    report['thinning_sensitivity'] = thinning
    return report


def appendix_table(columns, title, label, head, rows, note):
    return '\n'.join([
        r'\startlongtable', rf'\begin{{deluxetable}}{{{columns}}}',
        rf'\tablecaption{{{title}\label{{{label}}}}}',
        r'\tablehead{'+' & '.join(rf'\colhead{{{cell}}}' for cell in head)+'}',
        r'\startdata', *[' & '.join(row)+r' \\' for row in rows], r'\enddata',
        rf'\tablecomments{{{note}}}', r'\end{deluxetable}', '',
    ])


def supplement_tables(data, stats, residual):
    cases = ('bao_lenses', 'pantheon_plus', 'qso_grb', 'qso_grb_evolution', 'qso_grb_groups')
    values = {key: transform(data, key) for key in cases}
    parameters = (
        (r'$M_{B,*}$ (mag)', 'M70', 2, False),
        (r'$B$ (dex)', 'qso_B70', 3, False),
        (r'$\gamma$', 'qso_gamma', 3, False),
        (r'$\delta_{\rm QSO}$ (dex)', 'qso_scatter', 3, False),
        (r'$a$ (dex)', 'grb_a70', 2, False),
        (r'$b$', 'grb_b', 2, False),
        (r'$\delta_{\rm GRB}$ (dex)', 'grb_scatter', 3, False),
        (r'QSO evolution (dex)', 'qso_evolution', 3, False),
        (r'GRB evolution (dex)', 'grb_evolution', 3, False),
        (r'XXL offset (dex)', 'qso_group_4_offset', 3, False),
        (r'SDSS--Chandra offset (dex)', 'qso_group_6_offset', 3, False),
        (r'Nearby AGN offset (dex)', 'qso_group_7_offset', 3, False),
        (r'$\langle\lambda_{\rm int}\rangle$', 'lambda_mst', 3, False),
        (r'$\sigma(\lambda_{\rm int})$', 'ln_lambda_mst_sigma', 3, True),
        (r'$\alpha_{\lambda}$', 'alpha_lambda', 3, False),
        (r'$\langle a_{\rm ani}\rangle$', 'a_ani', 3, False),
        (r'$\sigma(a_{\rm ani})$', 'ln_a_ani_sigma', 3, True),
        (r'$\gamma_{\rm RXJ1131}$', 'gamma_pl_RXJ1131', 3, False),
    )
    group_keys = {name for name in values['qso_grb_groups'][0] if name.startswith('qso_group_')}
    if group_keys != {key for _, key, _, _ in parameters if key.startswith('qso_group_')}:
        raise ValueError(f'Survey group labels need verification for {group_keys}')
    nuisance = []
    for label, name, digits, exp in parameters:
        line = [label]
        for key in cases:
            v, w = values[key]
            line.append(interval(weighted_quantile(np.exp(v[name]) if exp else v[name], w), digits)
                        if name in v else r'\nodata')
        nuisance.append(line)
    absolute = appendix_table('lccccc', 'Standardization and Lens-population Posteriors', 'tab:nuisance',
                           ['Parameter', 'BAO + delays', '+ SN', '+ QSO + GRB', 'Evolution', 'Survey offsets'],
                           nuisance, r'Medians and central 68\% intervals. The final two columns vary '
                           r'standardization in the joint quasar--burst fit. SDSS--4XMM sets the group-offset zero point.')
    residual_cases = (*MAIN, 'pantheon_plus_lowz_matched', 'qso_grb_evolution', 'qso_grb_groups')
    fit_rows = []
    for key in residual_cases:
        fit_rows.append([LABELS[key], *[
            interval(residual[key][kind]['quantiles'], 3 if kind == 'sn' else 2) if kind in residual[key] else r'\nodata'
            for kind in ('bao', 'sn', 'qso', 'grb')
        ]])
    fit = appendix_table('lcccc', r'Conditional Residual Statistics, $\chi^2/N$', 'tab:resid',
                      ['Data or variant', 'BAO', 'SN', 'QSO', 'GRB'], fit_rows,
                      r'Medians and central 68\% intervals over joint posterior draws. $N$ is the number '
                      r'of measured components, including repeated supernova light curves. Statistics use '
                      r'the full covariance where supplied and the fitted variances for quasars and bursts.')
    priors = []
    for key, entry in stats['linear_prior'].items():
        priors.append([LABELS[key], interval(data[key][0]['quantiles']['H0'], 1),
                       interval(entry['quantiles']['H0'], 1)])
    linear = appendix_table('lcc', r'Shape of the $H_0$ Prior', 'tab:prior',
                         ['Data or variant', r'Uniform $\ln H_0$', r'Uniform $H_0$'], priors,
                         r'Central 68\% intervals in $\Hunit$ on the common support $40<H_0/(\Hunit)<105$.')
    return {'nuisance_table.tex': absolute, 'residual_table.tex': fit, 'priors_table.tex': linear}


def diagnostic_runs(data, folder, direct_smooth):
    header = ('case', 'source', 'H0_16', 'H0_50', 'H0_84', 'rd_16', 'rd_50', 'rd_84',
              'omega_16', 'omega_50', 'omega_84', 'H0_N_tau', 'rd_N_tau', 'omega_N_tau',
              'min_N_tau_sampled', 'acceptance', 'retained_steps', 'refined_power', 'refined_rx_power',
              'sampled_points', 'Kish', 'max_weight', 'H0_half_shift', 'H0_scramble_shift')
    with (folder/'numerical_diagnostics.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for key, (summary, *_) in data.items():
            for ens in summary['ensembles']:
                q = ens['combined']['parameters']
                t = ens['physical_retained_over_tau']
                halves = ens['weighted_half_summaries']
                scr = ens['quadrature_medians']
                chain_diag = ens['chain_diagnostics']
                writer.writerow([key, ens['source'], *q['H0'], *q['r_d'], *q['Omega_k'],
                                 *[t[k] for k in CORE[:3]], chain_diag.get('retained_over_tau_max'),
                                 chain_diag.get('acceptance'), ens['source_steps'], ens['refined_power'],
                                 ens['refined_rx_power'], ens['points'], ens['combined']['ess'], ens['combined']['max_weight'],
                                 abs(halves[0]['medians']['H0']-halves[1]['medians']['H0']),
                                 abs(scr[0]['H0']-scr[1]['H0'])])
    cases = (*MAIN, 'pantheon_plus_z023', 'pantheon_plus_lowz_matched', 'pantheon_plus_count_control',
             'qso_grb_evolution', 'qso_grb_groups', 'bao_lenses_smooth3', 'pantheon_plus_smooth3', 'qso_grb_smooth3',
             'wide_constant', 'wide_evolution', 'cut10', 'cut15')
    rows = []
    for key in cases:
        if key == 'pantheon_plus_smooth3':
            q = direct_smooth['quantiles']
            tau = direct_smooth['latest_n_over_tau']
            rows.append([LABELS[key], interval(q['H0'], 1), '/'.join(f'{tau[k]:.0f}' for k in CORE[:3]),
                         f"{direct_smooth['kish_ess']:.0f}", f"{100*direct_smooth['max_weight']:.2f}", r'\nodata'])
            continue
        ens = data[key][0]['ensembles']
        latest = ens[-1]
        q = latest['combined']['parameters']
        t = latest['physical_retained_over_tau']
        h0_medians = [x['combined']['parameters']['H0'][1] for x in ens]
        rows.append([LABELS.get(key) or EXTRA[key][2], interval(q['H0'], 1),
                     '/'.join(f'{t[k]:.0f}' for k in CORE[:3]),
                     f"{latest['combined']['ess']:.0f}", f"{100*latest['combined']['max_weight']:.2f}",
                     f'{np.ptp(h0_medians):.1f}' if len(ens)>1 else r'\nodata'])
    new = appendix_table('lccccc', 'Sampling and Lens-integral Diagnostics', 'tab:numerical',
                      ['Data or variant', r'Single-ensemble $H_0$', r'$N_{\rm step}/\tau$', 'Kish ESS',
                       r'$w_{\rm max}$ (\%)', r'$\Delta H_0$'], rows,
                      r'$N_{\rm step}/\tau$ is listed for $H_0$, $\rd$, and $\OmK$ in that order. '
                      r'Kish ESS and $w_{\rm max}$ refer to the normalized importance weights in the displayed ensemble. '
                      r'$\Delta H_0$ is the spread of $H_0$ medians across the independently sampled ensembles '
                      r'when available. $H_0$ and $\Delta H_0$ are in $\Hunit$.')
    (folder/'numerical_table.tex').write_text(new, encoding='utf8')


def macros(data, stats):
    summaries = {name: value[0]['quantiles'] for name, value in data.items()}
    matched, full = 'pantheon_plus_lowz_matched', 'pantheon_plus'
    control = 'pantheon_plus_count_control'
    reference, qg = 'bao_lenses', 'qso_grb'
    evolution = 'qso_grb_evolution'
    sift = stats['thinning_sensitivity'].values()
    sift_h = [x['quantiles']['H0'] for x in sift]
    sift_widths = [hi-lo for lo, _, hi in sift_h]
    wide = stats['wide_prior']
    def get_h(key):
        return summaries[key]['H0'][1]
    result = {
        'FullHwidth': f"{parameter_width(data, full, 'H0'):.1f}",
        'CutHwidth': f"{parameter_width(data, 'pantheon_plus_z023', 'H0'):.1f}",
        'ReferenceHwidth': f"{parameter_width(data, reference, 'H0'):.1f}",
        'QGwidth': f"{parameter_width(data, qg, 'H0'):.1f}",
        'ReferenceSmoothHwidth': f"{parameter_width(data, 'bao_lenses_smooth3', 'H0'):.1f}",
        'QGSmoothHwidth': f"{parameter_width(data, 'qso_grb_smooth3', 'H0'):.1f}",
        'ReferenceKwidth': f"{parameter_width(data, reference, 'Omega_k'):.3f}",
        'QGKwidth': f"{parameter_width(data, qg, 'Omega_k'):.3f}",
        'MatchHwidth': f"{parameter_width(data, matched, 'H0'):.1f}",
        'ControlHwidth': f"{parameter_width(data, control, 'H0'):.1f}",
        'ControlH': f"{get_h(control):.1f}", 'MatchH': f"{get_h(matched):.1f}",
        'ThinMin': f'{min(x[1] for x in sift_h):.1f}', 'ThinMax': f'{max(x[1] for x in sift_h):.1f}',
        'ThinWidthMin': f'{min(sift_widths):.1f}', 'ThinWidthMax': f'{max(sift_widths):.1f}',
        'ThinMinEss': f"{min(x['quadrature_ess'] for x in sift):.0f}",
        'MatchCorr': f"{data[matched][0]['corr_h0_e015']:.2f}",
        'ControlCorr': f"{data[control][0]['corr_h0_e015']:.2f}",
        'CutCorr': f"{data['pantheon_plus_z023'][0]['corr_h0_e015']:.2f}",
        'QGCorr': f"{data[qg][0]['corr_h0_e015']:.2f}",
        'QSOscat': f"{stats['scatter_mu']['qso'][1]:.2f}",
        'GRBscat': f"{stats['scatter_mu']['grb'][1]:.2f}",
        'SNerror': f"{stats['sn_marginal_error_median'][full]:.2f}",
        'EvoH': f"{get_h(evolution):.1f}", 'QGH': f"{get_h(qg):.1f}",
        'GroupH': f"{get_h('qso_grb_groups'):.1f}",
        'WideConstantH': interval(summaries['wide_constant']['H0'], 1),
        'WideEvolutionH': interval(summaries['wide_evolution']['H0'], 1),
        'WideTailConstant': f"{100*wide['wide_constant']['tail_probability']:.1f}",
        'WideTailEvolution': f"{100*wide['wide_evolution']['tail_probability']:.1f}",
        'WideEvoCorrQ': f"{wide['wide_evolution']['corr']['qso_evolution']:.2f}",
        'WideEvoCorrG': f"{wide['wide_evolution']['corr']['grb_evolution']:.2f}",
        'WideTruncConstantH': f"{wide['wide_constant']['truncated_H0'][1]:.1f}",
        'WideTruncEvolutionH': f"{wide['wide_evolution']['truncated_H0'][1]:.1f}",
        'QLowCount': str(data['cut10'][0]['counts']['qso_rows']),
        'QHighCount': str(data['cut15'][0]['counts']['qso_rows']),
        'MaxCutHshift': f"{max(abs(get_h(k)-get_h(qg)) for k in ('cut10', 'cut15')):.1f}",
        'BaseLinearShift': f"{stats['linear_prior'][reference]['quantiles']['H0'][1]-get_h(reference):.1f}",
        'SNLinearShift': f"{stats['linear_prior'][full]['quantiles']['H0'][1]-get_h(full):.1f}",
        'ReferenceTau': f"{data[reference][0]['ensembles'][-1]['physical_retained_over_tau']['H0']:.0f}",
        'QGTau': f"{data[qg][0]['ensembles'][-1]['physical_retained_over_tau']['H0']:.0f}",
    }
    return ''.join(rf'\newcommand{{\{k}}}{{{v}}}'+'\n' for k, v in result.items())


RUNS = ROOT/'aj_checks'/'aj_opus_revision'


SAMPLES = {
    'full': 'pantheon_plus',
    'count': 'pantheon_plus_count_control',
    'near': 'pantheon_plus_lowz_matched',
}


NODE_LABELS = {
    'full': 'Full Pantheon+',
    'count': 'Count-matched control',
    'near': 'Nearby-fraction match',
}


BASE_SEED = 904135


def figures(data, output):
    plot_redshift(output, legend_size=10)
    plot_posteriors(
        data, output, axes_fontsize=17, lab_fontsize=22, legend_fontsize=17,
        linewidth=1.8, linewidth_contour=1.55,
    )
    plot_nearby(data, output, row_size=12.5, axis_size=13)
    plot_expansion(data, output, legend_size=12, axis_size=12)


def width(result):
    low, _, high = result['quantiles']['H0']
    return high - low


def fit_entry(summary, *, sample, node, seed):
    return {
        'sample': sample,
        'first_node': node,
        'selection_seed': seed if sample != 'full' else None,
        'counts': summary['counts'],
        'quantiles': {key: summary['quantiles'][key] for key in ('H0', 'r_d', 'Omega_k', 'H0_r_d')},
        'kish_ess': summary['quadrature_ess'],
        'max_weight': summary['max_weight'],
        'latest_n_over_tau': summary['ensembles'][-1]['physical_retained_over_tau'],
    }


def load_direct(spec):
    directory = RUNS/spec['id']
    names, points, weights, refined, metadata = load_run(directory, spec['case'])
    if metadata['rate_nodes'][1] != spec['first_node'] or metadata['smooth'] != spec['smooth']:
        raise ValueError(f"Unexpected model settings for {spec['id']}")
    if metadata.get('sn_selection_seed', BASE_SEED) != spec['sn_seed']:
        raise ValueError(f"Unexpected sample selection for {spec['id']}")
    if metadata['seed'] != spec['seed'] or metadata['steps'] != spec['steps']:
        raise ValueError(f"Unexpected chain metadata for {spec['id']}")
    if metadata['sobol_power'] != spec['source_power'] or metadata['rx_power'] != spec['source_rx_power']:
        raise ValueError(f"Unexpected sampling quadrature for {spec['id']}")
    if refined['refined_power'] != spec['refined_power'] or refined['refined_rx_power'] != spec['refined_rx_power']:
        raise ValueError(f"Unexpected quadrature settings for {spec['id']}")
    if len(points) != refined['points'] or len(points) != spec['refined_draws']:
        raise ValueError(f"Unexpected weighted sample count for {spec['id']}")
    posterior = refined['combined']
    result = fit_entry(
        dict(quantiles=posterior['parameters'], counts={key: metadata[key] for key in (
            'sn_rows', 'sn_unique', 'sn_rows_z_lt_015', 'sn_unique_z_lt_015',
        )}, quadrature_ess=posterior['ess'], max_weight=posterior['max_weight'],
             ensembles=[refined]),
        sample=spec['sample'], node=spec['first_node'], seed=spec['sn_seed'],
    )
    parameters = dict(zip(names, points.T))
    h0 = REFERENCE_H*np.exp(parameters['log_H0_over_70'])
    h0rd = REFERENCE_H*parameters['H0rd_over_70']
    for name, values in (('H0', h0), ('r_d', h0rd/h0), ('Omega_k', parameters['Omega_k']), ('H0_r_d', h0rd)):
        if not np.allclose(posterior['parameters'][name], weighted_quantile(values, weights), rtol=1e-10):
            raise ValueError(f"Posterior quantiles of {name} differ from their draws in {spec['id']}")
    if not np.isclose(posterior['ess'], 1/(weights@weights), rtol=1e-10):
        raise ValueError(f"Incorrect Kish ESS for {spec['id']}")
    half_intervals = [
        weighted_quantile(values, w/w.sum()) for values, w in
        zip(np.array_split(h0, 2), np.array_split(weights, 2))
    ]
    result.update(
        run_id=spec['id'],
        group=spec['group'],
        mcmc_seed=metadata['seed'],
        steps=metadata['steps'],
        walkers=metadata['walkers'],
        burn_in=metadata['burn_in'],
        sampled_points=refined['points'],
        quadrature_powers=[refined['refined_power'], refined['refined_rx_power']],
        h0_half_medians=[row['medians']['H0'] for row in refined['weighted_half_summaries']],
        h0_scramble_medians=[row['parameters']['H0'][1] for row in refined['independent']],
        h0_half_widths=[hi-lo for lo, _, hi in half_intervals],
        h0_scramble_widths=[float(np.ptp(entry['parameters']['H0'])) for entry in refined['independent']],
    )
    return result, metadata


def checks(data, plan):
    nodes = {node: {} for node in (0.10, 0.15, 0.20)}
    coverage = {BASE_SEED: {}}
    for sample, case in SAMPLES.items():
        entry = fit_entry(data[case][0], sample=sample, node=.15, seed=BASE_SEED)
        nodes[.15][sample] = entry
        if sample != 'full':
            coverage[BASE_SEED][sample] = entry
    selections = {}
    runs = []
    repeated_coverage = {}
    for spec in plan:
        entry, meta = load_direct(spec)
        sample = spec['sample']
        if sample != 'full':
            key = (spec['sn_seed'], sample)
            indices = meta['sn_rows_selected']
            if selections.setdefault(key, indices) != indices:
                raise ValueError(f'Inconsistent supernova rows for {key}')
        if spec['group'] == 'node':
            nodes[spec['first_node']][sample] = entry
        elif spec['group'] == 'coverage':
            draws = coverage.setdefault(spec['sn_seed'], {})
            if sample in draws:
                repeated_coverage[spec['sn_seed']] = dict(draws)
            draws[sample] = entry
        else:
            direct_smooth = entry
        runs.append(entry)
    return dict(
        nodes=nodes, coverage=coverage, direct_smooth=direct_smooth,
        reweighted_smooth=fit_entry(data['pantheon_plus_smooth3'][0], sample='full', node=.15, seed=BASE_SEED),
        runs=runs, repeated_coverage=repeated_coverage,
    ), selections


def export_selections(selections, output):
    raw = pd.read_csv(AMATI/'review/Pantheon+SH0ES.dat', sep=r'\s+')
    eligible = raw.query('IS_CALIBRATOR == 0 and 0.023 <= zHD <= 2.26')
    qso, _ = read_qso(QSO_PATH, max_redshift=1.3)
    grb = pd.read_csv(GRB_PATH, dtype={'grb_id': str}).query('0 < z <= 2.4')
    probe_z = np.r_[qso.z, grb.z]
    with (output/'pantheon_selection_rows.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('selection', 'covariance_row_0_based', 'CID', 'z_HD', 'z_HEL'))
        for name, subset in [('full', eligible), ('z_ge_023', eligible.query('zHD >= 0.23'))]:
            for row in subset[['CID','zHD','zHEL']].itertuples():
                writer.writerow((name, row.Index, row.CID, row.zHD, row.zHEL))
        for (seed, sample), indices in sorted(selections.items()):
            reconstructed = sn_subsample(eligible, probe_z, sample == 'near', seed).tolist()
            if reconstructed != indices:
                raise ValueError(f'Selection {seed}/{sample} fails row-by-row replay')
            if seed == BASE_SEED:
                old = FOLLOWUP/f'{SAMPLES[sample]}_p7_rx8_s746.json'
                original = json.loads(old.read_text(encoding='utf-8'))
                if original['sn_rows_selected'] != indices:
                    raise ValueError(f'Selection {seed}/{sample} differs from the reference fit')
            subset = raw.loc[indices, ['CID','zHD','zHEL']]
            if subset.CID.nunique() != 882:
                raise ValueError(f'Selection {seed}/{sample} does not contain 882 supernovae')
            for row in subset.itertuples():
                writer.writerow((f'{seed}_{sample}', row.Index, row.CID, row.zHD, row.zHEL))
    with (output/'qso_grb_selections.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('selection', 'source_id', 'redshift'))
        for cutoff in (1., 1.3, 1.5):
            qso, _ = read_qso(QSO_PATH, max_redshift=cutoff)
            writer.writerows((f'qso_z_lt_{cutoff:.1f}', row.Name, row.z) for row in qso.itertuples())
        writer.writerows(('a118_z_le_2.4', row.grb_id, row.z) for row in grb.itertuples())


def export_diagnostics(report, output):
    with (output/'new_fit_diagnostics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('run_id','group','sample','first_node','selection_seed','mcmc_seed','steps','walkers',
                         'burn_in','H0_p16','H0_median','H0_p84','H0_N_tau','rd_N_tau','omega_N_tau',
                         'kish_ess','max_weight','H0_half1','H0_half2','H0_scramble1','H0_scramble2',
                         'H0_width_half1','H0_width_half2','H0_width_scramble1','H0_width_scramble2'))
        for run in report['runs']:
            writer.writerow([run[name] for name in (
                'run_id','group','sample','first_node','selection_seed','mcmc_seed','steps','walkers','burn_in',
            )] + run['quantiles']['H0'] + [run['latest_n_over_tau'][name] for name in ('H0','r_d','Omega_k')] +
                [run['kish_ess'],run['max_weight'],*run['h0_half_medians'],*run['h0_scramble_medians'],
                 *run['h0_half_widths'],*run['h0_scramble_widths']])


def ratio(fits):
    return width(fits['near']) / width(fits['count'])


def appendix_tables(report):
    node_rows = []
    for node, fits in sorted(report['nodes'].items()):
        node_rows.append([f'{node:.2f}', *(interval(fits[s]['quantiles']['H0'], 1) for s in SAMPLES),
                          f'{ratio(fits):.2f}'])
    grid_table = appendix_table(
        'lcccc', 'First Expansion Node under the Hierarchical Lens Likelihood', 'tab:nodecheck',
        ['First node', *NODE_LABELS.values(), r'$W_{\rm near}/W_{\rm control}$'], node_rows,
        r'$H_0$ in $\Hunit$. $W$ is the central 68\% interval width calculated from unrounded quantiles. '
        r'The two equal-count samples retain the same objects at each node. All fits use DESI DR2 BAO and the '
        r'TDCOSMO 2025 hierarchical likelihood.',
    )
    draw_rows = []
    for seed, fits in sorted(report['coverage'].items()):
        draw_rows.append([str(seed), str(fits['count']['counts']['sn_unique_z_lt_015']),
                          str(fits['near']['counts']['sn_unique_z_lt_015']),
                          f"{width(fits['count']):.2f}", f"{width(fits['near']):.2f}", f'{ratio(fits):.2f}'])
    draw_table = appendix_table(
        'lccccc', 'Equal-Count Supernova Realizations', 'tab:drawcheck',
        ['Selection seed', r'Nearby $N_{\rm control}$', r'Nearby $N_{\rm near}$',
         r'$W_{\rm control}$', r'$W_{\rm near}$', r'$W_{\rm near}/W_{\rm control}$'], draw_rows,
        r'Each fit uses 882 distinct supernovae and the first expansion node at $z=0.15$. '
        r'Nearby counts are for $z_{\rm HD}<0.15$. The widths $W$ are central 68\% widths of $H_0$ in $\Hunit$. '
        r'The ratios compare fitted realizations drawn from the same parent catalogue.',
    )
    return grid_table + draw_table


def smoothing_lines(direct):
    q = direct['quantiles']
    case = 'Pantheon+, smoothing width 3'
    curvature = interval(q['Omega_k'], 3).replace(r'\(-0.000', r'\(0.000')
    result = ' & '.join([case, interval(q['H0'], 1), interval(q['r_d'], 1),
                         curvature, interval(q['H0_r_d'], 0)]) + r' \\'
    autocorrelation = '/'.join(f"{direct['latest_n_over_tau'][p]:.0f}" for p in ('H0','r_d','Omega_k'))
    diagnostics = ' & '.join([case, interval(q['H0'], 1), autocorrelation, f"{direct['kish_ess']:.0f}",
                              f"{100*direct['max_weight']:.2f}", r'\nodata']) + r' \\'
    return result, diagnostics


def verify(root):
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    for relative, record in manifest['files'].items():
        with (root/relative).open('rb') as stream:
            actual = file_digest(stream, 'sha256').hexdigest()
        if actual != record['sha256']:
            raise ValueError(f'SHA-256 mismatch for {relative}')
    print(f"Verified {len(manifest['files'])} archived files.", flush=True)


def equal_summary(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(equal_summary(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(equal_summary(a, b) for a, b in zip(left, right))
    if isinstance(left, float) and isinstance(right, float):
        return isclose(left, right, rel_tol=1e-10, abs_tol=1e-10)
    return left == right


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        for row in csv.reader(handle):
            def value(cell):
                try:
                    return float(cell)
                except ValueError:
                    return cell
            yield [value(cell) for cell in row]


def compare_summary(before, after):
    if before.suffix == '.json':
        left = json.loads(before.read_text(encoding='utf-8'))
        right = json.loads(after.read_text(encoding='utf-8'))
    elif before.name.endswith('selections.csv') or before.name == 'pantheon_selection_rows.csv':
        left = before.read_text(encoding='utf-8')
        right = after.read_text(encoding='utf-8')
    else:
        left, right = list(csv_rows(before)), list(csv_rows(after))
    if not equal_summary(left, right):
        raise ValueError(f'Rebuilt output differs from {before.name}')


def export_width_diagnostics(report, path):
    """Differences in central-interval widths, not differences in medians."""
    with path.open('w', newline='', encoding='utf8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('comparison', 'value', 'control_width', 'nearby_match_width', 'ratio',
                         'width_difference', 'half_1_difference', 'half_2_difference',
                         'sobol_1_difference', 'sobol_2_difference'))
        for group, fits in (('node', report['nodes']), ('selection_seed', report['coverage'])):
            for key, pair in fits.items():
                control, match = (pair[name] for name in ('count', 'near'))
                differences = []
                for diagnostic in ('h0_half_widths', 'h0_scramble_widths'):
                    differences.extend([b-a for a, b in zip(control[diagnostic], match[diagnostic])]
                                       if diagnostic in control and diagnostic in match else [None, None])
                writer.writerow([group, key, width(control), width(match), ratio(pair),
                                 width(match)-width(control), *differences])


def build_results(root, output):
    data = collect()
    plan = json.loads((RUNS/'run_plan.json').read_text(encoding='utf-8-sig'))
    report, selections = checks(data, plan)
    (output/'robustness_results.json').write_text(json.dumps(report, indent=2), encoding='utf8')
    export_selections(selections, output)
    export_diagnostics(report, output)
    export_width_diagnostics(report, output/'width_diagnostics.csv')

    residual, sn_error = residuals(data)
    stats = sensitivity(data, sn_error, resample_nearby(data))
    stats['residuals'] = residual
    (output/'review_checks.json').write_text(json.dumps(stats, indent=2, allow_nan=False), encoding='utf8')
    (output/'results_macros.tex').write_text(macros(data, stats), encoding='utf8')
    (output/'posterior_summary.json').write_text(
        json.dumps({key: item[0] for key, item in data.items()}, indent=2, allow_nan=False), encoding='utf8')
    with (output/'posterior_summary.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('case','parameter','p16','p50','p84','quadrature_ess','max_weight'))
        for key, (summary, *_) in data.items():
            for name, quantiles in summary['quantiles'].items():
                writer.writerow([key, name, *quantiles, summary['quadrature_ess'], summary['max_weight']])
    with (output/'paper_summary.csv').open('w', newline='', encoding='utf8') as handle:
        writer = csv.writer(handle)
        writer.writerow(('case', 'parameter', 'p16', 'p50', 'p84', 'quadrature_ess', 'max_weight'))
        for key, (summary, *_) in data.items():
            if key == 'pantheon_plus_smooth3':
                direct = report['direct_smooth']
                quantiles, ess, peak = direct['quantiles'], direct['kish_ess'], direct['max_weight']
            else:
                quantiles, ess, peak = summary['quantiles'], summary['quadrature_ess'], summary['max_weight']
            for name in CORE[:4]:
                writer.writerow([key, name, *quantiles[name], ess, peak])

    table_dir = output/'tables'
    table_dir.mkdir(exist_ok=True)
    main_table = table(data, MAIN, 'Late-time Constraints with Different Luminosity-distance Indicators',
                       'tab:main')
    smooth_row, _ = smoothing_lines(report['direct_smooth'])
    robust = table(data, ROBUST, 'Redshift-coverage, Standardization, and Expansion-history Tests', 'tab:robust')
    robust = '\n'.join(smooth_row if line.startswith('Pantheon+, smoothing width 3 &') else line
                       for line in robust.split('\n'))
    robust = robust.replace(r'\(-0.000', r'\(0.000')
    additional = table(data, tuple(EXTRA), 'Quasar Redshift Cuts and Extended Hubble-constant Support',
                       'tab:additional', {key: row[2] for key, row in EXTRA.items()})
    outputs = dict(posterior_table=main_table, robustness_table=robust, sensitivity_table=additional,
                   node_and_draw_tables=appendix_tables(report))
    for name, content in outputs.items():
        (table_dir/f'{name}.tex').write_text(content, encoding='utf8')
    for name, content in supplement_tables(data, stats, residual).items():
        (table_dir/name).write_text(content, encoding='utf8')
    diagnostic_runs(data, table_dir, report['direct_smooth'])

    stored = root/'evidence'
    for name in ('robustness_results.json', 'new_fit_diagnostics.csv', 'pantheon_selection_rows.csv',
                 'qso_grb_selections.csv'):
        compare_summary(stored/name, output/name)
    for name in ('posterior_summary.csv', 'review_checks.json', 'numerical_diagnostics.csv'):
        produced = table_dir/name if name == 'numerical_diagnostics.csv' else output/name
        compare_summary(stored/f'baseline_{name}', produced)

    figures(data, output)
    print(f'Numerical summaries checked against the archive. Tables and figures saved to {output}.', flush=True)


def rerun_fit(identifier, output, workers):
    plan = json.loads((RUNS/'run_plan.json').read_text(encoding='utf-8-sig'))
    spec = next((row for row in plan if row['id'] == identifier), None)
    if spec is None:
        raise ValueError(f'Unknown fit: {identifier}. Available: {", ".join(row["id"] for row in plan)}')
    tag = f"{spec['case']}_p{spec['source_power']}_rx{spec['source_rx_power']}_s{spec['seed']}"
    start = RUNS/identifier/f'{tag}_initial.npy'
    target = output/identifier
    sample([
        '--case', spec['case'], '--steps', str(spec['steps']),
        '--power', str(spec['source_power']), '--rx-power', str(spec['source_rx_power']),
        '--seed', str(spec['seed']), '--walkers', str(spec['walkers']),
        '--first-node', str(spec['first_node']), '--sn-selection-seed', str(spec['sn_seed']),
        '--smooth', str(spec['smooth']), '--initial-positions', str(start),
        '--output', str(target), '--workers', str(workers),
    ])
    refine([
        spec['case'], '--source-power', str(spec['source_power']),
        '--source-rx-power', str(spec['source_rx_power']), '--source-seed', str(spec['seed']),
        '--power', str(spec['refined_power']), '--refine-rx-power', str(spec['refined_rx_power']),
        '--samples', str(spec['refined_draws']), '--source-dir', str(target),
        '--output', str(target), '--workers', str(workers),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=PROJECT/'results')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--verify-only', action='store_true')
    mode.add_argument('--fit', metavar='RUN_ID', help='Rerun one sensitivity fit instead of rebuilding results')
    parser.add_argument('--workers', type=int, default=1, help='Parallel processes for --fit')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    verify(PROJECT)
    if args.verify_only:
        return
    output = args.output.resolve()
    input_dirs = (ROOT, PROJECT/'evidence', PROJECT/'data', PROJECT/'inputs')
    if output == PROJECT or any(output.is_relative_to(path) for path in input_dirs):
        parser.error('Choose an output folder outside the archived input folders')
    output.mkdir(parents=True, exist_ok=True)
    if args.fit:
        rerun_fit(args.fit, output, args.workers)
    else:
        build_results(PROJECT, output)


if __name__ == '__main__':
    main()
