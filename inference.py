"""Likelihood, sampling and quadrature for the released distance-probe analysis."""

import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
import sys
from time import perf_counter

import emcee
from emcee.autocorr import integrated_time
import numpy as np
import pandas as pd
from numba import njit
from scipy.interpolate import CubicSpline
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.special import exprel, log_ndtr, logsumexp, ndtr, ndtri
from scipy.stats import qmc

PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT / 'lens_sn_grb_pilot'
AMATI = PROJECT / 'inputs' / 'amati'
# Use the licensed dependencies bundled with the archived lens configuration.
sys.path.insert(0, str(PROJECT / 'vendor.zip'))
from hierarc.Likelihood.lens_sample_likelihood import LensSampleLikelihood
from lenstronomy.Util.constants import c
from threadpoolctl import threadpool_limits


C_KMS = 299792.458


REFERENCE_DISTANCE_MPC = C_KMS / 70.0


MPC_CM = 3.085677581491367e24


# Expansion geometry and the Pantheon+ covariance projection.


RATE_Z = np.array([0.0, .15, .35, .65, 1., 1.5, 2.05, 2.4])


SPLINE_NODES = 100


REFERENCE_H = C_KMS / REFERENCE_DISTANCE_MPC


def distance_geometry(rate_log, k70, rate_z, targets):
    """Integrate 1/H analytically by segment and evaluate transverse distance."""
    grid = np.log1p(rate_z)
    t = np.log1p(targets)
    j = np.minimum(np.searchsorted(grid, t, side='right') - 1, len(grid) - 2)
    widths = np.diff(grid)
    slopes = np.diff(rate_log, axis=1) / widths
    amplitude = np.exp(grid[:-1] - rate_log[:, :-1])
    segment = amplitude * widths * exprel((1 - slopes) * widths)
    start = np.c_[np.zeros(len(rate_log)), np.cumsum(segment, axis=1)]
    dt = t - grid[j]
    chi = start[:, j] + amplitude[:, j] * dt * exprel((1 - slopes[:, j]) * dt)
    log_h = rate_log[:, j] + slopes[:, j] * dt
    # Series at k=0 avoids division by zero, while sin/sinh handles either sign.
    k_chi2 = k70[:, None] * chi**2
    arg = np.sqrt(np.abs(k_chi2))
    safe_arg = np.where(arg > 1e-4, arg, 1.)
    curved = np.where(k_chi2 > 0, np.sinh(safe_arg), np.sin(safe_arg)) / safe_arg
    ratio = np.where(arg > 1e-4, curved, 1 + k_chi2 / 6 + k_chi2**2 / 120)
    # Use the same branch as the endpoint distance-sum rule.
    domain = ~np.any((k_chi2 < 0) & (arg >= np.pi / 2), axis=1)
    return REFERENCE_DISTANCE_MPC * chi * ratio, REFERENCE_DISTANCE_MPC * np.exp(-log_h), domain


def prepare_sn(amati, nodes, min_z, rate_z=RATE_Z, row_selector=None):
    """Read the released Pantheon+ covariance and construct the spline basis."""

    path = amati / 'review'
    raw = pd.read_csv(path / 'Pantheon+SH0ES.dat', sep=r'\s+')
    with (path / 'Pantheon+SH0ES_STAT+SYS.cov').open(encoding='utf-8-sig') as handle:
        size = int(handle.readline().strip())
        cov = np.loadtxt(handle).reshape(size, size)
    if cov.shape != (len(raw), len(raw)):
        raise ValueError('Pantheon+ rows and covariance disagree')
    select = (raw.zHD.to_numpy() >= min_z) & (raw.zHD.to_numpy() <= 2.26) & (raw.IS_CALIBRATOR.to_numpy() == 0)
    if row_selector is not None:
        eligible = np.flatnonzero(select)
        requested = row_selector(raw.iloc[eligible])
        select[:] = False
        select[requested] = True
    sn = raw[select]
    z = sn.zHD.to_numpy()
    cov = cov[np.ix_(select, select)]
    cov = (cov + cov.T) / 2
    y = sn.m_b_corr.to_numpy() - 5 * np.log10((1 + sn.zHEL.to_numpy()) * z) - 25
    # Compress with the full covariance for a dense cubic interpolation basis.
    # The smooth interpolated quantity is 5 log10(D_M/z), not the magnitude itself.
    x = np.unique(np.r_[np.linspace(np.log(z.min()), np.log(z.max()), nodes),
                       np.log(rate_z[(rate_z > z.min()) & (rate_z < z.max())])])
    basis = CubicSpline(x, np.eye(len(x)))(np.log(z))
    whiten = cholesky(cov, lower=True)
    white_basis = solve_triangular(whiten, basis, lower=True)
    white_data = solve_triangular(whiten, y, lower=True)
    return dict(z=z, cid=sn.CID.to_numpy(), row_indices=np.flatnonzero(select), nodes=np.exp(x), basis=basis, whiten=whiten, white_data=white_data,
                precision=white_basis.T @ white_basis, rhs=white_basis.T @ white_data)


class GeometryLikelihood:
    def __init__(self, sn, lenses, grb, bao, bao_cov, use_bao, use_grb, smooth, rate_z=RATE_Z):
        self.sn, self.smooth = sn, smooth
        self.use_bao, self.use_grb = use_bao, use_grb
        self.rate_z = np.asarray(rate_z)
        self.n_rate = len(self.rate_z)
        self.names = ['k70', 'M_B', *[f'ln_H_over_70_z{z:g}' for z in self.rate_z]]
        self.bounds = [(-1.5, 3.), (-21., -17.), (np.log(40/70), np.log(105/70)),
                       *[(-1.0, 3.0)] * (self.n_rate - 1)]
        targets = [sn['nodes'], lenses[['z_d', 'z_s']].to_numpy().ravel()]
        self.lens_obs = np.log(lenses.ddt_mpc.to_numpy())
        self.lens_err = .5 * np.log((lenses.ddt_mpc + lenses.err_high_mpc) /
                                   (lenses.ddt_mpc - lenses.err_low_mpc)).to_numpy()
        self.n_sn = len(targets[0])
        self.lens_end = self.n_sn + len(targets[1])
        if use_grb:
            targets.append(grb.z.to_numpy())
            self.names += ['amati_a300', 'amati_b', 'amati_scatter']
            self.bounds += [(49., 56.), (0., 4.), (.03, 2.)]
            self.grb_x = np.log10(grb.ep_keV.to_numpy() / 300.)
            self.grb_xerr = (grb.ep_err_keV / grb.ep_keV / np.log(10)).to_numpy()
            self.grb_yoffset = np.log10(4*np.pi*MPC_CM**2*grb.fluence*(1+grb.z)).to_numpy()
            self.grb_yerr = (grb.fluence_err / grb.fluence / np.log(10)).to_numpy()
        self.grb_end = sum(map(len, targets))
        if use_bao:
            targets.append(bao.z.to_numpy())
            self.names.append('r_d')
            self.bounds.append((90., 230.))
            self.bao = bao
            self.bao_inverse = cho_solve(cho_factor(bao_cov), np.eye(len(bao)))
        self.targets = np.concatenate(targets)
        self.sn_base = 5 * np.log10(REFERENCE_DISTANCE_MPC) - 19.3
        base = np.full(self.n_sn, self.sn_base)
        self.sn_rhs = sn['rhs'] - sn['precision'] @ base

    def terms(self, p, *, include_lens_predictions=False):
        dm, dh, physical = distance_geometry(p[:, 2:2+self.n_rate], p[:, 0], self.rate_z, self.targets)
        with np.errstate(invalid='ignore'):
            model = 5 * np.log10(dm[:, :self.n_sn] / self.sn['nodes']) + p[:, 1, None] - self.sn_base
        sn_ll = -.5 * np.sum((model @ self.sn['precision'] - 2*self.sn_rhs) * model, axis=1)
        endpoint = dm[:, self.n_sn:self.lens_end].reshape(len(p), -1, 2)
        d_d, d_s = endpoint[:, :, 0] / REFERENCE_DISTANCE_MPC, endpoint[:, :, 1] / REFERENCE_DISTANCE_MPC
        k = p[:, 0, None]
        roots = np.sqrt(np.maximum(0., 1 + k*d_d**2))/d_d + np.sqrt(np.maximum(0., 1 + k*d_s**2))/d_s
        inv_ddt = (1/d_d - 1/d_s) * (1/d_d + 1/d_s) / roots
        with np.errstate(invalid='ignore', divide='ignore'):
            lens_log_prediction = np.log(REFERENCE_DISTANCE_MPC/inv_ddt)
            lens_ll = -.5 * np.sum(((lens_log_prediction - self.lens_obs) / self.lens_err)**2, axis=1)
        rates = p[:, 2:2+self.n_rate]
        slopes = np.diff(rates, axis=1) / np.diff(np.log1p(self.rate_z))
        smooth_ll = -.5 * np.sum((np.diff(slopes, axis=1) / self.smooth)**2, axis=1)
        log_likelihoods = dict(sn=sn_ll, lens=lens_ll, smooth=smooth_ll)
        if self.use_grb:
            a, b, scatter = p[:, 2+self.n_rate:5+self.n_rate].T[:, :, None]
            with np.errstate(invalid='ignore'):
                y = self.grb_yoffset + 2*np.log10(dm[:, self.lens_end:self.grb_end])
            variance = self.grb_yerr**2 + (b*self.grb_xerr)**2 + scatter**2
            log_likelihoods['grb'] = -.5 * np.sum((y-a-b*self.grb_x)**2/variance + np.log(variance), axis=1)
        if self.use_bao:
            dm_bao, dh_bao = dm[:, self.grb_end:], dh[:, self.grb_end:]
            dv = np.cbrt(self.bao.z.to_numpy() * dm_bao**2 * dh_bao)
            kind = self.bao.quantity.to_numpy()
            model = np.where(kind == 'DM_over_rs', dm_bao, np.where(kind == 'DH_over_rs', dh_bao, dv))
            residual = model / p[:, -1, None] - self.bao.value.to_numpy()
            log_likelihoods['bao'] = -.5 * np.sum((residual @ self.bao_inverse) * residual, axis=1)
        if include_lens_predictions:
            return log_likelihoods, physical, lens_log_prediction
        return log_likelihoods, physical

    def __call__(self, p):
        p = np.atleast_2d(p)
        lo, hi = np.asarray(self.bounds).T
        valid = np.all((p >= lo) & (p <= hi), axis=1)
        result = np.full(len(p), -np.inf)
        if valid.any():
            terms, physical = self.terms(p[valid])
            total = np.sum(list(terms.values()), axis=0)
            result[valid] = np.where(physical & np.isfinite(total), total, -np.inf)
        return result

    def exact_sn_check(self, theta):
        z = np.r_[self.sn['z'], self.sn['nodes']]
        dm, _, _ = distance_geometry(theta[None, 2:2+self.n_rate], theta[:1], self.rate_z, z)
        end = len(self.sn['z'])
        mag = 5*np.log10(dm[0]/z) + theta[1]
        approximate = self.sn['basis'] @ mag[end:]
        exact = mag[:end]
        residual = self.sn['white_data'] - solve_triangular(self.sn['whiten'], exact, lower=True)
        delta = solve_triangular(self.sn['whiten'], exact-approximate, lower=True)
        return dict(max_mag_error=float(np.max(np.abs(exact-approximate))),
                    exact_chi2=float(residual @ residual), chi2_error=float(2*(residual @ delta)+delta @ delta))


# Relative calibration and ensemble sampling.


SCALE_BOUNDS = np.log(np.array([40., 105.]) / REFERENCE_H)


NAMES = ['Omega_k', 'M70', *[f'ln_E_z{z:g}' for z in RATE_Z[1:]], 'H0rd_over_70']


BOUNDS = np.array([[-1.5, 3.], [-21., -17.], *[[-1., 3.]]*(len(RATE_Z)-1), [70., 260.]])


PROBABILITIES = [.16, .5, .84]


def pack_geometry(common, log_scale=0.):
    """Map common coordinates to the existing likelihood's dimensional parameters."""
    log_scale = np.broadcast_to(log_scale, len(common))
    scale = np.exp(log_scale)
    return np.column_stack([common[:, 0]*scale**2,
                            common[:, 1]+5*log_scale/np.log(10), log_scale,
                            common[:, 2:-1]+log_scale[:, None], common[:, -1]/scale])


def scale_terms(delta, errors):
    """Integrate the log-distance likelihood over the bounded log-scale prior."""
    precision = errors**-2
    sigma = 1/np.sqrt(precision.sum())
    mean = delta @ (precision*sigma**2)
    residual = (delta - mean[:, None])/errors
    lower, upper = (SCALE_BOUNDS[:, None]-mean)/sigma
    log_upper = log_ndtr(np.where(lower > 0, -lower, upper))
    log_lower = log_ndtr(np.where(lower > 0, -upper, lower))
    with np.errstate(divide='ignore'):
        log_mass = log_upper + np.log(-np.expm1(log_lower-log_upper))
    log_marginal = (-.5*np.sum(residual**2, axis=1) + log_mass
                    + np.log(sigma*np.sqrt(2*np.pi)/np.diff(SCALE_BOUNDS)[0]))
    return log_marginal, mean, sigma


class CalibrationLikelihood:
    def __init__(self, geometry, include_lenses):
        self.geometry = geometry
        self.include_lenses = include_lenses

    def evaluate(self, common):
        p = pack_geometry(common)
        terms, physical, predicted = self.geometry.terms(p, include_lens_predictions=True)
        delta = predicted-self.geometry.lens_obs
        relative = terms['sn'] + terms['bao'] + terms['smooth']
        relative = np.where(physical, relative, -np.inf)
        return relative, delta

    def __call__(self, common):
        common = np.atleast_2d(common)
        valid = np.all((common >= BOUNDS[:, 0]) & (common <= BOUNDS[:, 1]), axis=1)
        result = np.full(len(common), -np.inf)
        if valid.any():
            relative, delta = self.evaluate(common[valid])
            if self.include_lenses:
                active = np.isfinite(relative)
                relative[active] += scale_terms(delta[active], self.geometry.lens_err)[0]
            result[valid] = np.where(np.isfinite(relative), relative, -np.inf)
        return result


def run_chain(model, steps, seed, *, initial=None, widths=None, bounds=BOUNDS, progress=False, positions=None,
              walkers=48, save_initial=None):
    start = np.r_[[0., -19.35], .5*np.log(.3*(1+RATE_Z[1:])**3+.7), 142.] if initial is None else initial
    spread = np.r_[[.035, .015], np.full(len(RATE_Z)-1, .012), .6] if widths is None else widths
    scaled_bounds = (bounds-start[:, None])/spread[:, None]
    anchor = model(start)[0]

    def objective(x):
        lp = model(start+spread*x)[0]
        return anchor-lp if np.isfinite(lp) else 1e20

    rng = np.random.default_rng(seed)
    if positions is None:
        opt = minimize(objective, np.zeros(len(start)), bounds=scaled_bounds, method='Powell',
                       options={'maxiter': 2000, 'xtol': 1e-6, 'ftol': 1e-9})
        centre = start+spread*opt.x
        optimized = bool(opt.success)
        pos = centre + spread*rng.normal(size=(walkers, len(start)))
    else:
        centre = start
        optimized = None
        pos = positions.copy()
    n_walkers = len(pos)
    for attempt in range(30):
        bad = ~np.isfinite(model(pos))
        if not bad.any():
            break
        pos[bad] = centre + spread*rng.normal(size=(bad.sum(), len(start)))/(1+attempt//5)
    else:
        raise RuntimeError('Could not initialize physical ensemble')
    if save_initial is not None:
        np.save(save_initial, pos)
    moves = [(emcee.moves.DEMove(), .8), (emcee.moves.StretchMove(), .2)]
    sampler = emcee.EnsembleSampler(n_walkers, len(start), model, vectorize=True, moves=moves)
    sampler.random_state = np.random.RandomState(seed).get_state()
    sampler.run_mcmc(pos, steps, progress=progress)
    chain = sampler.get_chain(discard=steps//3)
    logp = sampler.get_log_prob(discard=steps//3)
    tau = emcee.autocorr.integrated_time(chain, quiet=True)
    return chain, logp, dict(retained_over_tau_max=float(len(chain)/tau.max()),
                             optimizer_converged=optimized, acceptance=float(sampler.acceptance_fraction.mean()))


def quantified(values):
    return {name: np.quantile(x, PROBABILITIES).tolist() for name, x in values.items()}


def weighted_quantile(values, normalized_weights):
    order = np.argsort(values)
    cumulative = np.cumsum(normalized_weights[order])-.5*normalized_weights[order]
    return np.interp(PROBABILITIES, cumulative, values[order]).tolist()


SHAPE_COLUMNS = np.r_[0, np.arange(2, len(NAMES))]


GRB_MAX_Z = 2.4


LENS_SKY = {'RX J1131-1231': (172.96462, -12.53298),
            'WGD 2038-4008': (309.51130, -40.13710),
            'B1608+656': (242.30808, 65.54108)}


def read_qso(path, max_redshift=1.479):
    if path.suffix == '.dat':
        raw = pd.read_fwf(path, colspecs=[(2,25), (28,38), (39,50), (51,57), (60,69),
                                             (72,79), (82,91), (94,101), (104,105)],
                         names=['Name','RAJ2000','DEJ2000','z','logFUV','e_logFUV',
                                'logFX','e_logFX','Group'], dtype=str)
    else:
        raw = pd.read_csv(path, sep='\t', comment='#', encoding='utf-8-sig', dtype=str)
    # The VizieR export starts with an optional units row and a separator row.
    raw['z'] = pd.to_numeric(raw['z'], errors='coerce')
    raw = raw.dropna(subset='z').copy()
    for name in ('logFUV', 'e_logFUV', 'logFX', 'e_logFX', 'Group', 'RAJ2000', 'DEJ2000'):
        raw[name] = pd.to_numeric(raw[name], errors='coerce')
    raw['Name'] = raw.Name.str.strip()
    if len(raw) != 2421 or raw.Name.duplicated().any():
        raise ValueError('Expected the complete VizieR Table 3 with unique source names')
    low = raw[(raw.z > 0) & (raw.z < max_redshift)].copy()
    if not np.isfinite(low.select_dtypes(include='number')).all().all():
        raise ValueError('A low-redshift quasar is missing a measurement')
    if not (low[['e_logFUV', 'e_logFX']] >= 0).all().all():
        raise ValueError('Flux errors must be nonnegative')
    rejected = np.zeros(len(low), dtype=bool)
    crossmatch = {}
    for name, (ra, dec) in LENS_SKY.items():
        dra = (low.RAJ2000.to_numpy() - ra)*np.cos(np.deg2rad(dec))
        dd = low.DEJ2000.to_numpy()-dec
        separation = 3600*np.hypot(dra, dd)
        overlap = separation < 15
        rejected |= overlap
        crossmatch[name] = dict(nearest_arcsec=float(separation.min()), excluded=low.Name[overlap].tolist())
    return low[~rejected], crossmatch


def uncapped_likelihood(smooth):
    # The shared implementation needs a matrix basis for SN. Empty arrays give exactly zero SN log likelihood.
    noise_free = dict(nodes=np.empty(0), precision=np.empty((0, 0)), rhs=np.empty(0))
    bao = pd.read_csv(ROOT/'bao_dr2'/'desi_mean.txt', sep=r'\s+', comment='#', names=['z', 'value', 'quantity'])
    cov = np.loadtxt(ROOT/'bao_dr2'/'desi_cov.txt')
    lenses = pd.read_csv(ROOT/'lenses.csv').dropna(subset='ddt_mpc')
    return GeometryLikelihood(noise_free, lenses, None, bao, cov, True, False, smooth), lenses.system.tolist()


class CandleLikelihood:
    def __init__(self, geometry, qso, grb, evolution, group_offsets, include_legacy_lenses=True):
        self.geometry, self.qso, self.grb = geometry, qso, grb
        self.evolution = evolution
        self.include_legacy_lenses = include_legacy_lenses
        self.names = [NAMES[i] for i in SHAPE_COLUMNS]
        self.bounds = BOUNDS[SHAPE_COLUMNS].tolist()
        self.initial = np.r_[0., .5*np.log(.3*(1+RATE_Z[1:])**3+.7), 142.].tolist()
        self.widths = [.035, *[.012]*(len(RATE_Z)-1), .6]
        self.qso_slice = slice(len(self.names), len(self.names)+3)
        if len(qso):
            self.add(['qso_B70', 'qso_gamma', 'qso_scatter'],
                     [(-3., 3.), (0.05, .98), (.02, 1.)], [0., .6, .23], [.035, .025, .02])
            self.qso_x = qso.logFUV.to_numpy()+27.5
            self.qso_y = qso.logFX.to_numpy()+31
            self.qso_xerr2 = qso.e_logFUV.to_numpy()**2
            self.qso_yerr2 = qso.e_logFX.to_numpy()**2
        self.grb_slice = slice(len(self.names), len(self.names)+3)
        if len(grb):
            self.add(['grb_a70', 'grb_b', 'grb_scatter'],
                     [(49., 56.), (0., 4.), (.03, 2.)], [52.9, 1.2, .45], [.035, .045, .025])
            self.grb_x = np.log10(grb.ep_keV.to_numpy()/300)
            self.grb_xerr2 = (grb.ep_err_keV/grb.ep_keV/np.log(10)).to_numpy()**2
            self.grb_offset = np.log10(4*np.pi*MPC_CM**2*grb.fluence*(1+grb.z)).to_numpy()
            self.grb_yerr2 = (grb.fluence_err/grb.fluence/np.log(10)).to_numpy()**2
        self.evolution_slice = slice(len(self.names), len(self.names)+2)
        if evolution:
            self.add(['qso_evolution', 'grb_evolution'], [(-1.5, 1.5), (-3., 3.)], [0., 0.], [.06, .08])
            self.qso_z_evo = np.log((1+qso.z.to_numpy())/1.8)
            self.grb_z_evo = np.log((1+grb.z.to_numpy())/2.3)
        self.group_mask = None
        if group_offsets:
            groups = sorted(set(qso.Group)-{5})
            self.group_mask = (qso.Group.to_numpy()[None, :] == np.asarray(groups)[:, None]).astype(float)
            self.add([f'qso_group_{g}_offset' for g in groups], [(-.7, .7)]*len(groups),
                     [0.]*len(groups), [.025]*len(groups))
        self.bounds = np.asarray(self.bounds)
        self.initial = np.asarray(self.initial)
        self.widths = np.asarray(self.widths)
        self.targets = np.r_[qso.z.to_numpy(), grb.z.to_numpy(), GRB_MAX_Z]

    def add(self, names, bounds, initial, widths):
        self.names.extend(names)
        self.bounds.extend(bounds)
        self.initial.extend(initial)
        self.widths.extend(widths)

    def relative_parameters(self, t):
        return np.insert(t[:, :len(SHAPE_COLUMNS)], 1, -19.35, axis=1)

    def evaluate(self, t, residual_chi2=None):
        common = self.relative_parameters(t)
        p = pack_geometry(common)
        terms, physical, lens_log = self.geometry.terms(p, include_lens_predictions=True)
        dm, _, branch = distance_geometry(p[:, 2:2+len(RATE_Z)], p[:, 0], RATE_Z, self.targets)
        logl = terms['bao']+terms['smooth']
        delta = lens_log-self.geometry.lens_obs
        physical &= branch
        if self.include_legacy_lenses and physical.any():
            logl[physical] += scale_terms(delta[physical], self.geometry.lens_err)[0]
        with np.errstate(invalid='ignore', divide='ignore'):
            if len(self.qso):
                intercept, slope, scatter = t[:, self.qso_slice].T[:, :, None]
                dl = dm[:, :len(self.qso)]*(1+self.qso.z.to_numpy())
                pred = intercept+slope*self.qso_x+2*(slope-1)*np.log10(dl/5000)
                if self.evolution:
                    pred += t[:, self.evolution_slice.start, None]*self.qso_z_evo
                if self.group_mask is not None:
                    offset = t[:, -len(self.group_mask):]
                    pred += offset @ self.group_mask
                    logl -= .5*np.sum((offset/.20)**2, axis=1)
                variance = self.qso_yerr2+slope**2*self.qso_xerr2+scatter**2
                square = (self.qso_y-pred)**2/variance
                logl -= .5*np.sum(square+np.log(variance), axis=1)
                if residual_chi2 is not None:
                    residual_chi2['qso'] = square.sum(axis=1)
            if len(self.grb):
                intercept, slope, scatter = t[:, self.grb_slice].T[:, :, None]
                energy = self.grb_offset+2*np.log10(dm[:, len(self.qso):-1])
                pred = intercept+slope*self.grb_x
                if self.evolution:
                    pred += t[:, self.evolution_slice.stop-1, None]*self.grb_z_evo
                variance = self.grb_yerr2+slope**2*self.grb_xerr2+scatter**2
                square = (energy-pred)**2/variance
                logl -= .5*np.sum(square+np.log(variance), axis=1)
                if residual_chi2 is not None:
                    residual_chi2['grb'] = square.sum(axis=1)
        return np.where(physical & np.isfinite(logl), logl, -np.inf), delta

    def __call__(self, t):
        t = np.atleast_2d(t)
        valid = np.all((t >= self.bounds[:, 0]) & (t <= self.bounds[:, 1]), axis=1)
        result = np.full(len(t), -np.inf)
        if valid.any():
            result[valid] = self.evaluate(t[valid])[0]
        return result


# Hierarchical TDCOSMO likelihood and parallel evaluation.


def _initialize_posterior(model):
    global _worker_model, _blas_limit
    _blas_limit = threadpool_limits(limits=1, user_api='blas')
    _worker_model = model


def _posterior_block(block):
    return _worker_model(block)


@contextmanager
def batched_posterior(model, workers):
    if workers == 1:
        yield model
        return

    with get_context('spawn').Pool(workers, initializer=_initialize_posterior, initargs=(model,)) as pool:
        def evaluate(points):
            points = np.atleast_2d(points)
            if len(points) <= 1:
                return model(points)
            blocks = np.array_split(points, min(workers, len(points)))
            return np.concatenate(pool.map(_posterior_block, blocks))

        yield evaluate


INPUTS = ROOT.parent/'tdc_inputs'


QSO_PATH = ROOT.parent/'data'/'J_A+A_642_A150.tar'/'table3.dat'


GRB_PATH = AMATI/'reanalysis/data/parents/a118_full.csv'


OUT = ROOT/'aj_checks'/'tdcosmo2025'


NUI_NAMES = ['log_H0_over_70', 'lambda_mst', 'ln_lambda_mst_sigma', 'alpha_lambda',
             'a_ani', 'ln_a_ani_sigma', 'gamma_pl_RXJ1131']


NUI_BOUNDS = np.array([SCALE_BOUNDS, [.5, 1.5], np.log([.001, .5]), [-1., 1.],
                       [.87, 1.12], np.log([.01, 1.]), [1.1, 2.9]])


NUI_START = [0., 1., np.log(.05), 0., 1., np.log(.10), 1.95]


NUI_WIDTHS = [.025, .025, .3, .025, .02, .25, .018]


SETTINGS = dict(lambda_mst_distribution='GAUSSIAN', anisotropy_sampling=True,
                anisotropy_model='const', anisotropy_distribution='GAUSSIAN',
                alpha_lambda_sampling=True, anisotropy_parameterization='TAN_RAD')


CASES = ('bao_lenses', 'pantheon_plus', 'qso_grb', 'pantheon_plus_z023',
         'qso', 'grb', 'qso_grb_evolution', 'qso_grb_groups',
         'pantheon_plus_lowz_matched', 'pantheon_plus_count_control')


SN_SUBSAMPLE_SEED = 904135


SN_LOW_Z = .15


MATCHED_SN_CASES = ('pantheon_plus_lowz_matched', 'pantheon_plus_count_control')


def sn_subsample(candidate, probe_z, match_low_z, seed=SN_SUBSAMPLE_SEED):
    # Select whole SNe by catalogue ID, then keep their covariance rows together.
    by_sn = candidate.groupby('CID', sort=True).zHD.median()
    low = by_sn[by_sn < SN_LOW_Z].index.to_numpy()
    high = by_sn[by_sn >= SN_LOW_Z].index.to_numpy()
    fraction = np.mean(probe_z[probe_z >= .023] < SN_LOW_Z)
    n_low = int(round(len(high)*fraction/(1-fraction)))
    rng = np.random.default_rng(seed)
    if match_low_z:
        chosen = np.r_[rng.choice(low, n_low, replace=False), high]
    else:
        chosen = rng.choice(by_sn.index.to_numpy(), len(high)+n_low, replace=False)
    return candidate.index[candidate.CID.isin(chosen)].to_numpy()


def source_positions(model, path, seed, walkers):
    rng = np.random.default_rng(seed)
    with np.load(path) as prior:
        names = prior['names'].tolist()
        chain = prior['chain']
        n = max(walkers, 2*len(model.names))
        source = chain[rng.integers(len(chain)//2, len(chain), n),
                       rng.integers(chain.shape[1], size=n)]
    pos = model.initial + .2*model.widths*rng.normal(size=(n, len(model.names)))
    for j, name in enumerate(model.names):
        if name in names:
            pos[:, j] = source[:, names.index(name)]
    return pos


def load_lenses():
    # These are the processed inputs distributed by the CosmoSIS lens module.
    lens_path = INPUTS/'tdcosmo2025_likelihood_processed_const_.pkl'
    jam_path = INPUTS/'kin_axi_jam_scaling'/'tdcosmo_correction.pickle'
    with lens_path.open('rb') as f:
        lenses = pickle.load(f)
    with jam_path.open('rb') as f:
        jam = {d['name']: d['correction_combined'] for d in pickle.load(f)}
    for lens in lenses:
        lens['vel_disp_scaling_distributions'] = jam[lens['name']]
    return lenses, {p.name: sha256(p.read_bytes()).hexdigest()
                    for p in (lens_path, jam_path, QSO_PATH, GRB_PATH)}


def log_normal(x):
    return -.5*x*x-.5*np.log(2*np.pi)


def kinematics_chi2(delta, scale, observation_cov, model_cov):
    # Contiguous Sobol draws let LLVM vectorize independent Cholesky operations.
    return _kinematics_chi2_draws(
        np.ascontiguousarray(delta.swapaxes(1, 2)),
        np.ascontiguousarray(scale.swapaxes(1, 2)), observation_cov, model_cov,
    )


@njit(cache=True, nogil=True)
def _kinematics_chi2_draws(delta, scale, observation_cov, model_cov):
    n_chains, n_bins, n_draws = delta.shape
    result = np.zeros((n_chains, n_draws))
    chol = np.empty((n_bins, n_bins, n_draws))
    whitened = np.empty((n_bins, n_draws))
    for i in range(n_chains):
        for k in range(n_bins):
            for l in range(k+1):
                row = chol[k, l]
                for j in range(n_draws):
                    row[j] = observation_cov[k, l] + model_cov[k, l]*scale[i, k, j]*scale[i, l, j]
                for m in range(l):
                    for j in range(n_draws):
                        row[j] -= chol[k, m, j]*chol[l, m, j]
                if k == l:
                    for j in range(n_draws):
                        row[j] = np.sqrt(row[j])
                else:
                    for j in range(n_draws):
                        row[j] /= chol[l, l, j]
            residual = whitened[k]
            for j in range(n_draws):
                residual[j] = delta[i, k, j]
            for l in range(k):
                for j in range(n_draws):
                    residual[j] -= chol[k, l, j]*whitened[l, j]
            for j in range(n_draws):
                residual[j] /= chol[k, k, j]
                result[i, j] += residual[j]*residual[j]
    return result


class IntegratedLens:
    """Deterministic quadrature with a mixture of the MST prior and time-delay KDE."""
    def __init__(self, raw, lens, power, seed, stratify_jam=False):
        self.lens = lens
        self.mst_offset = raw['lambda_scaling_property']
        self.gamma_slope = raw.get('gamma_pl_ddt_slope')
        self.gamma_pivot = raw.get('gamma_pl_pivot')
        self.fit_gamma = self.gamma_slope is not None
        self.beta_bounds = (lens._aniso_distribution._a_ani_min,
                            lens._aniso_distribution._a_ani_max)
        rand = qmc.Sobol(d=5, seed=seed).random_base2(power)
        self.prior_quantiles = ndtri(rand[:, 0])
        self.aniso_quantiles = rand[:, 1]
        self.kappa = lens._los._kappa_dist._cdf_inv_func(rand[:, 2])
        corrections = np.asarray(raw['vel_disp_scaling_distributions'])
        if stratify_jam:
            # Sorting preserves the empirical distribution and stratifies the
            # velocity-scale correction (cross-aperture structure is retained).
            order = np.argsort(corrections if corrections.ndim == 1 else corrections.mean(axis=1))
            corrections = corrections[order]
        self.jam = corrections[(rand[:, 3]*len(corrections)).astype(int)]**2
        if self.jam.ndim == 1:
            self.jam = self.jam[:, None]
        kde = lens._lens_type._tdLikelihood._score.__self__
        if kde.kernel != 'gaussian':
            raise ValueError('Expected a Gaussian time-delay KDE')
        kde_weights = np.asarray(kde.tree_.sample_weight)
        cdf = np.cumsum(kde_weights)/kde_weights.sum()
        kde_bins = np.asarray(kde.tree_.data).ravel()
        self.td_samples = kde_bins[np.searchsorted(cdf, rand[:, 4])]+kde.bandwidth*self.prior_quantiles
        # The likelihood is needed both at proposal draws and off-grid. Tabulate
        # the official KDE at 1/20 of a kernel bandwidth for cubic interpolation.
        grid = np.arange(kde_bins.min()-12*kde.bandwidth,
                         kde_bins.max()+12*kde.bandwidth, kde.bandwidth/20)
        self.kde = CubicSpline(grid, kde.score_samples(grid[:, None]), extrapolate=False)
        self.kin = lens._lens_type._kinlikelihood

    def anisotropy(self, mean, sigma):
        # hierArc accepts both signs of the TAN_RAD draw when beta is in range.
        bmin, bmax = self.beta_bounds
        lo = np.sqrt(max(0., 1-bmax))
        hi = np.sqrt(1-bmin)
        neg_start, neg_stop = ndtr((-hi-mean)/sigma), ndtr((-lo-mean)/sigma)
        pos_start, pos_stop = ndtr((lo-mean)/sigma), ndtr((hi-mean)/sigma)
        left = neg_stop-neg_start
        u = self.aniso_quantiles*(left+pos_stop-pos_start)
        a = mean + sigma*ndtri(np.where(u < left, u+neg_start, u-left+pos_start))
        return 1-a*a

    def __call__(self, ddt, dd, nuisance):
        _, m, log_s, alpha, a, log_a_s, gamma = nuisance.T[:, :, None]
        beta = self.anisotropy(a, np.exp(log_a_s))
        shift = (gamma-self.gamma_pivot)*self.gamma_slope if self.fit_gamma else 0.
        mean = m+alpha*self.mst_offset
        sigma = np.exp(log_s)
        slope = ddt[:, None]*(1-self.kappa)
        prior_draw = mean+sigma*self.prior_quantiles
        kde_draw = (self.td_samples+shift)/slope
        lam = np.concatenate([prior_draw, kde_draw], axis=1)
        scale = np.tile(slope, (1, 2))
        x = lam*scale-shift
        log_kde = self.kde(x)
        log_kde = np.nan_to_num(log_kde, nan=-np.inf)
        log_prior = log_normal((lam-mean)/sigma)-np.log(sigma)
        log_prop = np.logaddexp(log_prior, log_kde+np.log(scale))-np.log(2.)
        # The displace_prediction floor contributes only at vanishing Ddt,
        # over 12 KDE bandwidths outside the time-delay support of these lenses.
        with np.errstate(invalid='ignore'):
            weight = log_prior+log_kde-log_prop
        weight = np.where((lam > 0) & np.isfinite(weight), weight, -np.inf)
        if self.fit_gamma:
            gam = np.broadcast_to(gamma, beta.shape)
            j_scale = np.stack([s._f_ani.ev(beta, gam) for s in self.lens._j_scaling_ifu], axis=-1)
        else:
            j_scale = np.stack([s._f_ani(beta) for s in self.lens._j_scaling_ifu], axis=-1)
        j_scale *= self.jam
        j_scale = np.tile(j_scale, (1, 2, 1))
        ds_dds = np.maximum(x/(dd[:, None]*(1+self.lens.z_lens)), 0)
        velocity_scale = np.sqrt(ds_dds[:, :, None]*j_scale)*(c/1000)
        delta = self.kin._sigma_v_measured - np.sqrt(self.kin._j_model)*velocity_scale
        if self.kin.num_data == 1:
            variance = self.kin._error_cov_measurement.item() + self.kin._error_cov_j_sqrt.item()*velocity_scale[:, :, 0]**2
            log_kin = -.5*delta[:, :, 0]**2/variance
        else:
            log_kin = -.5*kinematics_chi2(delta, velocity_scale, self.kin._error_cov_measurement,
                                            self.kin._error_cov_j_sqrt)
        log_lens = logsumexp(weight+log_kin, axis=1)-np.log(weight.shape[1])
        if self.fit_gamma:
            log_lens -= .5*((gamma[:, 0]-1.95)/.05)**2
        return log_lens


class TDModel:
    def __init__(self, case, raw, official, power, seed, qso, grb, rx_power=None, stratify_jam=False,
                 smooth=1.5, h0_max=None, first_node=RATE_Z[1], sn_selection_seed=SN_SUBSAMPLE_SEED):
        self.candles = not case.startswith('pantheon_plus')
        self.rate_z = RATE_Z.copy()
        self.rate_z[1] = first_node
        if not RATE_Z[0] < first_node < RATE_Z[2]:
            raise ValueError(f'The first expansion node must lie in ({RATE_Z[0]}, {RATE_Z[2]})')
        if self.candles and first_node != RATE_Z[1]:
            raise ValueError('Alternative first nodes are implemented for the Pantheon+ likelihood')
        self.zd = np.array([x['z_lens'] for x in raw])
        self.zs = np.array([x['z_source'] for x in raw])
        self.lenses = [IntegratedLens(x, lens, rx_power if rx_power and x['name'] == 'RXJ1131-1231' else power,
                                     seed+i, stratify_jam)
                       for i, (x, lens) in enumerate(zip(raw, official._lens_list))]
        if self.candles:
            geo, _ = uncapped_likelihood(smooth)
            if case in ('bao_lenses', 'grb'):
                qso = qso.iloc[:0]
            if case in ('bao_lenses', 'qso'):
                grb = grb.iloc[:0]
            self.base = CandleLikelihood(geo, qso, grb, case.endswith('_evolution'),
                                         case.endswith('_groups'), include_legacy_lenses=False)
            names, bounds, initial, widths = (self.base.names, self.base.bounds, self.base.initial, self.base.widths)
        else:
            self.sn_z_min = .23 if case.endswith('_z023') else .023
            selector = None
            if case in MATCHED_SN_CASES:
                probe_z = np.r_[qso.z, grb.z]
                def selector(candidate):
                    return sn_subsample(candidate, probe_z, case == 'pantheon_plus_lowz_matched',
                                        seed=sn_selection_seed)
            sn = prepare_sn(AMATI, SPLINE_NODES, self.sn_z_min, self.rate_z, row_selector=selector)
            bao = pd.read_csv(ROOT/'bao_dr2'/'desi_mean.txt', sep=r'\s+', comment='#',
                              names=['z', 'value', 'quantity'])
            cov = np.loadtxt(ROOT/'bao_dr2'/'desi_cov.txt')
            lens_table = pd.read_csv(ROOT/'lenses.csv').dropna(subset='ddt_mpc')
            geo = GeometryLikelihood(sn, lens_table, None, bao, cov, True, False, smooth, rate_z=self.rate_z)
            self.base = CalibrationLikelihood(geo, False)
            names = [*NAMES[:2], *[f'ln_E_z{z:g}' for z in self.rate_z[1:]], NAMES[-1]]
            bounds = BOUNDS
            initial = np.r_[[0., -19.35], .5*np.log(.3*(1+self.rate_z[1:])**3+.7), 142.]
            widths = [.035, .015, *[.012]*(len(self.rate_z)-1), .6]
        self.dimension = len(names)
        self.names = list(names)+NUI_NAMES
        self.bounds = np.vstack([bounds, NUI_BOUNDS])
        if h0_max is not None:
            self.bounds[self.dimension, 1] = np.log(h0_max/REFERENCE_H)
        self.initial = np.r_[initial, NUI_START]
        self.widths = np.r_[widths, NUI_WIDTHS]

    def common(self, relative):
        return self.base.relative_parameters(relative) if self.candles else relative

    def geometry(self, common, log_scale):
        p = pack_geometry(common, log_scale)
        dm, _, domain = distance_geometry(p[:, 2:2+len(self.rate_z)], p[:, 0], self.rate_z,
                                          np.r_[self.zd, self.zs])
        dd, ds = np.hsplit(dm, 2)
        k = p[:, 0, None]/(299792.458/REFERENCE_H)**2
        root = np.sqrt(1+k*dd**2)
        root_s = np.sqrt(1+k*ds**2)
        dds = (ds**2-dd**2)/(ds*root+dd*root_s)
        return dd*ds/dds, dd/(1+self.zd), domain

    def __call__(self, values):
        values = np.atleast_2d(values)
        result = np.full(len(values), -np.inf)
        inside = np.flatnonzero(np.all((values >= self.bounds[:, 0]) & (values <= self.bounds[:, 1]), axis=1))
        if not len(inside):
            return result
        t = values[inside]
        relative, nuisance = t[:, :self.dimension], t[:, self.dimension:]
        ll = self.base(relative)
        active = np.flatnonzero(np.isfinite(ll))
        if not len(active):
            return result
        nuisance = nuisance[active]
        ddt, dd, domain = self.geometry(self.common(relative[active]), nuisance[:, 0])
        lens_ll = sum(lens(ddt[:, i], dd[:, i], nuisance) for i, lens in enumerate(self.lenses))
        combined = lens_ll+ll[active]
        result[inside[active]] = np.where(domain & np.isfinite(combined), combined, -np.inf)
        return result


def validation(raw, official, model):
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70, Om0=.3)
    rng = np.random.default_rng(1523)
    results = []
    for i, (raw_lens, official_lens, fast_lens) in enumerate(zip(raw, official._lens_list, model.lenses)):
        official_lens._num_distribution_draws = 20000
        zd, zs = official_lens.z_lens, official_lens.z_source
        dd = cosmo.angular_diameter_distance(zd).value
        ds = cosmo.angular_diameter_distance(zs).value
        dds = cosmo.angular_diameter_distance(zd, zs).value
        ddt = (1+zd)*dd*ds/dds
        rows = []
        for shift, mst, spread in zip([0., .06, -.06], [1., 1.06, .94], [.05, .1, .012]):
            nuisance = np.array([[shift, mst, np.log(spread), 0., 1., np.log(.10), 1.95]])
            dt = ddt*np.exp(-shift)
            dl = dd*np.exp(-shift)
            np.random.seed(int(rng.integers(1, 2**31)))
            reference = official_lens.hyper_param_likelihood(
                dt, dl, 0., kwargs_lens=dict(lambda_mst=mst, lambda_mst_sigma=spread,
                                             alpha_lambda=0., gamma_pl_list=[1.95]),
                kwargs_kin=dict(a_ani=1., a_ani_sigma=.1))
            reference += official_lens._lens_type._tdLikelihood._norm_factor
            high_values = [IntegratedLens(raw_lens, official_lens, 13, seed+i)(
                np.array([dt]), np.array([dl]), nuisance)[0] for seed in (180, 940)]
            high = np.logaddexp(*high_values)-np.log(2.)
            fast = fast_lens(np.array([dt]), np.array([dl]), nuisance)[0]
            rows.append(dict(hierarc_mc_normalized_kde=float(reference), sobol_reference=float(high),
                             independent_high_orders=[float(v) for v in high_values],
                             fit_quadrature=float(fast)))
        results.append(dict(lens=official_lens.name, points=rows))
    return results


def sample(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', default='bao_lenses', choices=CASES)
    parser.add_argument('--steps', type=int, default=13000)
    parser.add_argument('--power', type=int, default=6)
    parser.add_argument('--seed', type=int, default=155)
    parser.add_argument('--rx-power', type=int)
    starts = parser.add_mutually_exclusive_group()
    starts.add_argument('--warm-start', type=int, metavar='SOURCE_SEED')
    starts.add_argument('--start-from', type=Path)
    starts.add_argument('--initial-positions', type=Path, help='Replay saved starting walker positions')
    parser.add_argument('--save-initial', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--walkers', type=int, default=48)
    parser.add_argument('--workers', type=int, default=1, help='CPU processes for each emcee walker batch')
    parser.add_argument('--smooth', type=float, default=1.5)
    parser.add_argument('--qso-z-max', type=float, default=1.3)
    parser.add_argument('--h0-max', type=float, default=105.)
    parser.add_argument('--first-node', type=float, default=RATE_Z[1])
    parser.add_argument('--sn-selection-seed', type=int, default=SN_SUBSAMPLE_SEED)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    if args.smooth <= 0:
        parser.error('--smooth must be positive')
    if not .023 < args.qso_z_max <= GRB_MAX_Z:
        parser.error('--qso-z-max must lie in (0.023,2.4]')
    if not np.isfinite(args.h0_max) or args.h0_max <= REFERENCE_H:
        parser.error('--h0-max must exceed the initial H0')
    if not RATE_Z[0] < args.first_node < RATE_Z[2]:
        parser.error(f'--first-node must lie in ({RATE_Z[0]}, {RATE_Z[2]})')
    if args.first_node != RATE_Z[1] and not args.case.startswith('pantheon_plus'):
        parser.error('--first-node is only supported with Pantheon+')
    if args.sn_selection_seed < 0 or (args.sn_selection_seed != SN_SUBSAMPLE_SEED and args.case not in MATCHED_SN_CASES):
        parser.error('--sn-selection-seed requires a nonnegative seed and a matched Pantheon+ case')
    args.output.mkdir(parents=True, exist_ok=True)
    raw, hashes = load_lenses()
    official = LensSampleLikelihood(raw, kwargs_global_model=SETTINGS)
    qso, matches = read_qso(QSO_PATH, max_redshift=args.qso_z_max)
    grb = pd.read_csv(GRB_PATH, dtype={'grb_id': str}).query('0 < z <= @GRB_MAX_Z').copy()
    model = TDModel(args.case, raw, official, args.power, args.seed, qso, grb, args.rx_power,
                    stratify_jam=bool(args.rx_power), smooth=args.smooth, h0_max=args.h0_max,
                    first_node=args.first_node, sn_selection_seed=args.sn_selection_seed)
    if args.validate:
        result = validation(raw, official, model)
        (args.output/'likelihood_validation.json').write_text(json.dumps(result, indent=2), encoding='utf8')
        print(result, flush=True)
        return
    start = perf_counter()
    print(args.case, model(model.initial), len(qso), len(grb), flush=True)
    positions = None
    rx_tag = f'_rx{args.rx_power}' if args.rx_power else ''
    tag = f'{args.case}_p{args.power}{rx_tag}_s{args.seed}'
    if args.start_from is not None:
        positions = source_positions(model, args.start_from, args.seed, args.walkers)
    if args.warm_start is not None:
        start_path = OUT/f'{args.case}_p6_s{args.warm_start}_chain.npz'
        with np.load(start_path) as prior:
            positions = prior['chain'][-1]
    if args.initial_positions is not None:
        positions = np.load(args.initial_positions, allow_pickle=False)
        if positions.ndim != 2 or positions.shape[1] != len(model.names):
            parser.error('--initial-positions columns differ from the sampled parameters')
    walkers = len(positions) if positions is not None else max(args.walkers, 2*len(model.names))
    workers = min(args.workers, max(1, walkers//2))
    with batched_posterior(model, workers) as posterior:
        chain, logp, info = run_chain(posterior, args.steps, args.seed, initial=model.initial,
                                     widths=model.widths, bounds=model.bounds, progress=True, positions=positions,
                                     walkers=walkers,
                                     save_initial=args.output/f'{tag}_initial.npy' if args.save_initial else None)
    common = model.common(chain.reshape(-1, len(model.names))[:, :model.dimension])
    nuisance = chain.reshape(-1, len(model.names))[:, model.dimension:]
    log_scale = nuisance[:, 0]
    H0 = REFERENCE_H*np.exp(log_scale)
    estimates = quantified(dict(H0=H0, r_d=REFERENCE_H*common[:, -1]/H0, Omega_k=common[:, 0],
                                k70=common[:, 0]*np.exp(2*log_scale), H0_r_d=REFERENCE_H*common[:, -1]))
    flat_chain = chain.reshape(-1, len(model.names))
    estimates.update(quantified(dict(zip(model.names, flat_chain.T))))
    results = dict(parameters=estimates, diagnostics=info, steps=args.steps, walkers=chain.shape[1], workers=workers,
                   burn_in=args.steps//3, rate_nodes=model.rate_z.tolist(), sobol_power=args.power, rx_power=args.rx_power,
                   seed=args.seed, smooth=args.smooth, h0_max=args.h0_max, qso_z_max=args.qso_z_max, input_sha256=hashes, qso_rows=len(model.base.qso) if model.candles else 0,
                   grb_rows=len(model.base.grb) if model.candles else 0,
                   time_delay_source_crossmatch=matches, seconds=perf_counter()-start)
    if not model.candles:
        sample = model.base.geometry.sn
        results.update(sn_z_min=model.sn_z_min, sn_rows=len(sample['z']),
                       sn_unique=len(np.unique(sample['cid'])),
                       sn_rows_z_lt_015=int(np.count_nonzero(sample['z'] < SN_LOW_Z)),
                       sn_unique_z_lt_015=len(np.unique(sample['cid'][sample['z'] < SN_LOW_Z])))
        if args.case in MATCHED_SN_CASES:
            results['sn_selection_seed'] = args.sn_selection_seed
            results['sn_rows_selected'] = sample['row_indices'].tolist()
    np.savez_compressed(args.output/f'{tag}_chain.npz', chain=chain, log_prob=logp, names=model.names)
    (args.output/f'{tag}.json').write_text(json.dumps(results, indent=2), encoding='utf8')
    print(results, flush=True)


# Importance correction with two independently scrambled Sobol quadratures.


def evaluate_lens(task, raw_lenses, lenses, ddt, dd, nuisance, stratify_jam):
    lens_index, seed, power = task
    integral = IntegratedLens(raw_lenses[lens_index], lenses[lens_index], power, seed+lens_index,
                              stratify_jam=stratify_jam)
    return np.concatenate([
        integral(ddt[batch, lens_index], dd[batch, lens_index], nuisance[batch])
        for batch in np.array_split(np.arange(len(nuisance)), 128)
    ])


def initialize_worker(*inputs):
    global _quadrature_inputs, _blas_limit
    _blas_limit = threadpool_limits(limits=1, user_api='blas')
    _quadrature_inputs = inputs


def evaluate_in_worker(task):
    return evaluate_lens(task, *_quadrature_inputs)


def lens_quadratures(tasks, inputs, workers):
    if workers == 1:
        return [evaluate_lens(task, *inputs) for task in tasks]
    with get_context('spawn').Pool(min(workers, len(tasks)), initializer=initialize_worker,
                                   initargs=inputs) as pool:
        return pool.map(evaluate_in_worker, tasks, chunksize=1)


def refine(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=CASES)
    parser.add_argument('--samples', type=int, default=8192)
    parser.add_argument('--power', type=int, default=8)
    parser.add_argument('--source-power', type=int, default=6)
    parser.add_argument('--source-rx-power', type=int)
    parser.add_argument('--refine-rx-power', type=int)
    parser.add_argument('--source-seed', type=int, default=155)
    parser.add_argument('--workers', type=int, default=1, help='CPU processes for lens integrals and Sobol scrambles')
    parser.add_argument('--source-dir', type=Path, default=OUT)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    args.output.mkdir(parents=True, exist_ok=True)
    rx_tag = f'_rx{args.source_rx_power}' if args.source_rx_power else ''
    tag = f'{args.case}_p{args.source_power}{rx_tag}_s{args.source_seed}'
    metadata = json.loads((args.source_dir/f'{tag}.json').read_text(encoding='utf8'))
    with np.load(args.source_dir/f'{tag}_chain.npz') as archive:
        names = archive['names'].tolist()
        chain = archive['chain']
    rng = np.random.default_rng(145992)
    flat = chain.reshape(-1, chain.shape[-1])
    boundaries = np.linspace(0, len(flat), args.samples+1, dtype=int)
    idx = rng.integers(boundaries[:-1], boundaries[1:])
    points = flat[idx]

    raw, _ = load_lenses()
    official = LensSampleLikelihood(raw, kwargs_global_model=SETTINGS)
    qso, _ = read_qso(QSO_PATH, max_redshift=metadata.get('qso_z_max', 1.3))
    grb = pd.read_csv(GRB_PATH, dtype={'grb_id': str}).query('0 < z <= @GRB_MAX_Z')
    model = TDModel(args.case, raw, official, args.source_power, args.source_seed, qso, grb,
                    smooth=metadata.get('smooth', 1.5), h0_max=metadata.get('h0_max'),
                    first_node=metadata.get('rate_nodes', RATE_Z)[1],
                    sn_selection_seed=metadata.get('sn_selection_seed', SN_SUBSAMPLE_SEED))
    if model.names != names:
        raise ValueError('Saved chain columns do not match the likelihood')
    if 'sn_rows_selected' in metadata:
        if model.base.geometry.sn['row_indices'].tolist() != metadata['sn_rows_selected']:
            raise ValueError('The reconstructed supernova selection differs from the sampling run')
    nuisance = points[:, model.dimension:]
    common = model.common(points[:, :model.dimension])
    ddt, dd, valid = model.geometry(common, nuisance[:, 0])
    if not valid.all():
        raise ValueError('Saved posterior includes a point outside the distance-sum-rule domain')
    tasks = []
    for seed in (args.source_seed, 29753, 31825):
        use_source = seed == args.source_seed
        power = args.source_power if use_source else args.power
        for i, raw_lens in enumerate(raw):
            rx_power = args.source_rx_power if use_source else args.refine_rx_power
            if raw_lens['name'] != 'RXJ1131-1231':
                rx_power = None
            tasks.append((i, seed, rx_power or power))
    inputs = (raw, official._lens_list, ddt, dd, nuisance, bool(args.source_rx_power))
    print(f'Evaluating {len(tasks)} lens quadratures with {min(args.workers, len(tasks))} CPU worker(s)', flush=True)
    columns = lens_quadratures(tasks, inputs, args.workers)
    old, a, b = (np.vstack(part).T for part in np.split(np.asarray(columns), 3))
    refined = np.logaddexp(a, b)-np.log(2.)
    logw_by_run = [(x-old).sum(axis=1) for x in (refined, a, b)]
    H0 = REFERENCE_H*np.exp(nuisance[:, 0])
    parameters = dict(zip(names, points.T))
    parameters.update(H0=H0, r_d=REFERENCE_H*common[:, -1]/H0, Omega_k=common[:, 0],
                      k70=common[:, 0]*np.exp(2*nuisance[:, 0]),
                      H0_r_d=REFERENCE_H*common[:, -1])
    results = []
    for logw in logw_by_run:
        w = np.exp(logw-logw.max())
        w /= w.sum()
        results.append(dict(ess=float(1/(w@w)), max_weight=float(w.max()),
                            parameters={key: weighted_quantile(value, w) for key, value in parameters.items()}))
    absolute_h0 = REFERENCE_H*np.exp(chain[:, :, names.index('log_H0_over_70')])
    absolute_rd = REFERENCE_H*chain[:, :, names.index('H0rd_over_70')]/absolute_h0
    physical = np.stack([absolute_h0, absolute_rd, chain[:, :, 0]], axis=-1)
    n_over_tau = len(chain)/integrated_time(physical, quiet=True)
    logw = logw_by_run[0]
    halves = []
    for indices in np.array_split(np.arange(len(points)), 2):
        log_weights = logw[indices]
        w = np.exp(log_weights-log_weights.max())
        w /= w.sum()
        halves.append(dict(ess=float(1/(w@w)), medians={
            key: weighted_quantile(parameters[key][indices],w)[1] for key in ('H0','r_d','Omega_k')}))
    out = dict(source=tag, physical_retained_over_tau=dict(zip(('H0','r_d','Omega_k'),n_over_tau.tolist())),
               weighted_half_summaries=halves, refinements=[29753, 31825], source_steps=int(chain.shape[0]),
               workers=min(args.workers, len(tasks)),
               refined_power=args.power, refined_rx_power=args.refine_rx_power, points=args.samples,
               combined=results[0], independent=results[1:],
               lens_quadrature_rms={x['name']: float(v) for x, v in
                                    zip(raw, np.sqrt(np.mean((a-b)**2, axis=0)))})
    (args.output/f'{args.case}_refined.json').write_text(json.dumps(out, indent=2), encoding='utf8')
    np.savez_compressed(args.output/f'{args.case}_refined_draws.npz', points=points, source_flat_indices=idx,
                       log_relative_weight=logw_by_run[0], parameter_names=names)
    print({k: out['combined']['parameters'][k] for k in ('H0','r_d','Omega_k')}, results[0]['ess'], flush=True)
