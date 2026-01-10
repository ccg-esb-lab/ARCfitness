import os
import re
import time
import copy
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.patches import Patch, Ellipse
from matplotlib.ticker import MultipleLocator
from numpy.linalg import lstsq
import matplotlib.image as mpimg

import re
import pandas as pd


def _norm(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())



def load_exp_fitness(sheet, tab='fitness_aerobiosis'):
    ws = sheet.worksheet(tab)
    df = (get_as_dataframe(ws, header=0, evaluate_formulas=True)
          .dropna(how='all').dropna(axis=1, how='all'))
    rep_cols = [c for c in df.columns if str(c).strip().isdigit()]
    if 'mean' in df.columns:
        df['w_exp'] = pd.to_numeric(df['mean'], errors='coerce')
    else:
        df['w_exp'] = pd.to_numeric(df[rep_cols], errors='coerce').mean(axis=1)
    df = df.rename(columns={'ARC':'strain'}).dropna(subset=['strain','w_exp'])
    df['_key'] = df['strain'].map(_norm)
    return df[['strain','w_exp','_key']]

def load_all_fits(sheet, tab="params_fits", min_r2=0.9):
    ws = sheet.worksheet(tab)
    df = (get_as_dataframe(ws, header=0, evaluate_formulas=False)
          .dropna(how='all').dropna(axis=1, how='all'))
    if df.empty:
        raise ValueError(f"No rows in '{tab}' tab.")

    # Ensure correct types
    df = df.astype({'family':'string','strain':'string'})

    # Filter by R² if present
    if 'R2' in df.columns:
        df = df[df['R2'] >= min_r2]

    param_map = {}
    for _, r in df.iterrows():
        key = (str(r['family']), str(r['strain']))
        param_map[key] = {
            'Vmax': float(r['Vmax']),
            'K':          float(r['K']),
            'c':          float(r['c']),
            'N0':         float(r.get('N0_cells', 2e8)),  # fallback
            'R2':         float(r['R2'])
        }
    return df, param_map



def get_params(fits_df, family, strain):
    if family:  # only filter by family if provided
        row = fits_df[(fits_df['family'] == family) & (fits_df['strain'] == strain)]
    else:  # no family specified → only filter by strain
        row = fits_df[fits_df['strain'] == strain]

    if row.empty:
        raise ValueError(f"No parameters found for {strain} ({family if family else 'any family'})")

    return {
        'Vmax': row['Vmax'].iloc[0],
        'K':          row['K'].iloc[0],
        'c':          row['c'].iloc[0],
    }



# --- normalizer reused everywhere ---
def _norm(s):
    return re.sub(r'[^a-z0-9]+', '', str(s).lower())

# --- load experimental fitness (mean + std) from a given tab ---
def load_exp_fitness(sheet, tab='fitness_aerobiosis'):
    ws = sheet.worksheet(tab)
    df = (get_as_dataframe(ws, header=0, evaluate_formulas=True)
          .dropna(how='all').dropna(axis=1, how='all'))
    rep_cols = [c for c in df.columns if str(c).strip().isdigit()]

    # mean
    if 'mean' in df.columns:
        df['w_exp'] = pd.to_numeric(df['mean'], errors='coerce')
    else:
        df['w_exp'] = pd.to_numeric(df[rep_cols], errors='coerce').mean(axis=1)

    # std (desvest or compute from reps)
    if 'desvest' in df.columns:
        df['w_exp_std'] = pd.to_numeric(df['desvest'], errors='coerce')
    else:
        df['w_exp_std'] = pd.to_numeric(df[rep_cols], errors='coerce').std(axis=1, ddof=1)

    df = df.rename(columns={'ARC':'strain'}).dropna(subset=['strain','w_exp'])
    df['_key'] = df['strain'].map(_norm)
    return df[['strain','w_exp','w_exp_std','_key']]

def load_all_fits(sheet, tab="params_fits", min_r2=0.9):
    ws = sheet.worksheet(tab)
    df = (get_as_dataframe(ws, header=0, evaluate_formulas=False)
          .dropna(how='all').dropna(axis=1, how='all'))
    if df.empty:
        raise ValueError(f"No rows in '{tab}' tab.")

    df = df.astype({'family':'string','strain':'string'})

    # <<< make sure R2 is numeric
    if 'R2' in df.columns:
        df['R2'] = pd.to_numeric(df['R2'], errors='coerce')
        df = df[df['R2'] >= float(min_r2)]

    param_map = {}
    for _, r in df.iterrows():
        key = (str(r['family']), str(r['strain']))
        param_map[key] = {
            'Vmax': float(r['Vmax']),
            'K':          float(r['K']),
            'c':          float(r['c']),
            'N0':         float(r.get('N0_cells', 2e8)),
            'R2':         float(r.get('R2', float('nan')))
        }
    return df, param_map

def attach_family_to_exp(exp_df, fits_df, verbose=True, preview_n=10):
    import re
    def _norm(s): return re.sub(r'[^a-z0-9]+', '', str(s).lower())

    exp = exp_df.copy()
    fits = fits_df[['family','strain']].dropna(subset=['strain','family']).copy()

    # Dedup fits by exact strain first
    fits_dedup = fits.drop_duplicates(subset=['strain'], keep='first')

    if verbose:
        print(f"[attach] exp rows: {len(exp)}, fits rows: {len(fits)}, fits unique strains: {fits_dedup['strain'].nunique()}")

    # 1) exact merge
    merged = exp.merge(fits_dedup, on='strain', how='left', suffixes=('', '_fit_exact'))

    # 2) rescue with normalized key, but build a clean key->family map
    needs_rescue = merged['family'].isna()
    if needs_rescue.any():
        fits_key = fits.copy()
        fits_key['_key'] = fits_key['strain'].map(_norm)
        fits_key = fits_key.dropna(subset=['_key'])
        # keep first occurrence per normalized key to avoid many-to-many
        fits_key = fits_key.drop_duplicates(subset=['_key'], keep='first')

        key2fam = dict(zip(fits_key['_key'], fits_key['family']))
        merged['_key'] = merged['strain'].map(_norm)
        merged.loc[needs_rescue, 'family'] = merged.loc[needs_rescue, '_key'].map(key2fam)


    return merged.drop(columns=['_key'], errors='ignore')



# --- aggregate by family: simple mean/std across ARCs belonging to that family ---
def family_fitness_stats(exp_with_family_df, family):
    sub = exp_with_family_df[exp_with_family_df['family'] == family].dropna(subset=['w_exp'])
    if sub.empty:
        raise ValueError(f"No experimental fitness rows found for family '{family}'.")
    # simple across-ARC stats
    mean_w = float(sub['w_exp'].mean())
    std_w  = float(sub['w_exp'].std(ddof=1)) if len(sub) > 1 else float('nan')
    n      = int(len(sub))
    # optional: weighted mean using ARC-level std (if you prefer)
    if sub['w_exp_std'].notna().all() and (sub['w_exp_std'] > 0).any():
        w = 1.0 / (sub['w_exp_std']**2).replace([pd.NA, 0], pd.NA)
        if w.notna().any():
            w_mean = float((w * sub['w_exp']).sum() / w.sum())
        else:
            w_mean = mean_w
    else:
        w_mean = mean_w
    return {
        'family': family,
        'n_ARCs': n,
        'mean_w_exp': mean_w,
        'std_w_exp': std_w,
        'weighted_mean_w_exp': w_mean,
        'strains': (sub['strain'].unique().tolist())
    }

def get_family_w_exp(sheet, family, tab='fitness_aerobiosis', min_r2=0.9):
    exp = load_exp_fitness(sheet, tab=tab)
    fits_df, _ = load_all_fits(sheet, tab="params_fits", min_r2=min_r2)
    exp_wfam = attach_family_to_exp(exp, fits_df)
    return family_fitness_stats(exp_wfam, family)


def get_family_strains_and_stats(exp_wfam, family):
    sub = exp_wfam[exp_wfam['family'] == family].dropna(subset=['w_exp'])
    if sub.empty:
        raise ValueError(f"No experimental fitness data for family {family}")
    return {
        'family': family,
        'strains': sub['strain'].tolist(),
        'w_exp': sub['w_exp'].tolist(),
        'mean_w_exp': float(sub['w_exp'].mean()),
        'std_w_exp': float(sub['w_exp'].std(ddof=1)) if len(sub) > 1 else float('nan'),
        'n': len(sub)
    }

def get_family_exp_stats(family, fits_df, fitness_exp):
    """
    Given a family, the fits_df (with family/strain mapping),
    and the experimental fitness dataframe (with w_exp, w_exp_std),
    return ARC-level values and family-level summary.
    """
    # 1) Get all ARCs (strains) belonging to this family
    strains = fits_df.loc[fits_df['family'] == family, 'strain'].unique()

    rows = []
    for s in strains:
        val = fitness_exp.loc[fitness_exp['strain'] == s, ['w_exp', 'w_exp_std']]
        if not val.empty:
            mu_w, sigma_w = val.iloc[0,0], val.iloc[0,1]
            rows.append({'strain': s, 'w_exp': mu_w, 'w_exp_std': sigma_w})

    df_family = pd.DataFrame(rows)

    # 2) Family-level mean & std across ARCs
    mean_w = df_family['w_exp'].mean() if not df_family.empty else np.nan
    std_w  = df_family['w_exp'].std(ddof=1) if len(df_family) > 1 else np.nan

    return df_family, mean_w, std_w

def load_paired_family_exp(sheet, family,
                           tab_E='fitness_aerobiosis',
                           tab_G='fitness_anaerobiosis',
                           min_r2=0.9):
    # Load experimental fitness for E and G
    exp_E = load_exp_fitness(sheet, tab=tab_E)      # columns: strain, w_exp, w_exp_std, _key
    exp_G = load_exp_fitness(sheet, tab=tab_G)      # same columns
    fits_df, _ = load_all_fits(sheet, tab="params_fits", min_r2=min_r2)

    # attach family to both
    exp_E_fam = attach_family_to_exp(exp_E, fits_df)
    exp_G_fam = attach_family_to_exp(exp_G, fits_df)

    # filter family only if a specific family is requested
    if family != 'all':
        exp_E_fam = exp_E_fam[exp_E_fam['family'] == family]
        exp_G_fam = exp_G_fam[exp_G_fam['family'] == family]

    # merge on strain to get paired (E,G) per ARC
    paired = pd.merge(
        exp_E_fam[['strain','w_exp','w_exp_std']].rename(columns={'w_exp':'w_E','w_exp_std':'w_E_std'}),
        exp_G_fam[['strain','w_exp','w_exp_std']].rename(columns={'w_exp':'w_G','w_exp_std':'w_G_std'}),
        on='strain', how='inner'
    ).dropna(subset=['w_E','w_G'])

    return paired  # columns: strain, w_E, w_E_std, w_G, w_G_std




def estimate_family_bivariate_params(paired_df, shrink_r=0.05, ridge=1e-6):
    """
    Return mean vector, stds, correlation, and covariance matrix.
    shrink_r: convex combination with 0 to stabilize small-n correlations.
    ridge: diagonal jitter for numerical stability.
    """
    if paired_df.empty or paired_df['w_E'].count() < 3:
        # Not enough paired data: signal to fallback
        return None

    wE = paired_df['w_E'].astype(float).values
    wG = paired_df['w_G'].astype(float).values

    mu_E = float(np.mean(wE))
    mu_G = float(np.mean(wG))
    sd_E = float(np.std(wE, ddof=1)) if len(wE) > 1 else 0.0
    sd_G = float(np.std(wG, ddof=1)) if len(wG) > 1 else 0.0

    if sd_E == 0.0 or sd_G == 0.0:
        rho_hat = 0.0
    else:
        rho_hat = float(np.corrcoef(wE, wG)[0,1])

    # light shrinkage of correlation toward 0 to avoid overfitting with small n
    rho = (1.0 - shrink_r) * rho_hat

    cov = np.array([[sd_E**2, rho*sd_E*sd_G],
                    [rho*sd_E*sd_G, sd_G**2]], dtype=float)
    cov.flat[::3] += ridge  # add ridge to diagonal

    return {'mu': np.array([mu_E, mu_G], dtype=float),
            'sd': (sd_E, sd_G),
            'rho': rho,
            'cov': cov}

def sample_bivariate_w(n, mu, cov, clip_min=None, clip_max=None, seed=None):
    rng = np.random.default_rng(seed)
    samples = rng.multivariate_normal(mean=mu, cov=cov, size=n)
    if clip_min is not None or clip_max is not None:
        if clip_min is None: clip_min = -np.inf
        if clip_max is None: clip_max =  np.inf
        samples = np.clip(samples, clip_min, clip_max)
    # returns array shape (n,2): columns [w_E, w_G]
    return samples

# ---- Integration point ----
def build_synthetic_targets_for_family(sheet, family, n=1000,
                                       tab_E='fitness_aerobiosis',
                                       tab_G='fitness_anaerobiosis',
                                       min_r2=0.9,
                                       clip_range=(0.05, 3.0),
                                       seed=None):
    """
    Returns DataFrame with n target pairs (w_E, w_G) for given family,
    using bivariate Normal if paired data available; otherwise independent marginals.
    """
    paired = load_paired_family_exp(sheet, family, tab_E=tab_E, tab_G=tab_G, min_r2=min_r2)
    stats = estimate_family_bivariate_params(paired)

    if stats is not None:
        samp = sample_bivariate_w(n, stats['mu'], stats['cov'],
                                  clip_min=clip_range[0], clip_max=clip_range[1],
                                  seed=seed)
        out = pd.DataFrame(samp, columns=['w_E_target','w_G_target'])
        out['family'] = family
        out['method'] = 'bivariate'
        out['rho_used'] = stats['rho']
        return out

    # Fallback: independent marginals (your current approach)
    # Reuse your existing get_family_w_exp on each environment:
    exp_E = load_exp_fitness(sheet, tab=tab_E)
    exp_G = load_exp_fitness(sheet, tab=tab_G)
    fits_df, _ = load_all_fits(sheet, tab="params_fits", min_r2=min_r2)
    exp_E_fam = attach_family_to_exp(exp_E, fits_df)
    exp_G_fam = attach_family_to_exp(exp_G, fits_df)
    subE = exp_E_fam[exp_E_fam['family'] == family]['w_exp'].dropna().values
    subG = exp_G_fam[exp_G_fam['family'] == family]['w_exp'].dropna().values
    muE, sdE = float(np.mean(subE)), float(np.std(subE, ddof=1)) if len(subE)>1 else 0.0
    muG, sdG = float(np.mean(subG)), float(np.std(subG, ddof=1)) if len(subG)>1 else 0.0

    rng = np.random.default_rng(seed)
    wE = rng.normal(muE, sdE, size=n)
    wG = rng.normal(muG, sdG, size=n)
    if clip_range is not None:
        wE = np.clip(wE, clip_range[0], clip_range[1])
        wG = np.clip(wG, clip_range[0], clip_range[1])

    out = pd.DataFrame({'w_E_target': wE, 'w_G_target': wG,
                        'family': family, 'method': 'marginals', 'rho_used': 0.0})
    return out




# -------- Loader --------
def load_stability_aggregates(sheet, tab='stability_aggregates'):
    """Load aggregates from Google Sheet and coerce types."""
    ws = sheet.worksheet(tab)
    df = (get_as_dataframe(ws, header=0, evaluate_formulas=True)
          .dropna(how='all').dropna(axis=1, how='all'))
    if df.empty:
        raise ValueError(f"No data in tab '{tab}'.")

    # Coerce expected numeric columns if present
    num_cols = ['num_days','n_pairs','beta_per_day_mean','beta_per_day_sd',
                't_half_days_mean','t_half_days_sd','R2_mean','P30','P7']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Clean strings
    for c in ['family','schedule_name','schedule_str']:
        if c in df.columns:
            df[c] = df[c].astype(str)

    return df

# ---------- helpers ----------
def _read_tab(sheet, title):
    """Read a tab to DataFrame, drop fully-empty rows/cols."""
    ws = sheet.worksheet(title)
    df = get_as_dataframe(ws, header=0, evaluate_formulas=True)
    return df.dropna(how='all').dropna(axis=1, how='all')

def _coerce_numeric(df, cols, pct_cols=()):
    """
    Coerce listed columns to numeric. For pct_cols, if any values >1,
    assume percentages (0–100) and convert to 0–1.
    """
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in pct_cols:
        if c in df.columns:
            # convert 0–100 → 0–1 when needed
            vals = pd.to_numeric(df[c], errors='coerce')
            if np.nanmax(vals) is not None and np.nanmax(vals) > 1.5:
                df[c] = vals / 100.0
            else:
                df[c] = vals
    return df

def _ensure_cols(df, need, fill=np.nan):
    """Ensure required columns exist."""
    for c in need:
        if c not in df.columns:
            df[c] = fill
    return df

def load_stability_pairs(
    sheet,
    tab='stability_pairs',
):
    """
    Returns a cleaned pair_df (per-pair/per-run stats).
    Expected columns (some optional):
      family, schedule_name, schedule_str, pair_idx,
      beta_per_day, t_half_days, R2, n_points,
      outcome  (if you stored it; otherwise left NaN)
    """
    df = _read_tab(sheet, tab)

    needed = [
        'family','schedule_name','schedule_str','pair_idx',
        'f_end','persist_end','time','event','outcome'
    ]
    df = _ensure_cols(df, needed)

    df['family'] = df['family'].astype('string')
    df['schedule_name'] = df['schedule_name'].astype('string')
    df['schedule_str'] = df['schedule_str'].astype('string')

    df = _coerce_numeric(
        df,
        cols=['pair_idx','n_points']
    )

    df = df.sort_values(['family','schedule_name','pair_idx'], ignore_index=True)
    return df


# --- from df_pairs -> agg_df with fraction columns ---
def make_agg_from_pairs(df_pairs):
    """
    Build an aggregate table with one row per (family, schedule_str)
    and columns frac_cleared, frac_stable, frac_fixated that sum to ~1.
    Missing categories are filled with 0.
    """
    # counts per outcome
    counts = (
        df_pairs
        .groupby(['family','schedule_name', 'schedule_str', 'outcome'])
        .size()
        .rename('n')
        .reset_index()
    )

    # pivot to columns and fill missing outcomes with 0
    wide = (
        counts
        .pivot_table(index=['family','schedule_name','schedule_str'],
                     columns='outcome', values='n', fill_value=0)
        .reset_index()
    )

    # ensure all three columns exist
    for col in ['cleared','stable','fixated']:
        if col not in wide.columns:
            wide[col] = 0

    # totals and fractions
    wide['total'] = wide[['cleared','stable','fixated']].sum(axis=1).clip(lower=1)
    wide['frac_cleared']    = wide['cleared']    / wide['total']
    wide['frac_stable']  = wide['stable']  / wide['total']
    wide['frac_fixated'] = wide['fixated'] / wide['total']

    # final schema expected by plot_stability_stacks
    agg_df = wide[['family','schedule_name','schedule_str','frac_cleared','frac_stable','frac_fixated']].copy()
    return agg_df

