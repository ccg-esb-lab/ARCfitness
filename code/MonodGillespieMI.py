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
def upsert_to_sheet(sheet, tab_name, df_new, key_cols):
    """
    Upsert df_new into a Google Sheet tab by key columns.
    If the tab doesn't exist, it's created.
    """
    try:
        ws = sheet.worksheet(tab_name)
        df_old = (get_as_dataframe(ws, header=0, evaluate_formulas=True)
                  .dropna(how="all"))
    except Exception:
        ws = sheet.add_worksheet(title=tab_name, rows=2000, cols=40)
        df_old = pd.DataFrame()

    if df_old.empty:
        df_combined = df_new.copy()
    else:
        # concat then drop duplicates by key to simulate an upsert
        df_combined = (pd.concat([df_old, df_new], ignore_index=True, sort=False)
                       .drop_duplicates(subset=key_cols, keep='last'))

    ws.clear()
    set_with_dataframe(ws, df_combined, include_index=False, include_column_header=True)
    return len(df_new), len(df_combined)


def append_to_sheet(sheet, tab_name, df):
    """Append dataframe df to the end of a Google Sheet tab."""
    ws = sheet.worksheet(tab_name)
    existing_rows = len(ws.get_all_values())
    start_row = existing_rows + 1
    values = [df.columns.tolist()] + df.values.tolist() if existing_rows == 0 else df.values.tolist()
    ws.insert_rows(values, row=start_row)
    print(f"Appended {len(df)} rows to '{tab_name}' starting at row {start_row}.")


def gillespie_bacterial_growth_batch(strains, initial_resource, simulation_time, dt=0.1, num_simulations=3):
    """
    Simulates bacterial growth with multiple mutant levels, resource limitation, and antibiotic-induced death
    using a batch update approach. Runs multiple simulations and returns the mean population size.

    Parameters:
    - strains (list of dict): List of strain dictionaries with parameters:
        - 'initial_population' (float): Initial population size.
        - 'Vmax' (float): Birth rate.
        - 'death_rate' (float): Base death rate.
        - 'K' (float): Half-saturation constant for resource consumption.
        - 'A_half' (float): Half-saturation constant for antibiotic concentration.
        - 'c' (float): Resource consumption rate per birth event.
    - initial_resource (float): Initial resource level.
    - simulation_time (float): Maximum simulation time.
    - antibiotic_concentration (float): Antibiotic concentration affecting death rate.
    - dt (float): Time step for batch updates.
    - num_simulations (int): Number of independent simulations to average.

    Returns:
    - t_values (list): Time points where changes occurred.
    - mean_population_values (list of lists): Mean population sizes of each strain at corresponding time points.
    - mean_R_values (list): Mean resource levels at corresponding time points.
    """

    all_t_values = []
    all_population_values = []
    all_R_values = []

    for sim in range(num_simulations):
        # Initialize time and populations
        t = 0
        populations = [strain['initial_population'] for strain in strains]
        R = initial_resource

        # Record initial conditions
        t_values = [t]
        population_values = [populations[:]]
        R_values = [R]

        while t < simulation_time:
            # Check for extinction or resource depletion
            if sum(populations) <= 0 or R <= 0:
                t += dt
                t_values.append(t)
                population_values.append(populations[:])
                R_values.append(R)
                continue

            # Calculate birth and death events for each strain
            births = [
                np.random.poisson(strain['Vmax'] * populations[i] * (R / (R + strain['K'])) * dt)
                for i, strain in enumerate(strains)
            ]

            #deaths = [
            #    np.random.poisson(strain['death_rate'] * populations[i] * (antibiotic_concentration /
            #            (antibiotic_concentration + strain['A_half'])) * dt)
            #    for i, strain in enumerate(strains)
            #]

            # Update populations and resources after birth and death events
            for i in range(len(populations)):
                populations[i] += births[i] #- deaths[i]

            # Decrease resource based on total births across strains
            total_resource_consumption = sum(births[i] * strains[i]['c'] for i in range(len(strains)))
            R -= total_resource_consumption

            # Ensure populations and resources are non-negative
            populations = [max(0, pop) for pop in populations]
            R = max(R, 0)

            # Advance time and record results
            t += dt
            t_values.append(t)
            population_values.append(populations[:])
            R_values.append(R)

        # Store results from this simulation
        all_t_values.append(t_values)
        all_population_values.append(population_values)
        all_R_values.append(R_values)

    # Convert lists to NumPy arrays for averaging
    all_t_values = np.array(all_t_values)
    all_population_values = np.array(all_population_values)
    all_R_values = np.array(all_R_values)

    # Compute mean results across simulations
    mean_t_values = np.mean(all_t_values, axis=0)
    mean_population_values = np.mean(all_population_values, axis=0)
    mean_R_values = np.mean(all_R_values, axis=0)

    return mean_t_values, mean_population_values, mean_R_values




def plotSimulation(time_points, population_values, resource, pathFIGURES='',title=''):
    """
    Plots the results of the simulation for all strains and resource depletion using a blues colormap.
    The wild-type strain is plotted in black, and the colormap for mutants skips the first (white) color.

    Parameters:
    time_points (list): Time points from the simulation.
    population_values (list of list): Population sizes of each strain at corresponding time points.
    resource (list): Resource levels over time.
    """
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), constrained_layout=True)

    # Get the number of strains (only one type of population without plasmid-free/plasmid-bearing distinction)
    num_strains = len(population_values[0])

    # Define a colormap (Blues)
    cmap = plt.get_cmap('Blues')

    # Plot populations: wild-type in black, mutants in shades of blue
    for i in range(num_strains):
        linestyle = '-' if i == 0 else '--'  # Wild-type solid, mutants dashed
        color = 'black' if i == 0 else cmap(i / num_strains)
        axes[0].plot(time_points, [(pop[i]) for pop in population_values], label=f'Strain {i}', color=color, linestyle=linestyle)

    axes[0].set_xlabel('Time (hours)', fontsize=14)
    axes[0].set_ylabel('Population', fontsize=14)
    axes[0].set_yscale('log')  # Apply log scale to the population plot
    axes[0].tick_params(axis='both', labelsize=12)

    # Plot resource depletion
    axes[1].plot(time_points, resource, color='orange', label='Resource')
    axes[1].set_xlabel('Time (hours)', fontsize=14)
    axes[1].set_ylabel('Resource concentration', fontsize=14)
    axes[1].tick_params(axis='both', labelsize=12)
    axes[1].set_ylim([0, 1.1])


    if pathFIGURES != '':
            filename = f"{pathFIGURES}simulation_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()


    # --- Build two-strain inputs from raw parameter dicts ---
def make_two_strain_params_from_dicts(pA, pB, *,
                                      freqA0=0.5, Ntot0=2e8,
                                      death_rate=0.0, A_half=0.5):
    """
    pA, pB: dicts with {'Vmax','K','c'} (N0 is ignored here).
    Returns the 'strains' list ready for the simulator.
    """
    NA0 = float(Ntot0) * float(freqA0)
    NB0 = float(Ntot0) - NA0
    strains = [
        {'Vmax': float(pA['Vmax']),
         'death_rate': float(death_rate),
         'initial_population': float(NA0),
         'K': float(pA['K']),
         'c': float(pA['c']),
         'A_half': float(A_half)},
        {'Vmax': float(pB['Vmax']),
         'death_rate': float(death_rate),
         'initial_population': float(NB0),
         'K': float(pB['K']),
         'c': float(pB['c']),
         'A_half': float(A_half)},
    ]
    return strains

# --- Single competition runner  ---
def run_competition_once(sim_fn, strains, initial_resource, simulation_time):
    out = sim_fn(strains=strains,
                 initial_resource=initial_resource,
                 simulation_time=simulation_time)
    import numpy as np
    if len(out) == 2:
        t, pops = out; res = None
    else:
        t, pops, res = out
    return np.asarray(t, float), np.asarray(pops, float), res

# --- New unified function (dict-based), same overall structure ---
def run_pairwise_competition(sim_fn,
                             initial_resource, simulation_time,
                             *,
                             pA, pB,                  # <-- pass dicts instead of keys
                             freqA0=0.5, Ntot0=1e6, death_rate=0.0, A_half=0.5):
    """
    Runs a single A-vs-B competition using raw parameter dicts.

    Returns dict with:
      'pA','pB','freqA0','Ntot0','time','pops','resource','metrics'
    """
    strains = make_two_strain_params_from_dicts(
        pA, pB, freqA0=freqA0, Ntot0=Ntot0,
        death_rate=death_rate, A_half=A_half
    )
    t, pops, res = run_competition_once(sim_fn, strains, initial_resource, simulation_time)
    mets = competition_metrics(t, pops)
    return {
        'pA': pA, 'pB': pB,
        'freqA0': freqA0, 'Ntot0': Ntot0,
        'time': t, 'pops': pops, 'resource': res,
        'metrics': mets
    }

def competition_metrics(t, pops, early_hours=6.0):
    import numpy as np
    Na, Nb = pops[:,0], pops[:,1]
    Ntot = Na + Nb
    fA_end = Na[-1] / max(Ntot[-1], 1e-12)
    fB_end = Nb[-1] / max(Ntot[-1], 1e-12)
    with np.errstate(divide='ignore', invalid='ignore'):
        lnratio = np.log(np.maximum(Na,1e-12)/np.maximum(Nb,1e-12))
    mask = t <= min(early_hours, t[-1])
    s = np.polyfit(t[mask], lnratio[mask], 1)[0] if mask.sum() >= 2 else float('nan')
    return {'fA_end': fA_end, 'fB_end': fB_end, 's_early_per_hr': s}

def plot_competition(result, labels=None, colors=("C0","C1"), ylog=False, title=None, pathFIGURES=''):
    import matplotlib.pyplot as plt
    import numpy as np

    t = result['time']; pops = result['pops']

    # Try to get names from result (old path), else fall back
    famA = strainA = famB = strainB = None
    if 'keyA' in result and 'keyB' in result:
        try:
            (famA, strainA), (famB, strainB) = result['keyA'], result['keyB']
        except Exception:
            pass

    # Final labels
    if labels is None:
        if strainA is not None and strainB is not None:
            labels = (f"{strainA} ({famA})", f"{strainB} ({famB})")
        else:
            labels = ("Strain A", "Strain B")

    fig, axes = plt.subplots(1, 2, figsize=(10,4), sharey=False)

    # --- Left panel: population dynamics ---
    ax = axes[0]

    # Style: grey dotted if name contains 'pMBA'; otherwise colored solid
    def _style_for(name, color):
        if name and ("pmba" in name.lower()):
            return dict(color="0.5", ls=":", lw=2)
        return dict(color=color, ls="-", lw=2)

    styleA = _style_for(labels[0], colors[0])
    styleB = _style_for(labels[1], colors[1])

    ax.plot(t, pops[:,0], label=labels[0], **styleA)
    ax.plot(t, pops[:,1], label=labels[1], **styleB)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Cells/mL")
    if ylog:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

    # --- Right panel: log ratio ---
    ax2 = axes[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        lnratio = np.log(np.maximum(pops[:,0],1e-12)/np.maximum(pops[:,1],1e-12))
    ax2.plot(t, lnratio, '-', color='k', lw=1.8)
    ax2.axhline(0, color='0.7', lw=1)
    ax2.set_xlabel("Time (hours)")
    ax2.set_ylabel("ln(N_A / N_B)")
    ax2.grid(True, alpha=0.3)

    if title:
        fig.suptitle(title)

    plt.tight_layout()

    if pathFIGURES != '':
            filename = f"{pathFIGURES}competition_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return axes

def params_from_row(row):
    return {
        'Vmax': float(row['Vmax']),
        'K':          float(row['K']),
        'c':          float(row['c'])
    }

def _run_one_day(sim_fn, pA, pB, NA_init, NB_init, initial_resource, hours):
    strains = [
        {'Vmax': pA['Vmax'], 'death_rate': 0.0, 'initial_population': NA_init,
         'K': pA['K'], 'c': pA['c'], 'A_half': 0.5},
        {'Vmax': pB['Vmax'], 'death_rate': 0.0, 'initial_population': NB_init,
         'K': pB['K'], 'c': pB['c'], 'A_half': 0.5},
    ]
    out = sim_fn(
        strains=strains,
        initial_resource=initial_resource,
        simulation_time=hours
    )
    if len(out) == 2:
        t, pops = out; res = None
    else:
        t, pops, res = out
    return np.asarray(t, float), np.asarray(pops, float), res

def serial_transfer_competition(
    env_schedule,
    row_G, row_E,
    p0,
    sim_fn, initial_resource,
    day_hours=24.0,
    dilution=100.0,
    Ntot0_start=1e6,
    freqA0_start=0.5,
    p0_G=None, p0_E=None,
    N_extinct=1.0,
    fmin_extinct=None,          # <-- NEW: frequency-based extinction rule
    stochastic_bottleneck=True,
    rng=None
):
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(rng)

    Na0 = float(Ntot0_start * freqA0_start)
    Nb0 = float(Ntot0_start - Na0)

    all_t, all_pops, segments, day_rows = [], [], [], []
    t_offset = 0.0

    for day_idx, env in enumerate(env_schedule, start=1):
        env = str(env).upper().strip()
        if env not in {'G','E'}:
            raise ValueError(f"Environment must be 'G' or 'E', got {env}")

        pA = params_from_row(row_G) if env == 'G' else params_from_row(row_E)
        pB = (p0_G if (env == 'G' and p0_G is not None)
              else p0_E if (env == 'E' and p0_E is not None)
              else p0)

        # Simulate one day
        t_day, pops_day, _ = _run_one_day(sim_fn, pA, pB, Na0, Nb0, initial_resource, day_hours)

        # absolute-time segment
        t_abs = t_day + t_offset
        t_offset += day_hours

        # End-of-day sizes
        Na_end, Nb_end = float(pops_day[-1, 0]), float(pops_day[-1, 1])

        # --- Extinction rule 1: absolute cells ---
        extinctA = (Na_end < N_extinct)
        extinctB = (Nb_end < N_extinct)

        if extinctA: Na_end = 0.0
        if extinctB: Nb_end = 0.0

        # --- Optional extinction rule 2: frequency threshold ---
        if fmin_extinct is not None:
            denom = Na_end + Nb_end
            fA_tmp = (Na_end / denom) if denom > 0 else 0.0
            fB_tmp = (Nb_end / denom) if denom > 0 else 0.0

            if fA_tmp < float(fmin_extinct):
                extinctA = True
                Na_end = 0.0
            if fB_tmp < float(fmin_extinct):
                extinctB = True
                Nb_end = 0.0

        # Now compute summaries from the FORCED end counts
        Na_start, Nb_start = float(Na0), float(Nb0)
        Ntot_start = max(Na_start + Nb_start, 1e-12)
        fA_start, fB_start = Na_start / Ntot_start, Nb_start / Ntot_start

        denom_end = Na_end + Nb_end
        if denom_end <= 0:
            fA_end, fB_end = 0.0, 0.0
        else:
            fA_end, fB_end = Na_end / denom_end, Nb_end / denom_end

        # IMPORTANT: force the stored trajectory to match extinction at day end
        pops_day = pops_day.copy()
        pops_day[-1, 0] = Na_end
        pops_day[-1, 1] = Nb_end

        # store
        all_t.append(t_abs)
        all_pops.append(pops_day)
        segments.append((t_abs, pops_day))

        day_rows.append({
            'day': day_idx, 'env': env,
            'Na_start': Na_start, 'Nb_start': Nb_start,
            'Na_end': Na_end,     'Nb_end': Nb_end,
            'fA_start': fA_start, 'fB_start': fB_start,
            'fA_end': fA_end,     'fB_end': fB_end,
            'extinctA': bool(extinctA), 'extinctB': bool(extinctB)
        })

        # dilution
        Na0 = Na_end / float(dilution)
        Nb0 = Nb_end / float(dilution)

        # Once extinct, keep at zero
        if Na0 < N_extinct: Na0 = 0.0
        if Nb0 < N_extinct: Nb0 = 0.0

    time_all = np.concatenate(all_t) if all_t else np.array([])
    pops_all = np.vstack(all_pops)   if all_pops else np.zeros((0,2))
    day_summaries = pd.DataFrame(day_rows)

    return {'time': time_all, 'pops': pops_all,
            'day_summaries': day_summaries, 'segments': segments}



def _draw_env_band(ax, segments, day_summaries,
                   x_factor=1.0,                    # e.g. 1/24.0 for days
                   colors={'G': '#EBDD99', 'E': '#B63E36'},
                   alpha=0.12, ypad=0.0):
    """
    Draw environment background spans per day.
    segments: list of (t_abs, env_label) with times in HOURS.
    day_summaries: DataFrame with an 'env' column.
    x_factor: multiply times by this factor for the plot axis.
    """
    import numpy as np

    ymin, ymax = ax.get_ylim()
    ymin -= ypad; ymax += ypad

    for (t_abs, _), (_, row) in zip(segments, day_summaries.iterrows()):
        if t_abs is None or len(t_abs) == 0:
            continue
        x0 = float(t_abs[0]) * x_factor
        x1 = float(t_abs[-1]) * x_factor
        env = str(row.get('env', '')).strip()
        col = colors.get(env, '#BBBBBB')
        ax.axvspan(x0, x1, ymin=0, ymax=1, facecolor=col, alpha=alpha, edgecolor='none')

    ax.set_ylim(ymin, ymax)


# === 1) Full trajectory with env band ===
def plot_trajectories_single(res, labelA="control pMBA", labelB="BX",
                                    colorA="0.25", colorB="#440154", lsA="--", lsB="-",
                                    title=None,pathFIGURES=''):
    """
    Plot continuous trajectories (Na, Nb) across all days and draw a top band
    that encodes the environment per day (E=blue, G=orange).
    """
    t = res['time']
    pops = res['pops']
    Na, Nb = pops[:,0], pops[:,1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, Na, label=labelA, color=colorA, ls=lsA, lw=1)
    ax.plot(t, Nb, label=labelB, color=colorB, ls=lsB, lw=2)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Cells/mL')
    ax.set_xlim(0, t[-1])
    ax.set_ylim(0, 1.1*max(Na.max(), Nb.max()))
    ax.xaxis.set_major_locator(MultipleLocator(24))
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.)
    ax.legend(frameon=False, fontsize=14, loc='upper left')

    # Environment band
    _draw_env_band(ax, res['segments'], res['day_summaries'])

    plt.tight_layout()

    if pathFIGURES != '':
            filename = f"{pathFIGURES}trajectories_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return ax


def plot_final_frequencies_per_day_single(
    res,
    labelA=r"pMBA$_\emptyset$", labelB="ARC",
    colorA="0.25", colorB="#440154", pathFIGURES='',
    title=""
):
    """
    Plot frequencies (proportions) at the end of each day, with day 0 prepended
    from the start of day 1. Includes environment band aligned to time.
    Expects 'res' from serial_transfer_competition.
    """
    df = res['day_summaries'].copy()
    if df.empty:
        raise ValueError("day_summaries is empty in 'res'.")

    # Build plotting table with day 0 from starts of day 1
    day0 = {
        'day': 0,
        'env': df.iloc[0]['env'],
        'fA_end': df.iloc[0]['fA_start'],
        'fB_end': df.iloc[0]['fB_start']
    }
    df_plot = pd.concat([pd.DataFrame([day0]), df[['day','env','fA_end','fB_end']]], ignore_index=True)

    # X positions = absolute times: day 0 at 0; each day at its end time
    x_points = [0.0]
    for (t_abs, _), (_, _) in zip(res['segments'], res['day_summaries'].iterrows()):
        x_points.append(float(t_abs[-1]/ 24.0))

    fA = 100*df_plot['fA_end'].to_numpy(float)
    fB = 100*df_plot['fB_end'].to_numpy(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x_points, fA, 'o--', color=colorA, lw=1, label=labelA)
    ax.plot(x_points, fB, 'o-', color=colorB, lw=2, label=labelB)

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Frequency (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0, x_points[-1])
    ax.grid(True, alpha=0.)
    ax.axhline(50, color='0.5', lw=1, ls=':')
    if title:
        ax.set_title(title)
    ax.legend(frameon=False, fontsize=14, loc='lower left')

    # Top environment band
    _draw_env_band(ax, res['segments'], res['day_summaries'], x_factor=1/24.0)


    plt.tight_layout()

    if pathFIGURES != '':
            filename = f"{pathFIGURES}final_freq_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return ax




# --- 0) Slice day endpoints from a run ---------------------------------------
def _day_grid_and_freq(res, which='B'):
    """
    From a `serial_transfer_competition` result dict, return:
      t_days  : [0, day1_end, day2_end, ...] in DAYS
      f_series: matching frequencies for A or B at those times
    """
    # absolute end-of-day times (hours) -> to days
    x = [0.0]
    for (t_abs, _), _row in zip(res['segments'], res['day_summaries'].iterrows()):
        x.append(float(t_abs[-1]) / 24.0)
    x = np.asarray(x, float)

    df = res['day_summaries']
    fA0 = float(df.iloc[0]['fA_start'])
    fB0 = float(df.iloc[0]['fB_start'])
    fA  = np.concatenate([[fA0], df['fA_end'].to_numpy(float)])
    fB  = np.concatenate([[fB0], df['fB_end'].to_numpy(float)])

    return x, (fA if which.upper()=='A' else fB)



def run_serial_transfers(
    df_env, pair_indices, schedule, p0,
    sim_fn, initial_resource,
    day_hours=24.0, dilution=100.0,
    Ntot0_start=1e6, freqA0_start=0.5,
):
    """Run serial-transfer for each pair_idx; return list of `res` dicts."""
    results = []
    for pair_idx in pair_indices:
        row_G, row_E = _get_rows_for_pair(df_env, pair_idx)
        res = serial_transfer_competition(
            env_schedule=schedule,
            row_G=row_G, row_E=row_E,
            p0=p0,
            sim_fn=sim_fn,
            initial_resource=initial_resource,
            day_hours=day_hours,
            dilution=dilution,
            Ntot0_start=Ntot0_start,
            freqA0_start=freqA0_start,
        )
        results.append(res)
    return results


def run_replicates(
    R, schedule, row_G, row_E, p0,
    sim_fn, initial_resource,
    day_hours=24.0, dilution=100.0,
    Ntot0_start=1e6, freqA0_start=0.5,
):
    """Run the same pair R times (Gillespie stochasticity); return list of `res`."""
    out = []
    for _ in range(int(R)):
        res = serial_transfer_competition(
            env_schedule=schedule,
            row_G=row_G, row_E=row_E,
            p0=p0,
            sim_fn=sim_fn,
            initial_resource=initial_resource,
            day_hours=day_hours,
            dilution=dilution,
            Ntot0_start=Ntot0_start,
            freqA0_start=freqA0_start,
        )
        out.append(res)
    return out


# --- 0) Slice day endpoints from a run ---------------------------------------
def _day_grid_and_freq(res, which='B'):
    """
    From a `serial_transfer_competition` result dict, return:
      t_days  : [0, day1_end, day2_end, ...] in DAYS
      f_series: matching frequencies for A or B at those times
    """
    # absolute end-of-day times (hours) -> to days
    x = [0.0]
    for (t_abs, _), _row in zip(res['segments'], res['day_summaries'].iterrows()):
        x.append(float(t_abs[-1]) / 24.0)
    x = np.asarray(x, float)

    df = res['day_summaries']
    fA0 = float(df.iloc[0]['fA_start'])
    fB0 = float(df.iloc[0]['fB_start'])
    fA  = np.concatenate([[fA0], df['fA_end'].to_numpy(float)])
    fB  = np.concatenate([[fB0], df['fB_end'].to_numpy(float)])

    return x, (fA if which.upper()=='A' else fB)

def run_trajectories(
    df_env, pair_indices, schedule, p0,
    sim_fn, initial_resource,
    day_hours=24.0, dilution=100.0, Ntot0_start=1e6, freqA0_start=0.5,
):
    """
    Runs serial-transfer simulations for each pair_idx and returns a list of result dicts.
    Each result has: {'pair_idx','time','pops','segments','day_summaries'}.
    The first element can be used to draw the environment band.
    """
    results = []
    for pair_idx in pair_indices:
        row_G, row_E = _get_rows_for_pair(df_env, pair_idx)
        res = serial_transfer_competition(
            env_schedule=schedule,
            row_G=row_G, row_E=row_E,
            p0=p0,
            sim_fn=sim_fn,
            initial_resource=initial_resource,
            day_hours=day_hours,
            dilution=dilution,
            Ntot0_start=Ntot0_start,
            freqA0_start=freqA0_start
        )
        results.append({
            'pair_idx': pair_idx,
            'time': np.asarray(res['time'], float),
            'pops': np.asarray(res['pops'], float),   # columns: [Na, Nb]
            'segments': res['segments'],             # for env band
            'day_summaries': res['day_summaries'],   # for endpoints
        })
    return results




def plot_env_violin(
    df,
    *,
    family=None,                    # if provided, filters df['family']==family
    x="Environment",
    y="achieved_w",
    palette=None,                   # dict like {'G':'#EBDD99','E':'#B63E36'}
    ylim=(0.5, 1.5),
    baseline=1.0,
    title=None,
    show_inner="box",               # 'box', 'quartiles', or None
    figsize=(4,4),
    pathFIGURES=''
):
    """
    Violin plot of fitness by environment with consistent colors and baseline.
    """
    d = df.copy()
    if family is not None and "family" in d.columns:
        d = d[d["family"] == family]

    if palette is None:
        palette = {'G': '#EBDD99', 'E': '#B63E36'}

    fig, ax = plt.subplots(figsize=figsize)
    sns.violinplot(
        data=d,
        x=x,
        y=y,
        hue=x,                 # explicitly set hue to silence the FutureWarning
        palette=palette,
        legend=False,          # avoid duplicate legend
        inner=show_inner
    )
    ax.set_ylim(*ylim)
    if baseline is not None:
        ax.axhline(baseline, color="k", linestyle="--", lw=1)
    ax.set_ylabel("Relative fitness (w)")
    if title is None:
        title = f"Distribution of fitness effects ({family})" if family else "Distribution of fitness effects"
    ax.set_title(title)
    fig.tight_layout()

    if pathFIGURES != '':
            filename = f"{pathFIGURES}env_violin_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return fig, ax




def plot_trajectories_mean(
    res_list, *,
    labelA=r"pMBA$_\emptyset$", labelB_base="ARC",
    alpha_bg=0.5, figsize=(8.5,4.8),
    show_ci=False, ci_quantiles=(0.16, 0.84),
    this_color='#440154', n_grid=600,
    tick_every_hours=24*6,
    to_days=False, day_hours=24.0,
    title='',
    pathFIGURES=''
):
    """
    Overlay trajectories from replicate results (as returned by run_replicates or run_trajectories0).

    - Control (Na): ALL replicates drawn faint, dotted gray (no bold/labeled control).
    - Competitor (Nb): ALL replicates faint; ONLY the mean is bold & labeled.
    - Environment band: drawn from the first replicate.
    """
    fig, ax = plt.subplots(figsize=figsize)

    if not res_list:
        ax.text(0.5, 0.5, "No results to plot", ha='center', va='center')
        return fig, ax

    xscale = (1.0/day_hours) if to_days else 1.0

    # Collect time vectors and Nb curves for mean/CI
    t_list, Nb_list = [], []

    # Draw all replicates faint
    for r in res_list:
        t  = np.asarray(r['time'], float) * xscale
        pop = np.asarray(r['pops'], float)
        Na, Nb = pop[:,0], pop[:,1]

        # Control (faint dotted gray; no special first one)
        ax.plot(t, Na, color="0.25", ls="--", lw=1.0, alpha=0.12)

        # Competitor (faint colored)
        ax.plot(t, Nb, color=this_color, lw=1.2, alpha=alpha_bg)

        t_list.append(t)
        Nb_list.append(Nb)

    # Mean (and optional CI) of Nb on a common grid
    if Nb_list:
        tmin = max(np.min(t) for t in t_list)
        tmax = min(np.max(t) for t in t_list)
        t_grid = t_list[0] if tmax <= tmin else np.linspace(tmin, tmax, n_grid)

        Nb_interp = np.vstack([np.interp(t_grid, t, nb) for t, nb in zip(t_list, Nb_list)])
        Nb_mean = Nb_interp.mean(axis=0)

        # Bold mean line for Nb (the only labeled curve)
        ax.plot(t_grid, Nb_mean, color=this_color, lw=2.8, label=f"{labelB_base}")

        # Optional CI band
        if show_ci and Nb_interp.shape[0] >= 3:
            q_lo = np.quantile(Nb_interp, ci_quantiles[0], axis=0)
            q_hi = np.quantile(Nb_interp, ci_quantiles[1], axis=0)
            ax.fill_between(t_grid, q_lo, q_hi, color=this_color, alpha=0.18, label="CI")
        ax.set_xlim(0, t_grid[-1])

    # Cosmetics
    ax.set_xlabel('Time (days)' if to_days else 'Time (hours)', fontsize=16)
    ax.set_ylabel('Cells/mL', fontsize=16)
    ax.grid(True, alpha=0.)
    ax.legend(frameon=False, fontsize=14, loc='upper left')

    # Ticks (by hours or by days)
    if to_days:
        ax.xaxis.set_major_locator(MultipleLocator(tick_every_hours / day_hours))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(tick_every_hours))
    ax.tick_params(axis='both', labelsize=14)

    # Environment band from the first run (rescaled if needed)
    first = res_list[0]
    try:
        if to_days:
            for (t_abs, _), (_, row) in zip(first['segments'], first['day_summaries'].iterrows()):
                t0 = float(t_abs[0]) * xscale
                t1 = float(t_abs[-1]) * xscale
                env = str(row.get('env','G')).upper()
                col = "#B63E36" if env.startswith('E') else "#EBDD99"
                ax.axvspan(t0, t1, color=col, alpha=0.20, lw=0)
        else:
            _draw_env_band(ax, first['segments'], first['day_summaries'])
    except Exception:
        pass

    plt.tight_layout()

    if title != '':
        ax.set_title(title, fontsize=16)

    if pathFIGURES != '':
            filename = f"{pathFIGURES}trajectories_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return fig, ax


def _draw_env_band_simple(ax, segments, day_summaries, to_days=False, day_hours=24.0,
                          color_E="#B63E36", color_G="#EBDD99", alpha=0.20):
    """
    segments: list of tuples (t_abs_array, pops_array) per day
    day_summaries: DataFrame with column 'env' (values 'E' or 'G')
    """
    xscale = (1.0/day_hours) if to_days else 1.0
    for (t_abs, _), (_, row) in zip(segments, day_summaries.iterrows()):
        t0, t1 = float(t_abs[0])*xscale, float(t_abs[-1])*xscale
        c = color_E if str(row['env']).upper().startswith('E') else color_G
        ax.axvspan(t0, t1, color=c, alpha=alpha, lw=0)


def plot_replicates_daily_freq(
    res_list,
    *,
    which='B',                # 'A' or 'B'
    to_days=True, day_hours=24.0,
    percent=True,
    alpha_bg=0.05, show_mean=True, show_ci=True, ci_quantiles=(0.16, 0.84),
    this_color='#440154',
    figsize=(8, 4), legend_loc='upper left',
    pathFIGURES='',
    title=''
):
    """
    Uses `res['day_summaries']` to plot end-of-day frequencies for each replicate,
    prepending day 0 start. Highlights mean (and optional CI).
    """
    fig, ax = plt.subplots(figsize=figsize)
    xscale = (1.0/day_hours) if to_days else 1.0

    x_list, f_list = [], []

    res0 = res_list[0] if len(res_list) else None

    for res in res_list:
        # x: day 0 (0) + each day end time
        x_points = [0.0]
        for (t_abs, _), _row in zip(res['segments'], res['day_summaries'].iterrows()):
            x_points.append(float(t_abs[-1]) * xscale)

        df = res['day_summaries']
        if which.upper() == 'A':
            day0 = float(df.iloc[0]['fA_start'])
            f_end = df['fA_end'].to_numpy(float)
        else:
            day0 = float(df.iloc[0]['fB_start'])
            f_end = df['fB_end'].to_numpy(float)

        f = np.concatenate([[day0], f_end])
        if percent:
            f = 100.0 * f

        # faded individual
        ax.plot(x_points, f, color=this_color, lw=1.2, alpha=alpha_bg)

        x_list.append(np.asarray(x_points, float))
        f_list.append(np.asarray(f, float))

    # mean & CI on a common grid
    if f_list:
        same_grid = all(len(x) == len(x_list[0]) and np.allclose(x, x_list[0]) for x in x_list)
        if same_grid:
            x_grid = x_list[0]
            F = np.vstack(f_list)
        else:
            x_grid = np.unique(np.concatenate(x_list))
            F = np.vstack([np.interp(x_grid, x, y) for x, y in zip(x_list, f_list)])

        if show_mean:
            f_mean = F.mean(axis=0)
            ax.plot(x_grid, f_mean, color=this_color, lw=2.8, label=f"Mean Frequency")

        if show_ci and F.shape[0] >= 3:
            q_lo = np.quantile(F, ci_quantiles[0], axis=0)
            q_hi = np.quantile(F, ci_quantiles[1], axis=0)
            ax.fill_between(x_grid, q_lo, q_hi, color=this_color, alpha=0.18, label="CI")

        ax.set_xlim(x_grid[0], x_grid[-1])

    # environment band behind
    if res0 is not None:
        _draw_env_band_simple(ax, res0['segments'], res0['day_summaries'], to_days=to_days, day_hours=day_hours)

    # cosmetics
    ax.set_xlabel("Time (days)" if to_days else "Time (hours)")
    ax.set_ylabel(("Frequency (%)" if percent else "Frequency"))
    ax.axhline(50 if percent else 0.5, color='0.7', lw=1, ls='--')
    if not to_days:
        ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.set_ylim(0, 100 if percent else 1.0)
    ax.grid(True, alpha=0.1)


    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(frameon=False, loc=legend_loc)
    plt.tight_layout()

    #if title != '':
    #    ax.set_title(title, fontsize=16)

    if pathFIGURES != '':
            filename = f"{pathFIGURES}replicates_daily_freq_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()

    return fig, ax



def plot_mean_daily_freq(
    res_list,
    *,
    which='B',                # 'A' or 'B'
    to_days=True, day_hours=24.0,
    percent=True,
    alpha_bg=0.25, show_mean=True, show_ci=True, ci_quantiles=(0.16, 0.84),
    this_color='#440154',
    figsize=(8, 4), legend_loc='upper left',
    pathFIGURES='',
    title=''
):
    """
    Uses `res['day_summaries']` to plot end-of-day frequencies for each replicate,
    prepending day 0 start. Highlights mean (and optional CI).
    """
    fig, ax = plt.subplots(figsize=figsize)
    xscale = (1.0/day_hours) if to_days else 1.0

    x_list, f_list = [], []

    res0 = res_list[0] if len(res_list) else None

    for res in res_list:
        # x: day 0 (0) + each day end time
        x_points = [0.0]
        for (t_abs, _), _row in zip(res['segments'], res['day_summaries'].iterrows()):
            x_points.append(float(t_abs[-1]) * xscale)

        df = res['day_summaries']
        if which.upper() == 'A':
            day0 = float(df.iloc[0]['fA_start'])
            f_end = df['fA_end'].to_numpy(float)
        else:
            day0 = float(df.iloc[0]['fB_start'])
            f_end = df['fB_end'].to_numpy(float)

        f = np.concatenate([[day0], f_end])
        if percent:
            f = 100.0 * f

        # faded individual
        #ax.plot(x_points, f, color=this_color, lw=1.2, alpha=alpha_bg)

        x_list.append(np.asarray(x_points, float))
        f_list.append(np.asarray(f, float))

    # mean & CI on a common grid
    if f_list:
        same_grid = all(len(x) == len(x_list[0]) and np.allclose(x, x_list[0]) for x in x_list)
        if same_grid:
            x_grid = x_list[0]
            F = np.vstack(f_list)
        else:
            x_grid = np.unique(np.concatenate(x_list))
            F = np.vstack([np.interp(x_grid, x, y) for x, y in zip(x_list, f_list)])

        if show_mean:
            f_mean = F.mean(axis=0)
            ax.plot(x_grid, f_mean, color=this_color, lw=2.8, label=f"Mean Frequency")

        if show_ci and F.shape[0] >= 2:
            f_mean = F.mean(axis=0)
            f_se   = F.std(axis=0, ddof=1) #/ np.sqrt(F.shape[0])  # standard error of the mean
            ax.fill_between(x_grid, f_mean - f_se, f_mean + f_se,
                            color=this_color, alpha=0.18, label="±1 SD")


        ax.set_xlim(x_grid[0], x_grid[-1])

    # environment band behind
    if res0 is not None:
        _draw_env_band_simple(ax, res0['segments'], res0['day_summaries'], to_days=to_days, day_hours=day_hours)

    # cosmetics
    ax.set_xlabel("Time (days)" if to_days else "Time (hours)")
    ax.set_ylabel(("Frequency (%)" if percent else "Frequency"))
    ax.axhline(50 if percent else 0.5, color='0.7', lw=1, ls='--')
    if not to_days:
        ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.set_ylim(0, 100 if percent else 1.0)
    ax.grid(True, alpha=0.1)
    ax.legend(frameon=False, loc=legend_loc)

    # remove frame
    for spine in ax.spines.values():
        spine.set_visible(False)


    plt.tight_layout()


    if pathFIGURES != '':
            filename = f"{pathFIGURES}mean_daily_freq_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return fig, ax




def plot_freq_time_density(
    res_list,
    *,
    which='B',                  # 'A' or 'B'
    to_days=True, day_hours=24.0,
    percent=True,
    n_time=200,                 # time grid resolution
    bins_freq=20,               # vertical bins for density
    cmap='heat',
    alpha_env=0.20,
    show_mean=True,
    show_ci=True, ci_quantiles=(0.16, 0.84),
    mean_color='#ffffff',
    figsize=(8.75,4),
    legend_loc='upper left',
    pathFIGURES='',
    title=''
):
    """
    Build a time×frequency density (2D histogram) across replicates and plot as a heatmap.
    Overlays mean (and optional CI) trajectory. Uses end-of-day points (including day 0).
    """
    if not res_list:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No results", ha='center', va='center'); return fig, ax

    xscale = (1.0/day_hours) if to_days else 1.0

    # 1) Collect end-of-day (including day 0) trajectories from all replicates
    x_list, f_list = [], []
    for res in res_list:
        # time at day ends (prepend 0)
        x_points = [0.0]
        for (t_abs, _), _row in zip(res['segments'], res['day_summaries'].iterrows()):
            x_points.append(float(t_abs[-1]) * xscale)

        df = res['day_summaries']
        if which.upper() == 'A':
            day0 = float(df.iloc[0]['fA_start'])
            fend = df['fA_end'].to_numpy(float)
        else:
            day0 = float(df.iloc[0]['fB_start'])
            fend = df['fB_end'].to_numpy(float)

        f = np.concatenate([[day0], fend])
        if percent: f = 100.0 * f

        x_list.append(np.asarray(x_points, float))
        f_list.append(np.asarray(f, float))

    # 2) Common time grid and interpolation
    #    (Use union of time points so we never invent days)
    x_all = np.unique(np.concatenate(x_list))
    x_grid = np.linspace(x_all.min(), x_all.max(), n_time)
    F = np.vstack([np.interp(x_grid, x, y) for x, y in zip(x_list, f_list)])  # shape: (R, T)

    # 3) Build density per time column (histogram across replicates)
    f_min, f_max = (0.0, 100.0) if percent else (0.0, 1.0)
    edges = np.linspace(f_min, f_max, bins_freq + 1)
    centers = 0.5*(edges[:-1] + edges[1:])
    H = np.zeros((bins_freq, len(x_grid)), dtype=float)
    for j in range(len(x_grid)):
        col = F[:, j]
        hist, _ = np.histogram(col, bins=edges)
        H[:, j] = hist

    # normalize density per time (column) to max=1 for nice contrast (optional)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = H / np.maximum(H.max(axis=0, keepdims=True), 1e-12)

    # 4) Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    extent = [x_grid[0], x_grid[-1], f_min, f_max]
    #im = ax.imshow(
    #    H[::-1, :],             # flip so low freq at bottom
    #    aspect='auto',
    #    extent=extent,
    #    cmap=cmap,
    #    interpolation='nearest',
    #    vmin=0, vmax=1
    #)

    R = F.shape[0]
    H = H / max(R, 1)             # fraction of replicates per bin
    im = ax.imshow(
        H, origin='lower', aspect='auto', extent=extent, cmap=cmap,
        interpolation='nearest', vmin=0, vmax=H.max()
    )

    # 5) Overlay mean (and CI) trajectories
    if show_mean:
        f_mean = F.mean(axis=0)
        ax.plot(x_grid, f_mean, '--',color=mean_color, lw=2.2, label=f"Mean Frequency")


    # 6) Environment band (days) behind the plot if you have one replicate to copy
    try:
        res0 = res_list[0]
        # simple manual drawer that respects days vs hours
        for (t_abs, _), (_, row) in zip(res0['segments'], res0['day_summaries'].iterrows()):
            t0 = float(t_abs[0]) * xscale
            t1 = float(t_abs[-1]) * xscale
            env = str(row.get('env','G')).upper()
            col = "#B63E36" if env.startswith('E') else "#EBDD99"
            ax.axvspan(t0, t1, color=col, alpha=alpha_env, lw=0, zorder=-1)
    except Exception:
        pass

    # 7) Cosmetics
    ax.set_xlabel("Time (days)" if to_days else "Time (hours)")
    ax.set_ylabel("Frequency (%)" if percent else "Frequency")
    ax.set_ylim(f_min, f_max)
    if not to_days:
        ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.grid(True, alpha=0.1)
    if show_mean or show_ci:
        ax.legend(frameon=False, loc=legend_loc, labelcolor="#FFFFFF")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Relative density")

    plt.tight_layout()

    if pathFIGURES != '':
            filename = f"{pathFIGURES}freq_time_density_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return fig, ax

def plot_freq_time_with_highlights(
    res_list,
    *,
    which='B',                  # 'A' or 'B'
    to_days=True, day_hours=24.0,
    percent=True,
    n_time=200,                 # time grid resolution
    bins_freq=20,               # vertical bins for density
    cmap='Purples',
    alpha_env=0.20,
    show_mean=True,
    show_ci=False, ci_quantiles=(0.16, 0.84),
    mean_color='#000000',
    figsize=(8.,4),
    legend_loc='upper left',
    # --- highlight options ---
    highlight_ids=None,         # list of ids (e.g. pair_idx) to overlay
    fade_ids=None,
    id_key='pair_idx',          # key in each res dict with that id
    highlight_colors=None,      # dict {id: color}; else auto-cycle
    label_map=None,             # dict {id: "label text"}; default str(id)
    right_pad_frac=0.1,        # pad xlim to write labels
    alpha_bg_lines=0.8,          # (optional) fade all non-highlight lines (0 disables),
    pathFIGURES='',
    title=''
):
    """
    Time×frequency density (2D histogram) across replicates, with optional
    highlighted end-of-day trajectories overlaid and labeled at the right.
    """
    if not res_list:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No results", ha='center', va='center')
        return fig, ax

    xscale = (1.0/day_hours) if to_days else 1.0
    highlight_ids = set(highlight_ids or [])
    fade_ids = set(fade_ids or [])

    fig, ax = plt.subplots(figsize=figsize)

    # 1) Collect end-of-day trajectories from all replicates
    x_list, f_list, ids = [], [], []
    for res in res_list:
        # time at day ends (prepend 0)
        x_points = [0.0]
        for (t_abs, _), _row in zip(res['segments'], res['day_summaries'].iterrows()):
            x_points.append(float(t_abs[-1]) * xscale)

        df = res['day_summaries']
        if which.upper() == 'A':
            day0 = float(df.iloc[0]['fA_start'])
            fend = df['fA_end'].to_numpy(float)
        else:
            day0 = float(df.iloc[0]['fB_start'])
            fend = df['fB_end'].to_numpy(float)

        f = np.concatenate([[day0], fend])
        if percent:
            f = 100.0 * f

        # faded individual
        if res.get(id_key, None) in fade_ids:
            ax.plot(x_points, f, color="#999999", lw=1.0, alpha=alpha_bg_lines)

        x_list.append(np.asarray(x_points, float))
        f_list.append(np.asarray(f, float))
        ids.append(res.get(id_key, None))

    # 2) Common time grid and interpolation
    x_all = np.unique(np.concatenate(x_list))
    x_grid = np.linspace(x_all.min(), x_all.max(), n_time)
    F = np.vstack([np.interp(x_grid, x, y) for x, y in zip(x_list, f_list)])  # shape: (R, T)

    # 3) Build density per time column
    f_min, f_max = (0.0, 100.0) if percent else (0.0, 1.0)
    edges = np.linspace(f_min, f_max, bins_freq + 1)
    centers = 0.5*(edges[:-1] + edges[1:])
    H = np.zeros((bins_freq, len(x_grid)), dtype=float)
    for j in range(len(x_grid)):
        col = F[:, j]
        hist, _ = np.histogram(col, bins=edges)
        H[:, j] = hist

    # normalize each column to max=1 (nice contrast)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = H / np.maximum(H.max(axis=0, keepdims=True), 1e-12)

    extent = [x_grid[0], x_grid[-1], f_min, f_max]


    # 5) Overlay mean (and optional CI)
    if show_mean:
        f_mean = F.mean(axis=0)
        ax.plot(x_grid, f_mean, '--', color=mean_color, lw=2.2, label=f"Mean Frequency")
    if show_ci and F.shape[0] >= 3:
        q_lo = np.quantile(F, ci_quantiles[0], axis=0)
        q_hi = np.quantile(F, ci_quantiles[1], axis=0)
        ax.fill_between(x_grid, q_lo, q_hi, color=mean_color, alpha=0.18, label="CI")

    # 6) Environment band (days)
    try:
        res0 = res_list[0]
        for (t_abs, _), (_, row) in zip(res0['segments'], res0['day_summaries'].iterrows()):
            t0 = float(t_abs[0]) * xscale
            t1 = float(t_abs[-1]) * xscale
            env = str(row.get('env','G')).upper()
            col = "#B63E36" if env.startswith('E') else "#EBDD99"
            ax.axvspan(t0, t1, color=col, alpha=alpha_env, lw=0, zorder=-1)
    except Exception:
        pass

    # 8) Highlight selected ids, add right-side labels
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['w'])
    highlight_colors = highlight_colors or {}
    label_map = label_map or {}
    x_min, x_max = x_grid[0], x_grid[-1]
    pad = right_pad_frac * (x_max - x_min if x_max > x_min else 1.0)
    ax.set_xlim(x_min, x_max + pad)

    handles, labels = [], []

    for rid, x, f in zip(ids, x_list, f_list):
        if rid in highlight_ids:
            col = highlight_colors.get(rid, default_cycle[len(highlight_colors) % len(default_cycle)])
            highlight_colors.setdefault(rid, col)
            ax.plot(x, f, color=col, lw=2.6, alpha=1.0, zorder=3)
            # right-edge label
            txt = label_map.get(rid, str(rid))
            ax.text(x_max + 0.02*pad, f[-1], txt, color=col, fontsize=10, ha='left', va='center', clip_on=False)
            # legend entry
            h, = ax.plot([], [], color=col, lw=2.6)
            handles.append(h); labels.append(txt)

    # 9) Cosmetics
    ax.set_xlabel("Time (days)" if to_days else "Time (hours)")
    ax.set_ylabel("Frequency (%)" if percent else "Frequency")
    ax.set_ylim(f_min, f_max)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if not to_days:
        ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.grid(True, alpha=0.1)

    #if (show_mean or show_ci) or handles:
    #    ax.legend(handles, labels, frameon=False, loc=legend_loc)


    plt.tight_layout()

    if title != '':
        ax.set_title(title)


    if pathFIGURES != '':
            filename = f"{pathFIGURES}freq_time_with_highlights_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return fig, ax


def plot_freq_time_density_with_highlights(
    res_list,
    *,
    which='B',                  # 'A' or 'B'
    to_days=True, day_hours=24.0,
    percent=True,
    n_time=200,                 # time grid resolution
    bins_freq=20,               # vertical bins for density
    cmap='hot',
    alpha_env=0.20,
    show_mean=True,
    show_ci=False, ci_quantiles=(0.16, 0.84),
    mean_color='#ffffff',
    figsize=(9.5,4),
    legend_loc='upper left',
    # --- highlight options ---
    highlight_ids=None,         # list of ids (e.g. pair_idx) to overlay
    id_key='pair_idx',          # key in each res dict with that id
    highlight_colors=None,      # dict {id: color}; else auto-cycle
    label_map=None,             # dict {id: "label text"}; default str(id)
    right_pad_frac=0.1,        # pad xlim to write labels
    alpha_bg_lines=0.0,          # (optional) fade all non-highlight lines (0 disables),
    pathFIGURES='',
    title=''
):
    """
    Time×frequency density (2D histogram) across replicates, with optional
    highlighted end-of-day trajectories overlaid and labeled at the right.
    """
    if not res_list:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No results", ha='center', va='center')
        return fig, ax

    xscale = (1.0/day_hours) if to_days else 1.0
    highlight_ids = set(highlight_ids or [])

    # 1) Collect end-of-day trajectories from all replicates
    x_list, f_list, ids = [], [], []
    for res in res_list:
        # time at day ends (prepend 0)
        x_points = [0.0]
        for (t_abs, _), _row in zip(res['segments'], res['day_summaries'].iterrows()):
            x_points.append(float(t_abs[-1]) * xscale)

        df = res['day_summaries']
        if which.upper() == 'A':
            day0 = float(df.iloc[0]['fA_start'])
            fend = df['fA_end'].to_numpy(float)
        else:
            day0 = float(df.iloc[0]['fB_start'])
            fend = df['fB_end'].to_numpy(float)

        f = np.concatenate([[day0], fend])
        if percent:
            f = 100.0 * f

        x_list.append(np.asarray(x_points, float))
        f_list.append(np.asarray(f, float))
        ids.append(res.get(id_key, None))

    # 2) Common time grid and interpolation
    x_all = np.unique(np.concatenate(x_list))
    x_grid = np.linspace(x_all.min(), x_all.max(), n_time)
    F = np.vstack([np.interp(x_grid, x, y) for x, y in zip(x_list, f_list)])  # shape: (R, T)

    # 3) Build density per time column
    f_min, f_max = (0.0, 100.0) if percent else (0.0, 1.0)
    edges = np.linspace(f_min, f_max, bins_freq + 1)
    centers = 0.5*(edges[:-1] + edges[1:])
    H = np.zeros((bins_freq, len(x_grid)), dtype=float)
    for j in range(len(x_grid)):
        col = F[:, j]
        hist, _ = np.histogram(col, bins=edges)
        H[:, j] = hist

    # normalize each column to max=1 (nice contrast)
    #with np.errstate(divide='ignore', invalid='ignore'):
    #    H = H / np.maximum(H.max(axis=0, keepdims=True), 1e-12)
    #H = H / H.max()
    #print(H.max())


    # 4) Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    extent = [x_grid[0], x_grid[-1], f_min, f_max]

    R = F.shape[0]
    H = H / max(R, 1)             # fraction of replicates per bin
    im = ax.imshow(
        H, origin='lower', aspect='auto', extent=extent, cmap=cmap,
        interpolation='nearest', vmin=0, vmax=H.max()
    )


    handles, labels = [], []

    # 5) Overlay mean (and optional CI)
    if show_mean:
        f_mean = F.mean(axis=0)
        h, = ax.plot(x_grid, f_mean, '--', color=mean_color, lw=2.2, label=f"Mean Frequency")
        handles.append(h); labels.append("Mean Frequency")
    if show_ci and F.shape[0] >= 3:
        q_lo = np.quantile(F, ci_quantiles[0], axis=0)
        q_hi = np.quantile(F, ci_quantiles[1], axis=0)
        ax.fill_between(x_grid, q_lo, q_hi, color=mean_color, alpha=0.18, label="CI")

    # 6) Environment band (days)
    try:
        res0 = res_list[0]
        for (t_abs, _), (_, row) in zip(res0['segments'], res0['day_summaries'].iterrows()):
            t0 = float(t_abs[0]) * xscale
            t1 = float(t_abs[-1]) * xscale
            env = str(row.get('env','G')).upper()
            col = "#B63E36" if env.startswith('E') else "#EBDD99"
            ax.axvspan(t0, t1, color=col, alpha=alpha_env, lw=0, zorder=-1)
    except Exception:
        pass

    # 7) Optional overlay of individual lines (non-highlight faded)
    #    (Useful to see discrete paths over the density)
    #if alpha_bg_lines > 0:
    #    for x, f, rid in zip(x_list, f_list, ids):
    #        if rid not in highlight_ids:
    #            ax.plot(x, f, color='0.6', lw=1.0, alpha=alpha_bg_lines, zorder=1)

    # 8) Highlight selected ids, add right-side labels
    default_cycle = plt.rcParams['axes.prop_cycle'].by_key().get('color', ['w'])
    highlight_colors = highlight_colors or {}
    label_map = label_map or {}
    x_min, x_max = x_grid[0], x_grid[-1]
    pad = right_pad_frac * (x_max - x_min if x_max > x_min else 1.0)
    ax.set_xlim(x_min, x_max + pad)


    for rid, x, f in zip(ids, x_list, f_list):
        if rid in highlight_ids:
            col = highlight_colors.get(rid, default_cycle[len(highlight_colors) % len(default_cycle)])
            highlight_colors.setdefault(rid, col)
            ax.plot(x, f, color=col, lw=2.6, alpha=1.0, zorder=3)
            # right-edge label
            txt = label_map.get(rid, str(rid))
            ax.text(x_max + 0.04*pad, f[-1], txt, color=col, fontsize=10, ha='left', va='center', clip_on=False)
            # legend entry
            h, = ax.plot([], [], color=col, lw=2.6)
            handles.append(h); labels.append(txt)

    # 9) Cosmetics
    ax.set_xlabel("Time (days)" if to_days else "Time (hours)")
    ax.set_ylabel("Frequency (%)" if percent else "Frequency")
    ax.set_ylim(f_min, f_max)

    for spine in ax.spines.values():
        spine.set_visible(False)

    if not to_days:
        ax.xaxis.set_major_locator(MultipleLocator(24))
    ax.grid(True, alpha=0.)

    #if (show_mean or show_ci) or handles:
    #    ax.legend(handles, labels, frameon=False, loc=legend_loc, labelcolor="#FFFFFF")

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Relative density")

    plt.tight_layout()


    if pathFIGURES != '':
            filename = f"{pathFIGURES}freq_time_density_with_highlights_{title}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()
    return fig, ax

# --- Persistence probability over a horizon -------------------------------
def persistence_probability(res_list, H_days=30, fmin=0.01, which='B'):
    """
    Given a list of independent simulation results (same regime, different replicates),
    compute P_persist(H) = Pr(min_{t<=H} f(t) > fmin).
    """
    ok = 0; total = 0
    for res in res_list:
        t, f = _day_grid_and_freq(res, which=which)
        total += 1
        mask = (t <= float(H_days))
        f_sub = f[mask] if np.any(mask) else f[:1]  # at least day 0
        if np.all(np.isfinite(f_sub)) and (np.min(f_sub) > float(fmin)):
            ok += 1
    return ok / max(total, 1)



# ----- helpers (same as before) -----
def _fit_log_odds_half_life(day_times_h, fB_daily):
    t_days = np.asarray(day_times_h, float) / 24.0
    f = np.clip(np.asarray(fB_daily, float), 1e-9, 1-1e-9)
    y = np.log(f/(1-f))
    X = np.c_[np.ones_like(t_days), t_days]
    coef, _, _, _ = lstsq(X, y, rcond=None)
    alpha, beta = coef
    y_hat = X @ coef
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    R2 = 1 - ss_res/ss_tot if ss_tot > 0 else np.nan
    t_half = np.log(2)/abs(beta) if beta != 0 else np.inf
    return dict(beta_per_day=float(beta), t_half_days=float(t_half), R2=float(R2), n_points=int(len(t_days)))

def _persistence_over_horizon(df_day_summ, H_days=60, fmin=0.01):
    t_h = df_day_summ['t_end_h'].to_numpy(float)
    fB  = df_day_summ['fB_end'].to_numpy(float)
    mask = t_h <= (H_days*24.0 + 1e-9)
    return bool(np.all(fB[mask] > fmin))

def _run_one_serial(schedule, row_G, row_E, p0, sim_fn, initial_resource,
                    day_hours=24.0, dilution=10.0, Ntot0_start=1e6, freqA0_start=0.5):
    res = serial_transfer_competition(
        env_schedule=schedule, row_G=row_G, row_E=row_E, p0=p0,
        sim_fn=sim_fn, initial_resource=initial_resource,
        day_hours=day_hours, dilution=dilution,
        Ntot0_start=Ntot0_start, freqA0_start=freqA0_start,
        N_extinct=1.0, fmin_extinct=0.01
    )
    ds = res['day_summaries'].copy()
    if 't_end_h' not in ds.columns:
        ds['t_end_h'] = [(i+1)*day_hours for i in range(len(ds))]
    return ds, res

def _get_rows_for_pair(df_env, pair_idx):
    row_G = df_env[(df_env['Environment']=='G') & (df_env['pair_idx']==pair_idx)].iloc[0]
    row_E = df_env[(df_env['Environment']=='E') & (df_env['pair_idx']==pair_idx)].iloc[0]
    return row_G, row_E

def label_schedule(schedule):
    """
    Label a schedule sequence as 'aerobiosis', 'anaerobiosis', or 'switch_KX'.

    Parameters
    ----------
    schedule : list[str] of 'E' or 'G'

    Returns
    -------
    (label, schedule_str)
        label : str
            e.g. 'aerobiosis', 'anaerobiosis', or 'switch_K7'
        schedule_str : str
            compact string representation of the schedule (e.g. 'GGGGEEEE...')
    """
    s = ''.join(schedule)
    n = len(schedule)
    uniq = set(schedule)

    if uniq == {'E'}:
        return "aerobiosis", s
    elif uniq == {'G'}:
        return "anaerobiosis", s
    else:
        # detect the index of the first change
        switch_idx = next(
            (i for i in range(1, n) if schedule[i] != schedule[i - 1]),
            None
        )
        if switch_idx is None:
            # should never happen if uniq has both 'E' and 'G'
            return "switch", s
        # note: switch_idx days in the first environment
        return f"switch_K{switch_idx}", s


# ==========================================================
# Population-level analysis across all pairs
# ==========================================================

# --- robust slope/half-life from day_summaries only ---
def _fit_log_odds_half_life_from_day_summaries(ds, which='B', day_hours=24.0):
    """
    Fits y = logit(f_{which}) = log(f/(1-f)) vs time (in days) using the end-of-day points.
    Returns standardized dict: beta_per_day, t_half_days, R2, n_points.
    """
    if ds is None or len(ds) == 0:
        #return {'beta_per_day': np.nan, 't_half_days': np.nan, 'R2': np.nan, 'n_points': 0}
        return { 'n_points': 0}

    # time in days
    if 't_end_d' in ds.columns:
        t = ds['t_end_d'].to_numpy(float)
    elif 't_end_h' in ds.columns:
        t = (ds['t_end_h'].to_numpy(float)) / float(day_hours)
    else:
        # fallback: try 'time_end' in hours
        if 'time_end' in ds.columns:
            t = ds['time_end'].to_numpy(float) / float(day_hours)
        else:
            #return {'beta_per_day': np.nan, 't_half_days': np.nan, 'R2': np.nan, 'n_points': 0}
            return {'n_points': 0}

    # choose frequency column
    which = str(which).upper()
    col = 'fB_end' if which == 'B' else 'fA_end'
    if col not in ds.columns:
        # try lowercase variants
        cand = [c for c in ds.columns if c.lower() == col.lower()]
        if not cand:
            #return {'beta_per_day': np.nan, 't_half_days': np.nan, 'R2': np.nan, 'n_points': 0}
            return {'n_points': 0}
        col = cand[0]

    f = ds[col].to_numpy(float)

    # keep only (0,1) to avoid inf log-odds
    eps = 1e-9
    m = (f > eps) & (f < 1.0 - eps)
    t, f = t[m], f[m]
    if t.size < 3:  # need at least 3 points for a stable slope + R2
        return {'n_points': int(t.size)}

    y = np.log(f / (1.0 - f))  # log-odds

    # linear fit y = a + b t
    # np.polyfit returns [b, a] for deg=1
    #b, a = np.polyfit(t, y, 1)
    #yhat = a + b * t
    #ss_res = float(np.sum((y - yhat)**2))
    #ss_tot = float(np.sum((y - np.mean(y))**2))
    #R2 = (1.0 - ss_res/ss_tot) if ss_tot > 0 else np.nan

    #t_half_days = (np.log(2.0)/abs(b)) if b != 0 else np.inf

    return {
        'n_points': int(t.size)
    }


# --- little guard to fetch the two rows for a pair, with clear errors ---
def _safe_get_rows_for_pair(df_env, pair_idx):
    sub = df_env[df_env['pair_idx'] == pair_idx]
    if sub.empty:
        return None, None, "pair_idx not found"

    # Expect one row for G and one for E; if you store parameters on a single row, adapt here
    rowG = sub[sub['Environment'].str.upper() == 'G']
    rowE = sub[sub['Environment'].str.upper() == 'E']
    if rowG.empty or rowE.empty:
        return None, None, f"missing {'G' if rowG.empty else ''}{' and ' if rowG.empty and rowE.empty else ''}{'E' if rowE.empty else ''}"
    return rowG.iloc[0], rowE.iloc[0], None

def extract_survival_info(results, which='B', threshold=0.01, num_days=None):
    """
    From a list of serial_transfer_competition result dicts (each with 'day_summaries'),
    compute:
      - time to extinction (first day where freq < threshold)
      - censoring if survived entire run
      - final frequency

    Returns a tidy DataFrame with columns: ['pair_idx', 'time', 'event', 'f_end'].
    """
    rows = []
    for res in results:
        df = res['day_summaries']
        fcol = 'fB_end' if which.upper() == 'B' else 'fA_end'

        # get array of end-of-day frequencies
        f_series = df[fcol].to_numpy(float)
        days = df['day'].to_numpy(int)

        # extinction: first day when freq < threshold
        extinct_idx = np.where(f_series < threshold)[0]
        if extinct_idx.size > 0:
            tte = days[extinct_idx[0]]
            event = 1  # extinction event occurred
        else:
            tte = days[-1] if num_days is None else num_days
            event = 0  # censored (survived)

        f_end = float(f_series[-1])
        rows.append({'pair_idx': res['pair_idx'], 'time': tte, 'event': event, 'f_end': f_end})
    return pd.DataFrame(rows)

def analyze_schedule_population(
    df_strains, labels_of_interest, df_env, pair_indices, schedule, p0,
    sim_fn, initial_resource,
    *,
    day_hours=24.0, dilution=10.0,
    Ntot0_start=1e6, freqA0_start=0.5,
    fmin_end=0.01,                      # threshold for "persisted at end"
    outcome_thresholds=(0.1, 0.9),    # (cleared, fixated) thresholds for outcome labeling
    toPlot=True, title=None, family=None, pathFIGURES=None
):
    """
    Runs the schedule for all pair_indices and returns ONLY end-of-experiment stats.

    Returns
    -------
    dict with:
      - 'schedule'        : list[str] schedule used
      - 'pair_indices'    : list[int] pairs simulated
      - 'results'         : list of {'pair_idx','res','day_summaries'}
      - 'pair_stats'      : DataFrame with per-pair end-of-run info:
                            ['pair_idx','f_end','persist_end','time','event',
                             'outcome','schedule_name','schedule_str']
      - 'agg'             : {'n_pairs', 'P_end'}  (fraction persisted at final day)
      - 'outcome_counts'  : dict counts of cleared/stable/fixated/unknown
      - 'outcome_frac'    : dict fractions of same
    """
    import numpy as np
    import pandas as pd

    sched_name, sched_str = label_schedule(schedule)
    H_max = len(schedule)  # number of days in this run

    # If caller passed None or empty, derive from df_env
    if pair_indices is None or len(pair_indices) == 0:
        if 'pair_idx' not in df_env.columns:
            raise ValueError("df_env has no 'pair_idx' column; cannot determine pairs.")
        pair_indices = sorted(pd.unique(df_env['pair_idx'].dropna()))
        print(f"[analyze] derived {len(pair_indices)} pairs from df_env.")

    results = []
    stats_rows = []
    skipped = []

    def _classify_from_fend(f_end, f_cleared=0.05, f_fix=0.95):
        if not np.isfinite(f_end):
            return 'unknown'
        if f_end <= f_cleared:
            return 'cleared'
        elif f_end >= f_fix:
            return 'fixated'
        else:
            return 'stable'

    # --- core loop over pairs ---
    for pid in pair_indices:
        row_G, row_E, err = _safe_get_rows_for_pair(df_env, pid)
        if err is not None:
            skipped.append((pid, err))
            continue

        # Run one serial-transfer experiment for this pair
        ds, res = _run_one_serial(
            schedule, row_G, row_E, p0,
            sim_fn, initial_resource,
            day_hours=day_hours, dilution=dilution,
            Ntot0_start=Ntot0_start, freqA0_start=freqA0_start
        )
        results.append({'pair_idx': pid, 'res': res, 'day_summaries': ds})

        # --- per-pair end-of-run data ---
        # final frequency for B at last day
        try:
            f_end = float(ds['fA_end'].iloc[-1])
        except Exception:
            f_end = np.nan


        # persistence at end
        persist_end = bool(np.isfinite(f_end) and (f_end >= float(fmin_end)))

        # time-to-extinction (first day where fB_end < fmin_end), with censoring at H_max
        if ds is not None and ('fA_end' in ds.columns) and ('day' in ds.columns):
            below = ds.index[ds['fA_end'].to_numpy(float) < float(fmin_end)].tolist()
            if below:
                time = int(ds.loc[below[0], 'day'])
                event = 1  # extinction occurred
            else:
                time = int(H_max)  # censored at end
                event = 0
        else:
            time = int(H_max)
            event = 0

        # outcome bucket from final frequency (optional but handy)
        f_cleared, f_fix = outcome_thresholds
        outcome = _classify_from_fend(f_end, f_cleared=float(f_cleared), f_fix=float(f_fix))

        fA_last = float(ds['fA_end'].iloc[-1]) if ('fA_end' in ds) else np.nan
        fB_last = float(ds['fB_end'].iloc[-1]) if ('fB_end' in ds) else np.nan

        stats_rows.append({
            'pair_idx': pid,
            'f_end': fA_last,            # ARC final freq (primary)
            'fA_end': fA_last,           # explicit ARC column (redundant but clear)
            'fB_end': fB_last,           # control final freq (for checks / plots)
            'persist_end': persist_end,
            'time': time,
            'event': event,
            'outcome': outcome,
            'schedule_name': sched_name,
            'schedule_str': sched_str
        })


    pair_stats = pd.DataFrame(stats_rows)

    # Aggregate: fraction persisted at end
    if not pair_stats.empty:
        P_end = float(pair_stats['persist_end'].mean())
    else:
        P_end = float('nan')

    agg = {
        'n_pairs': int(len(pair_stats)),
        'P_end': P_end
    }

    # Outcome counts/fractions
    outcome_counts = {'cleared': 0, 'stable': 0, 'fixated': 0, 'unknown': 0}
    if not pair_stats.empty and 'outcome' in pair_stats.columns:
        vc = pair_stats['outcome'].value_counts(dropna=False)
        for k in outcome_counts.keys():
            outcome_counts[k] = int(vc.get(k, 0))
    total_pairs = max(1, int(len(pair_stats)))
    outcome_frac = {k: (v / total_pairs) for k, v in outcome_counts.items()}

    # Optional plotting (unchanged wiring; uses your existing helpers)
    if toPlot and len(results):
        # Build res_list_all for your plotters
        res_list_all = []
        for r in results:
            rr = r['res']
            res_list_all.append({
                'pair_idx': r['pair_idx'],
                'time': rr['time'],
                'pops': rr['pops'],
                'segments': rr.get('segments', []),
                'day_summaries': r['day_summaries'],
            })





    # Optional plotting (unchanged)
    if toPlot and len(results):

        fig_dist, ax_dist = plot_env_violin(df_env, family=family, pathFIGURES=pathFIGURES, title=family)



        fig4, ax4 = plot_mean_daily_freq(
            res_list_all,
            which='A',
            to_days=True,
            percent=True,
            legend_loc='upper left',
            pathFIGURES=pathFIGURES,
            title=f"{family}_{sched_name}"
        )

        fig2, ax2 = plot_replicates_daily_freq(
            res_list_all,
            which='A', to_days=True, percent=True,
            show_ci=False,
            legend_loc='upper left',
            pathFIGURES=pathFIGURES,
            title=f"{family}_{sched_name}"
        )

        fig, ax = plot_freq_time_density(
            res_list_all,
            which='A', to_days=True, percent=True,
            cmap='magma', show_mean=True, show_ci=True,
            pathFIGURES=pathFIGURES,
            title=f"{family}_{sched_name}"
        )


        mask = df_strains['label'].isin(labels_of_interest)
        subset = df_strains.loc[mask]
        to_highlight = subset['pair_idx'].dropna().tolist()
        to_fade = df_strains['pair_idx'].dropna().tolist()
        label_map = dict(zip(subset['pair_idx'], subset['label']))
        highlight_colors = {}

        fig5, ax5 = plot_freq_time_with_highlights(
            res_list_all, which='A', to_days=True, percent=True,
            highlight_ids=to_highlight, fade_ids=to_fade, id_key='pair_idx',
            label_map=label_map, highlight_colors=highlight_colors,
            alpha_bg_lines=0.2, legend_loc='upper left',
            pathFIGURES=pathFIGURES, title=f"{family}_{sched_name}"
        )

        fig, ax = plot_freq_time_density_with_highlights(
            res_list_all, which='A', to_days=True, percent=True,
            highlight_ids=to_highlight, id_key='pair_idx',
            label_map=label_map, highlight_colors=highlight_colors,
            alpha_bg_lines=0.5, legend_loc='upper left',
            pathFIGURES=pathFIGURES, title=f"{family}_{sched_name}"
        )



    if skipped:
        print(f"[analyze] skipped {len(skipped)} pairs: {skipped}")

    return {
        'schedule': list(schedule),
        'pair_indices': list(pair_indices),
        'results': results,
        'pair_stats': pair_stats,     # <- contains: f_end, persist_end, time, event, outcome, schedule labels
        'agg': agg,                   # <- contains: n_pairs, P_end
        'outcome_counts': outcome_counts,
        'outcome_frac': outcome_frac
    }



def _ellipse_axis_aligned(ax, x, y, nsig_levels=(1,2), color='black', alpha=0.8, lw=1.5):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 2:  # need at least 2 points for std
        return
    mu_x, mu_y = np.nanmean(x), np.nanmean(y)
    sx = np.nanstd(x, ddof=1) if x.size >= 2 else 0.0
    sy = np.nanstd(y, ddof=1) if y.size >= 2 else 0.0
    if not (np.isfinite(sx) and np.isfinite(sy)) or (sx == 0 and sy == 0):
        return
    for nsig in nsig_levels:
        w = 2 * nsig * sx
        h = 2 * nsig * sy
        if w > 0 and h > 0:
            ax.add_patch(Ellipse((mu_x, mu_y), width=w, height=h, angle=0.0,
                                 edgecolor=color, facecolor='none', lw=lw, alpha=alpha, zorder=3))

def _ellipse_rotated(ax, x, y, nsig_levels=(1,2), color='black', alpha=0.8, lw=1.5):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if x.size < 3:  # covariance needs ≥3 for stable ddof=1
        return
    mu_x, mu_y = np.nanmean(x), np.nanmean(y)
    cov = np.cov(np.vstack([x, y]), ddof=1)
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)          # ascending eigenvalues
    vals = np.clip(vals, 0.0, None)           # numeric safety
    angle_deg = np.degrees(np.arctan2(vecs[1,1], vecs[0,1]))  # major axis angle
    for nsig in nsig_levels:
        w = 2 * nsig * np.sqrt(vals[1])       # major axis
        h = 2 * nsig * np.sqrt(vals[0])       # minor axis
        if np.isfinite(w) and np.isfinite(h) and w > 0 and h > 0:
            ax.add_patch(Ellipse((mu_x, mu_y), width=w, height=h, angle=angle_deg,
                                 edgecolor=color, facecolor='none', lw=lw, alpha=alpha, zorder=3))

def plot_w_scatter(
    df_env,
    family=None,
    highlight=None,
    annotate=None,
    *,
    ellipse_mode='rotated',      # 'axis' | 'rotated' | 'auto'
    corr_threshold=0.2,       # used only when ellipse_mode='auto'
    nsig_levels=(1,2),
    ellipse_color='#666666',
    ellipse_alpha=0.8,
    ellipse_lw=1.5,
    pathFIGURES=''
):
    """
    Scatter of Relative fitness (w) in G (x) vs E (y) per label, with ellipses.

    df_env must have columns: ['label','Environment','achieved_w','family'].
    ellipse_mode:
      - 'axis'     : axis-aligned ellipse(s) using per-axis std
      - 'rotated'  : covariance ellipse(s) (tilted if correlation exists)
      - 'auto'     : rotated if |corr(G,E)| >= corr_threshold, else axis-aligned
    """
    highlight = set(highlight or [])
    annotate  = set(annotate or [])

    if family is not None:
        df_env = df_env[df_env['family'] == family].copy()
        if df_env.empty:
            raise ValueError(f"No rows found for family '{family}'.")

    # pivot to one row per label with E and G side-by-side
    pivot = (
        df_env.pivot_table(index="label", columns="Environment", values="achieved_w", aggfunc="mean")
              .reset_index()
    )
    if not {'E','G'}.issubset(pivot.columns):
        raise ValueError("Both E and G values are required (columns 'E' and 'G').")

    x = pivot['G'].to_numpy(float)  # G on x-axis
    y = pivot['E'].to_numpy(float)  # E on y-axis

    fig, ax = plt.subplots(figsize=(6,6))

    # base scatter
    ax.scatter(x, y, color='grey', alpha=0.2, s=20)

    # highlights
    if highlight:
        df_hl = pivot[pivot['label'].isin(highlight)]
        ax.scatter(df_hl['G'], df_hl['E'], color='yellow', alpha=0.7, s=20)

    # annotations
    if annotate:
        df_ann = pivot[pivot['label'].isin(annotate)]
        ax.scatter(df_ann['G'], df_ann['E'], color='red', alpha=0.8)
        for _, row in df_ann.iterrows():
            ax.text(float(row['G'])+0.01, float(row['E']), row['label'],
                    fontsize=9, ha='left', va='center', color='red')

    # fixed square axes
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_aspect('equal', adjustable='box')

    # cross at (1,1)
    ax.axhline(1, color='black', linestyle='--', alpha=0.7)
    ax.axvline(1, color='black', linestyle='--', alpha=0.7)

    # labels & title
    ax.set_xlabel("Relative fitness (anaero)")
    ax.set_ylabel("Relative fitness (aero)")
    ttl = ""
    if family:
        ttl += f" [{family}]"
    ax.set_title(ttl)

    # choose ellipse mode
    mode = ellipse_mode.lower()
    if mode == 'auto':
        # compute correlation (finite-only)
        m = np.isfinite(x) & np.isfinite(y)
        corr = np.corrcoef(x[m], y[m])[0,1] if m.sum() >= 2 else np.nan
        if np.isfinite(corr) and abs(corr) >= float(corr_threshold):
            _ellipse_rotated(ax, x, y, nsig_levels, ellipse_color, ellipse_alpha, ellipse_lw)
        else:
            _ellipse_axis_aligned(ax, x, y, nsig_levels, ellipse_color, ellipse_alpha, ellipse_lw)
    elif mode == 'rotated':
        _ellipse_rotated(ax, x, y, nsig_levels, ellipse_color, ellipse_alpha, ellipse_lw)
    else:  # 'axis'
        _ellipse_axis_aligned(ax, x, y, nsig_levels, ellipse_color, ellipse_alpha, ellipse_lw)

    plt.tight_layout()


    if pathFIGURES != '':
            filename = f"{pathFIGURES}w_scatter_{family}.pdf"
            plt.savefig(filename, format='pdf')
            print("Exporting %s" % filename)
    plt.show()

    return pivot




def plot_w_scatter_highlight(
    df_env,
    family=None,
    highlight=None,
    annotate=None,
    *,
    ellipse_mode='axis',      # 'axis' | 'rotated' | 'auto'
    nsig_levels=(1,2),
    ellipse_color='#666666',
    ellipse_alpha=0.8,
    ellipse_lw=1.5,
    color_highlight='red',
    color_annotate='black',
    pathFIGURES='',
    title='',
    ax=None
):
    """
    Scatter of Relative fitness (w) in G (x) vs E (y) per label, with ellipses.

    df_env must have columns: ['label','Environment','achieved_w','family'].
    """


    # --- do NOT create a new figure if ax is given ---
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), sharey=False)
        own_fig = True
    else:
        own_fig = False

    # ---- sanitize inputs ----
    highlight = set(highlight or [])
    annotate  = set(annotate or [])

    d = df_env.copy()
    if family is not None:
        d = d[d['family'] == family].copy()
    if d.empty:
        raise ValueError("No rows to plot for this selection.")

    # Pivot: one row per label; columns 'E' and 'G'
    piv = (d.pivot_table(index="label", columns="Environment",
                         values="achieved_w", aggfunc="mean"))
    # Ensure we have both E and G
    missing_env = [env for env in ('E','G') if env not in piv.columns]
    if missing_env:
        raise ValueError(f"Missing environment(s) in data: {missing_env}. Need both 'E' and 'G'.")

    # Work with a clean frame (labels as index)
    piv = piv[['G','E']].dropna(how='any').copy()
    if piv.empty:
        raise ValueError("After pivot/dropna, no paired (E,G) rows remain to plot.")

    # ---- aligned masks (indexed to piv.index = labels) ----
    labels_idx = piv.index.astype(str)
    mask_x = labels_idx.str.contains(r'_x', na=False)

    # membership masks
    mask_highlight = labels_idx.isin(highlight)
    # annotate-only = in annotate but NOT in highlight
    mask_annotate  = labels_idx.isin(annotate) & ~mask_highlight


    # base scatter: two alphas depending on '_x' in label
    ax.scatter(piv.loc[ mask_x, 'G'], piv.loc[ mask_x, 'E'],
               color='grey', alpha=0.10, s=20)
    ax.scatter(piv.loc[~mask_x, 'G'], piv.loc[~mask_x, 'E'],
               color='grey', alpha=0.80, s=20)

    # highlights (face+edge in highlight color)
    if mask_highlight.any():
        face_hl = mcolors.to_rgba(color_highlight, alpha=1.0)
        edge_hl = mcolors.to_rgba(color_highlight, alpha=1.00)
        ax.scatter(piv.loc[mask_highlight, 'G'], piv.loc[mask_highlight, 'E'],
                   facecolors=face_hl, edgecolors=edge_hl, s=20, linewidths=1.5, alpha=0.1)

    # annotations (only those not highlighted)
    if mask_annotate.any():
        face_ann = mcolors.to_rgba(color_highlight, alpha=1.0)
        edge_ann = mcolors.to_rgba("black", alpha=1.00)
        #ax.scatter(piv.loc[mask_annotate, 'G'], piv.loc[mask_annotate, 'E'],
        #           facecolors=face_ann, edgecolors=edge_ann, s=20, linewidths=1.2)

        mask_nonx_highlight = (~mask_x) & mask_highlight
        mask_x_highlight = (~mask_x) & (~mask_highlight)
        mask_x_highlight_annotate = (~mask_x) & (~mask_highlight) & (mask_annotate)

        ax.scatter(
            piv.loc[mask_nonx_highlight, 'G'],
            piv.loc[mask_nonx_highlight, 'E'],
            facecolors=face_ann, edgecolors=edge_ann, alpha=0.80, s=20
        )

        for lab, row in piv.loc[mask_nonx_highlight, ['G','E']].iterrows():
            ax.text(float(row['G'])+0.01, float(row['E']), str(lab),
                    fontsize=9, ha='left', va='center', color=color_annotate, alpha=0.8)

        ax.scatter(
            piv.loc[mask_x_highlight, 'G'],
            piv.loc[mask_x_highlight, 'E'],
            color=color_annotate, alpha=0.30, s=20
        )


        for lab, row in piv.loc[mask_x_highlight_annotate, ['G','E']].iterrows():
            ax.text(float(row['G'])+0.01, float(row['E']), str(lab),
                    fontsize=9, ha='left', va='center', color=color_annotate, alpha=0.8)


    # annotations (only those annotated)
    if mask_annotate.any():
        face_ann = mcolors.to_rgba("black", alpha=0.)
        edge_ann = mcolors.to_rgba("black", alpha=0.3)
        ax.scatter(piv.loc[mask_annotate, 'G'], piv.loc[mask_annotate, 'E'],
                   facecolors=face_ann, edgecolors=edge_ann, s=20, linewidths=1.2)




    # fixed square axes + cross at (1,1)
    ax.set_xlim(0.5, 1.5)
    ax.set_ylim(0.5, 1.5)
    ax.set_aspect('equal', adjustable='box')
    ax.axhline(1, color='black', linestyle='--', alpha=0.7)
    ax.axvline(1, color='black', linestyle='--', alpha=0.7)
    ax.set_xlabel("Relative fitness (anaerobiosis)")
    ax.set_ylabel("Relative fitness (aerobiosis)")

    ax.set_title(title or family)

    # ---- ellipses ----
    x = piv['G'].to_numpy(float)
    y = piv['E'].to_numpy(float)

    def _ellipse_axis_aligned(ax_, x_, y_, nsig=(1,2), color='#666', alpha=0.8, lw=1.5):
        mx, my = np.nanmean(x_), np.nanmean(y_)
        sx, sy = np.nanstd(x_),  np.nanstd(y_)
        for k in nsig:
            w = 2*k*sx
            h = 2*k*sy
            e = plt.matplotlib.patches.Ellipse((mx,my), width=w, height=h,
                                               facecolor='none', edgecolor=color,
                                               lw=lw, alpha=alpha)
            ax_.add_patch(e)

    def _ellipse_rotated(ax_, x_, y_, nsig=(1,2), color='#666', alpha=0.8, lw=1.5):
        X = np.column_stack([x_, y_])
        X = X[np.isfinite(X).all(axis=1)]
        if X.shape[0] < 3:
            _ellipse_axis_aligned(ax_, x_, y_, nsig, color, alpha, lw); return
        C = np.cov(X.T)
        vals, vecs = np.linalg.eigh(C)
        order = vals.argsort()[::-1]
        vals, vecs = vals[order], vecs[:,order]
        ang = np.degrees(np.arctan2(vecs[1,0], vecs[0,0]))
        mx, my = np.nanmean(x_), np.nanmean(y_)
        for k in nsig:
            w = 2*k*np.sqrt(vals[0])
            h = 2*k*np.sqrt(vals[1])
            e = plt.matplotlib.patches.Ellipse((mx,my), width=w, height=h,
                                               angle=ang, facecolor='none',
                                               edgecolor=color, lw=lw, alpha=alpha)
            ax_.add_patch(e)

    mode = str(ellipse_mode).lower()
    if mode == 'rotated':
        _ellipse_rotated(ax, x, y, nsig_levels, ellipse_color, ellipse_alpha, ellipse_lw)
    else:
        _ellipse_axis_aligned(ax, x, y, nsig_levels, ellipse_color, ellipse_alpha, ellipse_lw)

    if own_fig:
        plt.tight_layout()
        if pathFIGURES:
            filename = f"{pathFIGURES.rstrip('/')}/w_scatter_highlight_{family}.pdf"
            plt.savefig(filename, dpi=300)
            print(f"Exported: {filename}")
        plt.show()


    return ax

def print_schedules_readable(schedules_dict):
    """
    Prints schedules in a compact, human-readable way.
    """
    for name, sched in schedules_dict.items():
        s = "".join(sched)
        # Optionally compress consecutive environments (e.g. GGGEEEE -> G(3)E(4))
        comp = []
        last = s[0]
        count = 1
        for ch in s[1:]:
            if ch == last:
                count += 1
            else:
                comp.append(f"{last}({count})")
                last = ch
                count = 1
        comp.append(f"{last}({count})")
        compressed = " ".join(comp)

        print(f"{name:<12} : {compressed}   [{len(sched)} days]")
def make_switch_schedule(first_env="G", switch_day=10, num_days=100, second_env="E"):
    if switch_day < 0 or switch_day > num_days:
        raise ValueError("switch_day must be in [0, num_days]")
    return [first_env] * switch_day + [second_env] * (num_days - switch_day)

def build_switch_schedules(Ks, num_days, first_env="G", second_env="E"):
    """
    Returns dict: {'switch_K7': [...], 'switch_K30': [...], ...}
    """
    return {
        f"switch_K{K}": make_switch_schedule(first_env=first_env, switch_day=K, num_days=num_days, second_env=second_env)
        for K in Ks
    }


def upsert_to_sheet(sheet, tab_name, df_new, key_cols):
    """
    Upsert df_new into a Google Sheet tab by key columns.
    If the tab doesn't exist, it's created.
    """
    try:
        ws = sheet.worksheet(tab_name)
        df_old = (get_as_dataframe(ws, header=0, evaluate_formulas=True)
                  .dropna(how="all"))
    except Exception:
        ws = sheet.add_worksheet(title=tab_name, rows=2000, cols=40)
        df_old = pd.DataFrame()

    if df_old.empty:
        df_combined = df_new.copy()
    else:
        # concat then drop duplicates by key to simulate an upsert
        df_combined = (pd.concat([df_old, df_new], ignore_index=True, sort=False)
                       .drop_duplicates(subset=key_cols, keep='last'))

    ws.clear()
    set_with_dataframe(ws, df_combined, include_index=False, include_column_header=True)
    return len(df_new), len(df_combined)


def append_to_sheet(sheet, tab_name, df):
    """Append dataframe df to the end of a Google Sheet tab."""
    ws = sheet.worksheet(tab_name)
    existing_rows = len(ws.get_all_values())
    start_row = existing_rows + 1
    values = [df.columns.tolist()] + df.values.tolist() if existing_rows == 0 else df.values.tolist()
    ws.insert_rows(values, row=start_row)
    print(f"Appended {len(df)} rows to '{tab_name}' starting at row {start_row}.")



def load_paired_family_exp(sheet, family,
                           tab_E='fitness_aerobiosis',
                           tab_G='fitness_anaerobiosis',
                           min_r2=0.9):
    exp_E = load_exp_fitness(sheet, tab=tab_E)
    exp_G = load_exp_fitness(sheet, tab=tab_G)
    fits_df, _ = load_all_fits(sheet, tab="params_fits", min_r2=min_r2)

    exp_E_fam = attach_family_to_exp(exp_E, fits_df)
    exp_G_fam = attach_family_to_exp(exp_G, fits_df)
    exp_E_fam = exp_E_fam[exp_E_fam['family'] == family]
    exp_G_fam = exp_G_fam[exp_G_fam['family'] == family]

    paired = pd.merge(
        exp_E_fam[['strain','w_exp']].rename(columns={'w_exp':'w_E'}),
        exp_G_fam[['strain','w_exp']].rename(columns={'w_exp':'w_G'}),
        on='strain', how='inner'
    ).dropna(subset=['w_E','w_G'])
    return paired

def estimate_bivariate_params(paired_df, shrink_r=0.05, ridge=1e-6):
    if paired_df.empty or paired_df['w_E'].count() < 3:
        return None
    wE = paired_df['w_E'].astype(float).values
    wG = paired_df['w_G'].astype(float).values
    mu  = np.array([wE.mean(), wG.mean()], dtype=float)
    sdE = float(wE.std(ddof=1)) if len(wE) > 1 else 0.0
    sdG = float(wG.std(ddof=1)) if len(wG) > 1 else 0.0
    rho_hat = 0.0 if (sdE==0.0 or sdG==0.0) else float(np.corrcoef(wE, wG)[0,1])
    rho = (1.0 - shrink_r) * rho_hat
    cov = np.array([[sdE**2, rho*sdE*sdG],
                    [rho*sdE*sdG, sdG**2]], dtype=float)
    cov.flat[::3] += ridge
    return {'mu': mu, 'cov': cov, 'rho': rho}

def sample_bivariate(n, mu, cov, clip=(0.05, 3.0), seed=None):
    rng = np.random.default_rng(seed)
    S = rng.multivariate_normal(mu, cov, size=n)
    if clip is not None:
        S = np.clip(S, clip[0], clip[1])
    return S  # columns: [w_E, w_G]


def _validate_params(df_env):
    req = {'Vmax','K','c'}
    for env in ['E','G']:
        sub = df_env[df_env['Environment']==env]
        if not req.issubset(sub.columns):
            missing = list(req - set(sub.columns))
            raise ValueError(f"[{env}] missing columns: {missing}")
        bad = (~np.isfinite(sub['Vmax'])) | (~np.isfinite(sub['K'])) | (~np.isfinite(sub['c'])) \
              | (sub['Vmax'] < 0) | (sub['K'] < 0) | (sub['c'] < 0)
        if bad.any():
            print(sub.loc[bad, ['label','pair_idx','Environment','Vmax','K','c']].head(10))
            raise ValueError(f"[{env}] invalid parameter values detected.")


def get_existing_synthetic_labels(sheet, tab_name):
    """
    Return a set of (label, Environment) pairs already present in the sheet tab.
    Assumes the sheet has headers in row 1, with at least 'label' and 'Environment' columns.
    """
    ws = sheet.worksheet(tab_name)
    values = ws.get_all_values()
    if not values:
        return set()

    header = values[0]
    try:
        idx_label = header.index('label')
        idx_env   = header.index('Environment')
    except ValueError:
        # if these columns don't exist, nothing to dedupe against
        return set()

    seen = set()
    for row in values[1:]:
        # guard against short rows
        if len(row) <= max(idx_label, idx_env):
            continue
        lbl = row[idx_label].strip()
        env = row[idx_env].strip()
        # we only care about synthetic (_x) rows
        if lbl.endswith('_x0000') or '_x' in lbl:
            seen.add((lbl, env))
    return seen

def sanitize_for_export(df):
    """
    Return a copy of df where all NaN / inf / -inf are replaced with '' (empty string),
    so it can be JSON-serialized and pushed to Sheets.
    """
    clean = df.replace([np.inf, -np.inf], np.nan)
    clean = clean.fillna('')  # Sheets is happy with empty string cells
    return clean



def make_vertical_panel(pathFIGURES, family, sched_name, save=True, show=True):
    """
    Create a vertical panel of three figures for a given family and schedule.

    Parameters
    ----------
    pathFIGURES : str
        Directory where the figure pdf files are stored.
    family : str
        Name of the family (e.g. "dfr", "aa", "bla", "mix").
    sched_name : str
        Name of the schedule/environment (e.g. "anaerobiosis", "aerobiosis", "alternado").
    save : bool, optional
        If True, saves the combined panel as pdf in the same directory.
    show : bool, optional
        If True, displays the panel with plt.show().
    """

    # File templates
    filenames = [
        f"mean_daily_freq_{family}_{sched_name}.pdf",
        f"replicates_daily_freq_{family}_{sched_name}.pdf",
        f"freq_time_density_{family}_{sched_name}.pdf"
    ]

    # Build full paths
    filepaths = [os.path.join(pathFIGURES, fn) for fn in filenames]

    # Load images
    images = [mpimg.imread(fp) for fp in filepaths]

    # Create vertical panel
    fig, axes = plt.subplots(len(images), 1, figsize=(6, 12))
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")

    plt.tight_layout()

    # Save output
    if save:
        out_file = os.path.join(pathFIGURES, f"panel_{family}_{sched_name}.pdf")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved panel: {out_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def make_highlight_panel(pathFIGURES, family, sched_names=[], save=True, show=True):
    """
    Create a vertical panel with highlight density plots for all schedules of a given family.

    Parameters
    ----------
    pathFIGURES : str
        Directory where the figure pdf files are stored.
    family : str
        Name of the family (e.g. "dfr", "aa", "bla", "mix").
    save : bool, optional
        If True, saves the combined panel as pdf in the same directory.
    show : bool, optional
        If True, displays the panel with plt.show().
    """

    # Build file paths
    filenames = [
        f"freq_time_density_with_highlights_{family}_{sched_name}.pdf" for sched_name in sched_names
    ]
    filepaths = [os.path.join(pathFIGURES, fn) for fn in filenames]

    # Load images
    images = [mpimg.imread(fp) for fp in filepaths]

    # Plot vertically
    fig, axes = plt.subplots(len(images), 1, figsize=(6, 12))
    for ax, img, sched in zip(axes, images, sched_names):
        ax.imshow(img)
        ax.set_title(f"{sched}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    # Save
    if save:
        out_file = os.path.join(pathFIGURES, f"panel_highlight_{family}.pdf")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved panel: {out_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def make_vertical_multi_sched_panel(pathFIGURES, family, sched_names=[], prefix='_', save=True, show=True):
    """
    Create a vertical panel with one plot per schedule for a given family.

    Parameters
    ----------
    pathFIGURES : str
        Directory where the figure pdf files are stored.
    family : str
        Name of the family (e.g. "dfr", "aa", "bla", "mix").
    prefix : str
        File prefix (e.g. "freq_time_density_with_highlights_").
        The function will append family + sched_name + ".pdf".
    save : bool, optional
        If True, saves the combined panel as pdf in the same directory.
    show : bool, optional
        If True, displays the panel with plt.show().
    """


    # Build file paths
    filenames = [f"{prefix}{family}_{sched}.pdf" for sched in sched_names]
    filepaths = [os.path.join(pathFIGURES, fn) for fn in filenames]

    # Load images
    images = [mpimg.imread(fp) for fp in filepaths]

    # Plot vertically
    fig, axes = plt.subplots(len(images), 1, figsize=(6, 12))
    for ax, img, sched in zip(axes, images, sched_names):
        ax.imshow(img)
        #ax.set_title(sched, fontsize=10)
        ax.axis("off")

    plt.tight_layout()

    # Save
    if save:
        out_file = os.path.join(pathFIGURES, f"panel_{prefix}{family}.pdf")
        plt.savefig(out_file, dpi=300, bbox_inches="tight")
        print(f"Saved panel: {out_file}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


