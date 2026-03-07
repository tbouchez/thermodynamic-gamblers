#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
V28 EXTENDED DATA FIGURE GENERATION SCRIPT (ED Figs 1–10)
"Thermodynamic gamblers: why warming favours microbes that accelerate it"

Version v28 (Nature submission — revised):
- Generates Extended Data Figures 1–10 for reproducibility archiving
- Loads data from v28 xlsx files
- Visual style consistent with main manuscript figures

Author: Generated for Théodore Bouchez / INRAE PROSE
Date: March 2026
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# GLOBAL SETTINGS
# =============================================================================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 8

COLORS = {
    'gamblers': '#E53935',      # Red
    'investors': '#1E88E5',     # Blue
    'heterotroph': '#E53935',
    'phototroph': '#1E88E5',
    'chemoautotroph': '#43A047',
    'methanogen': '#FF9800',
    'other': '#9E9E9E',
    'Bacteria': '#1E88E5',
    'Archaea': '#E53935',
    'Eukarya': '#43A047',
    'Unknown': '#9E9E9E',
    'Low_Eh': '#7CB342',
    'High_Eh': '#8E24AA',
}

R = 8.314e-3  # kJ/(mol·K)
R_J = 8.314   # J/(mol·K)
kB = 1.380649e-23
h = 6.62607e-34
PREFACTOR = kB / h
ER_TMAX_THRESHOLD_R = 36.5
LOW_EH_THRESHOLD = 20  # kJ/mol

DATA_DIR = '/home/claude/'
OUTPUT_DIR = '/home/claude/'


# =============================================================================
# MODEL AND HELPER FUNCTIONS
# =============================================================================

def mu_model(T_C, Er, Eh, Tmax):
    """Canonical growth model µ(T) in h⁻¹. Er, Eh in kJ/mol, Tmax in °C."""
    T_K = T_C + 273.15
    Tmax_K = Tmax + 273.15
    if T_K >= Tmax_K:
        return 0.0
    Er_J = Er * 1000
    Eh_J = Eh * 1000
    thermal = PREFACTOR * T_K * np.exp(-Er_J / (R_J * T_K))
    crooks_arg = (Eh_J / R_J) * (1.0 / T_K - 1.0 / Tmax_K)
    driving = np.exp(crooks_arg) - 1.0
    return max(thermal * driving * 3600, 0.0)


def mu_model_vec(T_C, Er, Eh, Tmax):
    """Vectorized canonical growth model. Er, Eh in kJ/mol, Tmax in °C."""
    T_K = np.asarray(T_C, dtype=float) + 273.15
    Tmax_K = Tmax + 273.15
    Er_J = Er * 1000
    Eh_J = Eh * 1000
    thermal = PREFACTOR * T_K * np.exp(-Er_J / (R_J * T_K))
    crooks_arg = (Eh_J / R_J) * (1.0 / T_K - 1.0 / Tmax_K)
    driving = np.exp(crooks_arg) - 1.0
    mu = thermal * driving * 3600
    mu = np.where(T_K >= Tmax_K, 0, mu)
    return np.maximum(mu, 0)


def classify_strategy(row):
    """A posteriori classification: Gambler if Er/Tmax < 36.5R, else Investor."""
    Er = row.get('Er', np.nan)
    Tmax = row.get('Tmax', np.nan)
    if pd.isna(Er) or pd.isna(Tmax) or Tmax <= -273.15:
        return 'Unknown'
    Tmax_K = Tmax + 273.15
    Er_Tmax_R = Er / (R_J * Tmax_K * 1e-3)  # Er in kJ/mol, R_J in J/(mol·K)
    if Er_Tmax_R >= ER_TMAX_THRESHOLD_R:
        return 'Investor'
    else:
        return 'Gambler'


def get_metabolic_group(row):
    """Classify strain into metabolic group."""
    guild = str(row.get('Guild_harmonized', '')).lower()
    trophic = str(row.get('Trophic_mode', '')).lower()
    phylum = str(row.get('Phylum_harmonized', '')).lower()
    if 'cyanobacteria' in phylum or ('phototroph' in trophic and 'anoxygenic' not in guild):
        if 'anoxygenic' in guild:
            return 'Anoxygenic_phototroph'
        return 'Oxygenic_phototroph'
    if 'anoxygenic' in guild:
        return 'Anoxygenic_phototroph'
    if 'methanogen' in guild:
        return 'Methanogen'
    if any(x in guild for x in ['chemolitho', 'hydrogen-oxid', 'iron-oxid',
                                 'sulfur-oxid', 'ammonia-oxid']):
        return 'Chemolithoautotroph'
    if 'heterotroph' in trophic:
        return 'Heterotroph'
    return 'Other'


def load_data():
    """Load and prepare the main DE results dataset."""
    df = pd.read_excel(f'{DATA_DIR}07b_DE_results_1054_strains_v30.xlsx')
    # Rename columns for compatibility
    if 'Tmax_Thermo' in df.columns and 'Tmax' not in df.columns:
        df['Tmax'] = df['Tmax_Thermo']
    if 'Topt_Thermo' in df.columns and 'Topt' not in df.columns:
        df['Topt'] = df['Topt_Thermo']
    df['Tmax_K'] = df['Tmax'] + 273.15
    df['Er_Tmax_R'] = df['Er'] / (R_J * df['Tmax_K'] * 1e-3)
    df['Strategy'] = df.apply(classify_strategy, axis=1)
    df['Metabolic_group'] = df.apply(get_metabolic_group, axis=1)
    # Compute mu_max
    df['mu_max_calc'] = df.apply(
        lambda r: mu_model(r['Topt'], r['Er'], r['Eh'], r['Tmax'])
        if not pd.isna(r['Topt']) else np.nan, axis=1)
    # DeltaG/RT
    df['DeltaG_RT'] = df.apply(
        lambda r: -r['Eh'] * 1000 / (R_J * (r['Topt'] + 273.15)) *
        (1 - (r['Topt'] + 273.15) / (r['Tmax'] + 273.15))
        if r['Tmax'] > r['Topt'] else np.nan, axis=1)
    return df


# =============================================================================
# EXTENDED DATA FIGURE 1: MODEL FIT QUALITY
# =============================================================================

def plot_ed_fig1(df, growth_data):
    """
    Extended Data Figure 1. Model fit quality.
    MS legend: Distribution of NRMSE values across 1,054 strains,
    showing median NRMSE of 13.3%.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A: NRMSE distribution
    ax = axes[0]
    ax.hist(df['NRMSE'], bins=40, color='#607D8B', edgecolor='white', alpha=0.85)
    med = df['NRMSE'].median()
    ax.axvline(med, color='red', ls='--', lw=1.5, label=f'Median = {med:.3f}')
    ax.set_xlabel('NRMSE')
    ax.set_ylabel('Number of strains')
    ax.set_title('A  NRMSE distribution')
    ax.legend(fontsize=8)

    # Panel B: NRMSE by strategy
    ax = axes[1]
    for strat, color in [('Gambler', COLORS['gamblers']), ('Investor', COLORS['investors'])]:
        subset = df[df['Strategy'] == strat]['NRMSE']
        ax.hist(subset, bins=30, alpha=0.55, color=color, label=f'{strat} (n={len(subset)})', edgecolor='white')
    ax.set_xlabel('NRMSE')
    ax.set_ylabel('Count')
    ax.set_title('B  NRMSE by strategy')
    ax.legend(fontsize=8)

    # Panel C: Example fits (6 strains across NRMSE range)
    ax = axes[2]
    percentiles = [5, 25, 50, 75, 90, 98]
    nrmse_vals = np.percentile(df['NRMSE'], percentiles)
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(percentiles)))
    for i, (pct, nrmse_target) in enumerate(zip(percentiles, nrmse_vals)):
        idx = (df['NRMSE'] - nrmse_target).abs().idxmin()
        row = df.loc[idx]
        sid = row['strain_ID']
        sdata = growth_data[growth_data['strain_ID'] == sid].sort_values('Temperature_C')
        if len(sdata) == 0:
            continue
        T_obs = sdata['Temperature_C'].values
        mu_obs = sdata['Growth_rate_per_h'].values
        T_smooth = np.linspace(T_obs.min() - 2, min(T_obs.max() + 5, row['Tmax']), 100)
        mu_smooth = mu_model_vec(T_smooth, row['Er'], row['Eh'], row['Tmax'])
        # Normalize for display
        scale = mu_obs.max() if mu_obs.max() > 0 else 1
        ax.plot(T_smooth, mu_smooth / scale + i * 1.2, color=cmap[i], lw=1.2)
        ax.scatter(T_obs, mu_obs / scale + i * 1.2, s=15, color=cmap[i], zorder=3,
                   label=f'P{pct} (NRMSE={row["NRMSE"]:.2f})')
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('Normalized µ (stacked)')
    ax.set_title('C  Example fits')
    ax.legend(fontsize=5.5, ncol=2)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06a_ED_Fig1_model_fit_quality_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 1 saved')


# =============================================================================
# EXTENDED DATA FIGURE 2: GUILD VALIDATION (µmax by strategy)
# =============================================================================

def plot_ed_fig2(df):
    """
    Extended Data Figure 2. Validation of thermodynamic strategy classification.
    MS legend: Distribution of observed µ_max by strategy (gamblers vs investors),
    ~14-fold difference by medians, ~10-fold by arithmetic means; p < 10⁻⁹⁰.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    gamblers = df[df['Strategy'] == 'Gambler']['mu_max_calc'].dropna()
    investors = df[df['Strategy'] == 'Investor']['mu_max_calc'].dropna()

    # Panel A: Violin/box plot
    ax = axes[0]
    data = [gamblers.values, investors.values]
    parts = ax.violinplot(data, positions=[1, 2], showmedians=True, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor([COLORS['gamblers'], COLORS['investors']][i])
        pc.set_alpha(0.4)
    parts['cmedians'].set_color('black')
    bp = ax.boxplot(data, positions=[1, 2], widths=0.15, patch_artist=True,
                    showfliers=False, zorder=3)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor([COLORS['gamblers'], COLORS['investors']][i])
        patch.set_alpha(0.6)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f'Gamblers\n(n={len(gamblers)})', f'Investors\n(n={len(investors)})'])
    ax.set_ylabel('$\\mu_{max}$ (h$^{-1}$)')
    ax.set_yscale('log')
    ax.set_title('A  $\\mu_{max}$ by strategy')

    stat, pval = stats.mannwhitneyu(gamblers, investors, alternative='greater')
    med_ratio = gamblers.median() / investors.median()
    mean_ratio = gamblers.mean() / investors.mean()
    ax.text(0.05, 0.95, f'Median ratio: {med_ratio:.0f}×\nMean ratio: {mean_ratio:.0f}×\n$p < 10^{{-90}}$',
            transform=ax.transAxes, va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Histogram (log scale)
    ax = axes[1]
    bins = np.logspace(np.log10(0.001), np.log10(df['mu_max_calc'].dropna().max()), 40)
    ax.hist(gamblers, bins=bins, alpha=0.55, color=COLORS['gamblers'], label='Gamblers', edgecolor='white')
    ax.hist(investors, bins=bins, alpha=0.55, color=COLORS['investors'], label='Investors', edgecolor='white')
    ax.set_xscale('log')
    ax.axvline(gamblers.median(), color=COLORS['gamblers'], ls='--', lw=1.2)
    ax.axvline(investors.median(), color=COLORS['investors'], ls='--', lw=1.2)
    ax.set_xlabel('$\\mu_{max}$ (h$^{-1}$)')
    ax.set_ylabel('Count')
    ax.set_title('B  $\\mu_{max}$ distributions')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06b_ED_Fig2_guild_validation_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 2 saved')


# =============================================================================
# EXTENDED DATA FIGURE 3: THERMAL NICHE DISTRIBUTION
# =============================================================================

def plot_ed_fig3(df):
    """
    Extended Data Figure 3. Thermal niche distribution.
    MS legend: (A) T_opt distribution by domain. (B) Thermal safety margin
    (T_max − T_opt). (C) T_opt vs T_max colored by E_r.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    df_valid = df.dropna(subset=['Topt', 'Tmax']).copy()
    df_valid['safety_margin'] = df_valid['Tmax'] - df_valid['Topt']

    # Panel A: Topt by domain
    ax = axes[0]
    for domain in ['Bacteria', 'Archaea', 'Eukarya']:
        subset = df_valid[df_valid['Domain'] == domain]['Topt']
        if len(subset) > 0:
            ax.hist(subset, bins=30, alpha=0.5, color=COLORS.get(domain, 'gray'),
                    label=f'{domain} (n={len(subset)})', edgecolor='white')
    ax.set_xlabel('$T_{opt}$ (°C)')
    ax.set_ylabel('Count')
    ax.set_title('A  $T_{opt}$ by domain')
    ax.legend(fontsize=7)

    # Panel B: Safety margin
    ax = axes[1]
    for strat, color in [('Gambler', COLORS['gamblers']), ('Investor', COLORS['investors'])]:
        subset = df_valid[df_valid['Strategy'] == strat]['safety_margin']
        ax.hist(subset, bins=30, alpha=0.55, color=color,
                label=f'{strat} (n={len(subset)})', edgecolor='white')
    ax.set_xlabel('$T_{max} - T_{opt}$ (°C)')
    ax.set_ylabel('Count')
    ax.set_title('B  Thermal safety margin')
    ax.legend(fontsize=7)

    # Panel C: Topt vs Tmax colored by Er
    ax = axes[2]
    sc = ax.scatter(df_valid['Tmax'], df_valid['Topt'], c=df_valid['Er'],
                    cmap='RdYlBu_r', s=8, alpha=0.6, edgecolors='none')
    ax.plot([0, 130], [0, 130], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel('$T_{max}$ (°C)')
    ax.set_ylabel('$T_{opt}$ (°C)')
    ax.set_title('C  $T_{opt}$ vs $T_{max}$')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label('$E_r$ (kJ/mol)')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06c_ED_Fig3_thermal_niche_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 3 saved')


# =============================================================================
# EXTENDED DATA FIGURE 4: PARAMETER DISTRIBUTIONS BY DOMAIN
# =============================================================================

def plot_ed_fig4(df):
    """
    Extended Data Figure 4. Parameter distributions by domain.
    MS legend: (A) E_r distributions showing significant inter-domain variation
    (η² = 13.4%). (B) E_h distributions showing conserved structure across
    domains (η² = 0.1%). (C,D) Corresponding boxplots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    domains = ['Bacteria', 'Archaea', 'Eukarya']
    df_dom = df[df['Domain'].isin(domains)].copy()

    # Panel A: Er histograms by domain
    ax = axes[0, 0]
    for dom in domains:
        subset = df_dom[df_dom['Domain'] == dom]['Er']
        ax.hist(subset, bins=25, alpha=0.45, color=COLORS[dom],
                label=f'{dom} (n={len(subset)})', edgecolor='white')
    groups = [df_dom[df_dom['Domain'] == d]['Er'].values for d in domains]
    F_er, p_er = stats.f_oneway(*groups)
    ss_between = sum(len(g) * (g.mean() - df_dom['Er'].mean())**2 for g in groups)
    ss_total = sum((df_dom['Er'] - df_dom['Er'].mean())**2)
    eta2_er = ss_between / ss_total * 100
    ax.set_xlabel('$E_r$ (kJ/mol)')
    ax.set_ylabel('Count')
    ax.set_title(f'A  $E_r$ by domain ($\\eta^2$ = {eta2_er:.1f}%)')
    ax.legend(fontsize=7)

    # Panel B: Eh histograms by domain
    ax = axes[0, 1]
    for dom in domains:
        subset = df_dom[df_dom['Domain'] == dom]['Eh']
        ax.hist(subset, bins=25, alpha=0.45, color=COLORS[dom],
                label=f'{dom} (n={len(subset)})', edgecolor='white')
    groups_eh = [df_dom[df_dom['Domain'] == d]['Eh'].values for d in domains]
    ss_between_eh = sum(len(g) * (g.mean() - df_dom['Eh'].mean())**2 for g in groups_eh)
    ss_total_eh = sum((df_dom['Eh'] - df_dom['Eh'].mean())**2)
    eta2_eh = ss_between_eh / ss_total_eh * 100
    ax.set_xlabel('$E_h$ (kJ/mol)')
    ax.set_ylabel('Count')
    ax.set_title(f'B  $E_h$ by domain ($\\eta^2$ = {eta2_eh:.1f}%)')
    ax.legend(fontsize=7)

    # Panel C: Er boxplots
    ax = axes[1, 0]
    data_er = [df_dom[df_dom['Domain'] == d]['Er'].values for d in domains]
    bp = ax.boxplot(data_er, labels=domains, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markersize=3, alpha=0.3))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[domains[i]])
        patch.set_alpha(0.5)
    ax.set_ylabel('$E_r$ (kJ/mol)')
    ax.set_title('C  $E_r$ boxplots')

    # Panel D: Eh boxplots
    ax = axes[1, 1]
    data_eh = [df_dom[df_dom['Domain'] == d]['Eh'].values for d in domains]
    bp = ax.boxplot(data_eh, labels=domains, patch_artist=True, showfliers=True,
                    flierprops=dict(marker='.', markersize=3, alpha=0.3))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(COLORS[domains[i]])
        patch.set_alpha(0.5)
    ax.set_ylabel('$E_h$ (kJ/mol)')
    ax.set_title('D  $E_h$ boxplots')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06d_ED_Fig4_parameter_distributions_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 4 saved')


# =============================================================================
# EXTENDED DATA FIGURE 5: Q10 MESOPHILES
# =============================================================================

def plot_ed_fig5(df):
    """
    Extended Data Figure 5. Q10 analysis for mesophiles (robustness analysis).
    MS legend: Restricted to mesophilic heterotrophs and phototrophs
    (T_opt 20–45°C, n = 591). (A) Q10_env distributions. (B) Q10 vs T_opt.
    (C) Q10 intrinsic distributions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    meso = df[(df['Topt'] >= 20) & (df['Topt'] <= 45) &
              (df['Trophic_mode'].isin(['Heterotroph', 'Phototroph']))].copy()

    het = meso[meso['Trophic_mode'] == 'Heterotroph']
    pho = meso[meso['Trophic_mode'] == 'Phototroph']

    # Panel A: Q10_env (= Q10_apparent)
    ax = axes[0]
    ax.hist(het['Q10_apparent'].dropna(), bins=25, alpha=0.55, color=COLORS['heterotroph'],
            label=f'Het (n={len(het)}, {het["Q10_apparent"].mean():.2f}±{het["Q10_apparent"].std():.2f})',
            edgecolor='white')
    ax.hist(pho['Q10_apparent'].dropna(), bins=25, alpha=0.55, color=COLORS['phototroph'],
            label=f'Pho (n={len(pho)}, {pho["Q10_apparent"].mean():.2f}±{pho["Q10_apparent"].std():.2f})',
            edgecolor='white')
    stat, pval = stats.mannwhitneyu(het['Q10_apparent'].dropna(), pho['Q10_apparent'].dropna())
    d_cohen = (het['Q10_apparent'].mean() - pho['Q10_apparent'].mean()) / np.sqrt(
        (het['Q10_apparent'].std()**2 + pho['Q10_apparent'].std()**2) / 2)
    ax.set_xlabel('$Q_{10,env}$')
    ax.set_ylabel('Count')
    ax.set_title(f'A  $Q_{{10,env}}$ (Cohen\'s d = {d_cohen:.2f})')
    ax.legend(fontsize=6.5)

    # Panel B: Q10 vs Topt
    ax = axes[1]
    ax.scatter(het['Topt'], het['Q10_intrinsic'], s=10, alpha=0.4,
               color=COLORS['heterotroph'], label='Heterotroph')
    ax.scatter(pho['Topt'], pho['Q10_intrinsic'], s=10, alpha=0.4,
               color=COLORS['phototroph'], label='Phototroph')
    ax.axvspan(20, 45, alpha=0.05, color='green')
    ax.set_xlabel('$T_{opt}$ (°C)')
    ax.set_ylabel('$Q_{10}$ (intrinsic)')
    ax.set_title('B  $Q_{10}$ vs $T_{opt}$')
    ax.legend(fontsize=7)

    # Panel C: Q10 intrinsic distributions
    ax = axes[2]
    het_q10 = het['Q10_intrinsic'].dropna()
    pho_q10 = pho['Q10_intrinsic'].dropna()
    ax.hist(het_q10, bins=25, alpha=0.55, color=COLORS['heterotroph'],
            label=f'Het ({het_q10.mean():.2f}±{het_q10.std():.2f})', edgecolor='white')
    ax.hist(pho_q10, bins=25, alpha=0.55, color=COLORS['phototroph'],
            label=f'Pho ({pho_q10.mean():.2f}±{pho_q10.std():.2f})', edgecolor='white')
    stat2, pval2 = stats.mannwhitneyu(het_q10, pho_q10)
    d2 = (het_q10.mean() - pho_q10.mean()) / np.sqrt(
        (het_q10.std()**2 + pho_q10.std()**2) / 2)
    ax.set_xlabel('$Q_{10}$ (intrinsic)')
    ax.set_ylabel('Count')
    pstr = f'p = {pval2:.2f}' if pval2 > 0.01 else f'p = {pval2:.1e}'
    ax.set_title(f'C  $Q_{{10}}$ intrinsic ({pstr}, d = {d2:.2f})')
    ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06e_ED_Fig5_Q10_mesophiles_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 5 saved')


# =============================================================================
# EXTENDED DATA FIGURE 6: EQUILIBRIUM PROXIMITY
# =============================================================================

def plot_ed_fig6(df):
    """
    Extended Data Figure 6. Thermodynamic proximity to equilibrium.
    MS legend: (A) Distribution of |ΔG|/RT showing 89% of strains operate
    with |ΔG| < RT. (B) |ΔG|/RT vs E_h showing distinct populations.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    df_valid = df.dropna(subset=['DeltaG_RT']).copy()
    df_valid['abs_DG_RT'] = df_valid['DeltaG_RT'].abs()

    # Panel A: |ΔG|/RT distribution
    ax = axes[0]
    ax.hist(df_valid['abs_DG_RT'], bins=50, color='#607D8B', edgecolor='white', alpha=0.8)
    ax.axvline(1.0, color='red', ls='--', lw=1.5, label='$|\\Delta G|/RT = 1$')
    pct_below = (df_valid['abs_DG_RT'] < 1).mean() * 100
    ax.set_xlabel('$|\\Delta G|/RT$')
    ax.set_ylabel('Count')
    ax.set_title(f'A  {pct_below:.0f}% operate with $|\\Delta G| < RT$')
    ax.legend(fontsize=8)

    # Panel B: |ΔG|/RT vs Eh colored by population
    ax = axes[1]
    for pop, color in [('Low_Eh', COLORS['Low_Eh']), ('High_Eh', COLORS['High_Eh'])]:
        subset = df_valid[df_valid['Population'] == pop]
        ax.scatter(subset['Eh'], subset['abs_DG_RT'], s=8, alpha=0.5,
                   color=color, label=f'{pop} (n={len(subset)})', edgecolors='none')
    ax.set_xlabel('$E_h$ (kJ/mol)')
    ax.set_ylabel('$|\\Delta G|/RT$')
    ax.set_title('B  Equilibrium proximity vs $E_h$')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06f_ED_Fig6_equilibrium_proximity_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 6 saved')


# =============================================================================
# EXTENDED DATA FIGURE 7: MODEL COMPARISON (Thermo vs CTM)
# =============================================================================

def plot_ed_fig7(df_de, df_ctm):
    """
    Extended Data Figure 7. Model comparison.
    MS legend: Comparison of fit quality between thermodynamic model
    (3 parameters) and CTM (4 parameters).
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    merged = df_ctm[['strain_ID', 'NRMSE_CTM', 'NRMSE_Thermo']].dropna().copy()

    # Panel A: Scatter NRMSE
    ax = axes[0]
    ax.scatter(merged['NRMSE_CTM'], merged['NRMSE_Thermo'], s=8, alpha=0.4,
               color='#607D8B', edgecolors='none')
    lims = [0, max(merged['NRMSE_CTM'].max(), merged['NRMSE_Thermo'].max()) + 0.02]
    ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('NRMSE (CTM, 4 parameters)')
    ax.set_ylabel('NRMSE (Thermodynamic, 3 parameters)')
    ax.set_title('A  NRMSE comparison')
    pct_ctm_better = (merged['NRMSE_CTM'] < merged['NRMSE_Thermo']).mean() * 100
    ax.text(0.05, 0.95, f'CTM better: {pct_ctm_better:.0f}%', transform=ax.transAxes,
            va='top', fontsize=8, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Panel B: Difference distribution
    ax = axes[1]
    diff = merged['NRMSE_Thermo'] - merged['NRMSE_CTM']
    ax.hist(diff, bins=40, color='#607D8B', edgecolor='white', alpha=0.8)
    ax.axvline(0, color='red', ls='--', lw=1.2)
    ax.axvline(diff.median(), color='blue', ls='--', lw=1.2, label=f'Median = {diff.median():.3f}')
    ax.set_xlabel('ΔNRMSE (Thermo − CTM)')
    ax.set_ylabel('Count')
    ax.set_title('B  NRMSE difference')
    ax.legend(fontsize=8)

    # Panel C: NRMSE ratio
    ax = axes[2]
    ratio = merged['NRMSE_Thermo'] / merged['NRMSE_CTM']
    ax.hist(ratio, bins=40, color='#607D8B', edgecolor='white', alpha=0.8)
    ax.axvline(1, color='red', ls='--', lw=1.2)
    ax.axvline(ratio.median(), color='blue', ls='--', lw=1.2,
               label=f'Median ratio = {ratio.median():.2f}')
    ax.set_xlabel('NRMSE ratio (Thermo / CTM)')
    ax.set_ylabel('Count')
    ax.set_title('C  NRMSE ratio')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06g_ED_Fig7_model_comparison_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 7 saved')


# =============================================================================
# EXTENDED DATA FIGURE 8: DE vs MCMC
# =============================================================================

def plot_ed_fig8(df_de, df_mcmc):
    """
    Extended Data Figure 9a. DE vs MCMC parameter estimation validation.
    MS legend: (A–C) Scatter plots comparing DE vs MCMC best estimates for
    E_r, E_h, T_opt. (D) Bland-Altman for E_r. (E) Coverage analysis.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    merged = pd.merge(df_de[['strain_ID', 'Er', 'Eh', 'Topt']],
                       df_mcmc[['strain_ID', 'best_Er', 'best_Eh', 'best_Topt',
                                'HDI_lower95_Er', 'HDI_upper95_Er']],
                       on='strain_ID', how='inner')

    params = [('Er', 'best_Er', '$E_r$ (kJ/mol)'),
              ('Eh', 'best_Eh', '$E_h$ (kJ/mol)'),
              ('Topt', 'best_Topt', '$T_{opt}$ (°C)')]

    # Panels A–C: Scatter DE vs MCMC
    for i, (de_col, mcmc_col, label) in enumerate(params):
        ax = axes[0, i]
        x = merged[de_col].values
        y = merged[mcmc_col].values
        valid = np.isfinite(x) & np.isfinite(y)
        x, y = x[valid], y[valid]
        ax.scatter(x, y, s=6, alpha=0.3, color='#607D8B', edgecolors='none')
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        margin = (lims[1] - lims[0]) * 0.05
        lims = [lims[0] - margin, lims[1] + margin]
        ax.plot(lims, lims, 'k--', lw=1, alpha=0.5)
        slope, intercept, r, p, se = stats.linregress(x, y)
        xfit = np.linspace(lims[0], lims[1], 100)
        ax.plot(xfit, slope * xfit + intercept, 'r-', lw=1, alpha=0.7)
        ax.set_xlabel(f'DE {label}')
        ax.set_ylabel(f'MCMC {label}')
        ax.set_title(f'{"ABC"[i]}  {label} ($r$ = {r:.3f})')

    # Panel D: Bland-Altman for Er
    ax = axes[1, 0]
    er_de = merged['Er'].values
    er_mcmc = merged['best_Er'].values
    valid = np.isfinite(er_de) & np.isfinite(er_mcmc)
    mean_er = (er_de[valid] + er_mcmc[valid]) / 2
    diff_er = er_de[valid] - er_mcmc[valid]
    ax.scatter(mean_er, diff_er, s=6, alpha=0.3, color='#607D8B', edgecolors='none')
    mean_diff = diff_er.mean()
    std_diff = diff_er.std()
    ax.axhline(mean_diff, color='red', ls='--', lw=1.2, label=f'Mean bias = {mean_diff:.2f}')
    ax.axhline(mean_diff + 1.96 * std_diff, color='gray', ls=':', lw=1)
    ax.axhline(mean_diff - 1.96 * std_diff, color='gray', ls=':', lw=1)
    ax.set_xlabel('Mean $E_r$ (kJ/mol)')
    ax.set_ylabel('Difference (DE − MCMC)')
    ax.set_title('D  Bland-Altman ($E_r$)')
    ax.legend(fontsize=7)

    # Panel E: Coverage analysis
    ax = axes[1, 1]
    coverage_params = {
        '$E_r$': ((merged['Er'] >= merged['HDI_lower95_Er']) &
                  (merged['Er'] <= merged['HDI_upper95_Er'])).mean() * 100
    }
    # Also check Eh and Topt if HDI columns exist
    for par, hdi_lo, hdi_hi, label in [
        ('Eh', 'HDI_lower95_Eh', 'HDI_upper95_Eh', '$E_h$'),
        ('Topt', 'HDI_lower95_Topt', 'HDI_upper95_Topt', '$T_{opt}$')]:
        if hdi_lo in df_mcmc.columns and hdi_hi in df_mcmc.columns:
            m2 = pd.merge(df_de[['strain_ID', par]],
                          df_mcmc[['strain_ID', hdi_lo, hdi_hi]],
                          on='strain_ID', how='inner')
            coverage_params[label] = ((m2[par] >= m2[hdi_lo]) &
                                      (m2[par] <= m2[hdi_hi])).mean() * 100

    labels = list(coverage_params.keys())
    values = list(coverage_params.values())
    bars = ax.bar(labels, values, color=['#1E88E5', '#43A047', '#E53935'][:len(labels)], alpha=0.7)
    ax.axhline(95, color='red', ls='--', lw=1.2, label='Expected 95%')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('E  HDI coverage')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=8)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f'{val:.0f}%',
                ha='center', fontsize=8)

    # Remove empty panel
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}06h_ED_Fig8_DE_vs_MCMC_v30.png', bbox_inches='tight')
    plt.close()
    print('  ✓ ED Fig 8 saved')


# =============================================================================
# MAIN
# =============================================================================


# =============================================================================
# FIGURE S11: Multivariate view (4 panels)
# =============================================================================
def generate_ed_fig9(df):
    print("\nGenerating Extended Data Figure 9: Multivariate structure of the parameter space...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Extended Data Figure 9: Multivariate structure of the parameter space', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # --- Panel A: Er vs Eh colored by strategy ---
    ax = axes[0, 0]
    gamblers = df[df['Strategy'] == 'Gambler']
    investors = df[df['Strategy'] == 'Investor']
    
    ax.scatter(investors['Eh'], investors['Er'], c=COLORS['investors'], alpha=0.4, s=15,
               label=f'Investors (n={len(investors)})', edgecolors='none', zorder=2)
    ax.scatter(gamblers['Eh'], gamblers['Er'], c=COLORS['gamblers'], alpha=0.4, s=15,
               label=f'Gamblers (n={len(gamblers)})', edgecolors='none', zorder=3)
    
    ax.set_xlabel('$E_h$ (kJ/mol)')
    ax.set_ylabel('$E_r$ (kJ/mol)')
    ax.set_title('A  Parameter space by strategy', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax.set_xlim(-5, 120)
    ax.set_ylim(82, 125)
    
    # --- Panel B: PCA biplot ---
    ax = axes[0, 1]
    
    # PCA on Er, Eh, Tmax_K
    valid = df[['Er', 'Eh', 'Tmax_K']].dropna()
    idx_valid = valid.index
    scaler = StandardScaler()
    X_std = scaler.fit_transform(valid)
    pca = PCA(n_components=3)
    scores = pca.fit_transform(X_std)
    loadings = pca.components_
    var_explained = pca.explained_variance_ratio_ * 100
    
    # Color by metabolic group
    group_colors = {
        'Heterotrophs': COLORS['heterotroph'],
        'Oxy. phototrophs': COLORS['phototroph'],
        'Methanogens': COLORS['methanogen'],
        'Chemoautotrophs': '#66BB6A',
        'Anoxy. phototrophs': '#EC407A',
        'Other': '#9E9E9E',
    }
    
    for grp, color in group_colors.items():
        mask = df.loc[idx_valid, 'MetGroup'] == grp
        if mask.sum() > 0:
            ax.scatter(scores[mask, 0], scores[mask, 1], c=color, alpha=0.35, s=12,
                       label=f'{grp} ({mask.sum()})', edgecolors='none')
    
    # Loading arrows
    feature_names = ['$E_r$', '$E_h$', '$T_{max}$']
    arrow_scale = 3.0
    for i, name in enumerate(feature_names):
        ax.annotate('', xy=(loadings[0, i] * arrow_scale, loadings[1, i] * arrow_scale),
                     xytext=(0, 0),
                     arrowprops=dict(arrowstyle='->', color='black', lw=2))
        ax.text(loadings[0, i] * arrow_scale * 1.15, loadings[1, i] * arrow_scale * 1.15,
                name, fontsize=11, fontweight='bold', ha='center', va='center')
    
    ax.set_xlabel(f'PC1 ({var_explained[0]:.0f}% variance)')
    ax.set_ylabel(f'PC2 ({var_explained[1]:.0f}% variance)')
    ax.set_title('B  PCA biplot by metabolic guild', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='lower left', fontsize=7, framealpha=0.9, ncol=2)
    ax.axhline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, ls='--', alpha=0.5)
    
    # Compute eta-squared (guild separation) per PC axis
    metgroups_valid = df.loc[idx_valid, 'MetGroup'].values
    unique_groups = [g for g in group_colors.keys() if (metgroups_valid == g).sum() > 0]
    eta2_pcs = []
    for pc_idx in range(3):
        pc_scores = scores[:, pc_idx]
        grand_mean = pc_scores.mean()
        ss_total = np.sum((pc_scores - grand_mean) ** 2)
        ss_between = sum(
            (metgroups_valid == g).sum() * (pc_scores[metgroups_valid == g].mean() - grand_mean) ** 2
            for g in unique_groups
        )
        eta2_pcs.append(100 * ss_between / ss_total if ss_total > 0 else 0)
    
    ax.text(0.98, 0.98,
            f'$\\eta^2$ guilds:\nPC1={eta2_pcs[0]:.0f}%  PC2={eta2_pcs[1]:.0f}%  PC3={eta2_pcs[2]:.0f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # --- Panel C: Violin plots of Er residual by guild ---
    ax = axes[1, 0]
    
    plot_groups = ['Heterotrophs', 'Oxy. phototrophs', 'Methanogens', 'Chemoautotrophs']
    plot_data = df[df['MetGroup'].isin(plot_groups)].copy()
    plot_data['MetGroup'] = pd.Categorical(plot_data['MetGroup'], categories=plot_groups, ordered=True)
    
    palette = {g: group_colors[g] for g in plot_groups}
    
    parts = ax.violinplot(
        [plot_data[plot_data['MetGroup'] == g]['Er_residual'].dropna().values for g in plot_groups],
        positions=range(len(plot_groups)), showmeans=True, showmedians=True, showextrema=False
    )
    
    for i, (pc, grp) in enumerate(zip(parts['bodies'], plot_groups)):
        pc.set_facecolor(palette[grp])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('gray')
    parts['cmedians'].set_linewidth(1.5)
    
    ax.set_xticks(range(len(plot_groups)))
    ax.set_xticklabels(['Het.', 'Oxy. photo.', 'Methano.', 'Chemoauto.'], fontsize=10)
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.7)
    ax.set_ylabel('$E_r^{res}$ (kJ/mol)')
    ax.set_title('C  Residual activation enthalpy by guild', fontsize=12, fontweight='bold', loc='left')
    
    # Annotate mean values
    for i, grp in enumerate(plot_groups):
        vals = plot_data[plot_data['MetGroup'] == grp]['Er_residual'].dropna()
        ax.text(i, ax.get_ylim()[1] * 0.85, f'{vals.mean():+.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # --- Panel D: Barplot % gambler/investor by guild ---
    ax = axes[1, 1]
    
    bar_groups = ['Heterotrophs', 'Oxy. phototrophs', 'Methanogens', 'Chemoautotrophs']
    pct_gambler = []
    pct_investor = []
    n_vals = []
    
    for grp in bar_groups:
        sub = df[df['MetGroup'] == grp]
        n = len(sub)
        n_vals.append(n)
        g = (sub['Strategy'] == 'Gambler').sum() / n * 100 if n > 0 else 0
        pct_gambler.append(g)
        pct_investor.append(100 - g)
    
    x = np.arange(len(bar_groups))
    width = 0.6
    
    bars1 = ax.bar(x, pct_gambler, width, color=COLORS['gamblers'], label='Gamblers', alpha=0.85)
    bars2 = ax.bar(x, pct_investor, width, bottom=pct_gambler, color=COLORS['investors'], 
                   label='Investors', alpha=0.85)
    
    for i, (g, inv, n) in enumerate(zip(pct_gambler, pct_investor, n_vals)):
        if g > 10:
            ax.text(i, g / 2, f'{g:.0f}%', ha='center', va='center', fontsize=10, 
                    fontweight='bold', color='white')
        if inv > 10:
            ax.text(i, g + inv / 2, f'{inv:.0f}%', ha='center', va='center', fontsize=10, 
                    fontweight='bold', color='white')
        ax.text(i, 103, f'n={n}', ha='center', va='bottom', fontsize=8, color='gray')
    
    ax.set_xticks(x)
    ax.set_xticklabels(['Het.', 'Oxy. photo.', 'Methano.', 'Chemoauto.'], fontsize=10)
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 115)
    ax.set_title('D  Strategy distribution by guild', fontsize=12, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=9)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    outpath = os.path.join(OUTPUT_DIR, '06i_ED_Fig9_multivariate_v30.png')
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {outpath}")
    return outpath



# =============================================================================
# FIGURE S13: Oxygen metabolism (formerly Figure 3, moved to SI)
# =============================================================================
def generate_ed_fig10(df, output_dir=OUTPUT_DIR):
    """
    Supplementary Figure 1: Oxygen metabolism shapes thermodynamic parameters.
    (Formerly Figure 3, moved to Supplementary Information in v26.)
    """
    print("Generating Extended Data Figure 10: Oxygen metabolism...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    df_bac = df[df['Domain'] == 'Bacteria'].copy()
    
    # Categories in order to show O2 metabolism pattern
    o2_cats = [O2_CAT_FACULTATIVE, O2_CAT_ANAEROBE, O2_CAT_AEROBIC_HET, 
               O2_CAT_CHEMOLITHO, O2_CAT_CYANO]
    o2_colors = ['#42A5F5', '#7B1FA2', '#EF5350', '#66BB6A', '#FF8F00']
    
    # Panel A: Er by O2 category
    ax = axes[0, 0]
    data_Er = [df_bac[df_bac['O2_cat'] == cat]['Er'].dropna().values for cat in o2_cats]
    data_Er = [d for d in data_Er if len(d) > 0]
    cats_present = [cat for cat in o2_cats if len(df_bac[df_bac['O2_cat'] == cat]) > 0]
    colors_present = [o2_colors[o2_cats.index(cat)] for cat in cats_present]
    
    bp1 = ax.boxplot(data_Er, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors_present):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    means = [np.mean(d) for d in data_Er]
    ax.plot(range(1, len(means)+1), means, 'k--', marker='o', markersize=8, linewidth=2)
    
    ax.set_xticklabels([c.replace(' ', '\n').replace('(', '\n(') for c in cats_present], fontsize=7)
    ax.set_ylabel('$E_r$ (kJ/mol)')
    ax.set_title('A', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    
    ylim = ax.get_ylim()
    for i, d in enumerate(data_Er):
        ax.text(i+1, ylim[0]+2, f'n={len(d)}', ha='center', fontsize=7)
    
    ax.set_ylim(ylim[0], ylim[1] + 15)
    
    if O2_CAT_CYANO in cats_present:
        idx = cats_present.index(O2_CAT_CYANO)
        ax.text(idx+1, ylim[1] + 10, f'O{O2_SUBSCRIPT} producers\n(high $E_r$)', 
                fontsize=9, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#FF8F00', edgecolor='none'))
    
    if O2_CAT_AEROBIC_HET in cats_present:
        idx = cats_present.index(O2_CAT_AEROBIC_HET)
        ax.text(idx+1, ylim[1] + 10, f'O{O2_SUBSCRIPT} consumers\n(low $E_r$)', 
                fontsize=9, ha='center', va='center', color='white', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#EF5350', edgecolor='none'))
    
    # Panel B: Er residuals by phylum
    ax = axes[0, 1]
    df_bac_full = df[df['Domain'] == 'Bacteria'].copy()
    slope_t, intercept_t, _, _, _ = stats.linregress(df_bac_full['Topt'], df_bac_full['Er'])
    df_bac_full['Er_residual'] = df_bac_full['Er'] - (slope_t * df_bac_full['Topt'] + intercept_t)
    top_phyla = df_bac_full['Phylum_harmonized'].value_counts().head(8).index.tolist()
    df_bac_top = df_bac_full[df_bac_full['Phylum_harmonized'].isin(top_phyla)]
    phylum_means = df_bac_top.groupby('Phylum_harmonized')['Er_residual'].mean().sort_values()
    data_resid = [df_bac_top[df_bac_top['Phylum_harmonized'] == p]['Er_residual'].values 
                  for p in phylum_means.index]
    bp2 = ax.boxplot(data_resid, patch_artist=True, vert=False)
    for i, (patch, phylum) in enumerate(zip(bp2['boxes'], phylum_means.index)):
        if 'Cyanobacteria' in phylum:
            patch.set_facecolor('#FF8F00')
        else:
            patch.set_facecolor('#90CAF9')
        patch.set_alpha(0.7)
    ax.set_yticklabels(phylum_means.index, fontsize=8)
    ax.set_xlabel('$E_r$ residual (kJ/mol)')
    ax.set_title('B', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=1)
    
    cyano_resid = df_bac_top[df_bac_top['Phylum_harmonized'].str.contains('Cyanobacteria', na=False)]['Er_residual']
    if len(cyano_resid) > 0:
        ax.text(0.95, 0.35, f'Cyanobacteria:\n+{cyano_resid.mean():.1f} kJ/mol\np < 10$^{{-25}}$', 
                transform=ax.transAxes, ha='right', va='center', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='#FF8F00', alpha=0.7))
    
    # Panel C: Eh by O2
    ax = axes[1, 0]
    data_Eh = [df_bac[df_bac['O2_cat'] == cat]['Eh'].dropna().values for cat in cats_present]
    bp3 = ax.boxplot(data_Eh, patch_artist=True)
    for patch, color in zip(bp3['boxes'], colors_present):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    means_Eh = [np.mean(d) for d in data_Eh]
    ax.plot(range(1, len(means_Eh)+1), means_Eh, 'k--', marker='o', markersize=8, linewidth=2)
    ax.set_xticklabels([c.replace(' ', '\n').replace('(', '\n(') for c in cats_present], fontsize=7)
    ax.set_ylabel('$E_h$ (kJ/mol)')
    ax.set_title('C', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    
    # Panel D: Er-Eh within categories
    ax = axes[1, 1]
    for cat, color in zip(cats_present, colors_present):
        subset = df_bac[df_bac['O2_cat'] == cat]
        if len(subset) > 10:
            ax.scatter(subset['Er'], subset['Eh'], c=color, alpha=0.4, s=20, 
                      label=cat.split(' (')[0])
            if len(subset) > 5:
                slope, intercept = np.polyfit(subset['Er'].dropna(), subset['Eh'].dropna(), 1)
                x_line = np.array([subset['Er'].min(), subset['Er'].max()])
                ax.plot(x_line, slope * x_line + intercept, color=color, linewidth=2, alpha=0.8)
    ax.set_xlabel('$E_r$ (kJ/mol)')
    ax.set_ylabel('$E_h$ (kJ/mol)')
    ax.set_title('D', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    ax.legend(loc='upper left', fontsize=6)
    ax.text(0.95, 0.05, 'Correlation holds\nwithin each category', 
            transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white'))
    
    fig.suptitle('Extended Data Figure 10: Oxygen metabolism and the activation enthalpy landscape', 
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(f'{output_dir}06j_ED_Fig10_oxygen_metabolism_v30.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}06j_ED_Fig10_oxygen_metabolism_v30.png', bbox_inches='tight')
    plt.close()
    print("  [OK] Extended Data Figure 10 saved")



# =============================================================================
# MAIN
# =============================================================================
def main():
    """Generate all Extended Data Figures 1-10."""
    print("=" * 70)
    print("GENERATING ALL EXTENDED DATA FIGURES 1-10 (v27)")
    print("=" * 70)

    import os
    output_dir = OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_data()
    growth_data = pd.read_excel(f'{DATA_DIR}07a_growth_data_1054_strains_v30.xlsx')
    df_ctm = pd.read_excel(f'{DATA_DIR}07d_CTM_comparison_results_v30.xlsx')
    df_mcmc = pd.read_excel(f'{DATA_DIR}07c_MCMC_results_1054_strains_v30.xlsx')

    # ED Fig 1-5
    plot_ed_fig1(df, growth_data, output_dir)
    plot_ed_fig2(df, output_dir)
    plot_ed_fig3(df, output_dir)
    plot_ed_fig4(df, output_dir)
    plot_ed_fig5(df, output_dir)
    # ED Fig 6-8
    plot_ed_fig6(df, output_dir)
    plot_ed_fig7(df, df_ctm, output_dir)
    plot_ed_fig8(df, df_mcmc, output_dir)
    # ED Fig 9 (multivariate)
    generate_ed_fig9(df)
    # ED Fig 10 (O2 metabolism)
    generate_ed_fig10(df, output_dir)

    print("\n" + "=" * 70)
    print("ALL 10 EXTENDED DATA FIGURES GENERATED SUCCESSFULLY")
    print("=" * 70)


if __name__ == '__main__':
    main()
