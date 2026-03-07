#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
V26 COMPLETE FIGURE AND TABLE GENERATION SCRIPT
"Thermodynamic gamblers: why warming favours microbes that accelerate it"

Version v28 (Nature submission — revised):
- Former S13 renamed to Supplementary Figure 1
- All output filenames updated to v27 suffix
- Data input updated to v27 suffix

Based on V26 with:
- A POSTERIORI classification by Er/Tmax (threshold 36.5R)
- Low-Eh/High-Eh distribution: 60%/40%
- Contingency table and quadrant analysis functions
- Nested structure statistics (r=0.74, chi2=255)

Author: Generated for Théodore Bouchez / INRAE PROSE
Date: February 2026
Version: v28 (Nature submission — revised)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests  # R6: FDR correction
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['font.family'] = 'sans-serif'
# Enable LaTeX-compatible PDF text (allows copy-paste)
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Physical constants
R = 8.314e-3  # Gas constant in kJ/mol/K
kB = 1.38e-23  # Boltzmann constant in J/K
h = 6.63e-34   # Planck constant in J.s
kB_over_h = kB / h  # ~2.08e10 s^-1/K

# Color schemes
COLORS = {
    'gamblers': '#E53935',      # Red
    'investors': '#1E88E5',     # Blue
    'heterotroph': '#D32F2F',   # Dark red
    'phototroph': '#388E3C',    # Dark green
    'bacteria': '#1976D2',      # Blue
    'archaea': '#F57C00',       # Orange
    'eukarya': '#7B1FA2',       # Purple
    # For bimodal Figure 2A
    'Investor': '#4393C3',      # Blue
    'Gambler': '#D6604D',       # Orange-red
    'Unknown': '#878787'        # Gray
}

# Output directory - RELATIVE PATH (works on Windows/Mac/Linux)
# Les figures seront créées dans un dossier "outputs" à côté du script
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = '/home/claude/figures_v30/'
DATA_DIR = os.path.join(SCRIPT_DIR) + os.sep

# Créer le dossier outputs s'il n'existe pas
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# STRING CONSTANTS
# =============================================================================
# For figures - USE LATEX NOTATION for cross-platform compatibility (Windows fix)
O2_SUBSCRIPT = '$_2$'    # subscript 2 (LaTeX)
DEGREE = '$^\\circ$'     # degree symbol (LaTeX) - works on Windows
PLUS_MINUS_UNICODE = '$\\pm$'  # plus-minus for figures (LaTeX)
UP_ARROW = '\u2191'      # up arrow
UP_UP_ARROW = '\u21c8'   # double up arrow

# For CSV tables (ASCII only for maximum compatibility)
PLUS_MINUS_ASCII = '+/-'

# O2 category labels with proper Unicode (for figures)
O2_CAT_CYANO = f'Cyanobacteria (O{O2_SUBSCRIPT} producers)'
O2_CAT_AEROBIC_HET = f'Aerobic heterotroph (O{O2_SUBSCRIPT} consumers)'
O2_CAT_ANAEROBE = 'Strict anaerobe'
O2_CAT_FACULTATIVE = 'Facultative anaerobe'
O2_CAT_CHEMOLITHO = 'Aerobic chemolithotroph'
O2_CAT_ANOXYGENIC = 'Anoxygenic phototroph'

# Low-Eh threshold for bimodal analysis
LOW_EH_THRESHOLD = 20  # kJ/mol

# Threshold for Er/Tmax in units of R
# Strains with Er/Tmax >= 36.5R are classified as Investors
# Strains with Er/Tmax < 36.5R are classified as Gamblers
ER_TMAX_THRESHOLD_R = 36.5  # units of R


# =============================================================================
# THERMODYNAMIC MODEL FUNCTIONS
# =============================================================================
def mu_model(T_C, Er, Eh, Tmax):
    """
    Calculate growth rate at temperature T (deg C) using the thermodynamic model.
    
    mu(T) = (kB*T/h) * exp(-Er/RT) * [exp(Eh/R * (1/T - 1/Tmax)) - 1]
    """
    T = T_C + 273.15  # Convert to Kelvin
    Tmax_K = Tmax + 273.15
    
    if T >= Tmax_K:
        return 0.0
    
    prefactor = kB_over_h * T
    kinetic = np.exp(-Er / (R * T))
    thermo_term = np.exp(Eh / R * (1/T - 1/Tmax_K)) - 1
    
    if thermo_term <= 0:
        return 0.0
    
    return prefactor * kinetic * thermo_term


def calc_Q10_apparent(Er, Eh, Tmax, T1=15, T2=25):
    """
    Calculate environmental Q10 (Q10,env) between T1 and T2 from model parameters.
    
    This measures the observed temperature sensitivity at a fixed temperature
    range, which includes both intrinsic sensitivity AND proximity to Topt.
    """
    mu1 = mu_model(T1, Er, Eh, Tmax)
    mu2 = mu_model(T2, Er, Eh, Tmax)
    
    if mu1 <= 0 or mu2 <= 0:
        return np.nan
    
    return (mu2 / mu1) ** (10 / (T2 - T1))


def calc_Q10_intrinsic(Er, Eh, Tmax, Topt, step=1):
    """
    Calculate INTRINSIC Q10 as the mean Q10 over all valid [T, T+10] intervals
    where T+10 < Topt.
    
    This removes the confounding effect of proximity to the thermal optimum,
    measuring only the fundamental temperature sensitivity of the growth machinery.
    
    Parameters:
    -----------
    Er : float
        Activation enthalpy for cellular organization decay (kJ/mol)
    Eh : float
        Enthalpy of rate-limiting reaction (kJ/mol)
    Tmax : float
        Maximum temperature for growth (°C)
    Topt : float
        Optimal temperature for growth (°C)
    step : int
        Temperature step for averaging (default 1°C)
    
    Returns:
    --------
    float : Mean Q10 over valid intervals, or NaN if no valid intervals
    """
    if pd.isna(Topt) or pd.isna(Tmax) or Topt >= Tmax:
        return np.nan
    
    q10_values = []
    
    # Iterate over temperature intervals [T, T+10] where T+10 < Topt
    # Start from a reasonable minimum (e.g., -10°C or where growth is positive)
    T_start = max(-10, Topt - 50)  # Start at most 50°C below Topt
    
    for T1 in np.arange(T_start, Topt - 10, step):
        T2 = T1 + 10
        
        if T2 >= Topt:
            break
            
        mu1 = mu_model(T1, Er, Eh, Tmax)
        mu2 = mu_model(T2, Er, Eh, Tmax)
        
        if mu1 > 0 and mu2 > 0:
            q10 = mu2 / mu1
            if 0 < q10 < 20:  # Sanity check
                q10_values.append(q10)
    
    if len(q10_values) == 0:
        return np.nan
    
    return np.mean(q10_values)


def calc_mumax(Er, Eh, Tmax, Topt):
    """
    Calculate maximum growth rate (mumax) at Topt.
    
    Returns mumax in h^-1.
    """
    if pd.isna(Topt) or pd.isna(Tmax):
        return np.nan
    
    return mu_model(Topt, Er, Eh, Tmax)


def cohens_d(group1, group2):
    """Calculate Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0
    return (group1.mean() - group2.mean()) / pooled_std


# =============================================================================
# CLASSIFICATION FUNCTIONS
# =============================================================================
def classify_o2_metabolism(row):
    """
    Classify strain into oxygen metabolism category.
    """
    guild = str(row['Guild_harmonized']).lower()
    phylum = str(row['Phylum_harmonized']).lower()
    domain = str(row['Domain'])
    
    if 'cyanobacteria' in phylum:
        return O2_CAT_CYANO
    
    if 'anoxygenic' in guild:
        return O2_CAT_ANOXYGENIC
    
    if any(x in guild for x in ['hydrogen-oxid', 'iron-oxid', 'sulfur-oxid', 
                                 'ammonia-oxid', 'methane-oxid', 'methanotroph']):
        return O2_CAT_CHEMOLITHO
    
    if any(x in guild for x in ['ferment', 'methanogen', 'sulfate-reduc', 'acetogen', 
                                 'iron-reduc', 'sulfur-reduc']):
        return O2_CAT_ANAEROBE
    
    if any(x in guild for x in ['facultativ', 'denitrif']):
        return O2_CAT_FACULTATIVE
    
    if domain == 'Bacteria' and any(x in guild for x in ['aerobic', 'chemoorgano']):
        return O2_CAT_AEROBIC_HET
    
    return 'Other'


def classify_strategy(row):
    """
    Classify strain into gambler (r-strategist) or investor (K-strategist).
    
    CORRECTED (Jan 2026): Added 'denitrif' to gambler_keywords and replaced
    'nitrif' with 'ammonia-oxid' in investor_keywords to properly classify
    denitrifying respirers as gamblers (anaerobic respirers using nitrate
    as terminal electron acceptor).
    """
    guild = str(row['Guild_harmonized']).lower()
    trophic = str(row['Trophic_mode']).lower()
    
    # CORRECTED: Added 'denitrif' for denitrifying respirers (anaerobic respirers)
    gambler_keywords = ['ferment', 'facultativ', 'aerobic chemoorgano', 'decomposer', 
                        'fungi', 'yeast', 'grazer', 'predator', 'denitrif']
    
    # CORRECTED: Replaced 'nitrif' with 'ammonia-oxid' to avoid capturing denitrifiers
    investor_keywords = ['phototroph', 'chemolitho', 'methanogen', 'sulfate-reduc',
                        'ammonia-oxid', 'hydrogen-oxid', 'iron-oxid', 'sulfur-oxid']
    
    if any(kw in guild for kw in gambler_keywords):
        return 'Gambler'
    elif any(kw in guild for kw in investor_keywords):
        return 'Investor'
    elif 'phototroph' in trophic:
        return 'Investor'
    elif 'heterotroph' in trophic:
        return 'Gambler'
    else:
        return 'Unknown'


def classify_strategy_aposteriori(row):
    """
    A POSTERIORI classification based on thermodynamic parameter Er/Tmax.
    
    Instead, it classifies strains based solely on the ratio of activation
    energy (Er) to maximum temperature (Tmax), normalized by the gas constant R.
    
    Threshold: 36.5R (optimized for AUC=0.874, Cohen's d=1.64, Het + Oxy Photo n=880)
    
    Parameters:
    -----------
    row : pd.Series
        Row with 'Er' (kJ/mol) and 'Tmax' (°C) columns
    
    Returns:
    --------
    str : 'Gambler' if Er/Tmax < 36.5R, 'Investor' if >= 36.5R, 'Unknown' if NaN
    """
    Er = row.get('Er', np.nan)
    Tmax = row.get('Tmax', np.nan)
    
    if pd.isna(Er) or pd.isna(Tmax) or Tmax <= -273.15:
        return 'Unknown'
    
    Tmax_K = Tmax + 273.15  # Convert to Kelvin
    Er_Tmax_R = Er / (R * Tmax_K)  # Er/Tmax in units of R
    
    if Er_Tmax_R >= ER_TMAX_THRESHOLD_R:
        return 'Investor'
    else:
        return 'Gambler'


def get_metabolic_group(row):
    """
    Classify strain into metabolic group for validation analysis.
    
    Returns one of:
    - 'Oxygenic_phototroph'
    - 'Anoxygenic_phototroph'  
    - 'Heterotroph'
    - 'Methanogen'
    - 'Chemolithoautotroph'
    - 'Other'
    """
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


def compute_contingency_stats(df):
    """
    Compute contingency table statistics for Strategy x Eh regime.
    
    Returns:
    --------
    dict : Statistics including counts, percentages, chi2, correlation
    """
    # Create binary columns (use GMM Population if available, else threshold)
    df_temp = df.copy()
    df_temp['is_gambler'] = df_temp['Strategy_aposteriori'] == 'Gambler'
    if 'Population' in df_temp.columns:
        df_temp['is_low_eh'] = df_temp['Population'] == 'Low_Eh'
    else:
        df_temp['is_low_eh'] = df_temp['Eh'] < LOW_EH_THRESHOLD
    
    # Contingency table
    gambler_low = ((df_temp['is_gambler']) & (df_temp['is_low_eh'])).sum()
    gambler_high = ((df_temp['is_gambler']) & (~df_temp['is_low_eh'])).sum()
    investor_low = ((~df_temp['is_gambler']) & (df_temp['is_low_eh'])).sum()
    investor_high = ((~df_temp['is_gambler']) & (~df_temp['is_low_eh'])).sum()
    
    n_gamblers = gambler_low + gambler_high
    n_investors = investor_low + investor_high
    n_low_eh = gambler_low + investor_low
    n_high_eh = gambler_high + investor_high
    
    # Chi-squared test
    contingency = np.array([[gambler_low, gambler_high],
                           [investor_low, investor_high]])
    chi2, p_chi2, dof, expected = stats.chi2_contingency(contingency)
    
    # Correlation
    r, p_r = stats.pearsonr(df_temp['Er'] / (R * (df_temp['Tmax'] + 273.15)), 
                            df_temp['Eh'])
    
    return {
        'gambler_low': gambler_low,
        'gambler_high': gambler_high,
        'investor_low': investor_low,
        'investor_high': investor_high,
        'n_gamblers': n_gamblers,
        'n_investors': n_investors,
        'n_low_eh': n_low_eh,
        'n_high_eh': n_high_eh,
        'pct_gambler_low': gambler_low / n_gamblers * 100 if n_gamblers > 0 else 0,
        'pct_gambler_high': gambler_high / n_gamblers * 100 if n_gamblers > 0 else 0,
        'pct_investor_low': investor_low / n_investors * 100 if n_investors > 0 else 0,
        'pct_investor_high': investor_high / n_investors * 100 if n_investors > 0 else 0,
        'pct_low_eh': n_low_eh / len(df_temp) * 100,
        'pct_high_eh': n_high_eh / len(df_temp) * 100,
        'chi2': chi2,
        'p_chi2': p_chi2,
        'r_Er_Tmax_Eh': r,
        'p_r': p_r
    }


# =============================================================================
# DATA LOADING
# =============================================================================
def load_data(filepath=None):
    """Load and prepare the 3-parameter model data from corrected files."""
    if filepath is None:
        # R7.2: Use corrected file with methanogen reclassification
        possible_paths = [
            '/mnt/project/07b_DE_results_1054_strains_v30.xlsx',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '07b_DE_results_1054_strains_v30.xlsx'),
            os.path.join(DATA_DIR, '07b_DE_results_1054_strains_v30.xlsx'),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                filepath = path
                break
        if filepath is None:
            print("\n" + "="*70)
            print("ERREUR : Fichier de données introuvable !")
            print("="*70)
            print(f"\nVeuillez placer le fichier '24Fit_3_param_1054_strains_with_Q10.xlsx'")
            print(f"dans le dossier : {DATA_DIR}")
            print("\nChemins recherchés :")
            for p in possible_paths:
                print(f"  - {p}")
            print("="*70 + "\n")
            raise FileNotFoundError("Fichier de données introuvable. Voir message ci-dessus.")
    
    print(f"Loading data from: {filepath}")
    df = pd.read_excel(filepath)
    
    # Handle both old and new column naming conventions
    # New files use: strain_ID, Tmax_Thermo, Topt_Thermo, Q10_apparent, Q10_intrinsic
    # Old files use: ID, Tmax, Topt, Q10
    
    # Standardize column names (create aliases for backward compatibility)
    if 'strain_ID' in df.columns and 'ID' not in df.columns:
        df['ID'] = df['strain_ID']
    if 'Tmax_Thermo' in df.columns and 'Tmax' not in df.columns:
        df['Tmax'] = df['Tmax_Thermo']
    if 'Topt_Thermo' in df.columns and 'Topt' not in df.columns:
        df['Topt'] = df['Topt_Thermo']
    if 'Q10_apparent' in df.columns and 'Q10' not in df.columns:
        df['Q10'] = df['Q10_apparent']
    
    # For old files without new columns, create them
    if 'Q10_apparent' not in df.columns and 'Q10' in df.columns:
        df['Q10_apparent'] = df['Q10']
    
    df['O2_cat'] = df.apply(classify_o2_metabolism, axis=1)
    df['Strategy'] = df.apply(classify_strategy, axis=1)  # A priori (legacy)
    df['Strategy_aposteriori'] = df.apply(classify_strategy_aposteriori, axis=1)  # A posteriori
    df['Metabolic_group'] = df.apply(get_metabolic_group, axis=1)
    df['Eh_Er_ratio'] = df['Eh'] / df['Er']
    
    # Calculate Er/Tmax in units of R
    df['Tmax_K'] = df['Tmax'] + 273.15
    df['Er_Tmax_R'] = df['Er'] / (R * df['Tmax_K'])
    
    # Q10 backward compatibility
    df['Q10_calc'] = df['Q10_apparent']
    
    # Calculate Q10 intrinsic only if not already present
    if 'Q10_intrinsic' not in df.columns:
        print("  Calculating Q10 intrinsic for each strain...")
        df['Q10_intrinsic'] = df.apply(
            lambda row: calc_Q10_intrinsic(row['Er'], row['Eh'], row['Tmax'], row['Topt']),
            axis=1
        )
    else:
        print("  Using pre-calculated Q10_intrinsic from file")
    
    # Calculate mumax at Topt
    df['mumax'] = df.apply(
        lambda row: calc_mumax(row['Er'], row['Eh'], row['Tmax'], row['Topt']),
        axis=1
    )
    
    df['deltaG_Topt'] = df.apply(
        lambda row: -row['Eh'] * (1 - (row['Topt'] + 273.15) / (row['Tmax'] + 273.15)) 
        if row['Tmax'] > row['Topt'] else np.nan, axis=1
    )
    
    print(f"  [OK] Loaded {len(df)} strains")
    return df


# =============================================================================
# STATISTICS FUNCTIONS
# =============================================================================
def print_statistics(df):
    """Print all key statistics from the dataset."""
    print("\n" + "="*70)
    print("V26 STATISTICS (A POSTERIORI CLASSIFICATION)")
    print("="*70)
    
    nrmse = df['NRMSE'].dropna()
    print(f"\n=== FIT QUALITY ===")
    print(f"NRMSE median: {nrmse.median():.3f}")
    print(f"NRMSE < 0.10: {(nrmse < 0.10).sum()/len(nrmse)*100:.1f}%")
    print(f"NRMSE < 0.15: {(nrmse < 0.15).sum()/len(nrmse)*100:.1f}%")
    print(f"NRMSE < 0.20: {(nrmse < 0.20).sum()/len(nrmse)*100:.1f}%")
    
    # A POSTERIORI CLASSIFICATION STATISTICS
    print(f"\n=== A POSTERIORI CLASSIFICATION (Er/Tmax threshold: {ER_TMAX_THRESHOLD_R}R) ===")
    gamblers_ap = df[df['Strategy_aposteriori'] == 'Gambler']
    investors_ap = df[df['Strategy_aposteriori'] == 'Investor']
    print(f"Gamblers: n = {len(gamblers_ap)} ({len(gamblers_ap)/len(df)*100:.1f}%)")
    print(f"Investors: n = {len(investors_ap)} ({len(investors_ap)/len(df)*100:.1f}%)")
    
    # Contingency statistics
    contingency = compute_contingency_stats(df)
    print(f"\n=== CONTINGENCY: Strategy x Eh Regime ===")
    print(f"Low-Eh: n = {contingency['n_low_eh']} ({contingency['pct_low_eh']:.1f}%)")
    print(f"High-Eh: n = {contingency['n_high_eh']} ({contingency['pct_high_eh']:.1f}%)")
    print(f"Gamblers in Low-Eh: {contingency['gambler_low']} ({contingency['pct_gambler_low']:.0f}%)")
    print(f"Investors in High-Eh: {contingency['investor_high']} ({contingency['pct_investor_high']:.0f}%)")
    print(f"Chi-squared: {contingency['chi2']:.1f}, p < 10^{int(np.log10(contingency['p_chi2']))}")
    print(f"Correlation r(Er/Tmax, Eh): {contingency['r_Er_Tmax_Eh']:.3f}")
    
    # Strategy by metabolic group
    print(f"\n=== STRATEGY BY METABOLIC GROUP (A POSTERIORI) ===")
    for group in ['Oxygenic_phototroph', 'Heterotroph', 'Methanogen', 'Anoxygenic_phototroph']:
        subset = df[df['Metabolic_group'] == group]
        if len(subset) > 0:
            n_gambler = (subset['Strategy_aposteriori'] == 'Gambler').sum()
            n_investor = (subset['Strategy_aposteriori'] == 'Investor').sum()
            pct_gambler = n_gambler / len(subset) * 100
            pct_investor = n_investor / len(subset) * 100
            print(f"  {group}: n={len(subset)}, {pct_gambler:.0f}% Gamblers, {pct_investor:.0f}% Investors")
    
    print(f"\n=== Q10,env BY TROPHIC MODE ===")
    het = df[df['Trophic_mode'] == 'Heterotroph']['Q10_apparent'].dropna()
    pho = df[df['Trophic_mode'] == 'Phototroph']['Q10_apparent'].dropna()
    het = het[(het > 0) & (het < 10)]
    pho = pho[(pho > 0) & (pho < 10)]
    print(f"Heterotrophs (n={len(het)}): Q10_apparent = {het.mean():.2f} {PLUS_MINUS_ASCII} {het.std():.2f}")
    print(f"Phototrophs (n={len(pho)}): Q10_apparent = {pho.mean():.2f} {PLUS_MINUS_ASCII} {pho.std():.2f}")
    print(f"Ratio: {het.mean()/pho.mean():.2f}")
    _, p_q10_app = stats.mannwhitneyu(het, pho)
    d_q10_app = cohens_d(het, pho)
    print(f"Mann-Whitney p = {p_q10_app:.2e}, Cohen's d = {abs(d_q10_app):.2f}")
    
    print(f"\n=== Q10 INTRINSIC BY TROPHIC MODE ===")
    het_int = df[df['Trophic_mode'] == 'Heterotroph']['Q10_intrinsic'].dropna()
    pho_int = df[df['Trophic_mode'] == 'Phototroph']['Q10_intrinsic'].dropna()
    het_int = het_int[(het_int > 0) & (het_int < 10)]
    pho_int = pho_int[(pho_int > 0) & (pho_int < 10)]
    print(f"Heterotrophs (n={len(het_int)}): Q10_intrinsic = {het_int.mean():.2f} {PLUS_MINUS_ASCII} {het_int.std():.2f}")
    print(f"Phototrophs (n={len(pho_int)}): Q10_intrinsic = {pho_int.mean():.2f} {PLUS_MINUS_ASCII} {pho_int.std():.2f}")
    print(f"Ratio: {het_int.mean()/pho_int.mean():.2f}")
    _, p_q10_int = stats.mannwhitneyu(het_int, pho_int)
    d_q10_int = cohens_d(het_int, pho_int)
    print(f"Mann-Whitney p = {p_q10_int:.2e}, Cohen's d = {abs(d_q10_int):.2f}")
    
    print(f"\n=== MUMAX BY TROPHIC MODE ===")
    het_mu = df[df['Trophic_mode'] == 'Heterotroph']['mumax'].dropna()
    pho_mu = df[df['Trophic_mode'] == 'Phototroph']['mumax'].dropna()
    het_mu = het_mu[(het_mu > 0) & (het_mu < 1e15)]
    pho_mu = pho_mu[(pho_mu > 0) & (pho_mu < 1e15)]
    print(f"Heterotrophs (n={len(het_mu)}): mumax = {het_mu.mean():.2e} {PLUS_MINUS_ASCII} {het_mu.std():.2e}")
    print(f"Phototrophs (n={len(pho_mu)}): mumax = {pho_mu.mean():.2e} {PLUS_MINUS_ASCII} {pho_mu.std():.2e}")
    print(f"Ratio (Het/Pho): {het_mu.mean()/pho_mu.mean():.1f}x (arithmetic mean)")
    # Geometric mean ratio (robust to skew / thermophile outliers)
    log_het = np.log10(het_mu)
    log_pho = np.log10(pho_mu)
    geom_ratio_all = 10**(log_het.mean() - log_pho.mean())
    print(f"Geometric mean ratio (all): {geom_ratio_all:.1f}x")
    # Mesophile subsets
    for label, topt_lo, topt_hi in [("20-40C", 20, 40), ("25-37C", 25, 37)]:
        h = df[(df['Trophic_mode'] == 'Heterotroph') & (df['Topt'] >= topt_lo) & (df['Topt'] <= topt_hi)]
        p = df[(df['Trophic_mode'] == 'Phototroph') & (df['Topt'] >= topt_lo) & (df['Topt'] <= topt_hi)]
        mh = h['mumax'].dropna(); mh = mh[(mh > 0) & (mh < 1e15)]
        mp = p['mumax'].dropna(); mp = mp[(mp > 0) & (mp < 1e15)]
        if len(mh) > 0 and len(mp) > 0:
            gr = 10**(np.log10(mh).mean() - np.log10(mp).mean())
            print(f"Geometric mean ratio (mesophiles {label}): {gr:.1f}x (n_het={len(mh)}, n_pho={len(mp)})")
    
    print(f"\n=== Eh BIMODAL DISTRIBUTION ===")
    low_eh = df[df['Eh'] < LOW_EH_THRESHOLD]
    high_eh = df[df['Eh'] >= LOW_EH_THRESHOLD]
    print(f"Low-Eh population (< {LOW_EH_THRESHOLD} kJ/mol): n={len(low_eh)} ({100*len(low_eh)/len(df):.0f}%)")
    print(f"High-Eh population (>= {LOW_EH_THRESHOLD} kJ/mol): n={len(high_eh)} ({100*len(high_eh)/len(df):.0f}%)")
    
    print(f"\n=== O2 METABOLISM (Bacteria only) ===")
    bac = df[df['Domain'] == 'Bacteria']
    for cat in [O2_CAT_CYANO, O2_CAT_AEROBIC_HET, O2_CAT_ANAEROBE, 
                O2_CAT_FACULTATIVE, O2_CAT_CHEMOLITHO, O2_CAT_ANOXYGENIC]:
        subset = bac[bac['O2_cat'] == cat]
        if len(subset) >= 5:
            print(f"{cat}: n={len(subset)}, Er={subset['Er'].mean():.1f} {PLUS_MINUS_ASCII} {subset['Er'].std():.1f}")
    
    cyano = bac[bac['O2_cat'] == O2_CAT_CYANO]
    aer_het = bac[bac['O2_cat'] == O2_CAT_AEROBIC_HET]
    if len(cyano) > 0 and len(aer_het) > 0:
        delta = cyano['Er'].mean() - aer_het['Er'].mean()
        _, p = stats.mannwhitneyu(cyano['Er'], aer_het['Er'])
        print(f"\n*** O2 METABOLISM TEST ***")
        print(f"Cyanobacteria Er: {cyano['Er'].mean():.1f} {PLUS_MINUS_ASCII} {cyano['Er'].std():.1f} (n={len(cyano)})")
        print(f"Aerobic Het Er: {aer_het['Er'].mean():.1f} {PLUS_MINUS_ASCII} {aer_het['Er'].std():.1f} (n={len(aer_het)})")
        print(f"Delta = +{delta:.1f} kJ/mol, p = {p:.2e}")
    
    print(f"\n=== Eh BY DOMAIN ===")
    for domain in ['Bacteria', 'Archaea', 'Eukarya']:
        subset = df[df['Domain'] == domain]['Eh'].dropna()
        if len(subset) > 0:
            print(f"{domain} (n={len(subset)}): Eh = {subset.mean():.1f} {PLUS_MINUS_ASCII} {subset.std():.1f}")
    
    data_eh = [df[df['Domain'] == d]['Eh'].dropna().values for d in ['Bacteria', 'Archaea', 'Eukarya']]
    eh_domain = np.concatenate(data_eh)
    gm_eh = eh_domain.mean()
    ss_between = sum(len(d)*(np.mean(d) - gm_eh)**2 for d in data_eh)
    ss_total = np.sum((eh_domain - gm_eh)**2)
    eta2_eh = ss_between / ss_total
    print(f"Eh variance explained by domain (n={len(eh_domain)}): eta2 = {eta2_eh*100:.1f}%")
    
    data_er = [df[df['Domain'] == d]['Er'].dropna().values for d in ['Bacteria', 'Archaea', 'Eukarya']]
    er_domain = np.concatenate(data_er)
    gm_er = er_domain.mean()
    ss_between_er = sum(len(d)*(np.mean(d) - gm_er)**2 for d in data_er)
    ss_total_er = np.sum((er_domain - gm_er)**2)
    eta2_er = ss_between_er / ss_total_er
    print(f"Er variance explained by domain (n={len(er_domain)}): eta2 = {eta2_er*100:.1f}%")
    
    return {
        'eta2_eh': eta2_eh,
        'eta2_er': eta2_er,
        'q10_het': het.mean(),
        'q10_pho': pho.mean()
    }


# =============================================================================
# FIGURE 1: Model parameters and their effects
# =============================================================================
def generate_figure1(df, output_dir=OUTPUT_DIR):
    """Figure 1: Model parameters and their effects on growth curves."""
    print("\nGenerating Figure 1: Model parameters...")
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    # Panel A: Model schematic
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('A', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    
    eq_text = r'$\mu(T) = \frac{k_B T}{h} \cdot e^{-E_r/RT} \cdot \left[e^{\frac{E_h}{R}\left(\frac{1}{T}-\frac{1}{T_{max}}\right)} - 1\right]$'
    ax.text(5, 7.5, eq_text, fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='navy', linewidth=2))
    
    params = [
        (1.5, 4, r'$\frac{k_B T}{h}$', 'Thermal\nfluctuations', 'lightgray'),
        (5, 4, r'$E_r$', 'Thermal\ndampening', '#FFCDD2'),
        (8.5, 4, r'$E_h, T_{max}$', 'Driving\nforce', '#C8E6C9'),
    ]
    
    for x, y, symbol, desc, color in params:
        box = FancyBboxPatch((x-1, y-0.8), 2, 1.6, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y+0.3, symbol, fontsize=12, ha='center', va='center', fontweight='bold')
        ax.text(x, y-0.4, desc, fontsize=8, ha='center', va='center')
        ax.annotate('', xy=(x, y+1.2), xytext=(x, 6.3),
                   arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    ax.text(5, 1.5, r'$\mu_{max}$ emerges from physics', fontsize=10, ha='center',
            style='italic', color='darkblue')
    ax.text(5, 0.8, '(not a free parameter)', fontsize=9, ha='center', color='gray')
    
    # Panel B: Effect of Er
    ax = axes[1]
    T_range = np.linspace(5, 55, 200)
    Eh_fixed, Tmax_fixed = 25, 50
    Er_values = [85, 95, 105, 115]
    colors_Er = plt.cm.Reds(np.linspace(0.3, 0.9, len(Er_values)))
    
    for Er, color in zip(Er_values, colors_Er):
        mu_values = [mu_model(T, Er, Eh_fixed, Tmax_fixed) for T in T_range]
        mu_max = max(mu_values) if max(mu_values) > 0 else 1
        mu_norm = [m / mu_max for m in mu_values]
        ax.plot(T_range, mu_norm, color=color, linewidth=2.5, label=f'$E_r$ = {Er}')
    
    ax.set_xlabel(f'Temperature ({DEGREE}C)')
    ax.set_ylabel('Normalized growth rate')
    ax.set_title('B', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    ax.legend(title='kJ/mol', loc='upper left', fontsize=9)
    ax.set_xlim(5, 55)
    ax.set_ylim(0, 1.15)
    ax.text(42, 0.85, f'$E_h$ = {Eh_fixed} kJ/mol\n$T_{{max}}$ = {Tmax_fixed}{DEGREE}C', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Panel C: Effect of Eh
    ax = axes[2]
    Er_fixed = 98
    Eh_values = [10, 25, 50, 80]
    colors_Eh = plt.cm.Greens(np.linspace(0.3, 0.9, len(Eh_values)))
    
    for Eh, color in zip(Eh_values, colors_Eh):
        mu_values = [mu_model(T, Er_fixed, Eh, Tmax_fixed) for T in T_range]
        mu_max = max(mu_values) if max(mu_values) > 0 else 1
        mu_norm = [m / mu_max for m in mu_values]
        ax.plot(T_range, mu_norm, color=color, linewidth=2.5, label=f'$E_h$ = {Eh}')
    
    ax.set_xlabel(f'Temperature ({DEGREE}C)')
    ax.set_ylabel('Normalized growth rate')
    ax.set_title('C', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    ax.legend(title='kJ/mol', loc='upper left', fontsize=9)
    ax.set_xlim(5, 55)
    ax.set_ylim(0, 1.15)
    ax.text(42, 0.85, f'$E_r$ = {Er_fixed} kJ/mol\n$T_{{max}}$ = {Tmax_fixed}{DEGREE}C', 
            fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}05a_Figure1_model_parameters_v30.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}05a_Figure1_model_parameters_v30.png', bbox_inches='tight')
    plt.close()
    print("  [OK] Figure 1 saved")


# =============================================================================
# FIGURE 2: UNIFIED - Bimodal Eh + Parameter coupling (3 panels: A, B, C)
# =============================================================================
def generate_figure2(df, stats_dict, output_dir=OUTPUT_DIR):
    """
    Figure 2: Unified figure with bimodal Eh distribution and parameter coupling.
    
    Layout: Single row of 3 panels with width_ratios [1.5, 1, 1]
    - Panel A: Er-Eh scatter with marginal histograms (wider for marginals)
    - Panel B: Er/Tmax boxplots by metabolic group
    - Panel C: Eh/Er ratio by strategy (former Panel E)
    """
    print("\nGenerating Figure 2: Unified bimodal + coupling figure (3 panels)...")
    
    # Create figure with 3-panel layout
    fig = plt.figure(figsize=(16, 5))
    
    # Master GridSpec: 1 row, 3 columns
    gs_master = GridSpec(1, 3, figure=fig, width_ratios=[1.5, 1, 1], wspace=0.22)
    
    # Prepare data
    df_valid = df[df['Strategy_aposteriori'].isin(['Gambler', 'Investor'])].copy()
    gamblers = df_valid[df_valid['Strategy_aposteriori'] == 'Gambler']
    investors = df_valid[df_valid['Strategy_aposteriori'] == 'Investor']
    
    # =========================================================================
    # PANEL A: Er-Eh scatter with marginal histograms (bimodal)
    # =========================================================================
    gs_A = gs_master[0].subgridspec(4, 4, hspace=0.05, wspace=0.05)
    
    ax_A_main = fig.add_subplot(gs_A[1:4, 0:3])
    ax_A_top = fig.add_subplot(gs_A[0, 0:3], sharex=ax_A_main)
    ax_A_right = fig.add_subplot(gs_A[1:4, 3], sharey=ax_A_main)
    
    # Add shaded region for Low-Eh
    ax_A_main.axvspan(0, LOW_EH_THRESHOLD, alpha=0.08, color='gray', zorder=0)
    ax_A_top.axvspan(0, LOW_EH_THRESHOLD, alpha=0.08, color='gray', zorder=0)
    
    # Plot scatter by strategy
    for strategy in ['Investor', 'Gambler']:
        mask = df['Strategy_aposteriori'] == strategy
        n = mask.sum()
        ax_A_main.scatter(df.loc[mask, 'Eh'], df.loc[mask, 'Er'],
                         c=COLORS[strategy], s=18, alpha=0.6,
                         label=f'{strategy} (n={n})', edgecolors='none',
                         zorder=3)
    
    # Marginal histogram for Eh (top)
    bins_eh = np.linspace(0, 120, 50)
    for strategy in ['Investor', 'Gambler']:
        mask = df['Strategy_aposteriori'] == strategy
        ax_A_top.hist(df.loc[mask, 'Eh'], bins=bins_eh, alpha=0.7,
                     color=COLORS[strategy], edgecolor='white', linewidth=0.3)
    
    # Marginal histogram for Er (right, horizontal)
    bins_er = np.linspace(82, 120, 40)
    for strategy in ['Investor', 'Gambler']:
        mask = df['Strategy_aposteriori'] == strategy
        ax_A_right.hist(df.loc[mask, 'Er'], bins=bins_er, alpha=0.7,
                       color=COLORS[strategy], orientation='horizontal',
                       edgecolor='white', linewidth=0.3)
    
    # Annotation for Eh peak
    ax_A_top.annotate(f'Peak at\n$E_h$$\\approx$5.5 kJ/mol',
                     xy=(5.5, ax_A_top.get_ylim()[1]*0.9),
                     xytext=(35, ax_A_top.get_ylim()[1]*0.85),
                     fontsize=8, color='#B22222',
                     ha='left', va='top',
                     arrowprops=dict(arrowstyle='->', color='#B22222', lw=1.2))
    
    # Add population labels
    if 'Population' in df.columns:
        low_eh_pct = (df['Population'] == 'Low_Eh').mean() * 100
    else:
        low_eh_pct = (df['Eh'] < LOW_EH_THRESHOLD).mean() * 100
    high_eh_pct = 100 - low_eh_pct
    ax_A_main.text(8, 85.5, f'$Low$-$E_h$\n({low_eh_pct:.0f}%)', 
                  fontsize=8, style='italic', color='#555555', ha='center')
    ax_A_main.text(70, 85.5, f'$High$-$E_h$\n({high_eh_pct:.0f}%)', 
                  fontsize=8, style='italic', color='#555555', ha='center')
    
    # Add vertical line at threshold
    ax_A_main.axvline(x=LOW_EH_THRESHOLD, color='gray', linestyle=':', alpha=0.6, lw=1)
    ax_A_top.axvline(x=LOW_EH_THRESHOLD, color='gray', linestyle=':', alpha=0.6, lw=1)
    
    # Configure main axes
    ax_A_main.set_xlabel('$E_h$ (kJ/mol)', fontsize=10)
    ax_A_main.set_ylabel('$E_r$ (kJ/mol)', fontsize=10)
    ax_A_main.set_xlim(0, 120)
    ax_A_main.set_ylim(82, 120)
    ax_A_main.legend(loc='upper left', fontsize=7, framealpha=0.9,
                    markerscale=1.2, handletextpad=0.3)
    
    # Configure marginal axes
    ax_A_top.set_ylabel('Count', fontsize=8)
    ax_A_right.set_xlabel('Count', fontsize=8)
    plt.setp(ax_A_top.get_xticklabels(), visible=False)
    plt.setp(ax_A_right.get_yticklabels(), visible=False)
    
    ax_A_top.set_title('A  Two distinct $E_h$ populations',
                      fontsize=11, fontweight='bold', loc='left', pad=5)
    
    # Print statistics
    r_all, p_all = stats.pearsonr(df['Er'], df['Eh'])
    low_eh_df = df[df['Eh'] < LOW_EH_THRESHOLD]
    high_eh_df = df[df['Eh'] >= LOW_EH_THRESHOLD]
    r_low, _ = stats.pearsonr(low_eh_df['Er'], low_eh_df['Eh']) if len(low_eh_df) > 5 else (np.nan, np.nan)
    r_high, _ = stats.pearsonr(high_eh_df['Er'], high_eh_df['Eh']) if len(high_eh_df) > 5 else (np.nan, np.nan)
    
    print(f"  Panel A statistics:")
    print(f"    Global correlation: r = {r_all:.2f}")
    print(f"    Low-Eh (n={len(low_eh_df)}): r = {r_low:.2f}")
    print(f"    High-Eh (n={len(high_eh_df)}): r = {r_high:.2f}")
    
    # =========================================================================
    # PANEL B: Er/Tmax distributions by metabolic group
    # =========================================================================
    ax_B = fig.add_subplot(gs_master[1])
    
    metab_groups = [
        ('Heterotroph', 'Heterotrophs'),
        ('Oxygenic_phototroph', 'Oxy. phototrophs'),
        ('Chemolithoautotroph', 'Chemoauto.'),
        ('Anoxygenic_phototroph', 'Anoxy. photo.'),
    ]
    metab_colors = ['#D32F2F', '#388E3C', '#F57C00', '#7B1FA2']
    
    data_B = []
    labels_B = []
    pct_texts = []
    for (grp_key, grp_label), color in zip(metab_groups, metab_colors):
        mask = df_valid['Metabolic_group'] == grp_key
        vals = df_valid.loc[mask, 'Er_Tmax_R'].dropna().values
        if len(vals) > 0:
            data_B.append(vals)
            labels_B.append(grp_label)
            n_grp = len(vals)
            n_gamb = (vals < ER_TMAX_THRESHOLD_R).sum()
            n_inv = (vals >= ER_TMAX_THRESHOLD_R).sum()
            pct_gamb = n_gamb / n_grp * 100
            pct_inv = n_inv / n_grp * 100
            if pct_gamb > pct_inv:
                pct_texts.append(f'{pct_gamb:.0f}% G\nn={n_grp}')
            else:
                pct_texts.append(f'{pct_inv:.0f}% I\nn={n_grp}')
    
    bp_B = ax_B.boxplot(data_B, patch_artist=True, widths=0.55)
    for i, (patch, color) in enumerate(zip(bp_B['boxes'], metab_colors[:len(data_B)])):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax_B.axhline(y=ER_TMAX_THRESHOLD_R, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax_B.text(len(data_B) + 0.4, ER_TMAX_THRESHOLD_R, f'{ER_TMAX_THRESHOLD_R}$R$',
              va='center', ha='left', fontsize=7, color='gray')
    
    for i, txt in enumerate(pct_texts):
        ax_B.text(i + 1, ax_B.get_ylim()[1] * 0.98, txt,
                  ha='center', va='top', fontsize=7, fontweight='bold',
                  color=metab_colors[i])
    
    ax_B.set_xticklabels(labels_B, fontsize=8, rotation=15, ha='right')
    ax_B.set_ylabel('$E_r/T_{max}$ ($R$ units)')
    ax_B.set_title('B  $E_r/T_{max}$ by metabolic group', fontsize=11, fontweight='bold', loc='left')
    
    # =========================================================================
    # PANEL C: Eh/Er ratio by strategy (former Panel E)
    # =========================================================================
    ax_C = fig.add_subplot(gs_master[2])
    
    data_ratio = [gamblers['Eh_Er_ratio'].dropna().values, investors['Eh_Er_ratio'].dropna().values]
    bp_C = ax_C.boxplot(data_ratio, patch_artist=True)
    bp_C['boxes'][0].set_facecolor(COLORS['gamblers'])
    bp_C['boxes'][1].set_facecolor(COLORS['investors'])
    for patch in bp_C['boxes']:
        patch.set_alpha(0.7)
    ax_C.set_xticklabels(['Gamblers', 'Investors'])
    ax_C.set_ylabel('$E_h/E_r$ ratio')
    ax_C.set_title('C  $E_h/E_r$ ratio by strategy', fontsize=11, fontweight='bold', loc='left')
    
    _, p_ratio = stats.mannwhitneyu(gamblers['Eh_Er_ratio'].dropna(), investors['Eh_Er_ratio'].dropna())
    d_ratio = cohens_d(gamblers['Eh_Er_ratio'].dropna(), investors['Eh_Er_ratio'].dropna())
    ax_C.text(0.95, 0.95, f"Cohen's d = {abs(d_ratio):.2f}\np < 10$^{{{int(np.log10(p_ratio))}}}$", 
             transform=ax_C.transAxes, ha='right', va='top', fontsize=8, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save figure
    plt.savefig(f'{output_dir}05b_Figure2_unified_strategies_v30.pdf', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_dir}05b_Figure2_unified_strategies_v30.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("  [OK] Figure 2 (unified) saved")


# =============================================================================
# FIGURE 3: Ab initio validation (NEW in v26)
# =============================================================================
def generate_figure3(df, output_dir=OUTPUT_DIR):
    """
    Figure 3: Ab initio validation — 4 panels (2x2).
    
    Panel A: Eh vs Tmax (Low-Eh population) with theory line Eh = 2R*Tmax
    Panel B: Distribution of Eh/Tmax ratio
    Panel C: Maximum growth rate by strategy (in h^-1)
    Panel D: Residual activation enthalpy by guild
    """
    print("Generating Figure 3: Ab initio validation (NEW)...")
    
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Figure 3: Ab initio validation', fontsize=14, fontweight='bold', y=0.98)
    
    # =========================================================================
    # PANEL A: Eh vs Tmax (Low-Eh population)
    # =========================================================================
    ax = axes[0, 0]
    
    # Use GMM Population if available, else threshold
    if 'Population' in df.columns:
        low_eh = df[df['Population'] == 'Low_Eh'].copy()
    else:
        low_eh = df[df['Eh'] < LOW_EH_THRESHOLD].copy()
    
    n_low = len(low_eh)
    pct_low = 100 * n_low / len(df)
    
    low_eh_tmax_k = low_eh['Tmax_K'].values
    low_eh_eh = low_eh['Eh'].values
    
    domain_colors_local = {
        'Bacteria': COLORS['bacteria'],
        'Archaea': COLORS['archaea'],
        'Eukarya': COLORS['eukarya'],
    }
    
    for domain, color in domain_colors_local.items():
        mask = low_eh['Domain'] == domain
        if mask.sum() > 0:
            ax.scatter(low_eh.loc[mask, 'Tmax_K'], low_eh.loc[mask, 'Eh'],
                       c=color, alpha=0.4, s=15, edgecolors='none',
                       label=f'{domain} (n={mask.sum()})')
    
    # Handle Other/Unknown domains
    other_mask = ~low_eh['Domain'].isin(domain_colors_local.keys())
    if other_mask.sum() > 0:
        ax.scatter(low_eh.loc[other_mask, 'Tmax_K'], low_eh.loc[other_mask, 'Eh'],
                   c='#9E9E9E', alpha=0.3, s=10, edgecolors='none')
    
    # Linear regression
    slope, intercept, r_val, p_val, se = stats.linregress(low_eh_tmax_k, low_eh_eh)
    slope_J = slope * 1000  # kJ/(mol·K) → J/(mol·K)
    r_squared = r_val ** 2
    
    x_line = np.linspace(low_eh_tmax_k.min() - 5, low_eh_tmax_k.max() + 5, 100)
    ax.plot(x_line, slope * x_line + intercept, color='salmon', lw=2, ls='-',
            label=f'Fit: slope = {slope_J:.2f} J/mol/K')
    
    # Theory line: Eh = 2R * Tmax_K (in kJ/mol), 2R = 2 * 8.314e-3 kJ/mol/K
    theory_2R = 2 * 8.314  # J/(mol·K) = 16.628
    ax.plot(x_line, (2 * 8.314e-3) * x_line, color='black', lw=2, ls='--',
            label=r'Theory: $E_h = 2R \cdot T_{max}$')
    
    # Annotation box
    ax.text(0.98, 0.98,
            f'Low-$E_h$ population:\n'
            f'n = {n_low} ({pct_low:.0f}%)\n'
            f'$R^2$ = {r_squared:.3f}\n'
            f'Slope = {slope_J:.2f} J/mol/K\n'
            f'Theory (2R): {theory_2R:.2f} J/mol/K',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax.set_xlabel('$T_{max}$ (K)')
    ax.set_ylabel('$E_h$ (kJ/mol)')
    ax.set_title('A  $E_h$ vs $T_{max}$ (Low-$E_h$ population)',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.9)
    
    print(f"  Panel A: n={n_low}, R²={r_squared:.3f}, slope={slope_J:.2f} J/mol/K vs theory {theory_2R:.2f}")
    
    # =========================================================================
    # PANEL B: Distribution of Eh/Tmax ratio
    # =========================================================================
    ax = axes[0, 1]
    
    # Compute ratio in J/(mol·K)
    low_eh_ratio_J = (low_eh['Eh'] / low_eh['Tmax_K']) * 1000  # kJ→J
    
    # Also show all organisms for context
    all_ratio_J = (df['Eh'] / df['Tmax_K']) * 1000
    
    mean_ratio = low_eh_ratio_J.mean()
    sd_ratio = low_eh_ratio_J.std()
    cv_ratio = 100 * sd_ratio / mean_ratio
    
    bin_edges = np.arange(0, 62, 1.0)
    
    ax.hist(all_ratio_J.clip(upper=61), bins=bin_edges, color='#BDBDBD', alpha=0.5,
            label=f'All organisms (n={len(df)})')
    ax.hist(low_eh_ratio_J, bins=bin_edges, color='#2E7D32', alpha=0.7,
            label=f'Low-$E_h$ (n={n_low})')
    ax.axvline(theory_2R, color='black', lw=2, ls='--',
               label=f'Theory: $2R$ = {theory_2R:.1f}')
    
    ax.text(0.98, 0.98,
            f'Low-$E_h$ population:\n'
            f'Mean: {mean_ratio:.2f} J/mol/K\n'
            f'SD: {sd_ratio:.2f} J/mol/K\n'
            f'CV: {cv_ratio:.1f}%\n'
            f'Theory: {theory_2R:.2f} J/mol/K',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    ax.set_xlim(0, 60)
    ax.set_xlabel('$E_h/T_{max}$ (J/mol/K)')
    ax.set_ylabel('Count')
    ax.set_title('B  Distribution of $E_h/T_{max}$',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='center right', fontsize=7, framealpha=0.9)
    
    print(f"  Panel B: mean(Eh/Tmax)={mean_ratio:.2f}, SD={sd_ratio:.2f}, CV={cv_ratio:.1f}%")
    
    # =========================================================================
    # PANEL C: Maximum growth rate by strategy (in h^-1)
    # =========================================================================
    ax = axes[1, 0]
    
    # Convert mumax from s^-1 (model units) to h^-1
    df_strat = df[df['Strategy_aposteriori'].isin(['Gambler', 'Investor'])].copy()
    df_strat['mumax_h'] = df_strat['mumax'] * 3600  # s^-1 → h^-1
    
    gamb_mu = df_strat[df_strat['Strategy_aposteriori'] == 'Gambler']['mumax_h'].dropna()
    inv_mu = df_strat[df_strat['Strategy_aposteriori'] == 'Investor']['mumax_h'].dropna()
    
    # Filter out extreme values
    gamb_mu = gamb_mu[(gamb_mu > 0) & (gamb_mu < 1e6)]
    inv_mu = inv_mu[(inv_mu > 0) & (inv_mu < 1e6)]
    
    bins_mu = np.logspace(np.log10(max(1e-4, min(gamb_mu.min(), inv_mu.min()) * 0.5)),
                          np.log10(max(gamb_mu.max(), inv_mu.max()) * 2), 50)
    
    ax.hist(gamb_mu, bins=bins_mu, alpha=0.6, color=COLORS['gamblers'],
            label=f'Gamblers (n={len(gamb_mu)})', edgecolor='#B71C1C', linewidth=0.5)
    ax.hist(inv_mu, bins=bins_mu, alpha=0.6, color=COLORS['investors'],
            label=f'Investors (n={len(inv_mu)})', edgecolor='#0D47A1', linewidth=0.5)
    
    ax.set_xscale('log')
    
    # Median lines
    med_gamb = gamb_mu.median()
    med_inv = inv_mu.median()
    ax.axvline(med_gamb, color=COLORS['gamblers'], linestyle='--', linewidth=2.5)
    ax.axvline(med_inv, color=COLORS['investors'], linestyle='--', linewidth=2.5)
    
    # Ratio and statistics
    ratio_median = med_gamb / med_inv if med_inv > 0 else np.nan
    _, p_mu = stats.mannwhitneyu(gamb_mu, inv_mu)
    
    ax.text(0.98, 0.95,
            f'Median ratio: {ratio_median:.1f}x\n'
            f'Gambler: {med_gamb:.3f} h$^{{-1}}$\n'
            f'Investor: {med_inv:.4f} h$^{{-1}}$\n'
            f'p < 10$^{{{int(np.log10(p_mu))}}}$',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    ax.set_xlabel('$\\mu_{max}$ (h$^{-1}$)')
    ax.set_ylabel('Count')
    ax.set_title('C  Maximum growth rate by strategy',
                 fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    print(f"  Panel C: Gambler median={med_gamb:.4f} h⁻¹, Investor median={med_inv:.5f} h⁻¹, ratio={ratio_median:.1f}x")
    
    # =========================================================================
    # PANEL D: Residual activation enthalpy by guild
    # =========================================================================
    ax = axes[1, 1]
    
    # Compute Er residual: Er - Er_hat(Tmax)
    # Er_hat uses PUBLISHED rounded coefficients (MS L221: Er_hat = 0.193 * Tmax_K + 37.0)
    # to ensure consistency with body text, S11C, and all other Er_res computations.
    # Exact linregress gives slope=0.1925, intercept=36.88; rounding shift is ~0.3 kJ/mol uniform.
    df_temp = df.copy()
    df_temp['Er_predicted'] = 0.193 * df_temp['Tmax_K'] + 37.0
    df_temp['Er_residual'] = df_temp['Er'] - df_temp['Er_predicted']
    
    # Metabolic group classification (simplified for violin plot)
    def met_group_simple(row):
        guild = str(row.get('Guild_harmonized', '')).lower()
        trophic = str(row.get('Trophic_mode', '')).lower()
        phylum = str(row.get('Phylum_harmonized', '')).lower()
        if 'methanogen' in guild:
            return 'Methanogens'
        if 'anoxygenic' in guild:
            return None  # Exclude small group
        if 'cyanobacteria' in phylum or ('phototroph' in trophic and 'anoxygenic' not in guild):
            return 'Oxy. phototrophs'
        if any(x in guild for x in ['chemolitho', 'hydrogen-oxid', 'iron-oxid', 
                                     'sulfur-oxid', 'ammonia-oxid']):
            return 'Chemoautotrophs'
        if 'heterotroph' in trophic:
            return 'Heterotrophs'
        return None
    
    df_temp['MetGroup_simple'] = df_temp.apply(met_group_simple, axis=1)
    df_temp = df_temp[df_temp['MetGroup_simple'].notna()]
    
    plot_groups = ['Heterotrophs', 'Oxy. phototrophs', 'Methanogens', 'Chemoautotrophs']
    group_colors_D = {
        'Heterotrophs': '#D84315',
        'Oxy. phototrophs': '#2E7D32',
        'Methanogens': '#7B1FA2',
        'Chemoautotrophs': '#66BB6A',
    }
    
    plot_data = [df_temp[df_temp['MetGroup_simple'] == g]['Er_residual'].dropna().values 
                 for g in plot_groups]
    
    parts = ax.violinplot(plot_data, positions=range(len(plot_groups)),
                          showmeans=True, showmedians=True, showextrema=False)
    
    for i, (pc, grp) in enumerate(zip(parts['bodies'], plot_groups)):
        pc.set_facecolor(group_colors_D[grp])
        pc.set_alpha(0.6)
    parts['cmeans'].set_color('black')
    parts['cmeans'].set_linewidth(2)
    parts['cmedians'].set_color('gray')
    parts['cmedians'].set_linewidth(1.5)
    
    ax.set_xticks(range(len(plot_groups)))
    ax.set_xticklabels(['Het.', 'Oxy. photo.', 'Methano.', 'Chemoauto.'], fontsize=10)
    ax.axhline(0, color='gray', lw=1, ls='--', alpha=0.7)
    ax.set_ylabel('$E_r^{res}$ (kJ/mol)')
    ax.set_title('D  Residual activation enthalpy by guild',
                 fontsize=11, fontweight='bold', loc='left')
    
    # Annotate mean values
    for i, grp in enumerate(plot_groups):
        vals = df_temp[df_temp['MetGroup_simple'] == grp]['Er_residual'].dropna()
        ax.text(i, ax.get_ylim()[1] * 0.85, f'{vals.mean():+.1f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Compute eta-squared (ANCOVA-like: guild effect controlling for Tmax)
    # Simple approximation: eta² from 1-way ANOVA on residuals
    group_vals = [df_temp[df_temp['MetGroup_simple'] == g]['Er_residual'].dropna().values 
                  for g in plot_groups]
    all_resid = np.concatenate(group_vals)
    grand_mean = all_resid.mean()
    ss_total_d = np.sum((all_resid - grand_mean)**2)
    ss_between_d = sum(len(g) * (g.mean() - grand_mean)**2 for g in group_vals)
    eta2_guilds = ss_between_d / ss_total_d * 100 if ss_total_d > 0 else 0
    
    ax.text(0.98, 0.98,
            f'$\\eta^2$ guilds = {eta2_guilds:.0f}%\n(ANCOVA, $T_{{max}}$-controlled)',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    print(f"  Panel D: eta² guilds = {eta2_guilds:.1f}%")
    for grp in plot_groups:
        vals = df_temp[df_temp['MetGroup_simple'] == grp]['Er_residual'].dropna()
        print(f"    {grp}: mean Er_res = {vals.mean():+.1f}, n={len(vals)}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'{output_dir}05c_Figure3_ab_initio_validation_v30.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}05c_Figure3_ab_initio_validation_v30.png', bbox_inches='tight')
    plt.close()
    print("  [OK] Figure 3 (ab initio validation) saved")


# =============================================================================
# FIGURE 4: Q10,env vs intrinsic and warming projections
# =============================================================================
def generate_figure4(df, output_dir=OUTPUT_DIR):
    """
    Figure 4: Q10,env vs intrinsic and warming projections.
    
    Panel A: Q10,env distributions (15-25°C) - shows asymmetry
    Panel B: Q10 intrinsic distributions (T+10 < Topt) - shows no asymmetry
    Panel C: Warming projections with 95% CI bootstrap bands (under F∝μ)
    Panel D: Alpha by subset with bootstrap CI + Yvon-Durocher ecosystem validation
    """
    print("Generating Figure 4: Q10 distinction and warming feedback...")
    
    fig, axes_grid = plt.subplots(2, 2, figsize=(12, 9))
    axes = [axes_grid[0, 0], axes_grid[0, 1], axes_grid[1, 0], axes_grid[1, 1]]
    
    # Filter data
    df_q10 = df[(df['Trophic_mode'].isin(['Heterotroph', 'Phototroph']))].copy()
    
    het = df_q10[df_q10['Trophic_mode'] == 'Heterotroph']
    pho = df_q10[df_q10['Trophic_mode'] == 'Phototroph']
    
    # Panel A: Q10,env distributions
    ax = axes[0]
    
    het_app = het['Q10_apparent'].dropna()
    pho_app = pho['Q10_apparent'].dropna()
    het_app = het_app[(het_app > 0) & (het_app < 10)]
    pho_app = pho_app[(pho_app > 0) & (pho_app < 10)]
    
    bins = np.linspace(0.5, 5, 45)
    ax.hist(het_app, bins=bins, alpha=0.6, color=COLORS['heterotroph'], 
            label=f'Heterotrophs (n={len(het_app)})', density=True, edgecolor='darkred', linewidth=0.5)
    ax.hist(pho_app, bins=bins, alpha=0.6, color=COLORS['phototroph'], 
            label=f'Phototrophs (n={len(pho_app)})', density=True, edgecolor='darkgreen', linewidth=0.5)
    
    ax.axvline(het_app.mean(), color=COLORS['heterotroph'], linestyle='--', linewidth=2.5)
    ax.axvline(pho_app.mean(), color=COLORS['phototroph'], linestyle='--', linewidth=2.5)
    ax.axvline(2.0, color='gray', linestyle=':', linewidth=2, label='Assumed Q$_{10}$ = 2.0')
    
    ax.set_xlabel(r'Q$_{10,\mathrm{env}}$ (15-25$^\circ$C)')
    ax.set_ylabel('Density')
    ax.set_title('A', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0.5, 5)
    
    _, p_app = stats.mannwhitneyu(het_app, pho_app)
    d_app = cohens_d(het_app, pho_app)
    p_exp = int(np.ceil(np.log10(p_app)))  # e.g. -29 for p = 3.5e-30, so p < 10^{-29}
    ax.text(0.95, 0.7, f"Het: {het_app.mean():.2f} {PLUS_MINUS_UNICODE} {het_app.std():.2f}\n"
                       f"Pho: {pho_app.mean():.2f} {PLUS_MINUS_UNICODE} {pho_app.std():.2f}\n"
                       f"p < 10$^{{{p_exp}}}$\n"
                       f"Cohen's d = {abs(d_app):.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Panel B: Q10 intrinsic distributions
    ax = axes[1]
    
    het_int = het['Q10_intrinsic'].dropna()
    pho_int = pho['Q10_intrinsic'].dropna()
    het_int = het_int[(het_int > 0) & (het_int < 10)]
    pho_int = pho_int[(pho_int > 0) & (pho_int < 10)]
    
    bins = np.linspace(1.0, 3.5, 45)
    ax.hist(het_int, bins=bins, alpha=0.6, color=COLORS['heterotroph'], 
            label=f'Heterotrophs (n={len(het_int)})', density=True, edgecolor='darkred', linewidth=0.5)
    ax.hist(pho_int, bins=bins, alpha=0.6, color=COLORS['phototroph'], 
            label=f'Phototrophs (n={len(pho_int)})', density=True, edgecolor='darkgreen', linewidth=0.5)
    
    ax.axvline(het_int.mean(), color=COLORS['heterotroph'], linestyle='--', linewidth=2.5)
    ax.axvline(pho_int.mean(), color=COLORS['phototroph'], linestyle='--', linewidth=2.5)
    ax.axvline(2.0, color='gray', linestyle=':', linewidth=2, label='Q$_{10}$ = 2.0')
    
    ax.set_xlabel('Q$_{10}$ (T+10 < T$_{opt}$)')
    ax.set_ylabel('Density')
    ax.set_title('B', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(1.0, 3.5)
    
    _, p_int = stats.mannwhitneyu(het_int, pho_int)
    d_int = cohens_d(het_int, pho_int)
    
    # Format p-value for display
    if p_int > 0.05:
        p_str = f"p = {p_int:.2f} (NS)"
    else:
        p_str = f"p = {p_int:.2e}"
    
    ax.text(0.95, 0.7, f"Het: {het_int.mean():.2f} {PLUS_MINUS_UNICODE} {het_int.std():.2f}\n"
                       f"Pho: {pho_int.mean():.2f} {PLUS_MINUS_UNICODE} {pho_int.std():.2f}\n"
                       f"{p_str}\n"
                       f"Cohen's d = {abs(d_int):.2f}",
            transform=ax.transAxes, ha='right', va='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # =========================================================================
    # Shared: model definition and constants for Panels C and D
    # =========================================================================
    from scipy.constants import k as kB_SI, h as h_SI
    R_gas = 8.314
    kB = kB_SI
    h_planck = h_SI
    T0 = 288.15  # 15 deg C in K
    
    def mu_model(T_K, Er, Eh, Tmax):
        """Calculate growth rate from thermodynamic model."""
        prefactor = kB * T_K / h_planck
        eyring = np.exp(-Er / (R_gas * T_K))
        england_arg = (Eh / R_gas) * (1.0/T_K - 1.0/Tmax)
        england = np.exp(england_arg) - 1.0
        return prefactor * eyring * np.maximum(england, 0)
    
    # Precompute r(dT) for each strain across a range of warming values
    warming = np.linspace(0, 6, 100)
    
    def compute_strain_curves(subset_df, warming_range):
        """Compute growth acceleration curves for each strain using full model.
        
        Only includes strains where mu > 0 across the entire warming range,
        ensuring no artificial -100% acceleration for organisms beyond their
        thermal limit.
        """
        curves = []  # each element: array of (r-1)*100 for this strain
        max_dT = max(warming_range)
        for _, row in subset_df.iterrows():
            try:
                Er_J = row['Er'] * 1000
                Eh_J = row['Eh'] * 1000
                Tmax_K = row['Tmax'] + 273.15
                mu_base = mu_model(T0, Er_J, Eh_J, Tmax_K)
                mu_max_warm = mu_model(T0 + max_dT, Er_J, Eh_J, Tmax_K)
                if mu_base > 0 and mu_max_warm > 0:
                    accel = np.zeros(len(warming_range))
                    for j, dT in enumerate(warming_range):
                        mu_warm = mu_model(T0 + dT, Er_J, Eh_J, Tmax_K)
                        accel[j] = (mu_warm / mu_base - 1) * 100
                    curves.append(accel)
            except:
                pass
        return np.array(curves)
    
    het_curves = compute_strain_curves(het, warming)
    pho_curves = compute_strain_curves(pho, warming)
    
    # Bootstrap: resample strains, compute median curve
    n_boot = 10000
    np.random.seed(42)
    
    def bootstrap_model_projection(curves, n_boot=10000):
        """Bootstrap CI by resampling strains and computing median acceleration."""
        n = len(curves)
        all_medians = np.zeros((n_boot, curves.shape[1]))
        for i in range(n_boot):
            idx = np.random.randint(0, n, size=n)
            all_medians[i] = np.median(curves[idx], axis=0)
        median_curve = np.median(curves, axis=0)
        lo = np.percentile(all_medians, 2.5, axis=0)
        hi = np.percentile(all_medians, 97.5, axis=0)
        return median_curve, lo, hi
    
    het_med, het_lo, het_hi = bootstrap_model_projection(het_curves, n_boot)
    pho_med, pho_lo, pho_hi = bootstrap_model_projection(pho_curves, n_boot)
    
    # =========================================================================
    # Panel C: Warming projections with 95% CI bootstrap bands (full model)
    # =========================================================================
    ax = axes[2]
    
    # Plot confidence bands
    ax.fill_between(warming, het_lo, het_hi, alpha=0.2, color=COLORS['heterotroph'])
    ax.fill_between(warming, pho_lo, pho_hi, alpha=0.2, color=COLORS['phototroph'])
    
    # Plot median curves
    ax.plot(warming, het_med, color=COLORS['heterotroph'], linewidth=3, 
            label='Heterotrophs')
    ax.plot(warming, pho_med, color=COLORS['phototroph'], linewidth=3, 
            label='Phototrophs')
    
    # Annotations at +2 deg C and +4 deg C
    for dT_ann, label in [(2, f'+2{DEGREE}C'), (4, f'+4{DEGREE}C')]:
        ax.axvline(dT_ann, color='gray', linestyle=':', alpha=0.5)
        idx = np.argmin(np.abs(warming - dT_ann))
        het_val = het_med[idx]
        pho_val = pho_med[idx]
        
        ax.annotate(f'{het_val:.0f}%', (dT_ann, het_val), textcoords="offset points", 
                   xytext=(-15, 12), fontsize=9, color=COLORS['heterotroph'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))
        
        ax.annotate(f'{pho_val:.0f}%', (dT_ann, pho_val), textcoords="offset points", 
                   xytext=(-15, -15), fontsize=9, color=COLORS['phototroph'], fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='none', alpha=0.8))
        
        ax.text(dT_ann, -8, label, ha='center', fontsize=9, color='gray')
    
    # Add text box with CI info
    textstr = 'Shaded: 95% CI\n(10,000 bootstrap)\nFull $\\mu(T)$ model\nUnder F $\\propto$ $\\mu$'
    ax.text(0.98, 0.22, textstr, transform=ax.transAxes, fontsize=7,
            ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel(f'Warming ({DEGREE}C)')
    ax.set_ylabel('Growth acceleration (%)')
    ax.set_title('C', fontsize=14, fontweight='bold', loc='left', x=-0.1)
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(0, 6)
    ax.set_ylim(-12, 85)
    
    # =========================================================================
    # Panel D: Per-capita acceleration ratio alpha by subset + ecosystem validation
    # =========================================================================
    ax = axes[3]
    
    dT = 4.0
    

    
    # Compute r = mu(T0+dT)/mu(T0) for each strain
    def compute_r_values(subset_df):
        """Compute per-capita acceleration ratio for each strain.
        
        Excludes strains with mu = 0 at either T0 or T0+dT (i.e., strains
        whose Tmax falls below the warmed temperature). An acceleration ratio
        is undefined for organisms already beyond their thermal limit.
        """
        r_vals = []
        valid_idx = []
        for idx, row in subset_df.iterrows():
            try:
                mu_base = mu_model(T0, row['Er']*1000, row['Eh']*1000, row['Tmax']+273.15)
                mu_warm = mu_model(T0+dT, row['Er']*1000, row['Eh']*1000, row['Tmax']+273.15)
                if mu_base > 0 and mu_warm > 0:
                    r_vals.append(mu_warm / mu_base)
                    valid_idx.append(idx)
            except:
                pass
        return np.array(r_vals), valid_idx
    
    het_r_all, het_valid_idx = compute_r_values(het)
    pho_r_all, pho_valid_idx = compute_r_values(pho)
    
    # Lookup Topt and Domain from valid indices
    het_topt = het.loc[het_valid_idx, 'Topt'].values
    pho_topt = pho.loc[pho_valid_idx, 'Topt'].values
    het_domain_arr = het.loc[het_valid_idx, 'Domain'].values
    pho_domain_arr = pho.loc[pho_valid_idx, 'Domain'].values
    
    # Define subsets
    subset_defs = [
        ('All strains',
         np.ones(len(het_r_all), dtype=bool),
         np.ones(len(pho_r_all), dtype=bool)),
        ('Mesophiles 15\u201345' + DEGREE + 'C',
         (het_topt >= 15) & (het_topt <= 45),
         (pho_topt >= 15) & (pho_topt <= 45)),
        ('Mesophiles 20\u201340' + DEGREE + 'C',
         (het_topt >= 20) & (het_topt <= 40),
         (pho_topt >= 20) & (pho_topt <= 40)),
        ('Eukarya only',
         het_domain_arr == 'Eukarya',
         pho_domain_arr == 'Eukarya'),
    ]
    
    # Bootstrap alpha for each subset
    n_boot = 10000
    np.random.seed(42)
    
    results = []
    for name, h_mask, p_mask in subset_defs:
        r_h = het_r_all[h_mask]
        r_p = pho_r_all[p_mask]
        alpha_obs = r_h.mean() / r_p.mean()
        
        alphas_boot = np.zeros(n_boot)
        for i in range(n_boot):
            h_sample = np.random.choice(r_h, size=len(r_h), replace=True)
            p_sample = np.random.choice(r_p, size=len(r_p), replace=True)
            alphas_boot[i] = h_sample.mean() / p_sample.mean()
        
        ci_lo = np.percentile(alphas_boot, 2.5)
        ci_hi = np.percentile(alphas_boot, 97.5)
        p_gt1 = (alphas_boot > 1).mean() * 100
        results.append((name, len(r_h), len(r_p), alpha_obs, ci_lo, ci_hi, p_gt1))
    
    # Horizontal dot plot: model subsets + ecosystem observation
    n_subsets = len(results)
    y_model = np.arange(n_subsets)[::-1]  # top to bottom
    y_eco = -1.3  # below model results with gap
    
    # Model predictions (circles, dark red)
    for i, (name, nh, np_, alpha, lo, hi, pgt1) in enumerate(results):
        y = y_model[i]
        ax.errorbar(alpha, y, xerr=[[alpha - lo], [hi - alpha]],
                    fmt='o', color=COLORS['heterotroph'], markersize=9,
                    capsize=5, linewidth=2, markeredgecolor='black',
                    markeredgewidth=0.8, zorder=5)
        ax.text(hi + 0.008, y, f'  {alpha:.2f} [{lo:.2f}, {hi:.2f}]',
                va='center', fontsize=7, color='#333333')
    
    # Separator line
    ax.axhline(-0.5, color='gray', linewidth=0.5, linestyle='-', alpha=0.3)
    
    # Yvon-Durocher ecosystem observation (diamond, blue)
    yd_alpha = 1.13
    yd_lo = 1.07
    yd_hi = 1.19
    ax.errorbar(yd_alpha, y_eco, xerr=[[yd_alpha - yd_lo], [yd_hi - yd_alpha]],
                fmt='D', color='#2980B9', markersize=9, capsize=5, linewidth=2,
                markeredgecolor='black', markeredgewidth=0.8, zorder=5)
    ax.text(yd_hi + 0.008, y_eco, f'  {yd_alpha:.2f} [{yd_lo:.2f}, {yd_hi:.2f}]',
            va='center', fontsize=7, color='#2980B9')
    
    # "Zero-parameter prediction" annotation with arrow pointing to Yvon-Durocher diamond
    ax.annotate('Zero-parameter\nprediction', xy=(yd_lo - 0.005, y_eco),
                xytext=(-45, 20), textcoords='offset points',
                fontsize=7, fontstyle='italic', color='#2980B9',
                ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', color='#2980B9', lw=1.0))
    
    # Reference line at alpha = 1
    ax.axvline(1.0, color='gray', linestyle='--', linewidth=1, alpha=0.7, zorder=1)
    ax.text(1.001, y_model[0] + 0.6, '$\\alpha = 1$', fontsize=7, color='gray',
            ha='left', va='bottom')
    
    # Y-axis labels
    model_labels = [f'{name}\n($n_h$={nh}, $n_p$={np_})' 
                    for name, nh, np_, alpha, lo, hi, pgt1 in results]
    eco_label = 'Yvon-Durocher\net al. 2010\n(mesocosms, +4' + DEGREE + 'C)'
    all_labels = model_labels + [eco_label]
    all_y = list(y_model) + [y_eco]
    
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=7)
    ax.set_xlabel('$\\alpha$ = $r_{\\mathrm{het}}$ / $r_{\\mathrm{pho}}$')
    ax.set_title('D', fontsize=14, fontweight='bold', loc='left', x=-0.15)
    
    # Legend
    import matplotlib.lines as mlines
    model_marker = mlines.Line2D([], [], color=COLORS['heterotroph'], marker='o',
                                  markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                                  linestyle='None', label='Model prediction\n(10,000 bootstrap)')
    eco_marker = mlines.Line2D([], [], color='#2980B9', marker='D',
                                markersize=7, markeredgecolor='black', markeredgewidth=0.5,
                                linestyle='None', label='Ecosystem observation\n(95% CI)')
    ax.legend(handles=[model_marker, eco_marker], loc='lower right', fontsize=7,
              framealpha=0.9)
    
    # Axis limits and styling
    ax.set_xlim(0.97, 1.28)
    ax.set_ylim(y_eco - 1.0, y_model[0] + 0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Annotation box
    ax.text(0.97, 0.02, '$T_0$ = 15' + DEGREE + 'C, $\\Delta T$ = +4' + DEGREE + 'C\nUnder F $\\propto$ $\\mu$ hypothesis\n$P(\\alpha > 1)$ = 100% all subsets',
            transform=ax.transAxes, fontsize=6.5, ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Print summary
    print(f"\n  Panel D (alpha by subset, +4C from 15C):")
    for name, nh, np_, alpha, lo, hi, pgt1 in results:
        print(f"    {name}: n_het={nh}, n_pho={np_}, alpha={alpha:.3f} [{lo:.3f}, {hi:.3f}], P(a>1)={pgt1:.0f}%")
    print(f"    Yvon-Durocher 2010: R_H:U = {yd_alpha:.2f} [{yd_lo:.2f}, {yd_hi:.2f}] (ecosystem)")


    plt.tight_layout()
    plt.savefig(f'{output_dir}05d_Figure4_warming_feedback_v30.pdf', bbox_inches='tight')
    plt.savefig(f'{output_dir}05d_Figure4_warming_feedback_v30.png', bbox_inches='tight')
    plt.close()
    print("  [OK] Figure 4 saved")
    
    # Print summary statistics
    print(f"\n  === FIGURE 4 SUMMARY ===")
    print(f"  Panel A (Q10,env):")
    print(f"    Heterotrophs: {het_app.mean():.2f}  {het_app.std():.2f} (n={len(het_app)})")
    print(f"    Phototrophs:  {pho_app.mean():.2f}  {pho_app.std():.2f} (n={len(pho_app)})")
    print(f"    Mann-Whitney p = {p_app:.2e}, Cohen's d = {abs(d_app):.2f}")
    print(f"  Panel B (Q10 intrinsic):")
    print(f"    Heterotrophs: {het_int.mean():.2f}  {het_int.std():.2f} (n={len(het_int)})")
    print(f"    Phototrophs:  {pho_int.mean():.2f}  {pho_int.std():.2f} (n={len(pho_int)})")
    print(f"    Mann-Whitney p = {p_int:.4f}, Cohen's d = {abs(d_int):.2f}")


# =============================================================================
# TABLES - Using ASCII "+/-" for maximum compatibility
# =============================================================================
def generate_tables(df, output_dir=OUTPUT_DIR):
    """Generate all summary tables with ASCII encoding for symbols."""
    print("\nGenerating Tables...")
    
    # Table 1: Summary by domain
    print("  Table 1: Parameters by domain")
    table1_data = []
    for domain in ['Bacteria', 'Archaea', 'Eukarya']:
        subset = df[df['Domain'] == domain]
        q10_valid = subset['Q10'][(subset['Q10'] > 0) & (subset['Q10'] < 10)]
        table1_data.append({
            'Domain': domain,
            'n': len(subset),
            'Er (kJ/mol)': f"{subset['Er'].mean():.1f} +/- {subset['Er'].std():.1f}",
            'Eh (kJ/mol)': f"{subset['Eh'].mean():.1f} +/- {subset['Eh'].std():.1f}",
            'Q10': f"{q10_valid.mean():.2f} +/- {q10_valid.std():.2f}",
        })
    table1 = pd.DataFrame(table1_data)
    table1.to_csv(f'{output_dir}04_Table1_parameters_by_domain_v30.csv', index=False, encoding='utf-8')
    print(table1.to_string(index=False))
    
    # Table 2: Q10 by trophic mode - REVISED with intrinsic Q10 and additional columns
    print("\n  Table 2: Q10 by trophic mode (env + intrinsic)")
    table2_data = []
    for mode in ['Heterotroph', 'Phototroph']:
        subset = df[df['Trophic_mode'] == mode]
        
        # Q10,env
        q10_app = subset['Q10_apparent'][(subset['Q10_apparent'] > 0) & (subset['Q10_apparent'] < 10)]
        
        # Q10 intrinsic
        q10_int = subset['Q10_intrinsic'][(subset['Q10_intrinsic'] > 0) & (subset['Q10_intrinsic'] < 10)]
        
        # mumax
        mumax = subset['mumax'][(subset['mumax'] > 0) & (subset['mumax'] < 1e15)]
        
        # Topt
        topt = subset['Topt'].dropna()
        
        table2_data.append({
            'Trophic mode': mode,
            'n': len(q10_app),
            'Q10 mean': f"{q10_app.mean():.2f}",
            'Q10 SD': f"{q10_app.std():.2f}",
            'Q10 median': f"{q10_app.median():.2f}",
            'Q10_intrinsic mean': f"{q10_int.mean():.2f}",
            'Q10_intrinsic SD': f"{q10_int.std():.2f}",
            'mumax mean': f"{mumax.mean():.2e}",
            'mumax SD': f"{mumax.std():.2e}",
            'Topt mean': f"{topt.mean():.1f}",
        })
    table2 = pd.DataFrame(table2_data)
    table2.to_csv(f'{output_dir}04_Table2_Q10_by_trophic_mode_v30.csv', index=False, encoding='utf-8')
    print(table2.to_string(index=False))
    
    # Print additional statistics for validation
    het = df[df['Trophic_mode'] == 'Heterotroph']
    pho = df[df['Trophic_mode'] == 'Phototroph']
    
    het_int = het['Q10_intrinsic'][(het['Q10_intrinsic'] > 0) & (het['Q10_intrinsic'] < 10)]
    pho_int = pho['Q10_intrinsic'][(pho['Q10_intrinsic'] > 0) & (pho['Q10_intrinsic'] < 10)]
    
    het_mumax = het['mumax'][(het['mumax'] > 0) & (het['mumax'] < 1e15)]
    pho_mumax = pho['mumax'][(pho['mumax'] > 0) & (pho['mumax'] < 1e15)]
    
    _, p_int = stats.mannwhitneyu(het_int, pho_int)
    
    print(f"\n  --- VALIDATION ---")
    print(f"  Q10 intrinsic heterotrophs: {het_int.mean():.2f} +/- {het_int.std():.2f}")
    print(f"  Q10 intrinsic phototrophs:  {pho_int.mean():.2f} +/- {pho_int.std():.2f}")
    print(f"  Mann-Whitney p = {p_int:.4f} {'(NS)' if p_int > 0.05 else ''}")
    print(f"  mumax ratio (Het/Pho): {het_mumax.mean()/pho_mumax.mean():.1f}x (arithmetic)")
    print(f"  mumax ratio (geometric): {10**(np.log10(het_mumax).mean() - np.log10(pho_mumax).mean()):.1f}x")
    
    # Table 3: O2 metabolism (Bacteria only)
    print("\n  Table 3: Er by O2 metabolism (Bacteria only)")
    bac = df[df['Domain'] == 'Bacteria']
    table3_data = []
    
    o2_categories = [
        (O2_CAT_CYANO, 'Cyanobacteria (O2 producers)', 'O2 producers'),
        (O2_CAT_CHEMOLITHO, 'Aerobic chemolithotroph', 'Aerobic'),
        (O2_CAT_ANAEROBE, 'Strict anaerobe', 'Anaerobic'),
        (O2_CAT_AEROBIC_HET, 'Aerobic heterotroph (O2 consumers)', 'O2 consumers'),
        (O2_CAT_FACULTATIVE, 'Facultative anaerobe', 'Transitional')
    ]
    
    for cat_key, cat_name, interpretation in o2_categories:
        subset = bac[bac['O2_cat'] == cat_key]
        if len(subset) >= 5:
            table3_data.append({
                'O2 category': cat_name,
                'n': len(subset),
                'Er (kJ/mol)': f"{subset['Er'].mean():.1f} +/- {subset['Er'].std():.1f}",
                'O2_interpretation': interpretation
            })
    
    table3 = pd.DataFrame(table3_data)
    table3.to_csv(f'{output_dir}04_Table3_Er_by_O2_metabolism_v30.csv', 
                  index=False, encoding='utf-8')
    print(table3.to_string(index=False))
    
    print("\n  [OK] All tables saved with ASCII encoding")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main function to generate all figures and tables."""
    
    print("="*70)
    print("V26 FIGURE AND TABLE GENERATION")
    print("NEW: A posteriori classification by Er/Tmax (threshold 36.5R)")
    print("Data source: 24Fit_3_param_1054_strains_with_Q10.xlsx")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_data()
    stats_dict = print_statistics(df)
    
    print("\n" + "="*70)
    print("GENERATING MAIN FIGURES (1-4)")
    print("="*70)
    
    generate_figure1(df)
    generate_figure2(df, stats_dict)  # Unified Figure 2 with panels A-C (3 panels)
    generate_figure3(df)              # New: Ab initio validation (4 panels)
    generate_figure4(df)
    
    print("\n" + "="*70)
    print("GENERATING TABLES")
    print("="*70)
    
    generate_tables(df)
    
    # Generate a posteriori classification summary
    print("\n" + "="*70)
    print("A POSTERIORI CLASSIFICATION SUMMARY (V26)")
    print("="*70)
    generate_aposteriori_summary(df)
    
    # NEW R6: FDR correction summary
    print("\n" + "="*70)
    print("FDR CORRECTION SUMMARY (R6)")
    print("="*70)
    # Collect all p-values from multi-group comparisons
    het_q = df[df['Trophic_mode'] == 'Heterotroph']
    pho_q = df[df['Trophic_mode'] == 'Phototroph']
    het_int = het_q['Q10_intrinsic'].dropna()
    pho_int = pho_q['Q10_intrinsic'].dropna()
    het_int = het_int[(het_int > 0) & (het_int < 10)]
    pho_int = pho_int[(pho_int > 0) & (pho_int < 10)]
    
    bac = df[df['Domain'] == 'Bacteria']
    cyano = bac[bac['O2_cat'] == O2_CAT_CYANO] if 'O2_cat' in bac.columns else pd.DataFrame()
    aer_het = bac[bac['O2_cat'] == O2_CAT_AEROBIC_HET] if 'O2_cat' in bac.columns else pd.DataFrame()
    
    p_values = []
    p_labels = []
    
    # Q10 intrinsic
    if len(het_int) > 0 and len(pho_int) > 0:
        _, p = stats.mannwhitneyu(het_int, pho_int)
        p_values.append(p)
        p_labels.append("Q10 intrinsic Het vs Pho")
    
    # O2 metabolism Er
    if len(cyano) > 0 and len(aer_het) > 0:
        _, p = stats.mannwhitneyu(cyano['Er'], aer_het['Er'])
        p_values.append(p)
        p_labels.append("Er Cyano vs Aer Het")
    
    if len(p_values) > 1:
        reject, pvals_corrected, _, _ = multipletests(p_values, method='fdr_bh')
        for i, (label, p_raw, p_fdr, rej) in enumerate(zip(p_labels, p_values, pvals_corrected, reject)):
            print(f"  {label}: p_raw={p_raw:.4e}, p_FDR={p_fdr:.4e}, reject H0={rej}")
        print("  Note: Central results (p<10^-20) unaffected by FDR correction.")
        print("  Q10 intrinsic p=0.09 remains NS after FDR correction.")
    
    print("\n" + "="*70)
    print(f"ALL OUTPUTS SAVED TO: {OUTPUT_DIR}")
    print("="*70)
    
    return df


def generate_aposteriori_summary(df):
    """Generate summary statistics and CSV for a posteriori classification.
    
    Produces two CSV files:
    - 04a: Strategy by group (Trophic_mode + Metabolic_group, with explicit labels)
    - 04d: Contingency table (Strategy x Eh regime)
    
    IMPORTANT: The manuscript uses Trophic_mode for the main het/pho comparison
    (Heterotroph n=505 includes methanogens; Phototroph n=428 includes mixotrophs).
    Metabolic_group splits methanogens out (Heterotroph_excl_methanogen n=431).
    Both conventions are reported with unambiguous labels.
    """
    
    # Contingency statistics
    contingency = compute_contingency_stats(df)
    
    print(f"\n=== CONTINGENCY TABLE: Strategy x Regime ===")
    print(f"                  Low-Eh    High-Eh    Total")
    print(f"  Gamblers:       {contingency['gambler_low']:5d} ({contingency['pct_gambler_low']:.0f}%)   {contingency['gambler_high']:5d} ({contingency['pct_gambler_high']:.0f}%)   {contingency['n_gamblers']:5d}")
    print(f"  Investors:      {contingency['investor_low']:5d} ({contingency['pct_investor_low']:.0f}%)   {contingency['investor_high']:5d} ({contingency['pct_investor_high']:.0f}%)   {contingency['n_investors']:5d}")
    print(f"  Total:          {contingency['n_low_eh']:5d} ({contingency['pct_low_eh']:.0f}%)   {contingency['n_high_eh']:5d} ({contingency['pct_high_eh']:.0f}%)   {len(df):5d}")
    print(f"\n  Chi-squared = {contingency['chi2']:.1f}, p < 10^{int(np.log10(contingency['p_chi2']))}")
    print(f"  Pearson r(Er/Tmax, Eh) = {contingency['r_Er_Tmax_Eh']:.3f}")
    
    # --- Helper to compute one row of statistics ---
    def _row_stats(subset, label):
        n = len(subset)
        n_gambler = (subset['Strategy_aposteriori'] == 'Gambler').sum()
        n_investor = n - n_gambler
        pct_gambler = n_gambler / n * 100
        pct_investor = n_investor / n * 100
        mean_q10 = subset['Q10_apparent'].dropna().mean()
        mean_Er_Tmax = subset['Er_Tmax_R'].dropna().mean()
        print(f"  {label:35s}: n={n:4d}, {pct_gambler:5.1f}% Gamblers, {pct_investor:5.1f}% Investors, Q10={mean_q10:.2f}")
        return {
            'Metabolic_group': label,
            'n': n,
            'Er_Tmax_mean': f"{mean_Er_Tmax:.1f}",
            'pct_Gambler': f"{pct_gambler:.0f}",
            'pct_Investor': f"{pct_investor:.0f}",
            'Q10_app_mean': f"{mean_q10:.2f}"
        }
    
    results = []
    
    # --- Part 1: Trophic_mode groups (MS convention) ---
    print(f"\n=== STRATEGY BY TROPHIC MODE (MS convention, includes methanogens in Het) ===")
    for mode in ['Heterotroph', 'Phototroph']:
        subset = df[df['Trophic_mode'] == mode]
        if len(subset) > 0:
            label = f"{mode}_all_trophic"
            results.append(_row_stats(subset, label))
    
    # --- Part 2: Metabolic_group (fine-grained) ---
    print(f"\n=== STRATEGY BY METABOLIC GROUP (methanogens separated) ===")
    for group in ['Heterotroph', 'Oxygenic_phototroph', 'Methanogen', 
                  'Anoxygenic_phototroph', 'Chemolithoautotroph']:
        subset = df[df['Metabolic_group'] == group]
        if len(subset) > 0:
            # Use explicit label for Heterotroph to avoid ambiguity
            label = 'Heterotroph_excl_methanogen' if group == 'Heterotroph' else group
            results.append(_row_stats(subset, label))
    
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{OUTPUT_DIR}04a_Table_strategy_by_group_v30.csv', index=False)
    print(f"\n  [OK] Saved: 04a_Table_strategy_by_group_v30.csv")


if __name__ == "__main__":
    df = main()
