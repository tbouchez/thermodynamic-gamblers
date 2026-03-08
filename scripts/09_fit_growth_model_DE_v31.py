#!/usr/bin/env python3
"""
09_fit_growth_model_DE_v31.py

Differential Evolution fitting of the thermodynamic growth model.

This script fits the non-equilibrium thermodynamic growth model to microbial
growth rate data using Differential Evolution global optimization.

Model equation (derived from Crooks fluctuation theorem):
    μ(T) = (kB·T/h) · exp(-Er/RT) · [exp(Eh/R · (1/T - 1/Tmax)) - 1]

where:
    - kB: Boltzmann constant (1.380649e-23 J/K)
    - h: Planck constant (6.62607e-34 J·s)
    - R: Gas constant (8.314 J/mol/K = 8.314e-3 kJ/mol/K)
    - Er: Activation enthalpy (kJ/mol) - thermal impulse dampening
    - Eh: Inactivation enthalpy (kJ/mol) - thermodynamic driving force
    - Tmax: Maximum temperature (K) - upper thermal limit
    - T: Temperature (K)

The model has 3 free parameters: Er, Eh, Tmax
The optimal temperature Topt emerges from the constraint dμ/dT = 0

Author: Théodore Bouchez, INRAE
Date: January 2026
"""

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================
kB = 1.380649e-23    # Boltzmann constant (J/K)
h = 6.62607015e-34   # Planck constant (J·s)
R = 8.314e-3         # Gas constant (kJ/mol/K)
R_J = 8.314          # Gas constant (J/mol/K)

# Prefactor kB/h in s^-1
PREFACTOR = kB / h   # ≈ 2.08e10 s^-1

# =============================================================================
# THERMODYNAMIC GROWTH MODEL
# =============================================================================

def growth_rate(T_K, Er, Eh, Tmax_K):
    """
    Calculate growth rate using the thermodynamic model.
    
    Parameters
    ----------
    T_K : float or array
        Temperature in Kelvin
    Er : float
        Activation enthalpy in kJ/mol
    Eh : float
        Inactivation enthalpy in kJ/mol
    Tmax_K : float
        Maximum temperature in Kelvin
        
    Returns
    -------
    mu : float or array
        Growth rate in h^-1
    """
    # Convert Er, Eh to J/mol for calculation
    Er_J = Er * 1000
    Eh_J = Eh * 1000
    
    # Thermal impulse term: (kB·T/h) · exp(-Er/RT)
    thermal_impulse = PREFACTOR * T_K * np.exp(-Er_J / (R_J * T_K))
    
    # Net assembly probability (Crooks): [exp(Eh/R · (1/T − 1/Tmax)) − 1]
    # This term equals 0 at T = Tmax and diverges as T << Tmax
    crooks_arg = (Eh_J / R_J) * (1.0 / T_K - 1.0 / Tmax_K)
    driving_force = np.exp(crooks_arg) - 1.0
    
    # Ensure driving_force is non-negative (can be negative if T > Tmax)
    driving_force = np.maximum(driving_force, 0)
    
    # Growth rate in s^-1, convert to h^-1
    mu = thermal_impulse * driving_force * 3600
    
    return mu


def find_Topt(Er, Eh, Tmax_K, T_range=(273, 373)):
    """
    Find optimal temperature numerically.
    
    Parameters
    ----------
    Er, Eh : float
        Model parameters in kJ/mol
    Tmax_K : float
        Maximum temperature in K
    T_range : tuple
        Search range in K
        
    Returns
    -------
    Topt_K : float
        Optimal temperature in K
    """
    # Grid search for initial estimate
    T_grid = np.linspace(max(T_range[0], 250), min(T_range[1], Tmax_K - 0.1), 200)
    mu_grid = growth_rate(T_grid, Er, Eh, Tmax_K)
    Topt_init = T_grid[np.argmax(mu_grid)]
    
    # Refine with local optimization (minimize negative growth rate)
    def neg_mu(T):
        return -growth_rate(T[0], Er, Eh, Tmax_K)
    
    try:
        result = minimize(neg_mu, [Topt_init], 
                         bounds=[(T_range[0], Tmax_K - 0.1)],
                         method='L-BFGS-B')
        return result.x[0]
    except:
        return Topt_init


def calculate_Q10(Er, Tref_C=20):
    """
    Calculate intrinsic Q10 from activation enthalpy.
    
    Q10_intrinsic = exp(10 * Er / (R * Tref^2))
    
    This is the Q10 on the ascending phase of the thermal response curve,
    where the Boltzmann term dominates.
    
    Parameters
    ----------
    Er : float
        Activation enthalpy in kJ/mol
    Tref_C : float
        Reference temperature in Celsius (default: 20°C)
        
    Returns
    -------
    Q10 : float
        Temperature sensitivity coefficient
    """
    Tref_K = Tref_C + 273.15
    Q10 = np.exp(10 * Er / (R * Tref_K**2))
    return Q10


def calculate_Q10_apparent(T_data, mu_data, T_ref=20, T_range=10):
    """
    Calculate apparent Q10 from observed growth rate data.
    
    Uses linear regression of log(mu) vs T on the ascending phase.
    
    Parameters
    ----------
    T_data : array
        Temperature data in Celsius
    mu_data : array
        Growth rate data in h^-1
    T_ref : float
        Reference temperature for Q10 calculation
    T_range : float
        Range around T_ref to use for fitting
        
    Returns
    -------
    Q10_app : float
        Apparent Q10 coefficient
    """
    # Find ascending phase (temperatures below optimum)
    idx_max = np.argmax(mu_data)
    T_ascending = T_data[:idx_max+1]
    mu_ascending = mu_data[:idx_max+1]
    
    # Filter valid data points
    valid = (mu_ascending > 0) & np.isfinite(mu_ascending)
    if np.sum(valid) < 2:
        return np.nan
    
    T_valid = T_ascending[valid]
    mu_valid = mu_ascending[valid]
    
    # Linear regression of log(mu) vs T
    try:
        slope, intercept = np.polyfit(T_valid, np.log(mu_valid), 1)
        Q10_app = np.exp(10 * slope)
        return Q10_app
    except:
        return np.nan


# =============================================================================
# FITTING FUNCTIONS
# =============================================================================

def objective_function(params, T_data, mu_data):
    """
    Objective function for DE optimization (sum of squared errors).
    
    Parameters
    ----------
    params : array
        [Er, Eh, Tmax_K] - model parameters
    T_data : array
        Temperature data in Kelvin
    mu_data : array
        Growth rate data in h^-1
        
    Returns
    -------
    sse : float
        Sum of squared errors
    """
    Er, Eh, Tmax_K = params
    
    # Calculate predicted growth rates
    mu_pred = growth_rate(T_data, Er, Eh, Tmax_K)
    
    # Sum of squared errors
    residuals = mu_data - mu_pred
    sse = np.sum(residuals**2)
    
    return sse


def calculate_NRMSE(mu_data, mu_pred):
    """
    Calculate Normalized Root Mean Square Error.
    
    NRMSE = RMSE / (max(mu) - min(mu))
    
    Parameters
    ----------
    mu_data : array
        Observed growth rates
    mu_pred : array
        Predicted growth rates
        
    Returns
    -------
    nrmse : float
        Normalized RMSE (0-1 scale, lower is better)
    """
    rmse = np.sqrt(np.mean((mu_data - mu_pred)**2))
    range_mu = np.max(mu_data) - np.min(mu_data)
    if range_mu > 0:
        nrmse = rmse / range_mu
    else:
        nrmse = np.nan
    return nrmse


def fit_strain_DE(T_C, mu_data, maxiter=1000, seed=None):
    """
    Fit the thermodynamic model to a single strain using Differential Evolution.
    
    Parameters
    ----------
    T_C : array
        Temperature data in Celsius
    mu_data : array
        Growth rate data in h^-1
    maxiter : int
        Maximum number of DE iterations
    seed : int
        Random seed for reproducibility
        
    Returns
    -------
    results : dict
        Fitted parameters and diagnostics
    """
    # Convert temperature to Kelvin
    T_K = T_C + 273.15
    
    # Estimate initial bounds from data
    T_max_data = np.max(T_C)
    T_min_data = np.min(T_C)
    
    # Parameter bounds: [Er, Eh, Tmax_K]
    # Er: 50-200 kJ/mol (typical range for biological processes)
    # Eh: 0.1-300 kJ/mol (can be very small or large)
    # Tmax: must be > max observed temperature
    bounds = [
        (50, 200),                    # Er (kJ/mol)
        (0.1, 300),                   # Eh (kJ/mol)
        (T_max_data + 273.15, 400)    # Tmax (K), max 127°C
    ]
    
    # Run Differential Evolution
    result = differential_evolution(
        objective_function,
        bounds,
        args=(T_K, mu_data),
        maxiter=maxiter,
        seed=seed,
        strategy='best1bin',
        mutation=(0.5, 1.0),
        recombination=0.7,
        polish=True,
        tol=1e-8,
        atol=1e-8,
        workers=1
    )
    
    # Extract fitted parameters
    Er, Eh, Tmax_K = result.x
    
    # Calculate derived quantities
    Tmax_C = Tmax_K - 273.15
    Topt_K = find_Topt(Er, Eh, Tmax_K)
    Topt_C = Topt_K - 273.15
    
    # Calculate predicted values and NRMSE
    mu_pred = growth_rate(T_K, Er, Eh, Tmax_K)
    nrmse = calculate_NRMSE(mu_data, mu_pred)
    
    # Calculate Q10 values
    Q10_intrinsic = calculate_Q10(Er)
    Q10_apparent = calculate_Q10_apparent(T_C, mu_data)
    
    return {
        'Er': Er,
        'Eh': Eh,
        'Tmax_Thermo': Tmax_C,
        'Topt_Thermo': Topt_C,
        'NRMSE': nrmse,
        'Q10_intrinsic': Q10_intrinsic,
        'Q10_apparent': Q10_apparent,
        'n_points': len(T_C),
        'success': result.success,
        'n_iterations': result.nit
    }


# =============================================================================
# POPULATION CLASSIFICATION
# =============================================================================

def classify_population(Eh_values, threshold=20):
    """
    Classify strains into Low-Eh and High-Eh populations.
    
    Based on Gaussian mixture analysis, strains with Eh < 20 kJ/mol
    belong to the Low-Eh population, others to High-Eh.
    
    Parameters
    ----------
    Eh_values : array
        Eh values in kJ/mol
    threshold : float
        Threshold for classification (default: 20 kJ/mol)
        
    Returns
    -------
    population : array
        'Low_Eh' or 'High_Eh' for each strain
    """
    return np.where(Eh_values < threshold, 'Low_Eh', 'High_Eh')


def determine_trophic_mode(guild):
    """
    Determine trophic mode from guild annotation.
    
    CORRECTED R7.2: The original function had three errors:
      1. 'chemo' keyword matched chemolithoAUTOtrophs as Heterotroph
      2. 'methan' keyword matched hydrogenotrophic methanogens (CO2-fixers)
         as Heterotroph instead of Chemoautotroph
      3. Acetogens and nitrifiers fell through to 'Unknown'
    
    Classification logic (order matters — first match wins):
      1. Phototrophs: guild contains photo/cyano/alga
      2. Chemoautotrophs: chemolitho/nitrif/ammonia-oxid/hydrogen-oxid/
         iron-oxid/sulfur-oxid, OR hydrogenotrophic methanogens
      3. Other: methanotrophs (CH4-oxidizers, ambiguous C source)
      4. Heterotrophs: hetero/fermen/respir/predator/grazer/decomposer/
         halophilic/acetogen, OR versatile methanogens (Methanosarcina)
      5. Unknown: everything else
    
    Parameters
    ----------
    guild : str
        Guild annotation from database
        
    Returns
    -------
    trophic : str
        'Phototroph', 'Heterotroph', 'Chemoautotroph', 'Other', or 'Unknown'
    """
    guild_lower = str(guild).lower()
    
    # 1. Phototrophs (including mixotrophic phototrophs)
    if any(x in guild_lower for x in ['photo', 'cyano', 'alga']):
        return 'Phototroph'
    
    # 2. Chemoautotrophs: CO2-fixing chemolithotrophs
    #    Hydrogenotrophic methanogens fix CO2 via Wood-Ljungdahl pathway
    #    Note: 'nitrif' removed — captured by 'ammonia-oxid' instead,
    #    avoiding false match on 'denitrif' (denitrifiers are heterotrophs)
    if any(x in guild_lower for x in ['chemolitho', 'ammonia-oxid',
                                       'hydrogen-oxid', 'iron-oxid', 'sulfur-oxid']):
        return 'Chemoautotroph'
    if 'methanogen' in guild_lower and 'hydrogenotrophic' in guild_lower:
        return 'Chemoautotroph'
    
    # 3. Methanotrophs: CH4 as carbon source (ambiguous — neither strict 
    #    heterotroph nor autotroph); classified as 'Other'
    if 'methanotroph' in guild_lower:
        return 'Other'
    
    # 4. Heterotrophs: organic carbon consumers
    #    Includes versatile methanogens (Methanosarcina: acetoclastic pathway)
    #    and halophilic archaea (aerobic chemoorganotrophs)
    if any(x in guild_lower for x in ['hetero', 'fermen', 'respir', 'predator',
                                       'grazer', 'decomposer', 'halophilic',
                                       'acetogen']):
        return 'Heterotroph'
    if 'methanogen' in guild_lower:
        # Remaining methanogens (versatile/methylotrophic) → Heterotroph
        return 'Heterotroph'
    
    return 'Unknown'


def determine_domain(phylum):
    """
    Determine domain from phylum annotation.
    
    Parameters
    ----------
    phylum : str
        Phylum annotation
        
    Returns
    -------
    domain : str
        'Bacteria', 'Archaea', or 'Eukarya'
    """
    phylum_lower = str(phylum).lower()
    
    # Archaeal phyla
    archaea_phyla = ['euryarchaeota', 'crenarchaeota', 'thaumarchaeota', 
                    'korarchaeota', 'nanoarchaeota', 'archaea']
    
    # Eukaryotic groups
    eukarya_groups = ['fungi', 'ascomycota', 'basidiomycota', 'zygomycota',
                     'ciliophora', 'dinoflagellata', 'bacillariophyta', 'diatom',
                     'chlorophyta', 'rhodophyta', 'stramenopiles', 'alveolata',
                     'euglenozoa', 'amoebozoa', 'opisthokonta', 'microsporidia']
    
    if any(x in phylum_lower for x in archaea_phyla):
        return 'Archaea'
    elif any(x in phylum_lower for x in eukarya_groups):
        return 'Eukarya'
    else:
        return 'Bacteria'


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(input_file='07a_growth_data_1054_strains_v31.xlsx',
         output_file='07b_DE_results_1054_strains_v31.xlsx',
         seed=42):
    """
    Main function to fit all strains and save results.
    
    Parameters
    ----------
    input_file : str
        Path to growth data Excel file
    output_file : str
        Path to save results
    seed : int
        Random seed for reproducibility
    """
    print("=" * 60)
    print("THERMODYNAMIC GROWTH MODEL FITTING")
    print("Differential Evolution Optimization")
    print("=" * 60)
    print()
    
    # Load growth data
    print(f"Loading data from {input_file}...")
    growth_data = pd.read_excel(input_file)
    
    # Get unique strain IDs
    strain_ids = growth_data['strain_ID'].unique()
    n_strains = len(strain_ids)
    print(f"Found {n_strains} unique strains")
    print()
    
    # Initialize results list
    results_list = []
    
    # Fit each strain
    print("Fitting strains...")
    for i, strain_id in enumerate(strain_ids):
        # Get strain data
        strain_data = growth_data[growth_data['strain_ID'] == strain_id]
        
        # Extract temperature and growth rate
        T_C = strain_data['Temperature_C'].values
        mu = strain_data['Growth_rate_per_h'].values
        
        # Get metadata
        species = strain_data['Species'].iloc[0] if 'Species' in strain_data.columns else ''
        phylum = strain_data['Phylum_harmonized'].iloc[0] if 'Phylum_harmonized' in strain_data.columns else ''
        guild = strain_data['Guild_harmonized'].iloc[0] if 'Guild_harmonized' in strain_data.columns else ''
        thermal_class = strain_data['thermal_class'].iloc[0] if 'thermal_class' in strain_data.columns else ''
        source = strain_data['Source'].iloc[0] if 'Source' in strain_data.columns else ''
        
        # Skip if insufficient data
        if len(T_C) < 4:
            print(f"  Skipping {strain_id}: insufficient data points ({len(T_C)})")
            continue
        
        # Fit the model
        try:
            fit_result = fit_strain_DE(T_C, mu, seed=seed + i)
            
            # Determine classifications
            trophic_mode = determine_trophic_mode(guild)
            domain = determine_domain(phylum)
            
            # Build result row
            result_row = {
                'strain_ID': strain_id,
                'Species': species,
                'Phylum_harmonized': phylum,
                'Guild_harmonized': guild,
                'thermal_class': thermal_class,
                'Source': source,
                'n_points': fit_result['n_points'],
                'Er': fit_result['Er'],
                'Eh': fit_result['Eh'],
                'Tmax_Thermo': fit_result['Tmax_Thermo'],
                'Topt_Thermo': fit_result['Topt_Thermo'],
                'NRMSE': fit_result['NRMSE'],
                'NRMSE_ratio': fit_result['NRMSE'] / 0.133 if fit_result['NRMSE'] else np.nan,  # Ratio to median
                'Trophic_mode': trophic_mode,
                'Domain': domain,
                'Q10_apparent': fit_result['Q10_apparent'],
                'Q10_intrinsic': fit_result['Q10_intrinsic']
            }
            
            results_list.append(result_row)
            
            # Progress indicator
            if (i + 1) % 100 == 0:
                print(f"  Fitted {i + 1}/{n_strains} strains...")
                
        except Exception as e:
            print(f"  Error fitting {strain_id}: {str(e)}")
    
    # Create results DataFrame
    results_df = pd.DataFrame(results_list)
    
    # Add population classification
    results_df['Population'] = classify_population(results_df['Eh'].values)
    
    # Reorder columns to match expected output
    column_order = [
        'strain_ID', 'Species', 'Phylum_harmonized', 'Guild_harmonized',
        'thermal_class', 'Source', 'n_points', 'Er', 'Eh', 'Tmax_Thermo',
        'Topt_Thermo', 'NRMSE', 'NRMSE_ratio', 'Population', 'Trophic_mode',
        'Domain', 'Q10_apparent', 'Q10_intrinsic'
    ]
    results_df = results_df[column_order]
    
    # Save results
    print()
    print(f"Saving results to {output_file}...")
    results_df.to_excel(output_file, index=False)
    
    # Print summary statistics
    print()
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total strains fitted: {len(results_df)}")
    print()
    print("Parameter medians:")
    print(f"  Er: {results_df['Er'].median():.1f} kJ/mol")
    print(f"  Eh: {results_df['Eh'].median():.1f} kJ/mol")
    print(f"  Tmax: {results_df['Tmax_Thermo'].median():.1f} °C")
    print(f"  Topt: {results_df['Topt_Thermo'].median():.1f} °C")
    print(f"  NRMSE: {results_df['NRMSE'].median():.3f}")
    print()
    print("Population distribution:")
    print(results_df['Population'].value_counts())
    print()
    print("Trophic mode distribution:")
    print(results_df['Trophic_mode'].value_counts())
    print()
    print("Domain distribution:")
    print(results_df['Domain'].value_counts())
    print()
    print("Done!")
    
    return results_df


# =============================================================================
# VALIDATION FUNCTION
# =============================================================================

def validate_against_original(fitted_file, original_file):
    """
    Validate fitted results against original DE results.
    
    Parameters
    ----------
    fitted_file : str
        Path to newly fitted results
    original_file : str
        Path to original results file
    """
    fitted = pd.read_excel(fitted_file)
    original = pd.read_excel(original_file)
    
    # Merge on strain_ID
    merged = pd.merge(fitted, original, on='strain_ID', suffixes=('_new', '_orig'))
    
    print("=" * 60)
    print("VALIDATION AGAINST ORIGINAL RESULTS")
    print("=" * 60)
    print()
    
    for param in ['Er', 'Eh', 'Tmax_Thermo', 'Topt_Thermo', 'NRMSE']:
        col_new = f'{param}_new'
        col_orig = f'{param}_orig'
        
        if col_new in merged.columns and col_orig in merged.columns:
            corr = merged[col_new].corr(merged[col_orig])
            diff = (merged[col_new] - merged[col_orig]).abs().mean()
            print(f"{param}:")
            print(f"  Correlation: {corr:.4f}")
            print(f"  Mean absolute difference: {diff:.4f}")
            print()


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Fit thermodynamic growth model using DE')
    parser.add_argument('--input', '-i', default='07a_growth_data_1054_strains_v31.xlsx',
                       help='Input growth data file')
    parser.add_argument('--output', '-o', default='07b_DE_results_1054_strains_v31.xlsx',
                       help='Output results file')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--validate', '-v', default=None,
                       help='Original results file for validation')
    
    args = parser.parse_args()
    
    # Run fitting
    results = main(args.input, args.output, args.seed)
    
    # Optionally validate
    if args.validate:
        validate_against_original(args.output, args.validate)
