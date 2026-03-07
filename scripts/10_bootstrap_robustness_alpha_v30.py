#!/usr/bin/env python3
"""
10_bootstrap_robustness_alpha.py
=================================
Robustness analyses for the relative acceleration factor α = r_het / r_pho.

Implements two analyses claimed in SI Sections 12.9 and 12.10, specifically:

  1. Decreasing sample-size bootstrap (n = 300, 200, 100, 50, 30 per guild)
     → Verifies: "Down to n=30, P(α>1) = 100% across 10,000 iterations"
  
  2. Domain-stratified bootstrap (resample within Bacteria, Archaea, Eukarya)
     → Verifies: "α = 1.16 [1.13, 1.18]"

Also reproduces the existing subset analyses for cross-validation:
  - All strains:           α = 1.15 [1.13, 1.18]
  - Mesophiles 15–45°C:    α = 1.09 [1.07, 1.11]
  - Mesophiles 20–40°C:    α = 1.07 [1.05, 1.08]
  - Eukarya only:          α = 1.12 [1.09, 1.15]

Author: Théodore Bouchez
Date: 2025
"""

import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path

# =============================================================================
# CONSTANTS
# =============================================================================
kB = 1.380649e-23       # Boltzmann constant (J/K)
h_planck = 6.62607015e-34  # Planck constant (J·s)
R_gas = 8.314462        # Gas constant (J/(mol·K))

T0_K = 288.15           # Baseline temperature: 15°C in K
DT = 4.0                # Warming scenario: +4°C

N_BOOT = 10_000         # Bootstrap replicates
SEED = 42               # Reproducibility

# SI claims to verify
SI_CLAIMS = {
    'All strains':        {'alpha': 1.15, 'ci_lo': 1.13, 'ci_hi': 1.18},
    'Mesophiles 20–40°C': {'alpha': 1.07, 'ci_lo': 1.05, 'ci_hi': 1.08},
    'Eukarya only':       {'alpha': 1.12, 'ci_lo': 1.09, 'ci_hi': 1.15},
    'Domain-stratified':  {'alpha': 1.16, 'ci_lo': 1.13, 'ci_hi': 1.18},
    'n=30 P(α>1)':        1.00,  # 100%
}


# =============================================================================
# MODEL
# =============================================================================
def mu_model(T_K, Er_J, Eh_J, Tmax_K):
    """Thermodynamic growth model µ(T).
    
    µ(T) = (kB·T/h)·exp(-Er/RT)·[exp(Eh/R·(1/T - 1/Tmax)) - 1]
    
    Parameters in SI units (J/mol for Er, Eh; K for T, Tmax).
    """
    prefactor = kB * T_K / h_planck
    eyring = np.exp(-Er_J / (R_gas * T_K))
    england_arg = (Eh_J / R_gas) * (1.0 / T_K - 1.0 / Tmax_K)
    england = np.exp(england_arg) - 1.0
    return prefactor * eyring * max(england, 0.0)


def compute_r(row, T0=T0_K, dT=DT):
    """Compute per-capita acceleration r = µ(T0+dT)/µ(T0) for one strain.
    
    Returns NaN if µ = 0 at either temperature (strain beyond thermal limit).
    """
    Er_J = row['Er'] * 1000  # kJ/mol → J/mol
    Eh_J = row['Eh'] * 1000
    Tmax_K = row['Tmax_Thermo'] + 273.15
    
    mu_base = mu_model(T0, Er_J, Eh_J, Tmax_K)
    mu_warm = mu_model(T0 + dT, Er_J, Eh_J, Tmax_K)
    
    if mu_base > 0 and mu_warm > 0:
        return mu_warm / mu_base
    return np.nan


# =============================================================================
# BOOTSTRAP ENGINES
# =============================================================================
def bootstrap_alpha(r_het, r_pho, n_boot=N_BOOT, rng=None):
    """Standard bootstrap of α = mean(r_het) / mean(r_pho).
    
    Resamples het and pho independently with replacement.
    Returns: alpha_obs, ci_lo, ci_hi, P(α>1)
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    alpha_obs = np.nanmean(r_het) / np.nanmean(r_pho)
    
    alphas = np.empty(n_boot)
    for i in range(n_boot):
        h_sample = rng.choice(r_het, size=len(r_het), replace=True)
        p_sample = rng.choice(r_pho, size=len(r_pho), replace=True)
        alphas[i] = np.nanmean(h_sample) / np.nanmean(p_sample)
    
    ci_lo = np.percentile(alphas, 2.5)
    ci_hi = np.percentile(alphas, 97.5)
    p_gt1 = np.mean(alphas > 1)
    
    return alpha_obs, ci_lo, ci_hi, p_gt1


def bootstrap_alpha_subsampled(r_het, r_pho, n_sub, n_boot=N_BOOT, rng=None):
    """Bootstrap α with subsampled guild sizes.
    
    At each iteration:
      1. Draw n_sub strains (without replacement) from each guild
      2. Compute α from these subsamples
    
    This tests robustness to reduced sample size.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    n_het = len(r_het)
    n_pho = len(r_pho)
    
    if n_sub > n_het or n_sub > n_pho:
        return np.nan, np.nan, np.nan, np.nan
    
    alphas = np.empty(n_boot)
    for i in range(n_boot):
        h_idx = rng.choice(n_het, size=n_sub, replace=False)
        p_idx = rng.choice(n_pho, size=n_sub, replace=False)
        alphas[i] = np.mean(r_het[h_idx]) / np.mean(r_pho[p_idx])
    
    alpha_med = np.median(alphas)
    ci_lo = np.percentile(alphas, 2.5)
    ci_hi = np.percentile(alphas, 97.5)
    p_gt1 = np.mean(alphas > 1)
    
    return alpha_med, ci_lo, ci_hi, p_gt1


def bootstrap_alpha_domain_stratified(het_df, pho_df, r_het_arr, r_pho_arr,
                                       n_boot=N_BOOT, rng=None):
    """Domain-stratified bootstrap of α.
    
    Controls for phylogenetic composition by resampling WITHIN each domain
    (Bacteria, Archaea, Eukarya) separately, preserving domain proportions.
    
    For each bootstrap iteration:
      1. Within each domain, resample het strains with replacement
      2. Within each domain, resample pho strains with replacement
      3. Pool across domains → compute α
    
    This controls for the confounding effect of different domain compositions
    between het (Bacteria-dominated) and pho (Eukarya-dominated).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    
    domains = ['Bacteria', 'Archaea', 'Eukarya']
    
    # Build index arrays for each guild × domain
    het_by_domain = {}
    pho_by_domain = {}
    for d in domains:
        h_mask = het_df['Domain'].values == d
        p_mask = pho_df['Domain'].values == d
        h_r = r_het_arr[h_mask]
        p_r = r_pho_arr[p_mask]
        # Only include if domain is represented in this guild
        if len(h_r) > 0:
            het_by_domain[d] = h_r[~np.isnan(h_r)]
        if len(p_r) > 0:
            pho_by_domain[d] = p_r[~np.isnan(p_r)]
    
    # Report composition
    print("\n    Domain composition for stratified bootstrap:")
    for d in domains:
        nh = len(het_by_domain.get(d, []))
        np_ = len(pho_by_domain.get(d, []))
        print(f"      {d:12s}: het={nh:4d}, pho={np_:4d}")
    
    # Observed alpha (pooled across domains)
    all_h = np.concatenate([het_by_domain[d] for d in het_by_domain])
    all_p = np.concatenate([pho_by_domain[d] for d in pho_by_domain])
    alpha_obs = np.mean(all_h) / np.mean(all_p)
    
    # Bootstrap: resample within each domain
    alphas = np.empty(n_boot)
    for i in range(n_boot):
        h_samples = []
        p_samples = []
        for d in domains:
            if d in het_by_domain:
                h_d = het_by_domain[d]
                h_samples.append(rng.choice(h_d, size=len(h_d), replace=True))
            if d in pho_by_domain:
                p_d = pho_by_domain[d]
                p_samples.append(rng.choice(p_d, size=len(p_d), replace=True))
        
        h_pooled = np.concatenate(h_samples)
        p_pooled = np.concatenate(p_samples)
        alphas[i] = np.mean(h_pooled) / np.mean(p_pooled)
    
    ci_lo = np.percentile(alphas, 2.5)
    ci_hi = np.percentile(alphas, 97.5)
    p_gt1 = np.mean(alphas > 1)
    
    return alpha_obs, ci_lo, ci_hi, p_gt1


# =============================================================================
# VERIFICATION
# =============================================================================
def verify_claim(label, observed, expected, tol=0.015):
    """Compare observed result to SI claim within tolerance."""
    if isinstance(expected, dict):
        checks = []
        for key in ['alpha', 'ci_lo', 'ci_hi']:
            diff = abs(observed[key] - expected[key])
            ok = diff <= tol
            checks.append(ok)
            status = "✓" if ok else "✗"
            print(f"    {status} {key}: observed={observed[key]:.3f}, "
                  f"SI={expected[key]:.2f}, Δ={diff:.3f}")
        return all(checks)
    else:
        diff = abs(observed - expected)
        ok = diff <= tol
        status = "✓" if ok else "✗"
        print(f"    {status} observed={observed:.4f}, SI={expected:.2f}, Δ={diff:.4f}")
        return ok


# =============================================================================
# MAIN
# =============================================================================
def main():
    # Locate data file
    data_dir = Path('/mnt/project')
    data_file = data_dir / '07b_DE_results_1054_strains_v30.xlsx'
    
    if not data_file.exists():
        # Try relative path
        data_file = Path('07b_DE_results_1054_strains_v30.xlsx')
    
    print("=" * 78)
    print("BOOTSTRAP ROBUSTNESS ANALYSIS FOR α")
    print(f"Baseline T₀ = {T0_K - 273.15:.0f}°C, ΔT = +{DT:.0f}°C, "
          f"n_boot = {N_BOOT:,}")
    print("=" * 78)
    
    # -----------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------
    print(f"\n[1] Loading {data_file.name}...")
    df = pd.read_excel(data_file)
    print(f"    Total strains: {len(df)}")
    
    het = df[df['Trophic_mode'] == 'Heterotroph'].copy()
    pho = df[df['Trophic_mode'] == 'Phototroph'].copy()
    print(f"    Heterotrophs: {len(het)}")
    print(f"    Phototrophs:  {len(pho)}")
    
    # -----------------------------------------------------------------
    # 2. Compute r for all strains
    # -----------------------------------------------------------------
    print(f"\n[2] Computing r = µ(T₀+{DT:.0f})/µ(T₀) for each strain...")
    
    het['r'] = het.apply(compute_r, axis=1)
    pho['r'] = pho.apply(compute_r, axis=1)
    
    het_valid = het.dropna(subset=['r'])
    pho_valid = pho.dropna(subset=['r'])
    
    print(f"    Valid het (µ > 0 at both T): {len(het_valid)}/{len(het)}")
    print(f"    Valid pho (µ > 0 at both T): {len(pho_valid)}/{len(pho)}")
    
    r_het_all = het_valid['r'].values
    r_pho_all = pho_valid['r'].values
    
    # -----------------------------------------------------------------
    # 3. Reproduce existing subset analyses (cross-validation)
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[3] SUBSET BOOTSTRAP (reproducing Figure 4D / SI Table)")
    print("=" * 78)
    
    rng = np.random.default_rng(SEED)
    
    # Define subsets
    het_topt = het_valid['Topt_Thermo'].values
    pho_topt = pho_valid['Topt_Thermo'].values
    het_domain = het_valid['Domain'].values
    pho_domain = pho_valid['Domain'].values
    
    subsets = [
        ('All strains',
         np.ones(len(r_het_all), dtype=bool),
         np.ones(len(r_pho_all), dtype=bool)),
        ('Mesophiles 15–45°C',
         (het_topt >= 15) & (het_topt <= 45),
         (pho_topt >= 15) & (pho_topt <= 45)),
        ('Mesophiles 20–40°C',
         (het_topt >= 20) & (het_topt <= 40),
         (pho_topt >= 20) & (pho_topt <= 40)),
        ('Eukarya only',
         het_domain == 'Eukarya',
         pho_domain == 'Eukarya'),
    ]
    
    print(f"\n{'Subset':<25s} {'n_het':>6s} {'n_pho':>6s} "
          f"{'α':>7s} {'95% CI':>16s} {'P(α>1)':>8s}")
    print("-" * 78)
    
    subset_results = {}
    for name, h_mask, p_mask in subsets:
        rng_sub = np.random.default_rng(SEED)
        r_h = r_het_all[h_mask]
        r_p = r_pho_all[p_mask]
        
        alpha, lo, hi, p_gt1 = bootstrap_alpha(r_h, r_p, N_BOOT, rng_sub)
        
        print(f"  {name:<23s} {len(r_h):>6d} {len(r_p):>6d} "
              f"{alpha:>7.3f} [{lo:.3f}, {hi:.3f}] {p_gt1*100:>7.1f}%")
        
        subset_results[name] = {'alpha': alpha, 'ci_lo': lo, 'ci_hi': hi, 
                                'p_gt1': p_gt1}
    
    # Verify against SI claims
    print("\n  Verification against SI claims:")
    for name in ['All strains', 'Mesophiles 20–40°C', 'Eukarya only']:
        if name in SI_CLAIMS and name in subset_results:
            print(f"\n  {name}:")
            verify_claim(name, subset_results[name], SI_CLAIMS[name])
    
    # -----------------------------------------------------------------
    # 4. Decreasing sample-size bootstrap
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[4] DECREASING SAMPLE-SIZE BOOTSTRAP")
    print("    Testing: 'Down to n=30, P(α>1) = 100%'")
    print("=" * 78)
    
    sample_sizes = [400, 300, 200, 100, 50, 30]
    
    print(f"\n{'n_sub':>8s} {'α_median':>10s} {'95% CI':>18s} "
          f"{'P(α>1)':>10s} {'Status':>8s}")
    print("-" * 62)
    
    all_p_gt1_100 = True
    for n_sub in sample_sizes:
        rng_sub = np.random.default_rng(SEED)
        alpha_med, lo, hi, p_gt1 = bootstrap_alpha_subsampled(
            r_het_all, r_pho_all, n_sub, N_BOOT, rng_sub
        )
        
        if np.isnan(alpha_med):
            print(f"  {n_sub:>6d}   — insufficient strains in one guild —")
            continue
        
        status = "✓" if p_gt1 == 1.0 else "✗"
        if p_gt1 < 1.0:
            all_p_gt1_100 = False
        
        print(f"  {n_sub:>6d} {alpha_med:>10.4f} [{lo:.4f}, {hi:.4f}] "
              f"{p_gt1*100:>9.2f}% {status:>8s}")
    
    print(f"\n  SI claim verification (n=30, P(α>1) = 100%):")
    # Run n=30 specifically
    rng_30 = np.random.default_rng(SEED)
    _, _, _, p_gt1_30 = bootstrap_alpha_subsampled(
        r_het_all, r_pho_all, 30, N_BOOT, rng_30
    )
    verify_claim('n=30 P(α>1)', p_gt1_30, SI_CLAIMS['n=30 P(α>1)'])
    
    # Also test with bootstrap (with replacement) at n=30 — alternative reading
    print(f"\n  Alternative: bootstrap WITH replacement at n=30:")
    rng_30b = np.random.default_rng(SEED)
    alphas_30_wr = np.empty(N_BOOT)
    for i in range(N_BOOT):
        h_idx = rng_30b.choice(len(r_het_all), size=30, replace=True)
        p_idx = rng_30b.choice(len(r_pho_all), size=30, replace=True)
        alphas_30_wr[i] = np.mean(r_het_all[h_idx]) / np.mean(r_pho_all[p_idx])
    
    p_gt1_wr = np.mean(alphas_30_wr > 1)
    ci_lo_wr = np.percentile(alphas_30_wr, 2.5)
    ci_hi_wr = np.percentile(alphas_30_wr, 97.5)
    alpha_med_wr = np.median(alphas_30_wr)
    print(f"    α_median = {alpha_med_wr:.4f} [{ci_lo_wr:.4f}, {ci_hi_wr:.4f}], "
          f"P(α>1) = {p_gt1_wr*100:.2f}%")
    
    # -----------------------------------------------------------------
    # 5. Domain-stratified bootstrap
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[5] DOMAIN-STRATIFIED BOOTSTRAP")
    print("    Testing: 'α = 1.16 [1.13, 1.18]'")
    print("=" * 78)
    
    rng_strat = np.random.default_rng(SEED)
    
    alpha_strat, lo_strat, hi_strat, p_gt1_strat = \
        bootstrap_alpha_domain_stratified(
            het_valid, pho_valid, r_het_all, r_pho_all,
            N_BOOT, rng_strat
        )
    
    print(f"\n    Result: α = {alpha_strat:.3f} [{lo_strat:.3f}, {hi_strat:.3f}], "
          f"P(α>1) = {p_gt1_strat*100:.1f}%")
    
    print(f"\n  SI claim verification:")
    strat_result = {'alpha': alpha_strat, 'ci_lo': lo_strat, 'ci_hi': hi_strat}
    verify_claim('Domain-stratified', strat_result, SI_CLAIMS['Domain-stratified'])
    
    # -----------------------------------------------------------------
    # 6. Additional analysis: domain-stratified with equal weighting
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("  [5b] Variant: domain-WEIGHTED bootstrap (equal weight per domain)")
    print("       (weights each domain equally regardless of sample size)")
    print("-" * 78)
    
    domains = ['Bacteria', 'Archaea', 'Eukarya']
    rng_weighted = np.random.default_rng(SEED)
    
    # Compute mean r within each domain × guild
    het_domain_means = {}
    pho_domain_means = {}
    for d in domains:
        h_mask = het_valid['Domain'].values == d
        p_mask = pho_valid['Domain'].values == d
        h_r = r_het_all[h_mask]
        p_r = r_pho_all[p_mask]
        if len(h_r) > 0:
            het_domain_means[d] = h_r[~np.isnan(h_r)]
        if len(p_r) > 0:
            pho_domain_means[d] = p_r[~np.isnan(p_r)]
    
    # Shared domains (both guilds present)
    shared_domains = [d for d in domains 
                      if d in het_domain_means and d in pho_domain_means]
    print(f"    Domains with both guilds: {shared_domains}")
    
    alphas_weighted = np.empty(N_BOOT)
    for i in range(N_BOOT):
        h_means = []
        p_means = []
        for d in shared_domains:
            h_d = het_domain_means[d]
            p_d = pho_domain_means[d]
            h_means.append(np.mean(rng_weighted.choice(h_d, size=len(h_d), replace=True)))
            p_means.append(np.mean(rng_weighted.choice(p_d, size=len(p_d), replace=True)))
        # Equal weight per domain
        alphas_weighted[i] = np.mean(h_means) / np.mean(p_means)
    
    alpha_w = np.median(alphas_weighted)
    lo_w = np.percentile(alphas_weighted, 2.5)
    hi_w = np.percentile(alphas_weighted, 97.5)
    p_gt1_w = np.mean(alphas_weighted > 1)
    print(f"    Result: α = {alpha_w:.3f} [{lo_w:.3f}, {hi_w:.3f}], "
          f"P(α>1) = {p_gt1_w*100:.1f}%")
    
    # -----------------------------------------------------------------
    # 7. Summary
    # -----------------------------------------------------------------
    print("\n" + "=" * 78)
    print("[6] SUMMARY: CONCORDANCE WITH SI CLAIMS")
    print("=" * 78)
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ Analysis                  │ SI claim            │ This script       │
  ├─────────────────────────────────────────────────────────────────────┤
  │ All strains               │ 1.15 [1.13, 1.18]   │ {subset_results['All strains']['alpha']:.2f} [{subset_results['All strains']['ci_lo']:.2f}, {subset_results['All strains']['ci_hi']:.2f}]  │
  │ Mesophiles 20–40°C        │ 1.07 [1.05, 1.08]   │ {subset_results['Mesophiles 20–40°C']['alpha']:.2f} [{subset_results['Mesophiles 20–40°C']['ci_lo']:.2f}, {subset_results['Mesophiles 20–40°C']['ci_hi']:.2f}]  │
  │ Eukarya only              │ 1.12 [1.09, 1.15]   │ {subset_results['Eukarya only']['alpha']:.2f} [{subset_results['Eukarya only']['ci_lo']:.2f}, {subset_results['Eukarya only']['ci_hi']:.2f}]  │
  │ Domain-stratified         │ 1.16 [1.13, 1.18]   │ {alpha_strat:.2f} [{lo_strat:.2f}, {hi_strat:.2f}]  │
  │ n=30 P(α>1)              │ 100%                │ {p_gt1_30*100:.1f}%             │
  └─────────────────────────────────────────────────────────────────────┘
""")
    
    # Overall concordance
    discrepancies = []
    
    for name in ['All strains', 'Mesophiles 20–40°C', 'Eukarya only']:
        obs = subset_results[name]
        exp = SI_CLAIMS[name]
        for key in ['alpha', 'ci_lo', 'ci_hi']:
            if abs(obs[key] - exp[key]) > 0.015:
                discrepancies.append(f"{name}/{key}: {obs[key]:.3f} vs SI {exp[key]:.2f}")
    
    strat_exp = SI_CLAIMS['Domain-stratified']
    for key, val in [('alpha', alpha_strat), ('ci_lo', lo_strat), ('ci_hi', hi_strat)]:
        if abs(val - strat_exp[key]) > 0.015:
            discrepancies.append(f"Domain-stratified/{key}: {val:.3f} vs SI {strat_exp[key]:.2f}")
    
    if abs(p_gt1_30 - 1.0) > 0.001:
        discrepancies.append(f"n=30 P(α>1): {p_gt1_30:.4f} vs SI 1.00")
    
    if discrepancies:
        print("  ⚠ DISCREPANCIES FOUND:")
        for d in discrepancies:
            print(f"    • {d}")
        print("\n  → Review SI text or analysis assumptions.")
    else:
        print("  ✓ All results concordant with SI claims (tolerance ±0.015)")
    
    print("\nDone.")


if __name__ == '__main__':
    main()
