"""
Vector-Based Voting Simulator

Simulates voting using spatial models:
- Voters and candidates as vectors in R^n
- Distance-based utilities
- Method-appropriate ballot conversion
- True VSE based on spatial utilities

Architecture: Belief space → Distance → Utility → Ballot → Winner
"""

import numpy as np
import os
import argparse
from typing import Tuple, Dict, Any, Optional

from expanded_rules import (
    compute_distances, compute_d_max, compute_utilities,
    utilities_to_rankings, compute_vse_and_metrics,
    compute_margins_from_rankings, is_cycle, classify_cycle_type,
    VOTING_RULES, CARDINAL_RULES, ORDINAL_RULES
)

# Enable Windows ANSI colors
os.system('')

# ANSI color codes for Windows terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'

# =============================================================================
# Spatial Profile Generation Methods
# =============================================================================

def gen_spatial_uniform(n_voters: int, n_profiles: int, n_dim: int = 2
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Uniform spatial distribution: voters and candidates random in [0,1]^n.
    
    Most "unbiased" spatial distribution. Replacement for impartial_culture.
    
    Returns:
        voter_positions: (n_profiles, n_voters, n_dim)
        candidate_positions: (n_profiles, 3, n_dim)
    """
    voter_positions = np.random.uniform(0, 1, (n_profiles, n_voters, n_dim))
    candidate_positions = np.random.uniform(0, 1, (n_profiles, 3, n_dim))
    return voter_positions, candidate_positions


def gen_spatial_clustered(n_voters: int, n_profiles: int, n_dim: int = 2,
                          phi: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Clustered spatial distribution: voters Gaussian around random center.
    
    Replacement for Mallows model - phi interpreted as spatial variance.
    phi=0: all at same point, phi=1: wide scatter
    
    Returns:
        voter_positions: (n_profiles, n_voters, n_dim)
        candidate_positions: (n_profiles, 3, n_dim)
    """
    voter_positions = np.empty((n_profiles, n_voters, n_dim))
    candidate_positions = np.empty((n_profiles, 3, n_dim))
    
    # Variance scales with dimension
    voter_variance = phi * np.sqrt(n_dim) * 0.15  # Scale factor for reasonable spread
    candidate_variance = voter_variance * 0.5  # Candidates closer together
    
    for i in range(n_profiles):
        # Random center in unit hypercube
        center = np.random.uniform(0.3, 0.7, n_dim)  # Avoid edges
        
        # Voters: Gaussian around center, clipped to [0, 1]
        voter_positions[i] = np.clip(
            np.random.normal(center, voter_variance, (n_voters, n_dim)),
            0, 1
        )
        
        # Candidates: Also from cluster
        candidate_positions[i] = np.clip(
            np.random.normal(center, candidate_variance, (3, n_dim)),
            0, 1
        )
    
    return voter_positions, candidate_positions


def gen_spatial_single_peaked(n_voters: int, n_profiles: int, n_dim: int = 1
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Single-peaked preferences: 1D left-right spectrum.
    
    Forces n_dim=1 regardless of input. Candidates at fixed positions.
    Naturally produces single-peaked preferences (no Condorcet cycles).
    
    Returns:
        voter_positions: (n_profiles, n_voters, 1)
        candidate_positions: (n_profiles, 3, 1)
    """
    # Force 1D
    n_dim = 1
    
    voter_positions = np.random.uniform(0, 1, (n_profiles, n_voters, n_dim))
    
    # Fixed candidate positions: left, center, right
    candidate_positions = np.zeros((n_profiles, 3, n_dim))
    candidate_positions[:, 0, 0] = 0.2  # Candidate A (left)
    candidate_positions[:, 1, 0] = 0.5  # Candidate B (center)
    candidate_positions[:, 2, 0] = 0.8  # Candidate C (right)
    
    return voter_positions, candidate_positions


def gen_spatial_polarized(n_voters: int, n_profiles: int, n_dim: int = 2
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Polarized electorate: two opposing clusters with small center.
    
    Simulates political polarization with two factions.
    
    Returns:
        voter_positions: (n_profiles, n_voters, n_dim)
        candidate_positions: (n_profiles, 3, n_dim)
    """
    voter_positions = np.empty((n_profiles, n_voters, n_dim))
    candidate_positions = np.empty((n_profiles, 3, n_dim))
    
    # Define cluster centers
    left_center = np.full(n_dim, 0.15)
    right_center = np.full(n_dim, 0.85)
    middle_center = np.full(n_dim, 0.5)
    
    cluster_std = 0.08 * np.sqrt(n_dim)  # Scale with dimension
    
    for i in range(n_profiles):
        # Random faction sizes
        faction_left = int(n_voters * np.random.uniform(0.35, 0.50))
        faction_right = int(n_voters * np.random.uniform(0.35, 0.50))
        faction_middle = n_voters - faction_left - faction_right
        
        idx = 0
        
        # Left faction
        voter_positions[i, idx:idx+faction_left] = np.clip(
            np.random.normal(left_center, cluster_std, (faction_left, n_dim)),
            0, 1
        )
        idx += faction_left
        
        # Right faction
        voter_positions[i, idx:idx+faction_right] = np.clip(
            np.random.normal(right_center, cluster_std, (faction_right, n_dim)),
            0, 1
        )
        idx += faction_right
        
        # Middle faction
        voter_positions[i, idx:idx+faction_middle] = np.clip(
            np.random.normal(middle_center, cluster_std, (faction_middle, n_dim)),
            0, 1
        )
        
        # Shuffle voters
        np.random.shuffle(voter_positions[i])
        
        # Candidates: one near each cluster
        candidate_positions[i, 0] = np.clip(
            np.random.normal(left_center, cluster_std/2, n_dim), 0, 1
        )
        candidate_positions[i, 1] = np.clip(
            np.random.normal(middle_center, cluster_std/2, n_dim), 0, 1
        )
        candidate_positions[i, 2] = np.clip(
            np.random.normal(right_center, cluster_std/2, n_dim), 0, 1
        )
    
    return voter_positions, candidate_positions


def gen_spatial_1d(n_voters: int, n_profiles: int, n_dim: int = 1
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    1D spatial model: voters uniform, candidates at fixed positions.
    
    Alias for spatial_single_peaked with explicit 1D naming.
    """
    return gen_spatial_single_peaked(n_voters, n_profiles, 1)


def gen_spatial_2d(n_voters: int, n_profiles: int, n_dim: int = 2
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D spatial model: candidates form triangle, voters uniform in square.
    
    Returns:
        voter_positions: (n_profiles, n_voters, 2)
        candidate_positions: (n_profiles, 3, 2)
    """
    # Force 2D
    n_dim = 2
    
    voter_positions = np.random.uniform(0, 1, (n_profiles, n_voters, n_dim))
    
    # Fixed triangle positions for candidates
    candidate_positions = np.zeros((n_profiles, 3, n_dim))
    candidate_positions[:, 0] = [0.2, 0.3]  # Bottom left
    candidate_positions[:, 1] = [0.8, 0.3]  # Bottom right
    candidate_positions[:, 2] = [0.5, 0.8]  # Top center
    
    return voter_positions, candidate_positions


# Registry of generation methods
GENERATION_METHODS: Dict[str, Tuple[callable, str]] = {
    'spatial_uniform': (gen_spatial_uniform, "Uniform random in hypercube"),
    'spatial_clustered': (gen_spatial_clustered, "Clustered around center (phi=variance)"),
    'spatial_single_peaked': (gen_spatial_single_peaked, "1D left-right spectrum"),
    'spatial_polarized': (gen_spatial_polarized, "Two opposing clusters"),
    'spatial_1d': (gen_spatial_1d, "1D uniform distribution"),
    'spatial_2d': (gen_spatial_2d, "2D triangle + uniform voters"),
}


# =============================================================================
# Core Simulation Functions
# =============================================================================

def simulate_paradox(voting_rule: str, n_profiles: int, n_voters: int,
                     gen_method: str = 'spatial_uniform',
                     n_dim: int = 2, sigma_factor: float = 0.5,
                     utility_func: str = 'gaussian',
                     distance_metric: str = 'l2',
                     approval_policy: str = 'top_k',
                     approval_param: float = 0.5,
                     score_max: int = 5,
                     phi: float = 0.5,
                     epsilon: float = 1e-9,
                     rng_seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Simulate voting using spatial model.
    
    Args:
        voting_rule: Name of voting rule to use
        n_profiles: Number of election profiles to simulate
        n_voters: Number of voters per profile
        gen_method: Spatial generation method name
        n_dim: Dimension of vector space
        sigma_factor: Sigma = sigma_factor * sqrt(n_dim) for Gaussian utility
        utility_func: 'gaussian', 'quadratic', or 'linear'
        distance_metric: 'l2', 'l1', or 'cosine'
        approval_policy: For approval voting: 'top_k', 'delta', 'mean', 'absolute'
        approval_param: Parameter for approval policy
        score_max: Maximum score for score/STAR voting
        phi: Variance parameter for spatial_clustered
        epsilon: Epsilon for ranking ties
        rng_seed: Random seed for reproducibility
    
    Returns:
        Dictionary of simulation results
    """
    # Set RNG seed for reproducibility
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    # Get generation function
    if gen_method not in GENERATION_METHODS:
        raise ValueError(f"Unknown generation method: {gen_method}")
    gen_func = GENERATION_METHODS[gen_method][0]
    
    # Get rule info
    if voting_rule not in VOTING_RULES:
        raise ValueError(f"Unknown voting rule: {voting_rule}")
    rule_info = VOTING_RULES[voting_rule]
    rule_func = rule_info['func']
    rule_type = rule_info['type']
    
    # Generate spatial profiles
    if gen_method == 'spatial_clustered':
        voter_positions, candidate_positions = gen_func(n_voters, n_profiles, n_dim, phi)
    else:
        voter_positions, candidate_positions = gen_func(n_voters, n_profiles, n_dim)
    
    # Get actual dimension used (may be forced by generation method)
    actual_n_dim = voter_positions.shape[2]
    
    # Compute d_max once
    d_max = compute_d_max(actual_n_dim, distance_metric)
    
    # Initialize counters
    cycle_count = 0
    type1_count = 0
    type2_count = 0
    vse_values = []
    raw_social_utilities = []
    regrets = []
    rank_counts = np.zeros(3, dtype=int)
    
    # Process each profile
    for i in range(n_profiles):
        # 1. Compute distances (vectorized)
        distances = compute_distances(
            voter_positions[i],
            candidate_positions[i],
            metric=distance_metric
        )
        
        # 2. Compute utilities
        utilities = compute_utilities(
            distances, actual_n_dim, utility_func,
            sigma_factor, d_max, distance_metric
        )
        
        # 3. Compute rankings ONCE (for ordinal rules and cycle detection)
        rankings = utilities_to_rankings(utilities, epsilon)
        
        # 4. Compute winner based on rule type
        if rule_type == 'cardinal':
            if voting_rule == 'approval':
                winner_idx = rule_func(utilities, approval_policy, approval_param)
            elif voting_rule in ('score', 'star'):
                winner_idx = rule_func(utilities, score_max)
            else:
                winner_idx = rule_func(utilities)
        else:  # ordinal
            winner_idx = rule_func(rankings)
        
        # 5. Compute VSE and enhanced metrics
        vse, raw_utility, regret, winner_rank = compute_vse_and_metrics(
            utilities, winner_idx
        )
        vse_values.append(vse)
        raw_social_utilities.append(raw_utility)
        regrets.append(regret)
        rank_counts[winner_rank] += 1
        
        # 6. Condorcet cycle detection
        margins = compute_margins_from_rankings(rankings)
        if is_cycle(margins):
            cycle_count += 1
            cycle_type = classify_cycle_type(margins)
            if cycle_type == 1:
                type1_count += 1
            elif cycle_type == 2:
                type2_count += 1
    
    # Convert rank counts to percentages
    rank_pcts = 100 * rank_counts / n_profiles if n_profiles > 0 else np.zeros(3)
    
    return {
        'total_profiles': n_profiles,
        'n_voters': n_voters,
        'n_dim': actual_n_dim,
        'gen_method': gen_method,
        'voting_rule': voting_rule,
        'distance_metric': distance_metric,
        'utility_func': utility_func,
        'sigma_factor': sigma_factor,
        'cycle_count': cycle_count,
        'cycle_percentage': 100 * cycle_count / n_profiles if n_profiles > 0 else 0,
        'type1_count': type1_count,
        'type1_percentage': 100 * type1_count / n_profiles if n_profiles > 0 else 0,
        'type2_count': type2_count,
        'type2_percentage': 100 * type2_count / n_profiles if n_profiles > 0 else 0,
        'vse_mean': np.mean(vse_values) if vse_values else 0.0,
        'vse_std': np.std(vse_values) if len(vse_values) > 1 else 0.0,
        'vse_min': np.min(vse_values) if vse_values else 0.0,
        'vse_max': np.max(vse_values) if vse_values else 0.0,
        'vse_count': len(vse_values),
        'raw_utility_mean': np.mean(raw_social_utilities) if raw_social_utilities else 0.0,
        'regret_mean': np.mean(regrets) if regrets else 0.0,
        'regret_max': np.max(regrets) if regrets else 0.0,
        'winner_rank_1st_pct': rank_pcts[0],
        'winner_rank_2nd_pct': rank_pcts[1],
        'winner_rank_3rd_pct': rank_pcts[2],
        'rng_seed': rng_seed,
    }


# =============================================================================
# Table Formatting
# =============================================================================

def format_table(headers: list, rows: list, title: str = None, 
                 subtitle: str = None):
    """Format data as a pretty ASCII table."""
    C = Colors
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Add padding
    col_widths = [w + 2 for w in col_widths]
    total_width = sum(col_widths) + len(headers) + 1
    
    # Print table
    if title:
        print(f"\n{C.BOLD}{C.CYAN}{title}{C.RESET}")
    if subtitle:
        print(f"{C.DIM}{subtitle}{C.RESET}")
    print(f"{C.DIM}{'-' * total_width}{C.RESET}")
    
    # Headers
    header_str = "|"
    for i, header in enumerate(headers):
        header_str += f" {C.BOLD}{header:<{col_widths[i]-1}}{C.RESET}|"
    print(header_str)
    print(f"{C.DIM}{'-' * total_width}{C.RESET}")
    
    # Rows
    for row in rows:
        row_str = "|"
        for i, cell in enumerate(row):
            row_str += f" {str(cell):<{col_widths[i]-1}}|"
        print(row_str)
    
    print(f"{C.DIM}{'-' * total_width}{C.RESET}")


# =============================================================================
# Batch Simulation Functions
# =============================================================================

def run_all_methods(voting_rule: str, n_profiles: int, n_voters: int,
                    n_dim: int = 2, sigma_factor: float = 0.5,
                    utility_func: str = 'gaussian',
                    distance_metric: str = 'l2',
                    approval_policy: str = 'top_k',
                    approval_param: float = 0.5,
                    log_regret: bool = False,
                    log_winner_rank: bool = False,
                    log_raw_utility: bool = False):
    """Run simulation for all methods with a single voting rule."""
    C = Colors
    print(f"\n{C.BOLD}{C.GREEN}Running simulations for all methods with {voting_rule}...{C.RESET}")
    print(f"{C.DIM}Profiles: {n_profiles:,} x {n_voters} voters, {n_dim}D, "
          f"metric={distance_metric}, utility={utility_func}{C.RESET}")
    
    # Build headers
    headers = ["Method", "Cycles (%)", "VSE Mean", "VSE Min"]
    if log_regret:
        headers.append("Regret")
    if log_winner_rank:
        headers.extend(["1st (%)", "2nd (%)", "3rd (%)"])
    if log_raw_utility:
        headers.append("Raw U")
    
    rows = []
    
    for method_name in GENERATION_METHODS.keys():
        print(f"{C.DIM}  Simulating {method_name}...{C.RESET}", end="\r")
        results = simulate_paradox(
            voting_rule, n_profiles, n_voters, method_name,
            n_dim=n_dim, sigma_factor=sigma_factor,
            utility_func=utility_func, distance_metric=distance_metric,
            approval_policy=approval_policy, approval_param=approval_param
        )
        
        row = [
            method_name,
            f"{results['cycle_percentage']:.2f}%",
            f"{results['vse_mean']:.3f}",
            f"{results['vse_min']:.3f}",
        ]
        if log_regret:
            row.append(f"{results['regret_mean']:.4f}")
        if log_winner_rank:
            row.extend([
                f"{results['winner_rank_1st_pct']:.1f}%",
                f"{results['winner_rank_2nd_pct']:.1f}%",
                f"{results['winner_rank_3rd_pct']:.1f}%",
            ])
        if log_raw_utility:
            row.append(f"{results['raw_utility_mean']:.3f}")
        
        rows.append(row)
    
    print(" " * 60, end="\r")
    subtitle = f"N={n_profiles:,}, Voters={n_voters}, Dim={n_dim}, Metric={distance_metric}"
    format_table(headers, rows, f"Results for {voting_rule}", subtitle)


def run_all_rules(gen_method: str, n_profiles: int, n_voters: int,
                  n_dim: int = 2, sigma_factor: float = 0.5,
                  utility_func: str = 'gaussian',
                  distance_metric: str = 'l2',
                  approval_policy: str = 'top_k',
                  approval_param: float = 0.5,
                  log_regret: bool = False,
                  log_winner_rank: bool = False,
                  log_raw_utility: bool = False):
    """Run simulation for all voting rules with a single generation method."""
    C = Colors
    print(f"\n{C.BOLD}{C.GREEN}Running simulations for all rules with {gen_method}...{C.RESET}")
    print(f"{C.DIM}Profiles: {n_profiles:,} x {n_voters} voters, {n_dim}D, "
          f"metric={distance_metric}, utility={utility_func}{C.RESET}")
    
    # Build headers
    headers = ["Voting Rule", "Type", "Cycles (%)", "VSE Mean", "VSE Min"]
    if log_regret:
        headers.append("Regret")
    if log_winner_rank:
        headers.extend(["1st (%)", "2nd (%)", "3rd (%)"])
    if log_raw_utility:
        headers.append("Raw U")
    
    rows = []
    
    for rule_name in VOTING_RULES.keys():
        print(f"{C.DIM}  Simulating {rule_name}...{C.RESET}", end="\r")
        results = simulate_paradox(
            rule_name, n_profiles, n_voters, gen_method,
            n_dim=n_dim, sigma_factor=sigma_factor,
            utility_func=utility_func, distance_metric=distance_metric,
            approval_policy=approval_policy, approval_param=approval_param
        )
        
        rule_type = VOTING_RULES[rule_name]['type'][:4]  # 'card' or 'ordi'
        
        row = [
            rule_name,
            rule_type,
            f"{results['cycle_percentage']:.2f}%",
            f"{results['vse_mean']:.3f}",
            f"{results['vse_min']:.3f}",
        ]
        if log_regret:
            row.append(f"{results['regret_mean']:.4f}")
        if log_winner_rank:
            row.extend([
                f"{results['winner_rank_1st_pct']:.1f}%",
                f"{results['winner_rank_2nd_pct']:.1f}%",
                f"{results['winner_rank_3rd_pct']:.1f}%",
            ])
        if log_raw_utility:
            row.append(f"{results['raw_utility_mean']:.3f}")
        
        rows.append(row)
    
    print(" " * 60, end="\r")
    subtitle = f"N={n_profiles:,}, Voters={n_voters}, Dim={n_dim}, Metric={distance_metric}"
    format_table(headers, rows, f"Results for {gen_method}", subtitle)


def run_all_methods_all_rules(n_profiles: int, n_voters: int,
                              by_method: bool = False,
                              n_dim: int = 2, sigma_factor: float = 0.5,
                              utility_func: str = 'gaussian',
                              distance_metric: str = 'l2',
                              approval_policy: str = 'top_k',
                              approval_param: float = 0.5,
                              log_regret: bool = False,
                              log_winner_rank: bool = False,
                              log_raw_utility: bool = False):
    """Run simulations for all methods and all rules."""
    C = Colors
    print(f"\n{C.BOLD}{C.GREEN}Running comprehensive simulations...{C.RESET}")
    print(f"{C.DIM}Profiles: {n_profiles:,} x {n_voters} voters, {n_dim}D{C.RESET}")
    print(f"{C.DIM}Metric: {distance_metric}, Utility: {utility_func}, "
          f"Sigma factor: {sigma_factor}{C.RESET}")
    print(f"{C.DIM}Total simulations: {len(VOTING_RULES) * len(GENERATION_METHODS)}{C.RESET}")
    
    # Build headers
    headers = ["Item", "Cycles (%)", "VSE Mean", "VSE Min"]
    if log_regret:
        headers.append("Regret")
    if log_winner_rank:
        headers.extend(["1st (%)", "2nd (%)", "3rd (%)"])
    if log_raw_utility:
        headers.append("Raw U")
    
    if by_method:
        # Group by methods, compare rules within each method
        for method_name in GENERATION_METHODS.keys():
            rows = []
            for rule_name in VOTING_RULES.keys():
                print(f"{C.DIM}  {method_name} x {rule_name}...{C.RESET}", end="\r")
                results = simulate_paradox(
                    rule_name, n_profiles, n_voters, method_name,
                    n_dim=n_dim, sigma_factor=sigma_factor,
                    utility_func=utility_func, distance_metric=distance_metric,
                    approval_policy=approval_policy, approval_param=approval_param
                )
                
                row = [
                    rule_name,
                    f"{results['cycle_percentage']:.2f}%",
                    f"{results['vse_mean']:.3f}",
                    f"{results['vse_min']:.3f}",
                ]
                if log_regret:
                    row.append(f"{results['regret_mean']:.4f}")
                if log_winner_rank:
                    row.extend([
                        f"{results['winner_rank_1st_pct']:.1f}%",
                        f"{results['winner_rank_2nd_pct']:.1f}%",
                        f"{results['winner_rank_3rd_pct']:.1f}%",
                    ])
                if log_raw_utility:
                    row.append(f"{results['raw_utility_mean']:.3f}")
                
                rows.append(row)
            
            print(" " * 60, end="\r")
            format_table(headers, rows, f"{method_name} - Comparing Rules")
            print()
    else:
        # Group by rules, compare methods within each rule
        for rule_name in VOTING_RULES.keys():
            rows = []
            for method_name in GENERATION_METHODS.keys():
                print(f"{C.DIM}  {rule_name} x {method_name}...{C.RESET}", end="\r")
                results = simulate_paradox(
                    rule_name, n_profiles, n_voters, method_name,
                    n_dim=n_dim, sigma_factor=sigma_factor,
                    utility_func=utility_func, distance_metric=distance_metric,
                    approval_policy=approval_policy, approval_param=approval_param
                )
                
                row = [
                    method_name,
                    f"{results['cycle_percentage']:.2f}%",
                    f"{results['vse_mean']:.3f}",
                    f"{results['vse_min']:.3f}",
                ]
                if log_regret:
                    row.append(f"{results['regret_mean']:.4f}")
                if log_winner_rank:
                    row.extend([
                        f"{results['winner_rank_1st_pct']:.1f}%",
                        f"{results['winner_rank_2nd_pct']:.1f}%",
                        f"{results['winner_rank_3rd_pct']:.1f}%",
                    ])
                if log_raw_utility:
                    row.append(f"{results['raw_utility_mean']:.3f}")
                
                rows.append(row)
            
            print(" " * 60, end="\r")
            format_table(headers, rows, f"{rule_name} - Comparing Methods")
            print()


# =============================================================================
# Interactive Mode
# =============================================================================

def interactive_mode():
    """Interactive simulation mode."""
    C = Colors
    
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}       SPATIAL VOTING SIMULATOR{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")
    
    # Select voting rule
    print(f"\n{C.YELLOW}Available voting rules:{C.RESET}")
    rules = list(VOTING_RULES.keys())
    for i, rule in enumerate(rules, 1):
        rule_type = VOTING_RULES[rule]['type']
        desc = VOTING_RULES[rule]['description']
        print(f"  {C.GREEN}{i:2d}.{C.RESET} {rule:<18} [{rule_type[:4]}] {C.DIM}{desc}{C.RESET}")
    
    while True:
        try:
            choice = input(f"\n{C.CYAN}Select voting rule (1-{len(rules)}): {C.RESET}").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(rules):
                    voting_rule = rules[idx]
                    break
            elif choice in rules:
                voting_rule = choice
                break
            print(f"{C.RED}Invalid choice.{C.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{C.YELLOW}Exiting...{C.RESET}")
            return
    
    # Select generation method
    print(f"\n{C.YELLOW}Profile generation methods:{C.RESET}")
    methods = list(GENERATION_METHODS.keys())
    for i, method in enumerate(methods, 1):
        desc = GENERATION_METHODS[method][1]
        print(f"  {C.GREEN}{i}.{C.RESET} {method:<22} {C.DIM}- {desc}{C.RESET}")
    
    while True:
        try:
            choice = input(f"\n{C.CYAN}Select generation method (1-{len(methods)}): {C.RESET}").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(methods):
                    gen_method = methods[idx]
                    break
            elif choice in methods:
                gen_method = choice
                break
            print(f"{C.RED}Invalid choice.{C.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{C.YELLOW}Exiting...{C.RESET}")
            return
    
    # Get simulation parameters
    while True:
        try:
            n_profiles = int(input(f"\n{C.CYAN}Number of profiles to simulate: {C.RESET}"))
            if n_profiles > 0:
                break
            print(f"{C.RED}Please enter a positive number.{C.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{C.YELLOW}Exiting...{C.RESET}")
            return
    
    while True:
        try:
            n_voters = int(input(f"{C.CYAN}Number of voters per profile: {C.RESET}"))
            if n_voters > 0:
                break
            print(f"{C.RED}Please enter a positive number.{C.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{C.YELLOW}Exiting...{C.RESET}")
            return
    
    # Run simulation
    print(f"\n{C.DIM}Running simulation...{C.RESET}")
    print(f"{C.DIM}  Rule: {C.BOLD}{voting_rule}{C.RESET}")
    print(f"{C.DIM}  Method: {C.BOLD}{gen_method}{C.RESET}")
    print(f"{C.DIM}  Profiles: {n_profiles:,} x {n_voters} voters{C.RESET}")
    
    results = simulate_paradox(voting_rule, n_profiles, n_voters, gen_method)
    
    # Display results
    print(f"\n{C.BOLD}{C.GREEN}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}  RESULTS{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}{'=' * 60}{C.RESET}")
    
    print(f"\n{C.BOLD}Configuration:{C.RESET}")
    print(f"  {C.DIM}Rule:{C.RESET} {voting_rule} ({VOTING_RULES[voting_rule]['type']})")
    print(f"  {C.DIM}Method:{C.RESET} {gen_method}")
    print(f"  {C.DIM}Profiles:{C.RESET} {results['total_profiles']:,}")
    print(f"  {C.DIM}Dimension:{C.RESET} {results['n_dim']}D")
    print(f"  {C.DIM}Distance metric:{C.RESET} {results['distance_metric']}")
    print(f"  {C.DIM}Utility function:{C.RESET} {results['utility_func']}")
    
    cycle_pct = results['cycle_percentage']
    cycle_color = C.RED if cycle_pct > 10 else C.YELLOW if cycle_pct > 5 else C.GREEN
    print(f"\n{C.BOLD}Condorcet Cycles:{C.RESET} {cycle_color}{results['cycle_count']:,} "
          f"({cycle_pct:.2f}%){C.RESET}")
    
    print(f"\n{C.BOLD}{C.BLUE}Cycle Type Breakdown:{C.RESET}")
    print(f"  {C.CYAN}Type 1{C.RESET} (A>B>C>A): {results['type1_count']:,} "
          f"({results['type1_percentage']:.2f}%)")
    print(f"  {C.CYAN}Type 2{C.RESET} (B>A>C>B): {results['type2_count']:,} "
          f"({results['type2_percentage']:.2f}%)")
    
    print(f"\n{C.BOLD}{C.BLUE}Normalized VSE (Voter Satisfaction Efficiency):{C.RESET}")
    vse_color = C.GREEN if results['vse_mean'] > 0.8 else C.YELLOW if results['vse_mean'] > 0.5 else C.RED
    print(f"  {C.BOLD}Mean:{C.RESET} {vse_color}{results['vse_mean']:.3f}{C.RESET}")
    print(f"  {C.DIM}Std:  {results['vse_std']:.3f}  |  "
          f"Min: {results['vse_min']:.3f}  |  Max: {results['vse_max']:.3f}{C.RESET}")
    
    print(f"\n{C.BOLD}{C.BLUE}Enhanced Metrics:{C.RESET}")
    print(f"  {C.DIM}Raw social utility (mean):{C.RESET} {results['raw_utility_mean']:.4f}")
    print(f"  {C.DIM}Regret (mean):{C.RESET} {results['regret_mean']:.4f}")
    print(f"  {C.DIM}Regret (max):{C.RESET} {results['regret_max']:.4f}")
    print(f"\n{C.BOLD}{C.BLUE}Winner Rank Distribution:{C.RESET}")
    print(f"  {C.GREEN}1st best:{C.RESET} {results['winner_rank_1st_pct']:.1f}%")
    print(f"  {C.YELLOW}2nd best:{C.RESET} {results['winner_rank_2nd_pct']:.1f}%")
    print(f"  {C.RED}3rd best:{C.RESET} {results['winner_rank_3rd_pct']:.1f}%")
    
    print(f"\n{C.BOLD}{C.GREEN}{'=' * 60}{C.RESET}\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Spatial Voting Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paradox_simulator.py                           # Interactive mode
  python paradox_simulator.py -m -n 1000 -v 25          # Compare all methods
  python paradox_simulator.py -r -n 1000 -v 25          # Compare all rules
  python paradox_simulator.py -m -r -n 500 -v 25        # Comprehensive comparison
  python paradox_simulator.py -r --method spatial_polarized -n 1000 -v 25 -d 3
  python paradox_simulator.py -m --rule score --log-winner-rank
        """
    )
    
    # Basic flags
    parser.add_argument('-m', '--methods', action='store_true',
                        help='Simulate through all generation methods')
    parser.add_argument('-r', '--rules', action='store_true',
                        help='Simulate through all voting rules')
    parser.add_argument('-b', '--by-method', action='store_true',
                        help='Group by method instead of by rule (when both -m and -r)')
    
    # Simulation parameters
    parser.add_argument('-n', '--profiles', type=int, default=None,
                        help='Number of profiles to simulate')
    parser.add_argument('-v', '--voters', type=int, default=None,
                        help='Number of voters per profile')
    parser.add_argument('--rule', type=str, default=None,
                        help='Voting rule to use (when using -m flag)')
    parser.add_argument('--method', type=str, default=None,
                        help='Generation method to use (when using -r flag)')
    
    # Spatial configuration
    parser.add_argument('-d', '--n-dim', type=int, default=2,
                        help='Dimension of vector space (default: 2)')
    parser.add_argument('--distance-metric', type=str, default='l2',
                        choices=['l2', 'l1', 'cosine'],
                        help='Distance metric (default: l2)')
    
    # Utility function
    parser.add_argument('--utility-func', type=str, default='gaussian',
                        choices=['gaussian', 'quadratic', 'linear'],
                        help='Utility function (default: gaussian)')
    parser.add_argument('--sigma-factor', type=float, default=0.5,
                        help='Sigma = factor * sqrt(n_dim) for Gaussian (default: 0.5)')
    
    # Ballot conversion
    parser.add_argument('--approval-policy', type=str, default='top_k',
                        choices=['top_k', 'delta', 'mean', 'absolute'],
                        help='Approval threshold policy (default: top_k)')
    parser.add_argument('--approval-param', type=float, default=0.5,
                        help='Approval policy parameter (default: 0.5)')
    parser.add_argument('--score-max', type=int, default=5,
                        help='Maximum score for score/STAR voting (default: 5)')
    
    # Reproducibility
    parser.add_argument('--rng-seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--epsilon', type=float, default=1e-9,
                        help='Epsilon for tie-breaking (default: 1e-9)')
    
    # Enhanced logging
    parser.add_argument('--log-raw-utility', action='store_true',
                        help='Include raw social utility in output')
    parser.add_argument('--log-regret', action='store_true',
                        help='Include regret in output')
    parser.add_argument('--log-winner-rank', action='store_true',
                        help='Include winner rank distribution in output')
    
    args = parser.parse_args()
    
    # Set RNG seed if provided
    if args.rng_seed is not None:
        np.random.seed(args.rng_seed)
        print(f"{Colors.DIM}RNG seed: {args.rng_seed}{Colors.RESET}")
    
    # If no flags, run interactive mode
    if not args.methods and not args.rules:
        interactive_mode()
        return
    
    # Get number of profiles
    if args.profiles is None:
        try:
            n_profiles = int(input(f"{Colors.CYAN}Number of profiles to simulate: {Colors.RESET}"))
            if n_profiles <= 0:
                print(f"{Colors.RED}Please enter a positive number.{Colors.RESET}")
                return
        except (ValueError, KeyboardInterrupt):
            print(f"\n{Colors.YELLOW}Exiting...{Colors.RESET}")
            return
    else:
        n_profiles = args.profiles
    
    # Get number of voters
    if args.voters is None:
        try:
            n_voters = int(input(f"{Colors.CYAN}Number of voters per profile: {Colors.RESET}"))
            if n_voters <= 0:
                print(f"{Colors.RED}Please enter a positive number.{Colors.RESET}")
                return
        except (ValueError, KeyboardInterrupt):
            print(f"\n{Colors.YELLOW}Exiting...{Colors.RESET}")
            return
    else:
        n_voters = args.voters
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}       SPATIAL VOTING SIMULATOR{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    
    # Common parameters
    common_params = {
        'n_dim': args.n_dim,
        'sigma_factor': args.sigma_factor,
        'utility_func': args.utility_func,
        'distance_metric': args.distance_metric,
        'approval_policy': args.approval_policy,
        'approval_param': args.approval_param,
        'log_regret': args.log_regret,
        'log_winner_rank': args.log_winner_rank,
        'log_raw_utility': args.log_raw_utility,
    }
    
    # Handle different flag combinations
    if args.methods and args.rules:
        run_all_methods_all_rules(
            n_profiles, n_voters, by_method=args.by_method,
            **common_params
        )
    elif args.methods:
        # Only -m flag: compare all methods with a single rule
        if args.rule:
            voting_rule = args.rule
            if voting_rule not in VOTING_RULES:
                print(f"{Colors.RED}Invalid voting rule: {voting_rule}{Colors.RESET}")
                print(f"{Colors.YELLOW}Available rules: {', '.join(VOTING_RULES.keys())}{Colors.RESET}")
                return
        else:
            # Ask for voting rule
            print(f"\n{Colors.YELLOW}Available voting rules:{Colors.RESET}")
            rules = list(VOTING_RULES.keys())
            for i, rule in enumerate(rules, 1):
                print(f"  {Colors.GREEN}{i:2d}.{Colors.RESET} {rule}")
            
            try:
                choice = input(f"\n{Colors.CYAN}Select voting rule (1-{len(rules)}): {Colors.RESET}").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(rules):
                        voting_rule = rules[idx]
                    else:
                        print(f"{Colors.RED}Invalid choice.{Colors.RESET}")
                        return
                elif choice in rules:
                    voting_rule = choice
                else:
                    print(f"{Colors.RED}Invalid choice.{Colors.RESET}")
                    return
            except (ValueError, KeyboardInterrupt):
                print(f"\n{Colors.YELLOW}Exiting...{Colors.RESET}")
                return
        
        run_all_methods(voting_rule, n_profiles, n_voters, **common_params)
    
    elif args.rules:
        # Only -r flag: compare all rules with a single method
        if args.method:
            gen_method = args.method
            if gen_method not in GENERATION_METHODS:
                print(f"{Colors.RED}Invalid generation method: {gen_method}{Colors.RESET}")
                print(f"{Colors.YELLOW}Available methods: {', '.join(GENERATION_METHODS.keys())}{Colors.RESET}")
                return
        else:
            # Ask for generation method
            print(f"\n{Colors.YELLOW}Profile generation methods:{Colors.RESET}")
            methods = list(GENERATION_METHODS.keys())
            for i, method in enumerate(methods, 1):
                desc = GENERATION_METHODS[method][1]
                print(f"  {Colors.GREEN}{i}.{Colors.RESET} {method:<22} {Colors.DIM}- {desc}{Colors.RESET}")
            
            try:
                choice = input(f"\n{Colors.CYAN}Select generation method (1-{len(methods)}): {Colors.RESET}").strip()
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(methods):
                        gen_method = methods[idx]
                    else:
                        print(f"{Colors.RED}Invalid choice.{Colors.RESET}")
                        return
                elif choice in methods:
                    gen_method = choice
                else:
                    print(f"{Colors.RED}Invalid choice.{Colors.RESET}")
                    return
            except (ValueError, KeyboardInterrupt):
                print(f"\n{Colors.YELLOW}Exiting...{Colors.RESET}")
                return
        
        run_all_rules(gen_method, n_profiles, n_voters, **common_params)


if __name__ == "__main__":
    main()
