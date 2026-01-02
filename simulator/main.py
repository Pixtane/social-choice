"""
Main simulation orchestration module.

High-level functions for running voting simulations:
- generate_preferences: Generate spatial preferences
- run_simulation: Run simulation for a single voting rule
- run_experiment: Run complete experiment with multiple rules
"""

import numpy as np
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field

from .config import (
    SimulationConfig, GeometryConfig, ManipulationConfig,
    UtilityConfig, VotingRuleConfig, AVAILABLE_VOTING_RULES
)
from .geometry import GeometryGenerator, SpatialProfile
from .utility import UtilityComputer, utilities_to_rankings
from .voting_rules import VotingRuleEngine, VotingResult, get_rule_type, RuleType
from .manipulation import ManipulationEngine, ManipulationResult, compute_manipulation_impact
from .metrics import MetricsComputer, ProfileMetrics, AggregateMetrics
from .storage import save_experiment, get_storage_paths


@dataclass
class PreferenceProfile:
    """Complete preference profile with all data."""
    
    # Spatial data
    voter_positions: np.ndarray  # (n_profiles, n_voters, n_dim)
    candidate_positions: np.ndarray  # (n_profiles, n_candidates, n_dim)
    
    # Derived data
    utilities: np.ndarray  # (n_profiles, n_voters, n_candidates)
    rankings: np.ndarray  # (n_profiles, n_voters, n_candidates)
    
    # Metadata
    n_profiles: int
    n_voters: int
    n_candidates: int
    n_dim: int
    
    # Configuration used
    config: SimulationConfig
    rng_seed: Optional[int] = None


@dataclass
class SimulationResult:
    """Result of a simulation run."""
    
    # Configuration
    config: SimulationConfig
    voting_rule: str
    
    # Winners for each profile
    winners: np.ndarray  # (n_profiles,)
    
    # Per-profile metrics
    profile_metrics: List[ProfileMetrics]
    
    # Aggregate metrics
    aggregate_metrics: AggregateMetrics
    
    # Timing
    compute_time: float
    
    # Manipulation results (if enabled)
    manipulation_results: Optional[List[ManipulationResult]] = None
    manipulation_impact: Optional[List[Dict[str, Any]]] = None


@dataclass
class ExperimentResult:
    """Result of a complete experiment with multiple rules."""
    
    # Configuration
    config: SimulationConfig
    
    # Preference data
    preferences: PreferenceProfile
    
    # Results per voting rule
    rule_results: Dict[str, SimulationResult]
    
    # Total timing
    total_compute_time: float
    
    # Storage paths
    inputs_path: Optional[str] = None
    results_path: Optional[str] = None


def generate_preferences(
    config: SimulationConfig,
    rng_seed: Optional[int] = None
) -> PreferenceProfile:
    """
    Generate spatial preference profiles.
    
    This is the first step in the simulation pipeline.
    Generates voter and candidate positions, then computes
    utilities and rankings.
    
    Args:
        config: Simulation configuration
        rng_seed: Random seed (overrides config if provided)
        
    Returns:
        PreferenceProfile with all spatial and preference data
    """
    # Set up random generator
    seed = rng_seed if rng_seed is not None else config.rng_seed
    rng = np.random.default_rng(seed)
    
    # Generate spatial positions
    geometry_gen = GeometryGenerator(config.geometry, rng)
    spatial_profile = geometry_gen.generate(
        config.n_profiles,
        config.n_voters,
        config.n_candidates
    )
    
    # Compute utilities
    utility_computer = UtilityComputer(config.utility)
    
    # Process each profile
    n_profiles = config.n_profiles
    n_voters = config.n_voters
    n_candidates = config.n_candidates
    n_dim = spatial_profile.n_dim
    
    utilities = np.empty((n_profiles, n_voters, n_candidates))
    rankings = np.empty((n_profiles, n_voters, n_candidates), dtype=int)
    
    for i in range(n_profiles):
        # Compute distances and utilities for this profile
        distances = utility_computer.compute_distances(
            spatial_profile.voter_positions[i],
            spatial_profile.candidate_positions[i]
        )
        utilities[i] = utility_computer.compute_utilities(distances, n_dim)
        rankings[i] = utilities_to_rankings(utilities[i], config.epsilon)
    
    return PreferenceProfile(
        voter_positions=spatial_profile.voter_positions,
        candidate_positions=spatial_profile.candidate_positions,
        utilities=utilities,
        rankings=rankings,
        n_profiles=n_profiles,
        n_voters=n_voters,
        n_candidates=n_candidates,
        n_dim=n_dim,
        config=config,
        rng_seed=seed,
    )


def run_simulation(
    preferences: PreferenceProfile,
    voting_rule: str,
    config: Optional[SimulationConfig] = None,
    rng_seed: Optional[int] = None
) -> SimulationResult:
    """
    Run simulation for a single voting rule.
    
    Args:
        preferences: Generated preference profile
        voting_rule: Name of voting rule to apply
        config: Configuration (defaults to preferences.config)
        rng_seed: Random seed for tie-breaking
        
    Returns:
        SimulationResult with all metrics
    """
    start_time = time.perf_counter()
    
    config = config or preferences.config
    seed = rng_seed if rng_seed is not None else config.rng_seed
    rng = np.random.default_rng(seed)
    
    # Set up engines
    voting_engine = VotingRuleEngine(config.voting_rule_config, rng)
    metrics_computer = MetricsComputer(config.epsilon)
    
    n_profiles = preferences.n_profiles
    rule_type = get_rule_type(voting_rule)
    
    # Process with or without manipulation
    if config.manipulation.enabled:
        manipulation_engine = ManipulationEngine(config.manipulation, rng)
    else:
        manipulation_engine = None
    
    # Store results
    winners = np.empty(n_profiles, dtype=int)
    profile_metrics_list = []
    manipulation_results = [] if manipulation_engine else None
    manipulation_impacts = [] if manipulation_engine else None
    
    for i in range(n_profiles):
        utilities = preferences.utilities[i]
        rankings = preferences.rankings[i]
        
        # Apply manipulation if enabled
        if manipulation_engine:
            manip_result = manipulation_engine.apply_manipulation(
                utilities, rankings, voting_rule
            )
            manipulation_results.append(manip_result)
            
            # Use manipulated ballots for voting
            vote_utilities = manip_result.manipulated_utilities
            vote_rankings = manip_result.manipulated_rankings
        else:
            vote_utilities = utilities
            vote_rankings = rankings
        
        # Apply voting rule
        if rule_type == RuleType.CARDINAL:
            result = voting_engine.apply_rule(voting_rule, utilities=vote_utilities)
        else:
            result = voting_engine.apply_rule(voting_rule, rankings=vote_rankings)
        
        winner = result.winner
        
        # Handle case where no winner (e.g., no Condorcet winner)
        if winner < 0:
            # Fallback to utilitarian winner
            winner = np.argmax(np.mean(utilities, axis=0))
        
        winners[i] = winner
        
        # Compute metrics (using sincere utilities for true VSE)
        profile_metrics = metrics_computer.compute_profile_metrics(
            utilities, rankings, winner
        )
        profile_metrics_list.append(profile_metrics)
        
        # Compute manipulation impact if applicable
        if manipulation_engine and manip_result.n_manipulators > 0:
            # Get sincere result
            if rule_type == RuleType.CARDINAL:
                sincere_result = voting_engine.apply_rule(
                    voting_rule, utilities=utilities
                )
            else:
                sincere_result = voting_engine.apply_rule(
                    voting_rule, rankings=rankings
                )
            
            impact = compute_manipulation_impact(
                sincere_result, result,
                utilities, manip_result.manipulator_mask
            )
            manipulation_impacts.append(impact)
    
    # Aggregate metrics
    aggregate_metrics = metrics_computer.aggregate_metrics(
        profile_metrics_list, preferences.n_candidates
    )
    
    compute_time = time.perf_counter() - start_time
    
    return SimulationResult(
        config=config,
        voting_rule=voting_rule,
        winners=winners,
        profile_metrics=profile_metrics_list,
        aggregate_metrics=aggregate_metrics,
        compute_time=compute_time,
        manipulation_results=manipulation_results,
        manipulation_impact=manipulation_impacts,
    )


def run_experiment(
    config: SimulationConfig,
    save_results: bool = True,
    base_dir: Optional[str] = None,
    verbose: bool = False
) -> ExperimentResult:
    """
    Run complete experiment with multiple voting rules.
    
    This is the main entry point for running simulations.
    
    Args:
        config: Simulation configuration
        save_results: Whether to save results to disk
        base_dir: Base directory for storage
        verbose: Print progress information
        
    Returns:
        ExperimentResult with all data and metrics
    """
    total_start = time.perf_counter()
    
    # Generate preferences
    if verbose:
        print(f"Generating {config.n_profiles} preference profiles...")
    
    preferences = generate_preferences(config)
    
    if verbose:
        print(f"  Voters: {config.n_voters}, Candidates: {config.n_candidates}")
        print(f"  Dimensions: {preferences.n_dim}")
        print(f"  Geometry: {config.geometry.method}")
    
    # Run simulation for each voting rule
    rule_results = {}
    
    for rule_name in config.voting_rules:
        if verbose:
            print(f"Running {rule_name}...")
        
        result = run_simulation(preferences, rule_name, config)
        rule_results[rule_name] = result
        
        if verbose:
            agg = result.aggregate_metrics
            print(f"  VSE: {agg.vse_mean:.3f} ± {agg.vse_std:.3f}")
            print(f"  Cycles: {agg.cycle_percentage:.1f}%")
            print(f"  Time: {result.compute_time:.3f}s")
    
    total_compute_time = time.perf_counter() - total_start
    
    # Prepare result
    experiment_result = ExperimentResult(
        config=config,
        preferences=preferences,
        rule_results=rule_results,
        total_compute_time=total_compute_time,
    )
    
    # Save if requested
    if save_results:
        if verbose:
            print("Saving results...")
        
        # Prepare profile results for storage
        profile_results = _prepare_profile_results(experiment_result)
        
        # Get aggregate metrics from first rule (they share cycle info)
        first_rule = list(rule_results.keys())[0]
        aggregate_metrics = rule_results[first_rule].aggregate_metrics
        
        inputs_path, results_path = save_experiment(
            config=config,
            voter_positions=preferences.voter_positions,
            candidate_positions=preferences.candidate_positions,
            utilities=preferences.utilities,
            rankings=preferences.rankings,
            profile_results=profile_results,
            aggregate_metrics=aggregate_metrics,
            compute_time=total_compute_time,
            base_dir=base_dir,
        )
        
        experiment_result.inputs_path = inputs_path
        experiment_result.results_path = results_path
        
        if verbose:
            print(f"  Inputs: {inputs_path}")
            print(f"  Results: {results_path}")
    
    if verbose:
        print(f"Total time: {total_compute_time:.3f}s")
    
    return experiment_result


def _prepare_profile_results(experiment: ExperimentResult) -> List[Dict[str, Any]]:
    """
    Prepare per-profile results for storage.
    
    Args:
        experiment: ExperimentResult
        
    Returns:
        List of dictionaries, one per profile
    """
    results = []
    n_profiles = experiment.preferences.n_profiles
    
    for i in range(n_profiles):
        row = {
            'profile_index': i,
            'experiment_id': experiment.config.experiment_id,
        }
        
        # Add results for each voting rule
        for rule_name, rule_result in experiment.rule_results.items():
            prefix = f"{rule_name}_"
            
            # Winner
            row[f'{prefix}winner'] = int(rule_result.winners[i])
            
            # Metrics
            metrics = rule_result.profile_metrics[i]
            row[f'{prefix}vse'] = metrics.vse
            row[f'{prefix}winner_rank'] = metrics.winner_rank
            row[f'{prefix}regret'] = metrics.regret
            row[f'{prefix}condorcet_winner_exists'] = metrics.condorcet_winner_exists
            row[f'{prefix}is_condorcet_winner'] = metrics.is_condorcet_winner
            row[f'{prefix}has_cycle'] = metrics.has_cycle
            row[f'{prefix}cycle_type'] = metrics.cycle_type
            
            # Manipulation impact if available
            if rule_result.manipulation_impact and i < len(rule_result.manipulation_impact):
                impact = rule_result.manipulation_impact[i]
                row[f'{prefix}manipulation_winner_changed'] = impact['winner_changed']
                row[f'{prefix}manipulation_gain'] = impact['manipulator_gain']
                row[f'{prefix}manipulation_successful'] = impact['manipulation_successful']
        
        results.append(row)
    
    return results


def compare_rules(
    config: SimulationConfig,
    verbose: bool = True
) -> Dict[str, AggregateMetrics]:
    """
    Compare multiple voting rules on the same preference profiles.
    
    Convenience function for quick comparisons.
    
    Args:
        config: Configuration with voting_rules list
        verbose: Print comparison table
        
    Returns:
        Dictionary mapping rule names to aggregate metrics
    """
    experiment = run_experiment(config, save_results=False, verbose=False)
    
    results = {}
    for rule_name, rule_result in experiment.rule_results.items():
        results[rule_name] = rule_result.aggregate_metrics
    
    if verbose:
        _print_comparison_table(results, config)
    
    return results


def _print_comparison_table(
    results: Dict[str, AggregateMetrics],
    config: SimulationConfig
):
    """Print formatted comparison table."""
    print("\n" + "=" * 70)
    print("VOTING RULE COMPARISON")
    print("=" * 70)
    print(f"Profiles: {config.n_profiles} | Voters: {config.n_voters} | "
          f"Candidates: {config.n_candidates}")
    print(f"Geometry: {config.geometry.method} | Utility: {config.utility.function}")
    print("-" * 70)
    
    # Header
    print(f"{'Rule':<15} {'VSE':>8} {'VSE σ':>8} {'1st%':>8} "
          f"{'Cycles%':>8} {'CW Eff%':>8}")
    print("-" * 70)
    
    # Rows
    for rule_name, agg in results.items():
        print(f"{rule_name:<15} {agg.vse_mean:>8.3f} {agg.vse_std:>8.3f} "
              f"{agg.winner_rank_1st_pct:>7.1f}% "
              f"{agg.cycle_percentage:>7.1f}% {agg.condorcet_efficiency:>7.1f}%")
    
    print("=" * 70)


def quick_simulation(
    n_profiles: int = 1000,
    n_voters: int = 25,
    n_candidates: int = 3,
    voting_rules: Optional[List[str]] = None,
    geometry_method: str = 'uniform',
    n_dim: int = 2,
    rng_seed: Optional[int] = None,
    verbose: bool = True
) -> ExperimentResult:
    """
    Quick simulation with sensible defaults.
    
    Convenience function for rapid experimentation.
    
    Args:
        n_profiles: Number of election profiles
        n_voters: Number of voters per profile
        n_candidates: Number of candidates
        voting_rules: List of voting rule names (defaults to common ones)
        geometry_method: Spatial generation method
        n_dim: Spatial dimensions
        rng_seed: Random seed
        verbose: Print progress
        
    Returns:
        ExperimentResult
    """
    if voting_rules is None:
        voting_rules = ['plurality', 'borda', 'irv', 'approval', 'star', 'schulze']
    
    config = SimulationConfig(
        n_profiles=n_profiles,
        n_voters=n_voters,
        n_candidates=n_candidates,
        voting_rules=voting_rules,
        geometry=GeometryConfig(method=geometry_method, n_dim=n_dim),
        rng_seed=rng_seed,
    )
    
    return run_experiment(config, save_results=False, verbose=verbose)


# Module-level exports for convenience
__all__ = [
    'generate_preferences',
    'run_simulation', 
    'run_experiment',
    'compare_rules',
    'quick_simulation',
    'PreferenceProfile',
    'SimulationResult',
    'ExperimentResult',
]







