"""
Command-line interface for the voting simulator.

Provides both interactive mode and command-line argument support
for running voting simulations.
"""

import argparse
import os
import sys
from typing import Optional, List

import numpy as np

from .config import (
    SimulationConfig, GeometryConfig, ManipulationConfig,
    UtilityConfig, VotingRuleConfig,
    AVAILABLE_VOTING_RULES, AVAILABLE_GEOMETRY_METHODS,
    AVAILABLE_MANIPULATION_STRATEGIES
)
from .main import (
    run_experiment, compare_rules, quick_simulation,
    generate_preferences, run_simulation
)
from .storage import list_experiments, load_experiment
from .comparison import (
    ExperimentComparator, compare_experiments, get_comparison_modes
)


# Enable Windows ANSI colors
os.system('')


class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    RESET = '\033[0m'


def print_banner():
    """Print the simulator banner."""
    C = Colors
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}       SPATIAL VOTING SIMULATOR{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")


def print_section(title: str):
    """Print a section header."""
    C = Colors
    print(f"\n{C.BOLD}{C.YELLOW}{title}{C.RESET}")
    print(f"{C.DIM}{'-' * len(title)}{C.RESET}")


def format_table(headers: List[str], rows: List[List], title: str = None):
    """Format and print a table."""
    C = Colors
    
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    col_widths = [w + 2 for w in col_widths]
    total_width = sum(col_widths) + len(headers) + 1
    
    if title:
        print(f"\n{C.BOLD}{C.CYAN}{title}{C.RESET}")
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


def get_user_input(prompt: str, default=None, valid_options=None, cast_type=None):
    """Get validated user input."""
    C = Colors
    
    while True:
        try:
            if default is not None:
                full_prompt = f"{C.CYAN}{prompt} [{default}]: {C.RESET}"
            else:
                full_prompt = f"{C.CYAN}{prompt}: {C.RESET}"
            
            value = input(full_prompt).strip()
            
            if not value and default is not None:
                value = default
            
            if cast_type:
                value = cast_type(value)
            
            if valid_options and value not in valid_options:
                print(f"{C.RED}Invalid option. Choose from: {', '.join(map(str, valid_options))}{C.RESET}")
                continue
            
            return value
            
        except ValueError as e:
            print(f"{C.RED}Invalid input: {e}{C.RESET}")
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Operation cancelled.{C.RESET}")
            return None


def select_from_list(items: List, prompt: str, show_description: bool = False) -> Optional[int]:
    """Let user select from a numbered list."""
    C = Colors
    
    print(f"\n{C.YELLOW}{prompt}:{C.RESET}")
    for i, item in enumerate(items, 1):
        if isinstance(item, tuple) and show_description:
            name, desc = item
            print(f"  {C.GREEN}{i:2d}.{C.RESET} {name:<20} {C.DIM}{desc}{C.RESET}")
        else:
            print(f"  {C.GREEN}{i:2d}.{C.RESET} {item}")
    
    while True:
        try:
            choice = input(f"\n{C.CYAN}Select (1-{len(items)}): {C.RESET}").strip()
            if choice.isdigit():
                idx = int(choice) - 1
                if 0 <= idx < len(items):
                    return idx
            print(f"{C.RED}Invalid choice.{C.RESET}")
        except KeyboardInterrupt:
            print(f"\n{C.YELLOW}Cancelled.{C.RESET}")
            return None


def compare_experiments_cli(
    experiment_ids: List[str],
    mode: str = 'by_experiment',
    output_format: str = 'text'
):
    """
    Compare multiple experiments from the CLI.
    
    Args:
        experiment_ids: List of experiment IDs or filenames to compare
        mode: Comparison mode (by_experiment, by_rule, by_metric, differential)
        output_format: Output format (text, markdown, csv)
    """
    C = Colors
    
    print_banner()
    print(f"\n{C.BOLD}{C.YELLOW}Experiment Comparison Mode{C.RESET}")
    print(f"{C.DIM}{'=' * 40}{C.RESET}")
    
    # List available experiments
    all_experiments = list_experiments()
    
    if not all_experiments:
        print(f"\n{C.RED}No saved experiments found!{C.RESET}")
        print(f"{C.YELLOW}Run a simulation first to create experiments.{C.RESET}")
        return
    
    # Find matching experiments
    selected_paths = []
    unmatched = []
    
    for exp_id in experiment_ids:
        found = False
        for exp in all_experiments:
            # Match by experiment_id, filename, or partial match
            exp_name = exp.get('filename', '')
            exp_full_id = exp.get('experiment_id', '')
            
            if (exp_id == exp_name or 
                exp_id == exp_full_id or
                exp_id in exp_name or
                exp_id in exp_full_id):
                selected_paths.append(exp['inputs_path'])
                found = True
                print(f"  {C.GREEN}[+]{C.RESET} Found: {exp_name}")
                break
        
        if not found:
            unmatched.append(exp_id)
    
    if unmatched:
        print(f"\n{C.YELLOW}Warning: Could not find experiments:{C.RESET}")
        for exp_id in unmatched:
            print(f"  {C.RED}[-]{C.RESET} {exp_id}")
    
    if len(selected_paths) < 2:
        print(f"\n{C.RED}Need at least 2 experiments to compare.{C.RESET}")
        print(f"\n{C.YELLOW}Available experiments:{C.RESET}")
        for exp in all_experiments:
            print(f"  - {exp.get('filename', 'unknown')}")
        return
    
    print(f"\n{C.CYAN}Comparing {len(selected_paths)} experiments...{C.RESET}")
    print(f"{C.DIM}Mode: {mode}{C.RESET}")
    
    # Create comparator and run comparison
    try:
        comparator = ExperimentComparator.from_paths(selected_paths)
        result = comparator.compare(mode=mode)
        output = comparator.format_table(result, format=output_format)
        
        print(f"\n{C.BOLD}{C.GREEN}Comparison Results:{C.RESET}")
        print(output)
        
        # Show available modes
        print(f"\n{C.DIM}Available comparison modes:{C.RESET}")
        for mode_name, mode_desc in get_comparison_modes().items():
            print(f"  {C.CYAN}{mode_name:<15}{C.RESET} {C.DIM}{mode_desc}{C.RESET}")
        
    except Exception as e:
        print(f"\n{C.RED}Error comparing experiments: {e}{C.RESET}")
        import traceback
        traceback.print_exc()


def interactive_compare_mode():
    """Interactive mode for comparing experiments."""
    C = Colors
    
    print_banner()
    print(f"\n{C.BOLD}{C.YELLOW}Interactive Experiment Comparison{C.RESET}")
    print(f"{C.DIM}{'=' * 40}{C.RESET}")
    
    # List available experiments
    all_experiments = list_experiments()
    
    if not all_experiments:
        print(f"\n{C.RED}No saved experiments found!{C.RESET}")
        return
    
    print(f"\n{C.YELLOW}Available experiments:{C.RESET}")
    for i, exp in enumerate(all_experiments, 1):
        filename = exp.get('filename', 'unknown')
        exp_id = exp.get('experiment_id', 'unknown')[:8]
        n_profiles = exp.get('n_profiles', '?')
        n_voters = exp.get('n_voters', '?')
        print(f"  {C.GREEN}{i:2d}.{C.RESET} {filename} ({exp_id}) - {n_profiles} profiles, {n_voters} voters")
    
    print(f"\n{C.CYAN}Enter experiment numbers to compare (comma-separated, e.g., 1,2,3):{C.RESET}")
    print(f"{C.DIM}Or enter 'all' to compare all experiments{C.RESET}")
    
    try:
        selection = input(f"{C.CYAN}Selection: {C.RESET}").strip()
        
        if selection.lower() == 'all':
            selected_paths = [exp['inputs_path'] for exp in all_experiments]
        else:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_paths = []
            for idx in indices:
                if 0 <= idx < len(all_experiments):
                    selected_paths.append(all_experiments[idx]['inputs_path'])
                else:
                    print(f"{C.RED}Invalid index: {idx + 1}{C.RESET}")
        
        if len(selected_paths) < 2:
            print(f"\n{C.RED}Need at least 2 experiments to compare.{C.RESET}")
            return
        
        # Select comparison mode
        print(f"\n{C.YELLOW}Select comparison mode:{C.RESET}")
        modes = list(get_comparison_modes().items())
        for i, (mode_name, mode_desc) in enumerate(modes, 1):
            print(f"  {C.GREEN}{i}.{C.RESET} {mode_name:<15} {C.DIM}{mode_desc}{C.RESET}")
        
        mode_choice = input(f"\n{C.CYAN}Mode (1-{len(modes)}) [1]: {C.RESET}").strip()
        mode_idx = int(mode_choice) - 1 if mode_choice else 0
        mode_name = modes[mode_idx][0] if 0 <= mode_idx < len(modes) else 'by_experiment'
        
        # Select output format
        print(f"\n{C.YELLOW}Select output format:{C.RESET}")
        formats = ['text', 'markdown', 'csv']
        for i, fmt in enumerate(formats, 1):
            print(f"  {C.GREEN}{i}.{C.RESET} {fmt}")
        
        fmt_choice = input(f"\n{C.CYAN}Format (1-3) [1]: {C.RESET}").strip()
        fmt_idx = int(fmt_choice) - 1 if fmt_choice else 0
        output_format = formats[fmt_idx] if 0 <= fmt_idx < len(formats) else 'text'
        
        # Run comparison
        print(f"\n{C.CYAN}Comparing {len(selected_paths)} experiments...{C.RESET}")
        
        comparator = ExperimentComparator.from_paths(selected_paths)
        result = comparator.compare(mode=mode_name)
        output = comparator.format_table(result, format=output_format)
        
        print(f"\n{C.BOLD}{C.GREEN}Comparison Results:{C.RESET}")
        print(output)
        
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Comparison cancelled.{C.RESET}")
    except Exception as e:
        print(f"\n{C.RED}Error: {e}{C.RESET}")


def interactive_mode():
    """Run the interactive configuration wizard."""
    C = Colors
    
    print_banner()
    print(f"\n{C.DIM}Welcome to the Spatial Voting Simulator.{C.RESET}")
    print(f"{C.DIM}This wizard will guide you through configuring a simulation.{C.RESET}")
    
    # =========================================================================
    # Basic Parameters
    # =========================================================================
    print_section("Basic Parameters")
    
    n_profiles = get_user_input("Number of election profiles", 1000, cast_type=int)
    if n_profiles is None:
        return
    
    n_voters = get_user_input("Number of voters per profile", 25, cast_type=int)
    if n_voters is None:
        return
    
    n_candidates = get_user_input("Number of candidates", 3, cast_type=int)
    if n_candidates is None:
        return
    
    # =========================================================================
    # Geometry Selection
    # =========================================================================
    print_section("Spatial Geometry")
    
    geometry_items = [
        (name, desc) for name, desc in AVAILABLE_GEOMETRY_METHODS.items()
    ]
    geo_idx = select_from_list(geometry_items, "Select geometry method", show_description=True)
    if geo_idx is None:
        return
    geometry_method = list(AVAILABLE_GEOMETRY_METHODS.keys())[geo_idx]
    
    n_dim = get_user_input("Spatial dimensions", 2, cast_type=int)
    if n_dim is None:
        return
    
    # =========================================================================
    # Voting Rules Selection
    # =========================================================================
    print_section("Voting Rules")
    
    print(f"\n{C.YELLOW}Available voting rules:{C.RESET}")
    
    ordinal_rules = [name for name, info in AVAILABLE_VOTING_RULES.items() 
                     if info['type'] == 'ordinal']
    cardinal_rules = [name for name, info in AVAILABLE_VOTING_RULES.items() 
                      if info['type'] == 'cardinal']
    
    print(f"\n{C.CYAN}Ordinal (ranking-based):{C.RESET}")
    for i, rule in enumerate(ordinal_rules, 1):
        desc = AVAILABLE_VOTING_RULES[rule]['description']
        print(f"  {i:2d}. {rule:<15} {C.DIM}{desc}{C.RESET}")
    
    print(f"\n{C.CYAN}Cardinal (utility-based):{C.RESET}")
    for i, rule in enumerate(cardinal_rules, len(ordinal_rules) + 1):
        desc = AVAILABLE_VOTING_RULES[rule]['description']
        print(f"  {i:2d}. {rule:<15} {C.DIM}{desc}{C.RESET}")
    
    all_rules = ordinal_rules + cardinal_rules
    
    print(f"\n{C.DIM}Enter rule numbers separated by commas, or 'all' for all rules.{C.RESET}")
    rules_input = get_user_input("Select voting rules", "1,2,3,4")
    if rules_input is None:
        return
    
    if rules_input.lower() == 'all':
        selected_rules = all_rules
    else:
        try:
            indices = [int(x.strip()) - 1 for x in rules_input.split(',')]
            selected_rules = [all_rules[i] for i in indices if 0 <= i < len(all_rules)]
        except (ValueError, IndexError):
            print(f"{C.RED}Invalid selection, using defaults.{C.RESET}")
            selected_rules = ['plurality', 'borda', 'irv']
    
    # =========================================================================
    # Manipulation Configuration
    # =========================================================================
    print_section("Strategic Manipulation")
    
    enable_manipulation = get_user_input(
        "Enable strategic manipulation? (y/n)", "n",
        valid_options=['y', 'n', 'Y', 'N', 'yes', 'no']
    )
    if enable_manipulation is None:
        return
    enable_manipulation = enable_manipulation.lower() in ['y', 'yes']
    
    manipulation_config = ManipulationConfig(enabled=False)
    
    if enable_manipulation:
        manip_fraction = get_user_input(
            "Fraction of manipulating voters (0.0-1.0)", 0.2, cast_type=float
        )
        if manip_fraction is None:
            return
        
        strategy_items = list(AVAILABLE_MANIPULATION_STRATEGIES.items())
        strat_idx = select_from_list(
            [(k, v) for k, v in strategy_items],
            "Select manipulation strategy",
            show_description=True
        )
        if strat_idx is None:
            return
        strategy = list(AVAILABLE_MANIPULATION_STRATEGIES.keys())[strat_idx]
        
        manipulation_config = ManipulationConfig(
            enabled=True,
            manipulator_fraction=manip_fraction,
            strategy=strategy
        )
    
    # =========================================================================
    # Advanced Options
    # =========================================================================
    print_section("Advanced Options")
    
    show_advanced = get_user_input(
        "Configure advanced options? (y/n)", "n",
        valid_options=['y', 'n', 'Y', 'N']
    )
    
    utility_func = 'gaussian'
    distance_metric = 'l2'
    sigma_factor = 0.5
    rng_seed = None
    
    if show_advanced and show_advanced.lower() == 'y':
        utility_options = ['gaussian', 'quadratic', 'linear', 'exponential']
        util_idx = select_from_list(utility_options, "Select utility function")
        if util_idx is not None:
            utility_func = utility_options[util_idx]
        
        distance_options = ['l2', 'l1', 'cosine', 'chebyshev']
        dist_idx = select_from_list(distance_options, "Select distance metric")
        if dist_idx is not None:
            distance_metric = distance_options[dist_idx]
        
        sigma_factor = get_user_input("Sigma factor (for Gaussian)", 0.5, cast_type=float)
        if sigma_factor is None:
            sigma_factor = 0.5
        
        seed_input = get_user_input("Random seed (empty for random)", "")
        if seed_input:
            try:
                rng_seed = int(seed_input)
            except ValueError:
                pass
    
    # =========================================================================
    # Confirmation
    # =========================================================================
    print_section("Configuration Summary")
    
    print(f"  {C.DIM}Profiles:{C.RESET} {n_profiles:,}")
    print(f"  {C.DIM}Voters:{C.RESET} {n_voters}")
    print(f"  {C.DIM}Candidates:{C.RESET} {n_candidates}")
    print(f"  {C.DIM}Geometry:{C.RESET} {geometry_method} ({n_dim}D)")
    print(f"  {C.DIM}Rules:{C.RESET} {', '.join(selected_rules)}")
    print(f"  {C.DIM}Manipulation:{C.RESET} {'Enabled' if enable_manipulation else 'Disabled'}")
    print(f"  {C.DIM}Utility:{C.RESET} {utility_func}")
    print(f"  {C.DIM}Distance:{C.RESET} {distance_metric}")
    
    confirm = get_user_input("\nProceed with simulation? (y/n)", "y")
    if confirm is None or confirm.lower() != 'y':
        print(f"{C.YELLOW}Simulation cancelled.{C.RESET}")
        return
    
    # =========================================================================
    # Run Simulation
    # =========================================================================
    print(f"\n{C.BOLD}{C.GREEN}Starting simulation...{C.RESET}")
    
    # Build configuration
    config = SimulationConfig(
        n_profiles=n_profiles,
        n_voters=n_voters,
        n_candidates=n_candidates,
        voting_rules=selected_rules,
        geometry=GeometryConfig(method=geometry_method, n_dim=n_dim),
        manipulation=manipulation_config,
        utility=UtilityConfig(
            function=utility_func,
            distance_metric=distance_metric,
            sigma_factor=sigma_factor
        ),
        rng_seed=rng_seed,
    )
    
    # Run experiment
    try:
        result = run_experiment(config, save_results=True, verbose=True)
        
        # Display results
        print_section("Results")
        
        headers = ["Rule", "Type", "VSE", "VSE σ", "1st%", "Cycles%", "CW Eff%"]
        rows = []
        
        for rule_name, rule_result in result.rule_results.items():
            agg = rule_result.aggregate_metrics
            rule_type = AVAILABLE_VOTING_RULES[rule_name]['type'][:4]
            
            rows.append([
                rule_name,
                rule_type,
                f"{agg.vse_mean:.3f}",
                f"{agg.vse_std:.3f}",
                f"{agg.winner_rank_1st_pct:.1f}%",
                f"{agg.cycle_percentage:.1f}%",
                f"{agg.condorcet_efficiency:.1f}%",
            ])
        
        format_table(headers, rows, "Voting Rule Comparison")
        
        print(f"\n{C.GREEN}Results saved:{C.RESET}")
        print(f"  {C.DIM}Inputs:{C.RESET} {result.inputs_path}")
        print(f"  {C.DIM}Results:{C.RESET} {result.results_path}")
        print(f"  {C.DIM}Total time:{C.RESET} {result.total_compute_time:.2f}s")
        
    except Exception as e:
        print(f"{C.RED}Error running simulation: {e}{C.RESET}")
        import traceback
        traceback.print_exc()


def list_saved_experiments():
    """List saved experiments."""
    C = Colors
    
    print_banner()
    print_section("Saved Experiments")
    
    experiments = list_experiments()
    
    if not experiments:
        print(f"{C.YELLOW}No saved experiments found.{C.RESET}")
        return
    
    headers = ["#", "Filename", "Profiles", "Voters", "ID"]
    rows = []
    
    for i, exp in enumerate(experiments, 1):
        rows.append([
            str(i),
            exp.get('filename', 'N/A')[:30],
            str(exp.get('n_profiles', 'N/A')),
            str(exp.get('n_voters', 'N/A')),
            exp.get('experiment_id', 'N/A')[:8],
        ])
    
    format_table(headers, rows, "Saved Experiments")


def run_quick_comparison():
    """Run a quick comparison of voting rules."""
    C = Colors
    
    print_banner()
    print(f"\n{C.DIM}Running quick comparison with default settings...{C.RESET}")
    
    n_profiles = get_user_input("Number of profiles", 1000, cast_type=int)
    if n_profiles is None:
        return
    
    n_voters = get_user_input("Number of voters", 25, cast_type=int)
    if n_voters is None:
        return
    
    result = quick_simulation(
        n_profiles=n_profiles,
        n_voters=n_voters,
        verbose=True
    )
    
    print(f"\n{C.GREEN}Quick comparison complete!{C.RESET}")


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Spatial Voting Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m simulator.cli                          # Interactive mode
  python -m simulator.cli -q                       # Quick comparison
  python -m simulator.cli -l                       # List saved experiments
  python -m simulator.cli -n 1000 -v 25 -r plurality borda irv
  python -m simulator.cli --geometry polarized --dim 3
  
  # Compare experiments
  python -m simulator.cli --compare exp1 exp2 exp3
  python -m simulator.cli --compare exp1 exp2 --compare-mode by_rule
  python -m simulator.cli --compare-interactive   # Interactive comparison mode
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('-i', '--interactive', action='store_true',
                            help='Interactive configuration mode (default)')
    mode_group.add_argument('-q', '--quick', action='store_true',
                            help='Quick comparison with defaults')
    mode_group.add_argument('-l', '--list', action='store_true',
                            help='List saved experiments')
    mode_group.add_argument('--compare', nargs='+', metavar='EXP',
                            help='Compare experiments by ID or filename')
    mode_group.add_argument('--compare-interactive', action='store_true',
                            help='Interactive experiment comparison mode')
    
    # Simulation parameters
    parser.add_argument('-n', '--profiles', type=int, default=1000,
                        help='Number of profiles (default: 1000)')
    parser.add_argument('-v', '--voters', type=int, default=25,
                        help='Number of voters (default: 25)')
    parser.add_argument('-c', '--candidates', type=int, default=3,
                        help='Number of candidates (default: 3)')
    
    # Voting rules
    parser.add_argument('-r', '--rules', nargs='+', default=None,
                        help='Voting rules to compare')
    
    # Geometry
    parser.add_argument('--geometry', type=str, default='uniform',
                        choices=list(AVAILABLE_GEOMETRY_METHODS.keys()),
                        help='Spatial geometry method')
    parser.add_argument('--dim', type=int, default=2,
                        help='Spatial dimensions')
    
    # Utility
    parser.add_argument('--utility', type=str, default='gaussian',
                        choices=['gaussian', 'quadratic', 'linear', 'exponential'],
                        help='Utility function')
    parser.add_argument('--distance', type=str, default='l2',
                        choices=['l2', 'l1', 'cosine', 'chebyshev'],
                        help='Distance metric')
    
    # Manipulation
    parser.add_argument('--manipulation', action='store_true',
                        help='Enable strategic manipulation')
    parser.add_argument('--manip-fraction', type=float, default=0.2,
                        help='Fraction of manipulators')
    parser.add_argument('--manip-strategy', type=str, default='compromise',
                        choices=list(AVAILABLE_MANIPULATION_STRATEGIES.keys()),
                        help='Manipulation strategy')
    
    # Output
    parser.add_argument('--no-save', action='store_true',
                        help='Do not save results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', action='store_true',
                        help='Verbose output')
    
    # Comparison options
    parser.add_argument('--compare-mode', type=str, default='by_experiment',
                        choices=['by_experiment', 'by_rule', 'by_metric', 'differential'],
                        help='Comparison mode (default: by_experiment)')
    parser.add_argument('--compare-format', type=str, default='text',
                        choices=['text', 'markdown', 'csv'],
                        help='Comparison output format (default: text)')
    
    args = parser.parse_args()
    
    # Dispatch to appropriate mode
    if args.list:
        list_saved_experiments()
    elif args.compare:
        compare_experiments_cli(
            experiment_ids=args.compare,
            mode=args.compare_mode,
            output_format=args.compare_format
        )
    elif args.compare_interactive:
        interactive_compare_mode()
    elif args.quick:
        run_quick_comparison()
    elif args.interactive or (not args.rules):
        interactive_mode()
    else:
        # Command-line mode with explicit parameters
        C = Colors
        print_banner()
        
        rules = args.rules or ['plurality', 'borda', 'irv', 'approval', 'star']
        
        # Validate rules
        valid_rules = set(AVAILABLE_VOTING_RULES.keys())
        for rule in rules:
            if rule not in valid_rules:
                print(f"{C.RED}Unknown rule: {rule}{C.RESET}")
                print(f"{C.YELLOW}Available: {', '.join(sorted(valid_rules))}{C.RESET}")
                return
        
        config = SimulationConfig(
            n_profiles=args.profiles,
            n_voters=args.voters,
            n_candidates=args.candidates,
            voting_rules=rules,
            geometry=GeometryConfig(method=args.geometry, n_dim=args.dim),
            manipulation=ManipulationConfig(
                enabled=args.manipulation,
                manipulator_fraction=args.manip_fraction,
                strategy=args.manip_strategy,
            ),
            utility=UtilityConfig(
                function=args.utility,
                distance_metric=args.distance,
            ),
            rng_seed=args.seed,
        )
        
        try:
            result = run_experiment(
                config,
                save_results=not args.no_save,
                verbose=args.verbose or True
            )
            
            # Print summary
            headers = ["Rule", "VSE", "1st%", "Cycles%"]
            rows = []
            
            for rule_name, rule_result in result.rule_results.items():
                agg = rule_result.aggregate_metrics
                rows.append([
                    rule_name,
                    f"{agg.vse_mean:.3f}",
                    f"{agg.winner_rank_1st_pct:.1f}%",
                    f"{agg.cycle_percentage:.1f}%",
                ])
            
            format_table(headers, rows, "Results Summary")
            
            if result.inputs_path:
                print(f"\n{C.GREEN}Saved to:{C.RESET}")
                print(f"  {result.inputs_path}")
                print(f"  {result.results_path}")
            
        except Exception as e:
            print(f"{C.RED}Error: {e}{C.RESET}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()


