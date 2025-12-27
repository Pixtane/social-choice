import numpy as np
import os
import argparse
from voting_rules import VOTING_RULES, _get_positions

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
# Profile Generation Methods
# =============================================================================

def gen_impartial_culture(n_voters, n_profiles):
    """
    Impartial Culture (IC): Each voter independently picks a uniformly random ranking.
    This is the standard model for Condorcet paradox analysis.
    Expected cycle rate: ~8.77% for large n.
    """
    profiles = np.empty((n_profiles, n_voters, 3), dtype=np.int8)
    for i in range(n_profiles):
        for j in range(n_voters):
            profiles[i, j] = np.random.permutation(3)
    return profiles

def gen_impartial_anonymous(n_voters, n_profiles):
    """
    Impartial Anonymous Culture (IAC): All anonymous profiles are equally likely.
    Distribution of ranking counts is uniformly distributed over the simplex.
    """
    profiles = np.empty((n_profiles, n_voters, 3), dtype=np.int8)
    rankings = np.array([[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], dtype=np.int8)
    
    for i in range(n_profiles):
        # Dirichlet distribution for anonymous profile
        weights = np.random.dirichlet(np.ones(6))
        counts = np.random.multinomial(n_voters, weights)
        
        idx = 0
        for r, count in enumerate(counts):
            for _ in range(count):
                profiles[i, idx] = rankings[r]
                idx += 1
        
        # Shuffle to avoid ordering artifacts
        np.random.shuffle(profiles[i])
    
    return profiles

def gen_single_peaked(n_voters, n_profiles):
    """
    Single-peaked preferences: Voters are on a 1D spectrum (A-B-C order).
    Each voter has a peak and preferences decrease away from it.
    Guarantees NO Condorcet cycles (0% cycle rate).
    """
    profiles = np.empty((n_profiles, n_voters, 3), dtype=np.int8)
    # Single-peaked rankings with order A < B < C
    # Peak A: A > B > C  -> [0, 1, 2]
    # Peak B: B > A > C or B > C > A -> [1, 0, 2] or [1, 2, 0]
    # Peak C: C > B > A  -> [2, 1, 0]
    single_peaked_rankings = np.array([
        [0, 1, 2],  # Peak A: A > B > C
        [1, 0, 2],  # Peak B (left-leaning): B > A > C
        [1, 2, 0],  # Peak B (right-leaning): B > C > A
        [2, 1, 0],  # Peak C: C > B > A
    ], dtype=np.int8)
    
    for i in range(n_profiles):
        for j in range(n_voters):
            profiles[i, j] = single_peaked_rankings[np.random.randint(4)]
    
    return profiles

def gen_polarized(n_voters, n_profiles):
    """
    Polarized electorate: Two opposing factions with a small middle.
    One faction strongly prefers A, the other strongly prefers C.
    Creates high tension but may have clear Condorcet winner.
    """
    profiles = np.empty((n_profiles, n_voters, 3), dtype=np.int8)
    
    for i in range(n_profiles):
        # Random faction sizes
        faction_a = int(n_voters * np.random.uniform(0.35, 0.50))
        faction_c = int(n_voters * np.random.uniform(0.35, 0.50))
        faction_b = n_voters - faction_a - faction_c
        
        idx = 0
        # Faction A: A > B > C or A > C > B
        for _ in range(faction_a):
            if np.random.random() < 0.7:
                profiles[i, idx] = [0, 1, 2]  # A > B > C
            else:
                profiles[i, idx] = [0, 2, 1]  # A > C > B
            idx += 1
        
        # Faction C: C > B > A or C > A > B
        for _ in range(faction_c):
            if np.random.random() < 0.7:
                profiles[i, idx] = [2, 1, 0]  # C > B > A
            else:
                profiles[i, idx] = [2, 0, 1]  # C > A > B
            idx += 1
        
        # Middle faction B: prefers B
        for _ in range(faction_b):
            if np.random.random() < 0.5:
                profiles[i, idx] = [1, 0, 2]  # B > A > C
            else:
                profiles[i, idx] = [1, 2, 0]  # B > C > A
            idx += 1
        
        np.random.shuffle(profiles[i])
    
    return profiles

def gen_spatial_1d(n_voters, n_profiles):
    """
    1D Spatial model: Voters and candidates on a line.
    Candidates at positions 0.2 (A), 0.5 (B), 0.8 (C).
    Voters uniformly distributed, rank by distance.
    """
    profiles = np.empty((n_profiles, n_voters, 3), dtype=np.int8)
    candidate_pos = np.array([0.2, 0.5, 0.8])
    
    for i in range(n_profiles):
        voter_pos = np.random.uniform(0, 1, n_voters)
        for j in range(n_voters):
            distances = np.abs(candidate_pos - voter_pos[j])
            profiles[i, j] = np.argsort(distances)
    
    return profiles

def gen_spatial_2d(n_voters, n_profiles):
    """
    2D Spatial model: Voters and candidates in 2D space.
    Candidates form a triangle, voters uniformly distributed.
    """
    profiles = np.empty((n_profiles, n_voters, 3), dtype=np.int8)
    candidate_pos = np.array([[0.2, 0.3], [0.8, 0.3], [0.5, 0.8]])
    
    for i in range(n_profiles):
        voter_pos = np.random.uniform(0, 1, (n_voters, 2))
        for j in range(n_voters):
            distances = np.sqrt(np.sum((candidate_pos - voter_pos[j])**2, axis=1))
            profiles[i, j] = np.argsort(distances)
    
    return profiles

def gen_mallows(n_voters, n_profiles, phi=0.5):
    """
    Mallows model: Rankings cluster around a central ranking.
    phi=0 gives identical rankings, phi=1 gives uniform random.
    """
    profiles = np.empty((n_profiles, n_voters, 3), dtype=np.int8)
    central_ranking = np.array([0, 1, 2], dtype=np.int8)
    
    for i in range(n_profiles):
        # Random central ranking for this profile
        central = np.random.permutation(3)
        
        for j in range(n_voters):
            # Generate ranking based on Mallows model
            if np.random.random() < phi:
                # With probability phi, random perturbation
                profiles[i, j] = np.random.permutation(3)
            else:
                # Otherwise, close to central
                if np.random.random() < 0.7:
                    profiles[i, j] = central
                else:
                    # Swap adjacent elements
                    perm = central.copy()
                    swap_idx = np.random.randint(2)
                    perm[swap_idx], perm[swap_idx + 1] = perm[swap_idx + 1], perm[swap_idx]
                    profiles[i, j] = perm
    
    return profiles

# Registry of generation methods
GENERATION_METHODS = {
    'impartial_culture': (gen_impartial_culture, "Each voter picks uniformly random ranking"),
    'impartial_anonymous': (gen_impartial_anonymous, "Anonymous profiles equally likely"),
    'single_peaked': (gen_single_peaked, "Single-peaked prefs (no cycles)"),
    'polarized': (gen_polarized, "Two opposing factions"),
    'spatial_1d': (gen_spatial_1d, "1D spatial model"),
    'spatial_2d': (gen_spatial_2d, "2D spatial model"),
    'mallows': (gen_mallows, "Mallows model (clustered rankings)"),
}

# =============================================================================
# Core Functions
# =============================================================================

def compute_margins(profile):
    """Compute pairwise margins m_ab, m_bc, m_ca for a profile."""
    positions = _get_positions(profile)
    n = profile.shape[0]
    
    a_wins_ab = np.sum(positions[:, 0] < positions[:, 1])
    m_ab = (2 * a_wins_ab - n) / n
    
    b_wins_bc = np.sum(positions[:, 1] < positions[:, 2])
    m_bc = (2 * b_wins_bc - n) / n
    
    c_wins_ca = np.sum(positions[:, 2] < positions[:, 0])
    m_ca = (2 * c_wins_ca - n) / n
    
    return m_ab, m_bc, m_ca

def is_cycle(m_ab, m_bc, m_ca):
    """A cycle exists when all three margins have the same sign."""
    if m_ab == 0 or m_bc == 0 or m_ca == 0:
        return False
    return (m_ab > 0 and m_bc > 0 and m_ca > 0) or (m_ab < 0 and m_bc < 0 and m_ca < 0)

def classify_cycle_type(m_ab, m_bc, m_ca):
    """Classify cycle into Type 1 (A>B>C>A) or Type 2 (B>A>C>B)."""
    if m_ab > 0 and m_bc > 0 and m_ca > 0:
        return 1
    elif m_ab < 0 and m_bc < 0 and m_ca < 0:
        return 2
    return None

def rankings_to_utilities(profile):
    """Convert preference rankings to utilities (1st=1.0, 2nd=0.5, 3rd=0.0)."""
    positions = _get_positions(profile)
    utilities = np.zeros_like(positions, dtype=np.float64)
    
    for v in range(profile.shape[0]):
        for c in range(3):
            pos = positions[v, c]
            utilities[v, c] = 1.0 - pos * 0.5
    
    return utilities

def compute_vse(profile, winner_idx):
    """Compute Voter Satisfaction Efficiency (VSE)."""
    utilities = rankings_to_utilities(profile)
    u_elected = np.mean(utilities[:, winner_idx])
    avg_utilities = np.mean(utilities, axis=0)
    u_optimal = np.max(avg_utilities)
    u_random = np.mean(avg_utilities)
    
    if abs(u_optimal - u_random) < 1e-10:
        return 1.0 if abs(u_elected - u_optimal) < 1e-10 else 0.0
    
    return (u_elected - u_random) / (u_optimal - u_random)

def simulate_paradox(voting_rule, n_profiles, n_voters, gen_method='impartial_culture'):
    """Simulate voting profiles and calculate statistics."""
    C = Colors
    
    gen_func = GENERATION_METHODS.get(gen_method, GENERATION_METHODS['impartial_culture'])[0]
    profiles = gen_func(n_voters, n_profiles)
    rule_func = VOTING_RULES.get(voting_rule)
    
    cycle_count = 0
    type1_count = 0
    type2_count = 0
    vse_values = []
    
    for i in range(n_profiles):
        profile = profiles[i]
        m_ab, m_bc, m_ca = compute_margins(profile)
        
        if is_cycle(m_ab, m_bc, m_ca):
            cycle_count += 1
            cycle_type = classify_cycle_type(m_ab, m_bc, m_ca)
            if cycle_type == 1:
                type1_count += 1
            elif cycle_type == 2:
                type2_count += 1
        
        if rule_func:
            try:
                winner = rule_func(profile)
                winner_idx = {'A': 0, 'B': 1, 'C': 2}.get(winner)
                if winner_idx is not None:
                    vse_values.append(compute_vse(profile, winner_idx))
            except Exception:
                pass
    
    return {
        'total_profiles': n_profiles,
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
        'gen_method': gen_method
    }

def format_table(headers, rows, title=None):
    """Format data as a pretty table."""
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

def run_all_methods(voting_rule, n_profiles, n_voters):
    """Run simulation for all methods with a single voting rule."""
    C = Colors
    print(f"\n{C.BOLD}{C.GREEN}Running simulations for all methods with {voting_rule}...{C.RESET}")
    print(f"{C.DIM}Profiles: {n_profiles:,} × {n_voters} voters{C.RESET}")
    
    headers = ["Method", "Cycles (%)", "Type 1 (%)", "Type 2 (%)", "VSE Mean", "VSE Std"]
    rows = []
    
    for method_name in GENERATION_METHODS.keys():
        print(f"{C.DIM}  Simulating {method_name}...{C.RESET}", end="\r")
        results = simulate_paradox(voting_rule, n_profiles, n_voters, method_name)
        
        rows.append([
            method_name,
            f"{results['cycle_percentage']:.2f}%",
            f"{results['type1_percentage']:.2f}%",
            f"{results['type2_percentage']:.2f}%",
            f"{results['vse_mean']:.3f}",
            f"{results['vse_std']:.3f}"
        ])
    
    print(" " * 50, end="\r")  # Clear the status line
    format_table(headers, rows, f"Results for {voting_rule} (N={n_profiles:,}, Voters={n_voters})")

def run_all_rules(gen_method, n_profiles, n_voters):
    """Run simulation for all voting rules with a single generation method."""
    C = Colors
    print(f"\n{C.BOLD}{C.GREEN}Running simulations for all rules with {gen_method}...{C.RESET}")
    print(f"{C.DIM}Profiles: {n_profiles:,} × {n_voters} voters{C.RESET}")
    
    headers = ["Voting Rule", "Cycles (%)", "Type 1 (%)", "Type 2 (%)", "VSE Mean", "VSE Std"]
    rows = []
    
    for rule_name in VOTING_RULES.keys():
        print(f"{C.DIM}  Simulating {rule_name}...{C.RESET}", end="\r")
        results = simulate_paradox(rule_name, n_profiles, n_voters, gen_method)
        
        rows.append([
            rule_name,
            f"{results['cycle_percentage']:.2f}%",
            f"{results['type1_percentage']:.2f}%",
            f"{results['type2_percentage']:.2f}%",
            f"{results['vse_mean']:.3f}",
            f"{results['vse_std']:.3f}"
        ])
    
    print(" " * 50, end="\r")  # Clear the status line
    format_table(headers, rows, f"Results for {gen_method} (N={n_profiles:,}, Voters={n_voters})")

def run_all_methods_all_rules(n_profiles, n_voters, by_method=False):
    """Run simulations for all methods and all rules."""
    C = Colors
    print(f"\n{C.BOLD}{C.GREEN}Running comprehensive simulations...{C.RESET}")
    print(f"{C.DIM}Profiles: {n_profiles:,} × {n_voters} voters{C.RESET}")
    print(f"{C.DIM}Total simulations: {len(VOTING_RULES) * len(GENERATION_METHODS)}{C.RESET}")
    
    headers = ["Method/Rule", "Cycles (%)", "Type 1 (%)", "Type 2 (%)", "VSE Mean", "VSE Std"]
    
    if by_method:
        # Group by methods, compare rules within each method
        for method_name in GENERATION_METHODS.keys():
            rows = []
            for rule_name in VOTING_RULES.keys():
                print(f"{C.DIM}  {method_name} × {rule_name}...{C.RESET}", end="\r")
                results = simulate_paradox(rule_name, n_profiles, n_voters, method_name)
                
                rows.append([
                    rule_name,
                    f"{results['cycle_percentage']:.2f}%",
                    f"{results['type1_percentage']:.2f}%",
                    f"{results['type2_percentage']:.2f}%",
                    f"{results['vse_mean']:.3f}",
                    f"{results['vse_std']:.3f}"
                ])
            
            print(" " * 50, end="\r")
            format_table(headers, rows, f"{method_name} - Comparing Voting Rules (N={n_profiles:,}, Voters={n_voters})")
            print()
    else:
        # Group by rules, compare methods within each rule
        for rule_name in VOTING_RULES.keys():
            rows = []
            for method_name in GENERATION_METHODS.keys():
                print(f"{C.DIM}  {rule_name} × {method_name}...{C.RESET}", end="\r")
                results = simulate_paradox(rule_name, n_profiles, n_voters, method_name)
                
                rows.append([
                    method_name,
                    f"{results['cycle_percentage']:.2f}%",
                    f"{results['type1_percentage']:.2f}%",
                    f"{results['type2_percentage']:.2f}%",
                    f"{results['vse_mean']:.3f}",
                    f"{results['vse_std']:.3f}"
                ])
            
            print(" " * 50, end="\r")
            format_table(headers, rows, f"{rule_name} - Comparing Methods (N={n_profiles:,}, Voters={n_voters})")
            print()

def interactive_mode():
    """Interactive simulation of Condorcet paradox."""
    C = Colors
    
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}       CONDORCET PARADOX SIMULATOR{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")
    
    # Select voting rule
    print(f"\n{C.YELLOW}Available voting rules:{C.RESET}")
    rules = list(VOTING_RULES.keys())
    for i, rule in enumerate(rules, 1):
        print(f"  {C.GREEN}{i:2d}.{C.RESET} {rule}")
    
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
        print(f"  {C.GREEN}{i}.{C.RESET} {method:<20} {C.DIM}- {desc}{C.RESET}")
    
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
    
    # Get number of profiles
    while True:
        try:
            n_profiles = int(input(f"\n{C.CYAN}Number of profiles to simulate: {C.RESET}"))
            if n_profiles > 0:
                break
            print(f"{C.RED}Please enter a positive number.{C.RESET}")
        except (ValueError, KeyboardInterrupt):
            print(f"\n{C.YELLOW}Exiting...{C.RESET}")
            return
    
    # Get number of voters
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
    print(f"{C.DIM}  Profiles: {n_profiles:,} × {n_voters} voters{C.RESET}")
    
    results = simulate_paradox(voting_rule, n_profiles, n_voters, gen_method)
    
    # Display results
    print(f"\n{C.BOLD}{C.GREEN}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}  RESULTS{C.RESET}")
    print(f"{C.BOLD}{C.GREEN}{'=' * 60}{C.RESET}")
    
    print(f"\n{C.BOLD}Configuration:{C.RESET}")
    print(f"  {C.DIM}Rule:{C.RESET} {voting_rule}")
    print(f"  {C.DIM}Method:{C.RESET} {gen_method}")
    print(f"  {C.DIM}Profiles:{C.RESET} {results['total_profiles']:,}")
    
    cycle_pct = results['cycle_percentage']
    cycle_color = C.RED if cycle_pct > 10 else C.YELLOW if cycle_pct > 5 else C.GREEN
    print(f"\n{C.BOLD}Condorcet Cycles:{C.RESET} {cycle_color}{results['cycle_count']:,} ({cycle_pct:.2f}%){C.RESET}")
    
    print(f"\n{C.BOLD}{C.BLUE}Cycle Type Breakdown:{C.RESET}")
    print(f"  {C.CYAN}Type 1{C.RESET} (A>B>C>A): {results['type1_count']:,} ({results['type1_percentage']:.2f}%)")
    print(f"  {C.CYAN}Type 2{C.RESET} (B>A>C>B): {results['type2_count']:,} ({results['type2_percentage']:.2f}%)")
    
    if results['vse_count'] > 0:
        print(f"\n{C.BOLD}{C.BLUE}VSE (Voter Satisfaction Efficiency):{C.RESET}")
        vse_color = C.GREEN if results['vse_mean'] > 0.8 else C.YELLOW if results['vse_mean'] > 0.5 else C.RED
        print(f"  {C.BOLD}Mean:{C.RESET} {vse_color}{results['vse_mean']:.3f}{C.RESET}")
        print(f"  {C.DIM}Std:  {results['vse_std']:.3f}  |  Min: {results['vse_min']:.3f}  |  Max: {results['vse_max']:.3f}{C.RESET}")
    
    print(f"\n{C.BOLD}{C.GREEN}{'=' * 60}{C.RESET}\n")

def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Condorcet Paradox Simulator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python paradox_simulator.py                    # Interactive mode
  python paradox_simulator.py -m -n 1000 -v 25   # Compare all methods
  python paradox_simulator.py -r -n 1000 -v 25   # Compare all rules
  python paradox_simulator.py -m -r -n 1000 -v 25       # Compare methods for each rule
  python paradox_simulator.py -m -r -b -n 1000 -v 25    # Compare rules for each method
        """
    )
    
    parser.add_argument('-m', '--methods', action='store_true',
                        help='Simulate through all generation methods')
    parser.add_argument('-r', '--rules', action='store_true',
                        help='Simulate through all voting rules')
    parser.add_argument('-b', '--by-method', action='store_true',
                        help='When both -m and -r are given, group by method instead of by rule')
    parser.add_argument('-n', '--profiles', type=int, default=None,
                        help='Number of profiles to simulate')
    parser.add_argument('-v', '--voters', type=int, default=None,
                        help='Number of voters per profile')
    parser.add_argument('--rule', type=str, default=None,
                        help='Voting rule to use (when using -m flag)')
    parser.add_argument('--method', type=str, default=None,
                        help='Generation method to use (when using -r flag)')
    
    args = parser.parse_args()
    
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
    print(f"{Colors.BOLD}{Colors.CYAN}       CONDORCET PARADOX SIMULATOR{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'=' * 60}{Colors.RESET}")
    
    # Handle different flag combinations
    if args.methods and args.rules:
        # Both flags: run all methods and all rules
        run_all_methods_all_rules(n_profiles, n_voters, by_method=args.by_method)
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
        
        run_all_methods(voting_rule, n_profiles, n_voters)
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
                print(f"  {Colors.GREEN}{i}.{Colors.RESET} {method:<20} {Colors.DIM}- {desc}{Colors.RESET}")
            
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
        
        run_all_rules(gen_method, n_profiles, n_voters)

if __name__ == "__main__":
    main()
