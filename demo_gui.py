"""
Demo script showing different ways to launch the GUI.

This demonstrates both direct launch and programmatic configuration.
"""

from simulator.gui import VotingSimulatorApp, SimulationRunScreen
from simulator import SimulationConfig, GeometryConfig, ManipulationConfig


def demo_basic_launch():
    """
    Demo 1: Basic launch - starts at welcome screen.
    User can navigate through menus normally.
    """
    print("Demo 1: Basic GUI Launch")
    print("=" * 50)
    app = VotingSimulatorApp()
    app.run()


def demo_quick_start():
    """
    Demo 2: Quick start with pre-configured simulation.
    Jumps directly to running a simulation.
    """
    print("Demo 2: Quick Start with Config")
    print("=" * 50)
    
    # Create a quick configuration
    config = SimulationConfig(
        n_profiles=1000,
        n_voters=25,
        n_candidates=3,
        voting_rules=['plurality', 'borda', 'irv', 'approval'],
        geometry=GeometryConfig(method='uniform', n_dim=2),
    )
    
    # Launch directly into simulation
    app = VotingSimulatorApp()
    app.push_screen(SimulationRunScreen(config))
    app.run()


def demo_polarized_scenario():
    """
    Demo 3: Polarized electorate scenario.
    Tests voting rules under political polarization.
    """
    print("Demo 3: Polarized Electorate Analysis")
    print("=" * 50)
    
    config = SimulationConfig(
        n_profiles=2000,
        n_voters=51,
        n_candidates=3,
        voting_rules=['plurality', 'approval', 'star', 'irv', 'schulze'],
        geometry=GeometryConfig(method='polarized', n_dim=2),
    )
    
    app = VotingSimulatorApp()
    app.push_screen(SimulationRunScreen(config))
    app.run()


def demo_strategic_voting():
    """
    Demo 4: Strategic voting analysis.
    Enables manipulation to see impact on different rules.
    """
    print("Demo 4: Strategic Voting Impact")
    print("=" * 50)
    
    config = SimulationConfig(
        n_profiles=1000,
        n_voters=25,
        n_candidates=3,
        voting_rules=['plurality', 'borda', 'irv', 'approval', 'star'],
        geometry=GeometryConfig(method='uniform', n_dim=2),
        manipulation=ManipulationConfig(
            enabled=True,
            manipulator_fraction=0.2,
            strategy='compromise',
            information_level='polls'
        )
    )
    
    app = VotingSimulatorApp()
    app.push_screen(SimulationRunScreen(config))
    app.run()


def demo_condorcet_methods():
    """
    Demo 5: Compare Condorcet methods.
    Tests different Condorcet completion methods.
    """
    print("Demo 5: Condorcet Methods Comparison")
    print("=" * 50)
    
    config = SimulationConfig(
        n_profiles=5000,
        n_voters=50,
        n_candidates=4,
        voting_rules=['minimax', 'copeland', 'schulze', 'ranked_pairs', 'kemeny_young'],
        geometry=GeometryConfig(method='uniform', n_dim=2),
    )
    
    app = VotingSimulatorApp()
    app.push_screen(SimulationRunScreen(config))
    app.run()


def demo_single_peaked():
    """
    Demo 6: Single-peaked preferences (1D left-right).
    Classic political science scenario.
    """
    print("Demo 6: Single-Peaked Preferences")
    print("=" * 50)
    
    config = SimulationConfig(
        n_profiles=1000,
        n_voters=101,  # Odd number for clear medians
        n_candidates=3,
        voting_rules=['plurality', 'borda', 'irv', 'approval', 'median'],
        geometry=GeometryConfig(method='single_peaked', n_dim=1),
    )
    
    app = VotingSimulatorApp()
    app.push_screen(SimulationRunScreen(config))
    app.run()


if __name__ == "__main__":
    import sys
    
    demos = {
        '1': ('Basic Launch', demo_basic_launch),
        '2': ('Quick Start', demo_quick_start),
        '3': ('Polarized Electorate', demo_polarized_scenario),
        '4': ('Strategic Voting', demo_strategic_voting),
        '5': ('Condorcet Methods', demo_condorcet_methods),
        '6': ('Single-Peaked', demo_single_peaked),
    }
    
    if len(sys.argv) > 1 and sys.argv[1] in demos:
        demo_name, demo_func = demos[sys.argv[1]]
        print(f"\nRunning: {demo_name}\n")
        demo_func()
    else:
        print("Spatial Voting Simulator - GUI Demos")
        print("=" * 50)
        print("\nAvailable demos:")
        for key, (name, _) in demos.items():
            print(f"  {key}. {name}")
        print("\nUsage:")
        print(f"  python demo_gui.py [1-6]")
        print("\nExamples:")
        print(f"  python demo_gui.py 1    # Basic launch")
        print(f"  python demo_gui.py 4    # Strategic voting demo")
        print(f"\nOr run without arguments for this menu.")
        print(f"Then run: python demo_gui.py <number>")



