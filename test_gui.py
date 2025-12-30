"""
Test script for the GUI implementation.

Tests basic functionality without launching the full GUI.
Useful for verifying the installation and imports.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    
    try:
        import numpy
        print("  ✓ NumPy imported successfully")
    except ImportError as e:
        print(f"  ✗ NumPy import failed: {e}")
        return False
    
    try:
        import textual
        print("  ✓ Textual imported successfully")
    except ImportError as e:
        print(f"  ✗ Textual import failed: {e}")
        print("    Install with: pip install textual")
        return False
    
    try:
        import rich
        print("  ✓ Rich imported successfully")
    except ImportError as e:
        print(f"  ✗ Rich import failed: {e}")
        print("    Install with: pip install rich")
        return False
    
    try:
        from simulator import gui
        print("  ✓ GUI module imported successfully")
    except ImportError as e:
        print(f"  ✗ GUI module import failed: {e}")
        return False
    
    return True


def test_config_creation():
    """Test that configuration objects can be created."""
    print("\nTesting configuration creation...")
    
    try:
        from simulator import SimulationConfig, GeometryConfig, ManipulationConfig
        
        config = SimulationConfig(
            n_profiles=100,
            n_voters=25,
            n_candidates=3,
            voting_rules=['plurality', 'borda'],
            geometry=GeometryConfig(method='uniform', n_dim=2),
            manipulation=ManipulationConfig(enabled=False)
        )
        
        print("  ✓ SimulationConfig created successfully")
        print(f"    - Profiles: {config.n_profiles}")
        print(f"    - Voters: {config.n_voters}")
        print(f"    - Candidates: {config.n_candidates}")
        print(f"    - Rules: {', '.join(config.voting_rules)}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration creation failed: {e}")
        return False


def test_screen_classes():
    """Test that screen classes can be instantiated."""
    print("\nTesting screen classes...")
    
    try:
        from simulator.gui import (
            WelcomeScreen, ConfigurationScreen, QuickSimScreen,
            SimulationRunScreen, ResultsScreen, SavedExperimentsScreen,
            VotingSimulatorApp
        )
        from simulator import SimulationConfig
        
        # Test screen instantiation (without mounting)
        welcome = WelcomeScreen()
        print("  ✓ WelcomeScreen instantiated")
        
        config_screen = ConfigurationScreen()
        print("  ✓ ConfigurationScreen instantiated")
        
        quick = QuickSimScreen()
        print("  ✓ QuickSimScreen instantiated")
        
        # Test with config
        config = SimulationConfig(
            n_profiles=10,
            n_voters=5,
            n_candidates=3,
            voting_rules=['plurality']
        )
        
        sim_screen = SimulationRunScreen(config)
        print("  ✓ SimulationRunScreen instantiated")
        
        saved = SavedExperimentsScreen()
        print("  ✓ SavedExperimentsScreen instantiated")
        
        # Test app
        app = VotingSimulatorApp()
        print("  ✓ VotingSimulatorApp instantiated")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Screen class test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_constants():
    """Test that configuration constants are accessible."""
    print("\nTesting configuration constants...")
    
    try:
        from simulator.config import (
            AVAILABLE_VOTING_RULES,
            AVAILABLE_GEOMETRY_METHODS,
            AVAILABLE_MANIPULATION_STRATEGIES
        )
        
        print(f"  ✓ Found {len(AVAILABLE_VOTING_RULES)} voting rules")
        print(f"  ✓ Found {len(AVAILABLE_GEOMETRY_METHODS)} geometry methods")
        print(f"  ✓ Found {len(AVAILABLE_MANIPULATION_STRATEGIES)} manipulation strategies")
        
        # Show some examples
        print("\n  Sample voting rules:")
        for i, (name, info) in enumerate(list(AVAILABLE_VOTING_RULES.items())[:5]):
            print(f"    - {name}: {info['description']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Configuration constants test failed: {e}")
        return False


def test_simulator_integration():
    """Test that GUI can access simulator functions."""
    print("\nTesting simulator integration...")
    
    try:
        from simulator import run_experiment, SimulationConfig, GeometryConfig
        
        # Create minimal config
        config = SimulationConfig(
            n_profiles=5,  # Very small for quick test
            n_voters=10,
            n_candidates=3,
            voting_rules=['plurality', 'borda'],
            geometry=GeometryConfig(method='uniform', n_dim=2)
        )
        
        print("  Running tiny simulation (5 profiles)...")
        result = run_experiment(config, save_results=False, verbose=False)
        
        print(f"  ✓ Simulation completed in {result.total_compute_time:.3f}s")
        print(f"  ✓ Generated {len(result.rule_results)} rule results")
        
        for rule_name, rule_result in result.rule_results.items():
            vse = rule_result.aggregate_metrics.vse_mean
            print(f"    - {rule_name}: VSE = {vse:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Simulator integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_file_structure():
    """Test that required files exist."""
    print("\nTesting file structure...")
    
    required_files = [
        'simulator/gui.py',
        'simulator/__init__.py',
        'simulator/cli.py',
        'simulator/main.py',
        'simulator/config.py',
        'run_gui.py',
        'demo_gui.py',
        'requirements.txt',
        'GUI_README.md',
        'QUICKSTART_GUI.md',
        'GUI_FEATURES.md',
        'GUI_SUMMARY.md',
        'GUI_INDEX.md',
    ]
    
    all_exist = True
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            all_exist = False
    
    return all_exist


def test_run_gui_function():
    """Test that run_gui function is accessible."""
    print("\nTesting run_gui function...")
    
    try:
        from simulator import run_gui
        print("  ✓ run_gui function imported successfully")
        print("  Note: Not calling it (would launch GUI)")
        return True
    except ImportError as e:
        print(f"  ✗ run_gui import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("GUI IMPLEMENTATION TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration Creation", test_config_creation),
        ("Screen Classes", test_screen_classes),
        ("Configuration Constants", test_config_constants),
        ("Simulator Integration", test_simulator_integration),
        ("File Structure", test_file_structure),
        ("Run GUI Function", test_run_gui_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'=' * 60}")
        print(f"TEST: {test_name}")
        print('=' * 60)
        
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n✨ All tests passed! GUI is ready to use.")
        print("\nTo launch the GUI, run:")
        print("  python run_gui.py")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


