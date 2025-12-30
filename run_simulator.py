# run_simulator.py (in the project root)
"""
Launcher for the Spatial Voting Simulator.

Supports both CLI and GUI modes.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def show_menu():
    """Display launcher menu."""
    print("\n" + "=" * 60)
    print("  SPATIAL VOTING SIMULATOR")
    print("=" * 60)
    print("\nChoose an interface:")
    print("  1. GUI (Textual) - Interactive graphical interface")
    print("  2. CLI - Command-line interface")
    print("  3. Exit")
    print("\n" + "-" * 60)
    
    while True:
        choice = input("Enter choice (1-3): ").strip()
        if choice in ['1', '2', '3']:
            return choice
        print("Invalid choice. Please enter 1, 2, or 3.")


def main():
    """Main launcher."""
    # Check for command-line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--gui', '-g', 'gui']:
            from simulator.gui import main as gui_main
            gui_main()
        elif sys.argv[1] in ['--cli', '-c', 'cli']:
            from simulator.cli import main as cli_main
            cli_main()
        elif sys.argv[1] in ['--help', '-h']:
            print("Spatial Voting Simulator")
            print("\nUsage:")
            print("  python run_simulator.py           # Interactive menu")
            print("  python run_simulator.py --gui     # Launch GUI directly")
            print("  python run_simulator.py --cli     # Launch CLI directly")
            print("\nOr use dedicated launchers:")
            print("  python run_gui.py                 # GUI")
            print("  python -m simulator.cli           # CLI")
        else:
            print(f"Unknown option: {sys.argv[1]}")
            print("Use --help for usage information")
    else:
        # Interactive menu
        choice = show_menu()
        
        if choice == '1':
            print("\nLaunching GUI...")
            from simulator.gui import main as gui_main
            gui_main()
        elif choice == '2':
            print("\nLaunching CLI...")
            from simulator.cli import main as cli_main
            cli_main()
        elif choice == '3':
            print("\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()