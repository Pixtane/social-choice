# Spatial Voting Simulator - GUI Application

A rich terminal-based GUI application built with [Textual](https://textual.textualize.io/) for running and analyzing spatial voting simulations.

## Features

### 🎨 Interactive Interface
- **Welcome Screen**: Main menu with quick access to all features
- **Configuration Screen**: Comprehensive simulation setup with real-time validation
- **Simulation Runner**: Live progress tracking with detailed logs
- **Results Viewer**: Multi-tab interface for exploring metrics and statistics
- **Experiment Browser**: View and load saved experiments

### ⚙️ Configuration Options

#### Basic Parameters
- Number of election profiles (1-100,000)
- Number of voters per profile (3-1,000)
- Number of candidates (2-20)

#### Spatial Geometry
- **Methods**: uniform, clustered, single_peaked, polarized, 1d, 2d
- **Dimensions**: 1-10D spatial representation
- Custom geometry parameters (phi, variance, etc.)

#### Voting Rules
Choose from 20+ voting rules including:
- **Ordinal**: Plurality, Borda, IRV, Condorcet methods (Minimax, Copeland, Schulze, Ranked Pairs, Kemeny-Young), Coombs, Bucklin, Nanson, Baldwin
- **Cardinal**: Approval, Score, STAR, Utilitarian, Median, Quadratic

#### Strategic Manipulation
- Enable/disable strategic voting
- Manipulator fraction (0.0-1.0)
- Strategies: bullet, compromise, burial, pushover, optimal
- Information levels: none, polls, full

#### Utility Functions
- **Functions**: Gaussian, Quadratic, Linear, Exponential
- **Distance Metrics**: L2 (Euclidean), L1 (Manhattan), Cosine, Chebyshev
- Configurable sigma factor for Gaussian utilities

### 📊 Results Display

#### Summary View
- Aggregate metrics table with VSE (Voter Satisfaction Efficiency)
- Condorcet efficiency and cycle statistics
- Winner rank distributions
- Compute time tracking

#### Detailed View
- Per-rule performance metrics
- Winner statistics breakdown
- Condorcet properties analysis
- Configuration review

#### Saved Experiments
- Browse previous simulations
- Load and compare results
- Quick experiment lookup

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

Required packages:
- `textual>=0.47.0` - TUI framework
- `rich>=13.0.0` - Rich text formatting
- `numpy>=1.20.0` - Numerical computations

### Quick Start
```bash
# Launch the GUI
python run_gui.py

# Or from Python
python -m simulator.gui

# Or programmatically
from simulator import run_gui
run_gui()
```

## Usage Guide

### Running a Simulation

1. **Launch the Application**
   ```bash
   python run_gui.py
   ```

2. **Configure Your Simulation**
   - Select "New Simulation" from the main menu
   - Set basic parameters (profiles, voters, candidates)
   - Choose spatial geometry method and dimensions
   - Select voting rules to compare (multi-select supported)
   - Optionally enable strategic manipulation
   - Configure utility function and distance metric
   - Set random seed for reproducibility (optional)

3. **Run the Simulation**
   - Press `Ctrl+S` or click "Run Simulation"
   - Monitor progress with real-time updates
   - View detailed logs of each step

4. **Analyze Results**
   - Browse aggregate metrics in table format
   - Switch between tabs for per-rule details
   - Review configuration settings
   - Results are automatically saved

### Quick Simulation

For rapid testing with sensible defaults:
1. Select "Quick Simulation" from main menu
2. Enter basic parameters only
3. Uses default voting rules: Plurality, Borda, IRV, Approval, STAR, Schulze

### Viewing Saved Experiments

1. Select "View Saved Experiments" from main menu
2. Browse through saved simulations
3. Load experiments to view details
4. Press `R` to refresh the list

## Keyboard Shortcuts

### Global
- `Q` - Quit application
- `D` - Toggle dark mode
- `Escape` - Go back/cancel

### Configuration Screen
- `Ctrl+S` - Run simulation
- `Tab` - Navigate between fields

### Results Screen
- `D` - Open detailed view

### Saved Experiments
- `L` - Load selected experiment
- `R` - Refresh experiment list

## Configuration Tips

### For Quick Comparisons
- Use 1,000-10,000 profiles
- Keep voters at 25-50
- 3 candidates for classic scenarios
- 2D uniform geometry is fast and interpretable

### For Research-Grade Results
- Use 10,000+ profiles for statistical significance
- Vary voters (25, 50, 100) to test scalability
- Test with 3-5 candidates
- Try different geometries: uniform, polarized, single_peaked

### For Strategic Voting Analysis
- Enable manipulation with 20% fraction
- Compare results with/without manipulation
- Test different strategies (compromise, burial, bullet)
- Use full information level for theoretical analysis

### For Performance Testing
- Start small (100 profiles) to test configuration
- Scale up gradually
- Complex rules (Kemeny-Young, optimal manipulation) are computationally intensive
- Use random seed for reproducible benchmarks

## Output Files

Simulations automatically save to:
- **Inputs**: `simulator/inputs/YYYY-MM-DD_HH-MM-SS_<id>.npz`
  - Voter positions
  - Candidate positions
  - Utilities and rankings
  - Configuration metadata

- **Results**: `simulator/results/YYYY-MM-DD_HH-MM-SS_<id>.csv`
  - Per-profile metrics for each voting rule
  - Winner selections
  - VSE calculations
  - Condorcet properties
  - Manipulation impact (if enabled)

## Architecture

### Screen Flow
```
WelcomeScreen
    ├── ConfigurationScreen → SimulationRunScreen → ResultsScreen → DetailedResultsScreen
    ├── QuickSimScreen → SimulationRunScreen → ResultsScreen
    └── SavedExperimentsScreen
```

### Key Components

- **WelcomeScreen**: Main menu hub
- **ConfigurationScreen**: Two-column layout with basic and advanced settings
- **QuickSimScreen**: Modal dialog for quick setup
- **SimulationRunScreen**: Background worker with progress tracking
- **ResultsScreen**: DataTable with sortable columns
- **DetailedResultsScreen**: Tabbed interface with per-rule analysis
- **SavedExperimentsScreen**: File browser for experiments

### Threading Model
- Simulations run in background threads (via `@work` decorator)
- UI remains responsive during computation
- Progress updates via `call_from_thread`
- Proper cancellation support

## Troubleshooting

### Import Errors
```bash
# Ensure you're in the project root
cd social-choice
python run_gui.py

# Or use module syntax
python -m simulator.gui
```

### Terminal Size
Textual requires a minimum terminal size. If layout issues occur:
- Maximize your terminal window
- Use a terminal with at least 80 columns × 24 rows
- Try full-screen mode

### Performance Issues
- Reduce number of profiles for faster results
- Disable complex voting rules (Kemeny-Young)
- Use fewer dimensions in spatial model
- Disable manipulation for baseline comparisons

### Data Loading
- Ensure `simulator/inputs/` and `simulator/results/` directories exist
- Check file permissions for write access
- Verify NumPy version compatibility for .npz files

## Advanced Usage

### Custom Themes
Modify the CSS in `simulator/gui.py` to customize colors and layout:
```python
CSS = """
Screen {
    background: $background;
}
Button {
    background: $primary;
}
"""
```

### Programmatic Access
```python
from simulator import SimulationConfig, GeometryConfig
from simulator.gui import SimulationRunScreen, VotingSimulatorApp

# Create custom config
config = SimulationConfig(
    n_profiles=5000,
    n_voters=50,
    n_candidates=4,
    voting_rules=['borda', 'schulze', 'star'],
    geometry=GeometryConfig(method='polarized', n_dim=2)
)

# Launch GUI with config
app = VotingSimulatorApp()
app.push_screen(SimulationRunScreen(config))
app.run()
```

## Comparison with CLI

| Feature | GUI | CLI |
|---------|-----|-----|
| Ease of Use | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Visual Feedback | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Scripting | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Real-time Progress | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Batch Processing | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Results Exploration | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Use GUI for**: Interactive exploration, quick prototyping, visual analysis
**Use CLI for**: Automation, batch jobs, HPC clusters, scripting

## Contributing

The GUI is built with modularity in mind. To add features:

1. **New Screen**: Extend `Screen` or `ModalScreen`
2. **New Widget**: Use Textual's built-in widgets or create custom ones
3. **Background Tasks**: Use `@work` decorator for non-blocking operations
4. **CSS Styling**: Add styles in the `CSS` class variable

## Credits

- Built with [Textual](https://textual.textualize.io/) by Textualize
- Uses [Rich](https://rich.readthedocs.io/) for text formatting
- Powered by the Spatial Voting Simulator engine

## License

Same as the parent Spatial Voting Simulator project.



