# 🗳️ Spatial Voting Simulator - GUI Edition

A beautiful, interactive terminal-based GUI for exploring and analyzing voting systems.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## ✨ What's New

A complete **Textual-based GUI** has been added to the Spatial Voting Simulator! Now you can:

- 🎨 **Configure simulations visually** with an intuitive interface
- 📊 **View results in real-time** with live progress tracking
- 🔍 **Explore metrics interactively** with tabbed detailed views
- 💾 **Browse saved experiments** with a built-in file browser
- ⚡ **Run quick comparisons** with one-click defaults
- 🎯 **Test strategic voting** with easy manipulation settings

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install textual rich numpy

# Launch the GUI
python run_gui.py
```

That's it! You're ready to explore voting systems.

### Your First Simulation (30 seconds)

1. **Launch**: `python run_gui.py`
2. **Select**: "⚡ Quick Simulation"
3. **Click**: "Run"
4. **View**: Results appear automatically!

See 6 voting rules compared in under a minute.

## 📚 Documentation

We've created comprehensive documentation to help you get started:

| Document | Purpose | Read Time |
|----------|---------|-----------|
| **[QUICKSTART_GUI.md](QUICKSTART_GUI.md)** | Get started in 5 minutes | 5 min |
| **[GUI_README.md](GUI_README.md)** | Complete user manual | 20 min |
| **[GUI_FEATURES.md](GUI_FEATURES.md)** | Visual tour with screenshots | 15 min |
| **[GUI_SUMMARY.md](GUI_SUMMARY.md)** | Technical implementation details | 10 min |
| **[GUI_INDEX.md](GUI_INDEX.md)** | Documentation index & navigation | 5 min |

**Start here**: [QUICKSTART_GUI.md](QUICKSTART_GUI.md) for a 5-minute introduction.

## 🎯 Key Features

### 🖥️ Seven Interactive Screens

1. **Welcome Screen** - Main menu hub
2. **Configuration Screen** - Full parameter control
3. **Quick Simulation** - One-click testing
4. **Simulation Runner** - Live progress tracking
5. **Results Viewer** - Sortable metrics table
6. **Detailed Analysis** - Per-rule deep dive
7. **Experiment Browser** - Saved results explorer

### ⚙️ Comprehensive Configuration

- **20+ Voting Rules**: Plurality, Borda, IRV, Approval, STAR, Schulze, and more
- **6 Geometry Methods**: Uniform, clustered, polarized, single-peaked, 1D, 2D
- **Strategic Manipulation**: Test 5 manipulation strategies
- **Utility Functions**: Gaussian, quadratic, linear, exponential
- **Full Customization**: Control every simulation parameter

### 📊 Rich Results Display

- **VSE (Voter Satisfaction Efficiency)** - Primary performance metric
- **Condorcet Properties** - Winner existence, efficiency, cycles
- **Winner Statistics** - Rank distributions
- **Comparative Analysis** - Side-by-side rule comparison
- **Detailed Metrics** - Per-rule breakdowns

### 🎨 Beautiful Interface

- Modern terminal UI with colors and formatting
- Keyboard-first navigation (mouse optional)
- Real-time progress bars and status updates
- Responsive layout that adapts to terminal size
- Dark mode optimized (toggle with `D`)

## 🎮 How to Use

### Method 1: Simple Launch
```bash
python run_gui.py
```

### Method 2: Interactive Menu
```bash
python run_simulator.py
# Choose: 1 (GUI) or 2 (CLI)
```

### Method 3: Direct Launch
```bash
python run_simulator.py --gui
```

### Method 4: Demo Scenarios
```bash
python demo_gui.py 1    # Basic launch
python demo_gui.py 3    # Polarized electorate
python demo_gui.py 4    # Strategic voting analysis
```

### Method 5: Programmatic
```python
from simulator import run_gui
run_gui()
```

## 📖 Example Workflows

### Quick Comparison
```
1. Launch GUI
2. Select "Quick Simulation"
3. Accept defaults (1000 profiles, 25 voters, 3 candidates)
4. Click "Run"
5. View results comparing 6 voting rules
```

### Custom Analysis
```
1. Launch GUI
2. Select "New Simulation"
3. Configure:
   - Profiles: 5000
   - Voters: 50
   - Candidates: 4
   - Geometry: polarized
   - Rules: plurality, approval, star, schulze
4. Run simulation
5. Explore detailed results in tabs
```

### Strategic Voting Study
```
1. Launch GUI
2. Select "New Simulation"
3. Enable manipulation:
   - Fraction: 0.2 (20% strategic voters)
   - Strategy: compromise
4. Compare with/without manipulation
5. Analyze impact on different rules
```

## 🎓 Learning Resources

### For Beginners
- Start with [QUICKSTART_GUI.md](QUICKSTART_GUI.md)
- Try "Quick Simulation" mode
- Explore different voting rules
- Read result explanations

### For Researchers
- Read [GUI_README.md](GUI_README.md) for full details
- Use custom configurations
- Enable strategic manipulation
- Export results for analysis

### For Developers
- Study [GUI_SUMMARY.md](GUI_SUMMARY.md)
- Examine `simulator/gui.py` source
- Run `python test_gui.py` to verify installation
- Extend with custom features

## 🔧 Technical Details

### Built With
- **[Textual](https://textual.textualize.io/)** - Modern TUI framework
- **[Rich](https://rich.readthedocs.io/)** - Beautiful terminal formatting
- **NumPy** - Numerical computations

### Architecture
- 7 screen classes with full navigation
- Background threading for non-blocking simulations
- Real-time progress updates via worker threads
- Automatic result saving (NPZ + CSV)
- Comprehensive error handling and validation

### Requirements
```
numpy>=1.20.0
textual>=0.47.0
rich>=13.0.0
```

## 🎯 Use Cases

### 1. **Education**
Demonstrate voting systems to students with visual, interactive examples.

### 2. **Research**
Configure controlled experiments to study voting system properties.

### 3. **Exploration**
Quickly test hypotheses about different voting rules and scenarios.

### 4. **Comparison**
Evaluate multiple voting rules side-by-side with identical conditions.

### 5. **Strategic Analysis**
Study manipulation resistance of different voting systems.

## ⌨️ Keyboard Shortcuts

### Global
- `Q` - Quit application
- `D` - Toggle dark mode
- `Escape` - Go back / Cancel

### Configuration Screen
- `Ctrl+S` - Run simulation
- `Tab` - Navigate fields

### Results Screen
- `D` - Open detailed view
- Arrow keys - Navigate table

### Saved Experiments
- `L` - Load selected
- `R` - Refresh list

## 📊 Output Files

All simulations automatically save to:

- **Inputs**: `simulator/inputs/YYYY-MM-DD_HH-MM-SS_<id>.npz`
  - Voter positions
  - Candidate positions
  - Utilities and rankings
  - Configuration metadata

- **Results**: `simulator/results/YYYY-MM-DD_HH-MM-SS_<id>.csv`
  - Per-profile metrics
  - Winner selections
  - VSE calculations
  - Condorcet properties

## 🧪 Testing

Verify your installation:

```bash
python test_gui.py
```

This runs a comprehensive test suite checking:
- Dependencies installed correctly
- GUI modules import successfully
- Screen classes instantiate properly
- Simulator integration works
- File structure is complete

## 🆚 GUI vs CLI

| Feature | GUI | CLI |
|---------|-----|-----|
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Visual Feedback** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Scripting** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Batch Jobs** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Real-time Progress** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Results Exploration** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Learning Curve** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

**Use GUI for**: Interactive exploration, learning, quick testing, visual analysis

**Use CLI for**: Automation, batch processing, scripting, HPC clusters

## 🎨 Screenshots (ASCII)

### Welcome Screen
```
┌────────────────────────────────────────┐
│  🗳️  SPATIAL VOTING SIMULATOR          │
│                                        │
│  [▶ New Simulation]                   │
│  [📊 View Saved Experiments]          │
│  [⚡ Quick Simulation]                 │
│  [❌ Exit]                             │
└────────────────────────────────────────┘
```

### Results Table
```
┌─────────────────────────────────────────────┐
│ Rule      │ VSE   │ 1st % │ Cycles │ CW Eff│
├───────────┼───────┼───────┼────────┼───────┤
│ plurality │ 0.792 │ 78.5  │ 12.3   │ 82.1  │
│ borda     │ 0.876 │ 89.2  │ 12.3   │ 94.5  │
│ irv       │ 0.854 │ 85.7  │ 12.3   │ 91.2  │
│ approval  │ 0.889 │ 91.4  │ 12.3   │ 96.8  │
│ star      │ 0.912 │ 94.1  │ 12.3   │ 98.2  │
└─────────────────────────────────────────────┘
```

See [GUI_FEATURES.md](GUI_FEATURES.md) for complete visual tour!

## 🚀 What's Included

### Code Files
- `simulator/gui.py` - Main GUI implementation (900+ lines)
- `run_gui.py` - Simple launcher
- `demo_gui.py` - 6 pre-configured demos
- `test_gui.py` - Test suite

### Documentation
- `QUICKSTART_GUI.md` - 5-minute quick start
- `GUI_README.md` - Complete user manual
- `GUI_FEATURES.md` - Visual feature tour
- `GUI_SUMMARY.md` - Technical details
- `GUI_INDEX.md` - Documentation index
- `README_GUI.md` - This file

### Configuration
- `requirements.txt` - Python dependencies
- `simulator/__init__.py` - Updated with `run_gui()`
- `run_simulator.py` - Enhanced with GUI option

## 🎉 Get Started Now!

```bash
# 1. Install dependencies (30 seconds)
pip install textual rich numpy

# 2. Launch GUI (instant)
python run_gui.py

# 3. Run your first simulation (30 seconds)
# - Select "Quick Simulation"
# - Click "Run"
# - View results!

# Total time: ~1 minute to first results! 🚀
```

## 💡 Pro Tips

1. **Start Small**: Test with 100 profiles first, then scale up
2. **Use Keyboard**: Tab to navigate, Ctrl+S to run, Escape to go back
3. **Save Configs**: Note successful configurations for later
4. **Compare Pairs**: Test 2-3 rules at a time for clearer insights
5. **Set Seeds**: Use random seed for reproducible experiments
6. **Read Docs**: Check [QUICKSTART_GUI.md](QUICKSTART_GUI.md) for tips

## 🐛 Troubleshooting

### "Module not found"
```bash
pip install textual rich numpy
```

### "Layout looks broken"
- Maximize terminal window (need 80x24 minimum)
- Try full-screen mode

### "Simulation is slow"
- Reduce number of profiles
- Use fewer voting rules
- Disable complex rules (Kemeny-Young)

### More help
- Check [GUI_README.md](GUI_README.md) troubleshooting section
- Run `python test_gui.py` to diagnose issues

## 🌟 Highlights

- ✅ **Complete Implementation** - All features working
- ✅ **Fully Documented** - 5 comprehensive guides
- ✅ **Tested** - Test suite included
- ✅ **Production Ready** - Error handling and validation
- ✅ **Easy to Use** - 5-minute learning curve
- ✅ **Powerful** - Access to all simulator features
- ✅ **Beautiful** - Modern terminal UI
- ✅ **Fast** - Background threading keeps UI responsive

## 📞 Next Steps

1. **Install**: `pip install textual rich numpy`
2. **Launch**: `python run_gui.py`
3. **Learn**: Read [QUICKSTART_GUI.md](QUICKSTART_GUI.md)
4. **Explore**: Try different configurations
5. **Analyze**: Compare voting systems
6. **Share**: Export results for presentations

## 🎓 Additional Resources

- **Voting Theory**: See `VOTING_PATTERNS.md` in project root
- **CLI Documentation**: Run `python -m simulator.cli --help`
- **Source Code**: Explore `simulator/` directory
- **Textual Docs**: https://textual.textualize.io/

## 🤝 Contributing

The GUI is built with modularity in mind:
- Screens are independent classes
- Easy to add new widgets
- CSS-based styling
- Well-documented code

Feel free to extend and customize!

## 📄 License

Same as the parent Spatial Voting Simulator project.

---

**Ready to explore voting systems?** 🗳️

```bash
python run_gui.py
```

**Enjoy!** ✨

---

*Built with [Textual](https://textual.textualize.io/) by Textualize*

*Part of the Spatial Voting Simulator project*

*Version 1.0.0 - December 2025*









