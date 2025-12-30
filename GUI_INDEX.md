# GUI Documentation Index

Quick reference to all GUI-related documentation and files.

## 🚀 Quick Links

### Getting Started
- **[QUICKSTART_GUI.md](QUICKSTART_GUI.md)** - 5-minute quick start guide
  - Installation steps
  - Your first simulation
  - Understanding results
  - Common use cases

### Comprehensive Guide
- **[GUI_README.md](GUI_README.md)** - Complete user manual
  - Full feature documentation
  - Installation and setup
  - Usage guide with examples
  - Troubleshooting
  - Advanced usage

### Visual Guide
- **[GUI_FEATURES.md](GUI_FEATURES.md)** - Visual walkthrough
  - ASCII mockups of all screens
  - Feature descriptions
  - Workflow examples
  - UI/UX details

### Implementation Details
- **[GUI_SUMMARY.md](GUI_SUMMARY.md)** - Technical summary
  - What was implemented
  - Architecture overview
  - Code structure
  - Design principles

## 📁 File Reference

### Executable Files
- **`run_gui.py`** - Simple GUI launcher
- **`demo_gui.py`** - Pre-configured demo scenarios
- **`run_simulator.py`** - Unified launcher (CLI or GUI)

### Source Code
- **`simulator/gui.py`** - Main GUI implementation (~900 lines)
  - 7 complete screens
  - Background threading
  - Full feature set

### Configuration
- **`requirements.txt`** - Python dependencies
  - textual>=0.47.0
  - rich>=13.0.0
  - numpy>=1.20.0

## 📖 Documentation by Purpose

### For First-Time Users
1. Start here: [QUICKSTART_GUI.md](QUICKSTART_GUI.md)
2. Try running: `python run_gui.py`
3. Explore: [GUI_FEATURES.md](GUI_FEATURES.md)

### For Regular Users
1. Reference: [GUI_README.md](GUI_README.md)
2. Try demos: `python demo_gui.py 1`
3. Check keyboard shortcuts in any guide

### For Developers
1. Architecture: [GUI_SUMMARY.md](GUI_SUMMARY.md)
2. Source code: `simulator/gui.py`
3. Integration: See examples in [GUI_README.md](GUI_README.md)

### For Researchers
1. Configuration options: [GUI_README.md](GUI_README.md#configuration-options)
2. Output format: [GUI_README.md](GUI_README.md#output-files)
3. Reproducibility: Set random seed in config screen

## 🎯 Quick Command Reference

```bash
# Launch GUI
python run_gui.py

# Launch with menu (choose CLI or GUI)
python run_simulator.py

# Direct launch
python run_simulator.py --gui

# Module launch
python -m simulator.gui

# Demo scenarios
python demo_gui.py 1    # Basic launch
python demo_gui.py 2    # Quick start
python demo_gui.py 3    # Polarized electorate
python demo_gui.py 4    # Strategic voting
python demo_gui.py 5    # Condorcet methods
python demo_gui.py 6    # Single-peaked preferences
```

## 📚 Documentation Sections

### QUICKSTART_GUI.md
- Installation (2 commands)
- First simulation (2 methods)
- Understanding results
- Next steps
- Common use cases
- Tips for best results
- Troubleshooting
- Examples to try

### GUI_README.md
- Feature overview
- Installation prerequisites
- Usage guide
  - Running simulations
  - Quick simulation mode
  - Viewing saved experiments
- Keyboard shortcuts
- Configuration tips
- Output files
- Architecture
- Troubleshooting
- Advanced usage
- CLI comparison

### GUI_FEATURES.md
- Screen-by-screen visual guide
  - Welcome screen
  - Configuration screen
  - Simulation runner
  - Results display
  - Detailed results
  - Saved experiments
  - Quick simulation modal
- UI/UX features
- Workflow examples
- Customization options
- Data flow
- Advanced features
- Learning path
- Best practices

### GUI_SUMMARY.md
- Implementation overview
- Files created
- Features implemented
- Dependencies
- How to run (6 methods)
- Visual design
- Configuration options
- Output & results
- Use cases
- Code architecture
- Advanced features
- Known limitations
- Future enhancements

## 🗺️ Navigation Map

```
Documentation Structure
│
├── Quick Start
│   └── QUICKSTART_GUI.md (5 min read)
│
├── User Guides
│   ├── GUI_README.md (comprehensive)
│   └── GUI_FEATURES.md (visual)
│
└── Technical
    └── GUI_SUMMARY.md (implementation)

Code Structure
│
├── Launchers
│   ├── run_gui.py (simple)
│   ├── run_simulator.py (menu)
│   └── demo_gui.py (examples)
│
├── Implementation
│   └── simulator/gui.py (main code)
│
└── Configuration
    └── requirements.txt (dependencies)
```

## 🎓 Learning Path

### Beginner (30 minutes)
1. Read: QUICKSTART_GUI.md
2. Run: `python run_gui.py`
3. Try: Quick Simulation
4. Explore: Configuration screen

### Intermediate (2 hours)
1. Read: GUI_README.md
2. Try: All demos in demo_gui.py
3. Experiment: Different voting rules
4. Test: Strategic manipulation

### Advanced (1 day)
1. Read: GUI_SUMMARY.md
2. Study: simulator/gui.py source
3. Customize: Modify CSS/layout
4. Extend: Add new features

## 🔍 Find What You Need

### "How do I install?"
→ QUICKSTART_GUI.md or GUI_README.md (Installation section)

### "How do I run a simulation?"
→ QUICKSTART_GUI.md (Your First Simulation)

### "What do the results mean?"
→ QUICKSTART_GUI.md (Understanding Results)

### "What keyboard shortcuts exist?"
→ GUI_README.md (Keyboard Shortcuts)

### "How do I configure strategic voting?"
→ GUI_README.md (Configuration Tips > Strategic Voting)

### "What screens are available?"
→ GUI_FEATURES.md (complete visual guide)

### "How does it work internally?"
→ GUI_SUMMARY.md (Architecture section)

### "Can I use it programmatically?"
→ GUI_README.md (Advanced Usage) or GUI_SUMMARY.md (How to Run)

### "What voting rules are supported?"
→ GUI_README.md (Configuration Options) or GUI_SUMMARY.md

### "Where are results saved?"
→ GUI_README.md (Output Files)

### "How do I customize the appearance?"
→ GUI_FEATURES.md (Customization Options)

### "What examples can I try?"
→ QUICKSTART_GUI.md (Examples) or demo_gui.py

## 📊 Statistics

**Total Documentation**: ~10,000+ words
**Code Written**: ~900 lines
**Screens Implemented**: 7
**Demo Scenarios**: 6
**Supported Voting Rules**: 20+
**Configuration Parameters**: 20+
**Time to First Simulation**: <5 minutes

## 🎨 Visual Tour

For a complete visual tour with ASCII mockups of every screen, see:
- **[GUI_FEATURES.md](GUI_FEATURES.md)**

Screenshots show:
- Welcome screen layout
- Configuration interface (split panels)
- Simulation progress with logs
- Results table with metrics
- Detailed tabs for each rule
- Saved experiments browser
- Quick simulation modal

## 🛠️ Development Reference

### Adding New Features
1. Study: `simulator/gui.py` structure
2. Extend: Screen or Widget classes
3. Update: Documentation
4. Test: With demo_gui.py

### Modifying Styles
1. Find: CSS strings in gui.py
2. Edit: Colors, spacing, borders
3. Test: Launch and view changes
4. Document: In GUI_FEATURES.md

### Contributing
1. Follow: Existing code style
2. Document: New features
3. Test: All workflows
4. Update: Relevant documentation

## 🔗 External Resources

- **Textual Framework**: https://textual.textualize.io/
- **Rich Library**: https://rich.readthedocs.io/
- **NumPy**: https://numpy.org/doc/

## 📞 Getting Help

1. Check documentation:
   - Quick issue? → QUICKSTART_GUI.md
   - Detailed question? → GUI_README.md
   - Visual confusion? → GUI_FEATURES.md
   - Technical issue? → GUI_SUMMARY.md

2. Review examples:
   - Run: `python demo_gui.py`
   - Study: demo_gui.py source

3. Check source:
   - Read: simulator/gui.py
   - Understand: Implementation

## ✨ Feature Highlights

### What Makes This GUI Special?

1. **Comprehensive**: Access to all simulator features
2. **Intuitive**: No command-line knowledge needed
3. **Responsive**: Background threading keeps UI smooth
4. **Beautiful**: Modern terminal UI with colors
5. **Powerful**: 20+ voting rules, full configuration
6. **Fast**: Real-time progress and quick navigation
7. **Documented**: 4 comprehensive guides
8. **Demonstrated**: 6 ready-to-run examples

## 🎉 Get Started Now!

```bash
# 1. Install (if needed)
pip install textual rich numpy

# 2. Launch
python run_gui.py

# 3. Enjoy!
# Select "Quick Simulation" and press Run
# See results in ~10 seconds
```

## 📝 Documentation Checklist

- ✅ Quick start guide
- ✅ Comprehensive user manual
- ✅ Visual feature tour
- ✅ Technical implementation summary
- ✅ Demo examples
- ✅ Code comments
- ✅ Keyboard shortcuts
- ✅ Troubleshooting guide
- ✅ Use cases and examples
- ✅ This index document

Everything you need to use the GUI is documented! 🗳️✨

---

**Last Updated**: December 30, 2025
**Version**: 1.0.0
**Status**: Complete and ready to use


