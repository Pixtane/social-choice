# GUI Installation & Verification Guide

Complete guide to installing and verifying the Spatial Voting Simulator GUI.

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Terminal**: Any modern terminal (Windows Terminal, PowerShell, bash, zsh, etc.)
- **Terminal Size**: Minimum 80 columns × 24 rows (recommended: maximize window)

## 🚀 Installation Steps

### Step 1: Install Dependencies

```bash
pip install textual rich numpy
```

**Expected output:**
```
Successfully installed textual-0.47.0 rich-13.7.0 numpy-1.24.0
```

### Step 2: Verify Installation

```bash
python test_gui.py
```

**Expected output:**
```
============================================================
GUI IMPLEMENTATION TEST SUITE
============================================================

============================================================
TEST: Imports
============================================================
Testing imports...
  ✓ NumPy imported successfully
  ✓ Textual imported successfully
  ✓ Rich imported successfully
  ✓ GUI module imported successfully

... (more tests) ...

============================================================
TEST SUMMARY
============================================================
✓ PASS   Imports
✓ PASS   Configuration Creation
✓ PASS   Screen Classes
✓ PASS   Configuration Constants
✓ PASS   Simulator Integration
✓ PASS   File Structure
✓ PASS   Run GUI Function

7/7 tests passed

✨ All tests passed! GUI is ready to use.

To launch the GUI, run:
  python run_gui.py
```

### Step 3: Launch GUI

```bash
python run_gui.py
```

You should see the welcome screen!

## ✅ Verification Checklist

Run through this checklist to ensure everything works:

### Basic Functionality
- [ ] GUI launches without errors
- [ ] Welcome screen displays correctly
- [ ] Can navigate with arrow keys
- [ ] Can press Q to quit
- [ ] Can toggle dark mode with D

### Configuration Screen
- [ ] "New Simulation" opens configuration screen
- [ ] Can enter numbers in input fields
- [ ] Can select from dropdown menus
- [ ] Can check/uncheck voting rules
- [ ] Can navigate with Tab key
- [ ] Escape goes back to welcome screen

### Quick Simulation
- [ ] "Quick Simulation" opens modal
- [ ] Can enter basic parameters
- [ ] "Run" starts simulation
- [ ] Progress bar shows progress
- [ ] Log displays output
- [ ] Results screen appears automatically

### Results Display
- [ ] Results table displays correctly
- [ ] Can navigate table with arrow keys
- [ ] Metrics are readable
- [ ] "Detailed View" button works
- [ ] Detailed view has multiple tabs
- [ ] Can switch between tabs

### File Operations
- [ ] Simulations save to `simulator/inputs/`
- [ ] Results save to `simulator/results/`
- [ ] "View Saved Experiments" shows files
- [ ] Can navigate experiment list

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'textual'"

**Solution:**
```bash
pip install textual
```

If that doesn't work:
```bash
python -m pip install textual
```

### Issue: "ModuleNotFoundError: No module named 'rich'"

**Solution:**
```bash
pip install rich
```

### Issue: "ModuleNotFoundError: No module named 'numpy'"

**Solution:**
```bash
pip install numpy
```

### Issue: "ModuleNotFoundError: No module named 'simulator'"

**Solution:**
Make sure you're running from the project root directory:
```bash
cd c:\Programming\fun\social-choice
python run_gui.py
```

### Issue: Layout looks broken or text is garbled

**Solutions:**
1. **Maximize terminal window** - Need at least 80×24 characters
2. **Use a modern terminal** - Windows Terminal, iTerm2, or similar
3. **Check terminal encoding** - Should be UTF-8
4. **Try different terminal** - Some terminals have better support

### Issue: Colors don't display

**Solutions:**
1. **Enable ANSI colors** in your terminal settings
2. **Use Windows Terminal** on Windows (better color support)
3. **Check terminal compatibility** - Most modern terminals support colors

### Issue: Simulation is very slow

**Solutions:**
1. **Reduce number of profiles** - Start with 100 instead of 1000
2. **Use fewer voting rules** - Select 2-3 rules instead of all
3. **Disable complex rules** - Kemeny-Young is computationally expensive
4. **Check CPU usage** - Make sure system isn't overloaded

### Issue: GUI freezes during simulation

**Solution:**
This shouldn't happen (simulations run in background threads), but if it does:
1. Press **Escape** to cancel
2. Try a smaller simulation first
3. Check `test_gui.py` passes all tests
4. Report the issue with details

### Issue: Can't see cursor or selection

**Solution:**
1. Use **arrow keys** to navigate
2. Press **Tab** to move between fields
3. Check terminal supports cursor display
4. Try toggling dark mode with **D**

### Issue: Results don't save

**Solution:**
1. Check `simulator/inputs/` and `simulator/results/` directories exist
2. Verify write permissions
3. Check disk space
4. Look for error messages in the log panel

## 🔍 Detailed Verification

### Test 1: Basic Import
```python
python -c "from simulator import gui; print('✓ GUI module loaded')"
```

Expected: `✓ GUI module loaded`

### Test 2: Configuration Creation
```python
python -c "from simulator import SimulationConfig; c = SimulationConfig(); print(f'✓ Config created: {c.n_profiles} profiles')"
```

Expected: `✓ Config created: 1000 profiles`

### Test 3: Run Tiny Simulation
```python
python -c "from simulator import run_experiment, SimulationConfig; result = run_experiment(SimulationConfig(n_profiles=5, n_voters=10, n_candidates=3, voting_rules=['plurality']), save_results=False, verbose=False); print(f'✓ Simulation completed in {result.total_compute_time:.2f}s')"
```

Expected: `✓ Simulation completed in 0.XX s`

### Test 4: Full Test Suite
```bash
python test_gui.py
```

Expected: All 7 tests pass

## 📦 Package Versions

### Minimum Versions
- `numpy>=1.20.0`
- `textual>=0.47.0`
- `rich>=13.0.0`

### Check Your Versions
```bash
pip show textual rich numpy
```

### Upgrade if Needed
```bash
pip install --upgrade textual rich numpy
```

## 🎯 Quick Verification Script

Save this as `verify_gui.py` and run it:

```python
#!/usr/bin/env python
"""Quick verification script."""

def verify():
    print("Verifying GUI installation...")
    
    # Test imports
    try:
        import numpy
        import textual
        import rich
        from simulator import gui
        print("✓ All imports successful")
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test config
    try:
        from simulator import SimulationConfig
        config = SimulationConfig(n_profiles=10, n_voters=5, n_candidates=3)
        print(f"✓ Config created: {config.n_profiles} profiles")
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        return False
    
    # Test simulation
    try:
        from simulator import run_experiment
        result = run_experiment(config, save_results=False, verbose=False)
        print(f"✓ Simulation works: {result.total_compute_time:.3f}s")
    except Exception as e:
        print(f"✗ Simulation failed: {e}")
        return False
    
    print("\n✨ All checks passed! Ready to use.")
    print("Launch with: python run_gui.py")
    return True

if __name__ == "__main__":
    verify()
```

Run it:
```bash
python verify_gui.py
```

## 🎓 Next Steps After Installation

### 1. Read Quick Start
```bash
# Open in your editor or browser
QUICKSTART_GUI.md
```

### 2. Try a Demo
```bash
python demo_gui.py 1
```

### 3. Run Your First Simulation
```bash
python run_gui.py
# Select "Quick Simulation"
# Click "Run"
```

### 4. Explore Documentation
- `README_GUI.md` - Overview
- `QUICKSTART_GUI.md` - Quick start
- `GUI_README.md` - Full manual
- `GUI_FEATURES.md` - Visual tour

## 📊 Performance Benchmarks

Expected performance on modern hardware:

| Configuration | Time | Notes |
|--------------|------|-------|
| 100 profiles, 25 voters, 3 candidates, 3 rules | ~1s | Quick test |
| 1,000 profiles, 25 voters, 3 candidates, 6 rules | ~10s | Quick simulation |
| 5,000 profiles, 50 voters, 4 candidates, 5 rules | ~60s | Standard research |
| 10,000 profiles, 100 voters, 5 candidates, 10 rules | ~5min | Large-scale analysis |

Note: Kemeny-Young method is significantly slower than others.

## 🔧 Advanced Installation

### Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install textual rich numpy

# Launch GUI
python run_gui.py
```

### Development Installation

```bash
# Install in development mode
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
python test_gui.py
```

### Offline Installation

If you need to install without internet:

1. Download wheels on a connected machine:
```bash
pip download textual rich numpy -d packages/
```

2. Transfer `packages/` directory to offline machine

3. Install from local files:
```bash
pip install --no-index --find-links=packages/ textual rich numpy
```

## 🌐 Platform-Specific Notes

### Windows
- **Recommended**: Windows Terminal (better color support)
- **PowerShell**: Works well
- **CMD**: Works but limited colors
- **Enable ANSI**: Run `os.system('')` (done automatically)

### Linux
- Most terminals work perfectly
- **Recommended**: GNOME Terminal, Konsole, Alacritty
- UTF-8 encoding should be default

### macOS
- **Recommended**: iTerm2 or Terminal.app
- Both work excellently
- Full color support

## 🎨 Terminal Recommendations

### Best Experience
1. **Windows Terminal** (Windows)
2. **iTerm2** (macOS)
3. **Alacritty** (Cross-platform)
4. **GNOME Terminal** (Linux)

### Good Experience
- PowerShell (Windows)
- Terminal.app (macOS)
- Konsole (Linux)
- xterm (Linux)

### Limited Experience
- CMD (Windows) - works but limited colors
- Basic terminals - may have layout issues

## ✨ Success Indicators

You know it's working when:

1. ✅ Welcome screen displays with colors
2. ✅ Can navigate with keyboard
3. ✅ Buttons respond to clicks/Enter
4. ✅ Configuration screen has two panels
5. ✅ Simulations show progress bar
6. ✅ Results display in table format
7. ✅ Files save to simulator/inputs and simulator/results
8. ✅ No error messages in logs

## 🎉 You're Ready!

If you've completed the verification checklist, you're all set!

**Launch the GUI:**
```bash
python run_gui.py
```

**Try a quick simulation:**
1. Select "⚡ Quick Simulation"
2. Click "Run"
3. View results!

**Read the docs:**
- Start with `QUICKSTART_GUI.md`
- Then explore `GUI_README.md`

**Have fun exploring voting systems!** 🗳️✨

---

**Need help?** Check the troubleshooting section above or read `GUI_README.md` for more details.

**Found a bug?** Run `python test_gui.py` and report the output.

**Want to contribute?** Check `GUI_SUMMARY.md` for architecture details.


