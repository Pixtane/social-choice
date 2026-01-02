# GUI Quick Reference Card

One-page reference for the Spatial Voting Simulator GUI.

## 🚀 Launch Commands

```bash
python run_gui.py              # Simple launch
python run_simulator.py        # Interactive menu
python run_simulator.py --gui  # Direct GUI
python demo_gui.py 1           # Demo scenario
python test_gui.py             # Verify installation
```

## ⌨️ Keyboard Shortcuts

### Global
| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `D` | Toggle dark mode |
| `Escape` | Go back / Cancel |
| `Tab` | Next field |
| `Shift+Tab` | Previous field |
| `Enter` | Activate button / Confirm |

### Configuration Screen
| Key | Action |
|-----|--------|
| `Ctrl+S` | Run simulation |
| `Tab` | Navigate fields |

### Results Screen
| Key | Action |
|-----|--------|
| `D` | Detailed view |
| `↑↓` | Navigate table |

### Saved Experiments
| Key | Action |
|-----|--------|
| `L` | Load selected |
| `R` | Refresh list |
| `↑↓` | Navigate list |

## 📊 Key Metrics

| Metric | Meaning | Good Value |
|--------|---------|------------|
| **VSE** | Voter Satisfaction Efficiency | >0.85 |
| **1st %** | Winner is top-rated | >85% |
| **Cycles %** | Condorcet cycles | <15% |
| **CW Eff %** | Condorcet efficiency | >90% |

## 🗳️ Voting Rules

### Ordinal (Ranking-based)
- `plurality` - First-past-the-post
- `borda` - Borda count
- `irv` - Instant runoff
- `schulze` - Schulze method
- `minimax` - Minimax Condorcet
- `copeland` - Copeland method
- `ranked_pairs` - Tideman method
- `kemeny_young` - Kemeny-Young

### Cardinal (Score-based)
- `approval` - Approve/disapprove
- `score` - Range voting
- `star` - Score then runoff
- `utilitarian` - Maximum utility
- `quadratic` - Quadratic voting

## 🌐 Geometry Methods

| Method | Description |
|--------|-------------|
| `uniform` | Random uniform distribution |
| `clustered` | Clustered around center |
| `polarized` | Two opposing groups |
| `single_peaked` | 1D left-right spectrum |
| `1d` | 1D uniform |
| `2d` | 2D with triangle placement |

## 🎯 Manipulation Strategies

| Strategy | Description |
|----------|-------------|
| `bullet` | Vote only for top choice |
| `compromise` | Rank viable alternative higher |
| `burial` | Rank competitor lower |
| `pushover` | Support weak opponent |
| `optimal` | Compute best strategy |

## 📐 Utility Functions

| Function | Formula | Use Case |
|----------|---------|----------|
| `gaussian` | exp(-d²/2σ²) | Default, smooth |
| `quadratic` | 1 - d² | Simple, fast |
| `linear` | 1 - d | Proportional |
| `exponential` | exp(-λd) | Sharp decay |

## 🔧 Quick Configurations

### Quick Test (1 minute)
```
Profiles: 100
Voters: 25
Candidates: 3
Rules: plurality, borda, irv
```

### Standard Analysis (10 seconds)
```
Profiles: 1000
Voters: 25
Candidates: 3
Rules: plurality, borda, irv, approval, star
```

### Research Grade (1 minute)
```
Profiles: 10000
Voters: 50
Candidates: 4
Rules: All ordinal + All cardinal
```

### Polarization Study
```
Profiles: 2000
Voters: 51
Candidates: 3
Geometry: polarized
Rules: plurality, approval, star
```

### Strategic Voting
```
Profiles: 1000
Voters: 25
Candidates: 3
Manipulation: Enabled (20%, compromise)
Rules: plurality, borda, irv, approval
```

## 📁 File Locations

| Type | Path |
|------|------|
| Inputs | `simulator/inputs/YYYY-MM-DD_HH-MM-SS_<id>.npz` |
| Results | `simulator/results/YYYY-MM-DD_HH-MM-SS_<id>.csv` |

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| Module not found | `pip install textual rich numpy` |
| Layout broken | Maximize terminal (need 80×24) |
| No colors | Enable ANSI in terminal settings |
| Slow simulation | Reduce profiles or rules |
| Can't see cursor | Use arrow keys to navigate |

## 📚 Documentation Quick Links

| Need | Read |
|------|------|
| Get started (5 min) | `QUICKSTART_GUI.md` |
| Full manual (20 min) | `GUI_README.md` |
| Visual tour (15 min) | `GUI_FEATURES.md` |
| Technical details | `GUI_SUMMARY.md` |
| Installation help | `INSTALL_GUI.md` |

## 🎯 Common Tasks

### Run Quick Simulation
1. Launch: `python run_gui.py`
2. Select: "⚡ Quick Simulation"
3. Click: "Run"

### Compare Voting Rules
1. Launch GUI
2. Select: "▶ New Simulation"
3. Choose rules to compare
4. Click: "▶ Run Simulation"

### Test Strategic Voting
1. Launch GUI
2. Select: "▶ New Simulation"
3. Check: "Enable manipulation"
4. Set: Fraction and strategy
5. Run and compare

### View Saved Results
1. Launch GUI
2. Select: "📊 View Saved Experiments"
3. Navigate list
4. Press: `L` to load

## 💡 Pro Tips

1. **Start small** - Test with 100 profiles first
2. **Use keyboard** - Faster than mouse
3. **Set seed** - For reproducible results
4. **Compare pairs** - 2-3 rules at a time
5. **Read logs** - Watch for warnings
6. **Save configs** - Note successful setups
7. **Check docs** - Comprehensive guides available

## 📊 Performance Guide

| Profiles | Voters | Rules | Time | Use Case |
|----------|--------|-------|------|----------|
| 100 | 25 | 3 | ~1s | Quick test |
| 1,000 | 25 | 6 | ~10s | Standard |
| 5,000 | 50 | 5 | ~60s | Research |
| 10,000 | 100 | 10 | ~5min | Large-scale |

## 🎨 Screen Navigation

```
Welcome
  ├─→ New Simulation → Configuration → Simulation → Results → Detailed
  ├─→ Quick Simulation → Simulation → Results → Detailed
  └─→ Saved Experiments → (Load) → Results → Detailed
```

## 🔍 Result Interpretation

### VSE (Voter Satisfaction Efficiency)
- **0.90-1.00**: Excellent
- **0.85-0.90**: Very good
- **0.75-0.85**: Good
- **0.60-0.75**: Fair
- **<0.60**: Poor

### Condorcet Efficiency
- **>95%**: Excellent
- **90-95%**: Very good
- **80-90%**: Good
- **70-80%**: Fair
- **<70%**: Poor

## 🎓 Learning Path

### Beginner (30 min)
1. Read `QUICKSTART_GUI.md`
2. Run quick simulation
3. Try 3 different voting rules
4. View detailed results

### Intermediate (2 hours)
1. Read `GUI_README.md`
2. Try all 6 demos
3. Configure custom simulations
4. Test manipulation

### Advanced (1 day)
1. Read `GUI_SUMMARY.md`
2. Study source code
3. Run research-grade simulations
4. Analyze exported data

## 🆘 Get Help

1. **Quick issue**: Check this reference
2. **Installation**: Read `INSTALL_GUI.md`
3. **Usage**: Read `QUICKSTART_GUI.md`
4. **Details**: Read `GUI_README.md`
5. **Technical**: Read `GUI_SUMMARY.md`
6. **Test**: Run `python test_gui.py`

## ✅ Quick Checklist

Before running a simulation:
- [ ] Dependencies installed
- [ ] Terminal size adequate (80×24+)
- [ ] Configuration makes sense
- [ ] At least one voting rule selected
- [ ] Output directories exist

After simulation:
- [ ] Results displayed correctly
- [ ] Metrics are reasonable
- [ ] Files saved successfully
- [ ] Ready for next simulation

## 🎉 Quick Start

```bash
# 1. Install (30 seconds)
pip install textual rich numpy

# 2. Verify (10 seconds)
python test_gui.py

# 3. Launch (instant)
python run_gui.py

# 4. Run (30 seconds)
# Select "Quick Simulation" → Run

# Total: ~70 seconds to results! 🚀
```

---

**Print this page for quick reference!**

**Version**: 1.0.0 | **Date**: December 30, 2025

**More help**: See `GUI_INDEX.md` for complete documentation index







