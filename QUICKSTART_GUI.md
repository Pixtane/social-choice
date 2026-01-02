# Quick Start Guide - GUI

Get started with the Spatial Voting Simulator GUI in 5 minutes!

## Installation

```bash
# 1. Install dependencies
pip install textual rich numpy

# 2. Launch the GUI
python run_gui.py
```

## Your First Simulation

### Option 1: Quick Simulation (Fastest)

1. Launch: `python run_gui.py`
2. Select: **⚡ Quick Simulation**
3. Enter:
   - Profiles: `1000` (default)
   - Voters: `25` (default)
   - Candidates: `3` (default)
4. Click: **Run**
5. Wait ~5-10 seconds
6. View results automatically!

**What you get**: Comparison of 6 voting rules (Plurality, Borda, IRV, Approval, STAR, Schulze) with default settings.

### Option 2: Custom Simulation

1. Launch: `python run_gui.py`
2. Select: **▶ New Simulation**
3. Configure (left panel):
   ```
   Profiles: 1000
   Voters: 25
   Candidates: 3
   Geometry: uniform
   Dimensions: 2
   ```
4. Select voting rules (right panel):
   - Check: `plurality`, `borda`, `irv`
   - Or select all you want to compare
5. Click: **▶ Run Simulation** (or press `Ctrl+S`)
6. View results!

## Understanding Results

### Results Table
```
Rule      Type  VSE    VSE σ  1st %  Cycles %  CW Eff %
--------  ----  -----  -----  -----  --------  --------
plurality ord   0.792  0.124  78.5   12.3      82.1
borda     ord   0.876  0.091  89.2   12.3      94.5
irv       ord   0.854  0.098  85.7   12.3      91.2
approval  card  0.889  0.087  91.4   12.3      96.8
star      card  0.912  0.079  94.1   12.3      98.2
```

**Key Metrics**:
- **VSE**: Voter Satisfaction Efficiency (0-1, higher is better)
  - 1.0 = always picks the utilitarian winner
  - 0.0 = random selection
- **1st %**: How often the winner is the top-rated candidate
- **Cycles %**: Frequency of Condorcet cycles (voting paradoxes)
- **CW Eff %**: Condorcet winner efficiency (picks CW when exists)

### Interpreting Your Results

**High VSE (>0.85)**: Excellent voting rule
**Medium VSE (0.75-0.85)**: Good performance
**Low VSE (<0.75)**: Poor voter satisfaction

**High CW Efficiency (>90%)**: Respects majority preferences
**High 1st Place %**: Winners are widely preferred

## Next Steps

### Compare Different Scenarios

#### Test Polarization
```
Configuration Screen:
  Geometry: polarized
  Dimensions: 2
```
See how rules perform when voters cluster in opposing groups.

#### Add Strategic Voting
```
Configuration Screen (right panel):
  ☑ Enable manipulation
  Manipulator fraction: 0.2
  Strategy: compromise
```
See impact of 20% strategic voters.

#### Test More Candidates
```
Configuration Screen:
  Candidates: 5
```
See how complexity affects rule performance.

### Explore Detailed Results

After viewing results:
1. Click: **📈 Detailed View**
2. Browse tabs for each voting rule
3. See full statistics breakdown
4. Review configuration used

### Save and Compare

All simulations auto-save to:
- `simulator/inputs/` - Input data
- `simulator/results/` - Result metrics

From main menu:
1. Select: **📊 View Saved Experiments**
2. Browse previous runs
3. Load and compare

## Keyboard Shortcuts

While using the GUI:
- `Q` - Quit anytime
- `Escape` - Go back
- `Ctrl+S` - Run simulation (on config screen)
- `D` - Toggle dark mode
- `Tab` - Navigate fields

## Common Use Cases

### 1. "Which voting rule is best?"
```
Quick Simulation with defaults → Compare VSE scores
```

### 2. "How does IRV compare to Approval?"
```
New Simulation → Select only: irv, approval → Run
```

### 3. "Test strategic voting resistance"
```
New Simulation → Enable manipulation → 
Set fraction to 0.3 → Compare with/without
```

### 4. "Analyze polarized electorate"
```
New Simulation → Geometry: polarized →
Select all rules → Compare results
```

### 5. "Research-grade simulation"
```
New Simulation →
  Profiles: 10000
  Voters: 100
  Candidates: 4
  Enable manipulation
→ Run → Export results
```

## Tips for Best Results

1. **Start Small**: Test with 100 profiles first
2. **Compare 3-5 Rules**: Too many clutters results
3. **Vary One Thing**: Change one parameter at a time
4. **Use Random Seed**: For reproducible experiments
5. **Save Configs**: Note settings for successful runs

## Troubleshooting

**"No voting rules selected"**: Check at least one rule in the list

**Slow simulation**: Reduce profiles or disable complex rules

**Layout issues**: Maximize terminal window (need 80x24 minimum)

**Can't see colors**: Enable ANSI color support in your terminal

## Examples to Try

### Example 1: Classic Comparison
```
Profiles: 1000
Voters: 25
Candidates: 3
Geometry: uniform
Rules: plurality, borda, irv, approval, star
```

### Example 2: Condorcet Methods
```
Profiles: 5000
Voters: 50
Candidates: 3
Geometry: uniform
Rules: minimax, copeland, schulze, ranked_pairs
```

### Example 3: Strategic Voting Impact
Run twice, compare:
```
Config 1: Manipulation disabled
Config 2: Manipulation enabled (20%, compromise)
```

### Example 4: Polarization Study
```
Profiles: 2000
Voters: 51
Candidates: 3
Geometry: polarized
Rules: plurality, approval, star, schulze
```

## Getting Help

- Read `GUI_README.md` for comprehensive documentation
- Check `VOTING_PATTERNS.md` for voting theory
- Explore `simulator/` code for implementation details

## What's Next?

Once comfortable with the GUI:
- Try the CLI for automation: `python -m simulator.cli`
- Use Python API for custom analysis
- Explore manipulation strategies
- Test with custom geometries
- Export data for visualization

Enjoy exploring voting systems! 🗳️







