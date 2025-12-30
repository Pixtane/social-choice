# GUI Features & Screenshots

Visual guide to the Spatial Voting Simulator GUI features.

## 🎯 Main Features

### 1. Welcome Screen

```
┌─────────────────────────────────────────────────────────┐
│ 🗳️  SPATIAL VOTING SIMULATOR                            │
│ A comprehensive voting systems analysis tool            │
│                                                          │
│ ┌────────────────────────────────────────────────────┐  │
│ │          ▶ New Simulation                          │  │
│ └────────────────────────────────────────────────────┘  │
│ ┌────────────────────────────────────────────────────┐  │
│ │          📊 View Saved Experiments                 │  │
│ └────────────────────────────────────────────────────┘  │
│ ┌────────────────────────────────────────────────────┐  │
│ │          📈 Compare Experiments                    │  │
│ └────────────────────────────────────────────────────┘  │
│ ┌────────────────────────────────────────────────────┐  │
│ │          ⚡ Quick Simulation                        │  │
│ └────────────────────────────────────────────────────┘  │
│ ┌────────────────────────────────────────────────────┐  │
│ │          ❌ Exit                                    │  │
│ └────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Features**:

- Clean, minimal interface
- Quick access to all major functions
- Keyboard shortcuts (Q to quit)

---

### 2. Configuration Screen

#### Two-Panel Layout

**Left Panel - Basic Parameters**:

```
┌─ ⚙️ Basic Parameters ──────────────┐
│ Number of profiles:                │
│ ┌────────────────────────────────┐ │
│ │ 1000                           │ │
│ └────────────────────────────────┘ │
│ Number of voters:                  │
│ ┌────────────────────────────────┐ │
│ │ 25                             │ │
│ └────────────────────────────────┘ │
│ Number of candidates:              │
│ ┌────────────────────────────────┐ │
│ │ 3                              │ │
│ └────────────────────────────────┘ │
└────────────────────────────────────┘

┌─ 🌐 Spatial Geometry ──────────────┐
│ Geometry method:                   │
│ ┌────────────────────────────────┐ │
│ │ ▼ uniform                      │ │
│ └────────────────────────────────┘ │
│ Spatial dimensions:                │
│ ┌────────────────────────────────┐ │
│ │ 2                              │ │
│ └────────────────────────────────┘ │
└────────────────────────────────────┘
```

**Right Panel - Advanced Options**:

```
┌─ 🗳️ Voting Rules ──────────────────┐
│ ☑ plurality                        │
│ ☑ borda                            │
│ ☑ irv                              │
│ ☐ condorcet                        │
│ ☐ minimax                          │
│ ☐ copeland                         │
│ ☑ schulze                          │
│ ☐ approval                         │
│ ☑ star                             │
└────────────────────────────────────┘

┌─ 🎯 Strategic Manipulation ────────┐
│ ☐ Enable manipulation              │
│ Manipulator fraction (0.0-1.0):    │
│ ┌────────────────────────────────┐ │
│ │ 0.2                            │ │
│ └────────────────────────────────┘ │
│ Strategy:                          │
│ ┌────────────────────────────────┐ │
│ │ ▼ compromise                   │ │
│ └────────────────────────────────┘ │
└────────────────────────────────────┘
```

**Features**:

- Split-screen layout for better organization
- Real-time input validation
- Dropdown menus for method selection
- Multi-select for voting rules
- Scrollable panels for many options
- Keyboard navigation with Tab
- Quick run with Ctrl+S

---

### 3. Simulation Runner Screen

```
┌─────────────────────────────────────────────────────────┐
│ ⚙️ Running Simulation                                    │
│                                                          │
│ Profiles: 1,000 | Voters: 25 | Candidates: 3           │
│ Rules: plurality, borda, irv, approval, star           │
│ Geometry: uniform (2D)                                  │
│                                                          │
│ Status: Running borda...                                │
│ ████████████████████░░░░░░░░░ 65% (ETA: 3s)            │
│                                                          │
│ ┌─ Simulation Log ─────────────────────────────────┐   │
│ │ Starting simulation...                            │   │
│ │ Generating 1000 preference profiles...           │   │
│ │ Running plurality...                              │   │
│ │   VSE: 0.792 ± 0.124                             │   │
│ │   Cycles: 12.3%                                   │   │
│ │   Time: 0.231s                                    │   │
│ │ Running borda...                                  │   │
│ │   VSE: 0.876 ± 0.091                             │   │
│ │   Cycles: 12.3%                                   │   │
│ │   Time: 0.289s                                    │   │
│ │ Running irv...                                    │   │
│ │                                                   │   │
│ └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Features**:

- Real-time progress bar with ETA
- Live status updates
- Scrolling log with detailed output
- Rule-by-rule progress tracking
- Background thread execution (UI stays responsive)
- Escape to cancel
- Auto-navigation to results when complete

---

### 4. Results Screen

```
┌─ 📊 Simulation Results ─────────────────────────────────┐
│ Profiles: 1,000 | Voters: 25 | Candidates: 3           │
│ Time: 2.45s | Geometry: uniform                        │
│                                                          │
│ ┌────────────────────────────────────────────────────┐  │
│ │ Rule      │Type│VSE  │VSE σ│1st %│Cycles%│CW Eff%│  │
│ ├───────────┼────┼─────┼─────┼─────┼───────┼───────┤  │
│ │ plurality │ord │0.792│0.124│78.5 │12.3   │82.1   │  │
│ │ borda     │ord │0.876│0.091│89.2 │12.3   │94.5   │  │
│ │ irv       │ord │0.854│0.098│85.7 │12.3   │91.2   │  │
│ │ approval  │card│0.889│0.087│91.4 │12.3   │96.8   │  │
│ │ star      │card│0.912│0.079│94.1 │12.3   │98.2   │  │
│ └────────────────────────────────────────────────────┘  │
│                                                          │
│ [◀ Back to Menu]  [📈 Detailed View]                    │
└─────────────────────────────────────────────────────────┘
```

**Features**:

- Sortable data table
- Zebra striping for readability
- Cursor navigation (arrow keys)
- Color-coded columns
- Summary statistics in header
- Quick comparison across rules
- Direct navigation to detailed view

---

### 5. Detailed Results Screen (Tabbed)

#### Summary Tab

```
┌─ Detailed Results ──────────────────────────────────────┐
│ ┌Summary┐┌plurality┐┌borda┐┌irv┐┌approval┐┌star┐┌Config┐│
│ └───────┘└─────────┘└─────┘└───┘└────────┘└────┘└──────┘│
│                                                          │
│  Aggregate Metrics                                      │
│  ┌──────────────────────────────────────────────────┐   │
│  │ Rule      │ VSE Mean │ VSE Std │ 1st % │ CW Eff │   │
│  ├───────────┼──────────┼─────────┼───────┼────────┤   │
│  │ plurality │  0.7924  │  0.1243 │ 78.45 │  82.12 │   │
│  │ borda     │  0.8762  │  0.0912 │ 89.23 │  94.56 │   │
│  │ irv       │  0.8541  │  0.0982 │ 85.67 │  91.23 │   │
│  │ approval  │  0.8893  │  0.0871 │ 91.45 │  96.78 │   │
│  │ star      │  0.9124  │  0.0793 │ 94.12 │  98.23 │   │
│  └──────────────────────────────────────────────────┘   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Per-Rule Tab

```
┌─ Detailed Results ──────────────────────────────────────┐
│ ┌Summary┐│plurality│┌borda┐┌irv┐┌approval┐┌star┐┌Config┐│
│ └───────┘└─────────┘└─────┘└───┘└────────┘└────┘└──────┘│
│                                                          │
│  Voting Rule: plurality                                 │
│                                                          │
│  Performance Metrics:                                   │
│    VSE Mean: 0.7924                                     │
│    VSE Std: 0.1243                                      │
│    VSE Min: 0.4123                                      │
│    VSE Max: 0.9876                                      │
│                                                          │
│  Winner Statistics:                                     │
│    1st Place: 78.45%                                    │
│    2nd Place: 18.23%                                    │
│    3rd Place: 3.21%                                     │
│    4th+ Place: 0.11%                                    │
│                                                          │
│  Condorcet Properties:                                  │
│    CW Exists: 87.67%                                    │
│    CW Efficiency: 82.12%                                │
│    Cycles: 12.33%                                       │
│                                                          │
│  Compute Time: 0.231s                                   │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

#### Config Tab

```
┌─ Detailed Results ──────────────────────────────────────┐
│ ┌Summary┐┌plurality┐┌borda┐┌irv┐┌approval┐┌star┐│Config││
│ └───────┘└─────────┘└─────┘└───┘└────────┘└────┘└──────┘│
│                                                          │
│  {                                                       │
│    'experiment_id': 'beb22357',                         │
│    'n_profiles': 1000,                                  │
│    'n_voters': 25,                                      │
│    'n_candidates': 3,                                   │
│    'voting_rules': ['plurality', 'borda', ...],         │
│    'geometry_method': 'uniform',                        │
│    'geometry_n_dim': 2,                                 │
│    'utility_function': 'gaussian',                      │
│    'manipulation_enabled': False,                       │
│    ...                                                  │
│  }                                                       │
│                                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Features**:

- Tabbed interface for organized information
- One tab per voting rule + summary + config
- Rich text formatting with colors
- Syntax highlighting for config (JSON)
- Easy navigation with Tab/Shift+Tab
- Scrollable content for long outputs
- Comprehensive metrics breakdown

---

### 6. Saved Experiments Browser

```
┌─ 📁 Saved Experiments ──────────────────────────────────┐
│                                                          │
│ ┌────────────────────────────────────────────────────┐  │
│ │#│Filename              │Profiles│Voters│Cands│ID  │  │
│ ├─┼──────────────────────┼────────┼──────┼─────┼────┤  │
│ │1│2025-12-30_17-33-19..│1000    │25    │3    │beb2│  │
│ │2│2025-12-30_18-02-20..│5000    │50    │4    │f2e3│  │
│ │3│2025-12-29_14-22-11..│2000    │25    │3    │ac0e│  │
│ │4│2025-12-29_10-15-33..│1000    │100   │5    │7f9a│  │
│ │5│2025-12-28_16-45-22..│10000   │25    │3    │2d4b│  │
│ └────────────────────────────────────────────────────┘  │
│                                                          │
│ [◀ Back]  [🔄 Refresh]  [📂 Load Selected]              │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Features**:

- Scrollable list of all saved experiments
- Key metadata at a glance
- Cursor navigation (arrow keys)
- Quick load with Enter or button
- Refresh to detect new files
- Sort by date (most recent first)
- Keyboard shortcuts (L to load, R to refresh)

---

### 7. Quick Simulation Modal

```
┌─────────────────────────────────────────────────────────┐
│                                                          │
│        ┌─ ⚡ Quick Simulation ──────────────────┐        │
│        │                                        │        │
│        │ Number of profiles:                   │        │
│        │ ┌────────────────────────────────────┐│        │
│        │ │ 1000                               ││        │
│        │ └────────────────────────────────────┘│        │
│        │ Number of voters:                     │        │
│        │ ┌────────────────────────────────────┐│        │
│        │ │ 25                                 ││        │
│        │ └────────────────────────────────────┘│        │
│        │ Number of candidates:                 │        │
│        │ ┌────────────────────────────────────┐│        │
│        │ │ 3                                  ││        │
│        │ └────────────────────────────────────┘│        │
│        │                                        │        │
│        │  [Cancel]  [Run]                      │        │
│        └────────────────────────────────────────┘        │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

**Features**:

- Modal overlay (doesn't leave current screen)
- Minimal configuration (3 fields only)
- Uses sensible defaults for all other settings
- Quick escape with Cancel or Escape key
- Perfect for rapid testing

---

## 🎨 UI/UX Features

### Color Scheme

- **Accent Colors**: Cyan/Blue for headers and important text
- **Status Colors**:
  - Green for success/completion
  - Yellow for warnings/in-progress
  - Red for errors/cancellation
  - Dim/Gray for secondary information
- **Zebra Striping**: Alternating row colors in tables
- **Syntax Highlighting**: Color-coded config display

### Responsive Design

- Adapts to terminal size
- Minimum 80x24 recommended
- Scrollable panels for overflow content
- Fixed headers and footers
- Proper text wrapping

### Accessibility

- Keyboard-only navigation supported
- Clear visual hierarchy
- High contrast text
- Status indicators beyond just color
- Screen reader compatible (Textual framework)

### Performance

- Background threading for simulations
- Non-blocking UI during computation
- Efficient table rendering
- Lazy loading for large datasets
- Progress feedback for long operations

---

## 🎯 Workflow Examples

### Example 1: Quick Test Run

1. Start → Quick Simulation
2. Accept defaults
3. Run
4. View results (30 seconds total)

### Example 2: Custom Comparison

1. Start → New Simulation
2. Configure parameters (2 minutes)
3. Select 5 voting rules
4. Run simulation (1 minute)
5. View results
6. Open detailed view
7. Compare metrics across tabs (5 minutes)

### Example 3: Research Session

1. Start → New Simulation
2. Configure high-fidelity settings (5 minutes)
3. Run multiple scenarios (30 minutes)
4. Browse saved experiments
5. Load and compare results (20 minutes)
6. Export data for further analysis

---

## 🔧 Customization Options

### Themes

- Default: Dark mode (optimal for terminals)
- Toggle with `D` key
- Custom CSS support in code

### Layout

- Modify `CSS` strings in gui.py
- Adjust panel sizes
- Change color schemes
- Customize spacing and borders

### Widgets

- Add new input fields
- Create custom visualizations
- Extend screens with new features
- Integrate additional metrics

---

## 📊 Data Flow

```
Welcome Screen
    ↓
Configuration Screen (collect parameters)
    ↓
SimulationConfig object
    ↓
SimulationRunScreen (background thread)
    ↓
run_experiment() → ExperimentResult
    ↓
ResultsScreen (display table)
    ↓
DetailedResultsScreen (tabs for deep dive)
```

### Auto-Save

Every simulation automatically saves:

- Input data (positions, utilities, rankings)
- Results (metrics for all rules)
- Configuration (for reproducibility)
- Metadata (timestamp, IDs)

---

## 🚀 Advanced Features

### Concurrent Operations

- Multiple simulations can be queued
- Background processing doesn't block UI
- Worker cancellation on demand
- Progress tracking per task

### Data Export

- CSV format for results
- NPZ format for arrays
- JSON config embedded
- Compatible with pandas/numpy

### Integration

- Call from Python scripts
- Embed in larger workflows
- Export to visualization tools
- Batch processing support

---

## 🎓 Learning Path

### Beginner

1. Run quick simulation
2. Try different voting rules
3. View basic results
4. Understand VSE metric

### Intermediate

1. Configure custom parameters
2. Test different geometries
3. Compare multiple scenarios
4. Enable strategic voting

### Advanced

1. Design controlled experiments
2. Vary systematic parameters
3. Analyze manipulation resistance
4. Export for statistical analysis

---

## 💡 Tips & Tricks

### Performance Tips

- Start with 100-1000 profiles for testing
- Use quick simulation for rapid iteration
- Disable expensive rules (Kemeny-Young) for speed
- Lower dimensions for faster computation

### Analysis Tips

- Run with fixed seed for reproducibility
- Compare pairs of rules at a time
- Test extreme scenarios (polarized, single-peaked)
- Look for consistent patterns across runs

### Workflow Tips

- Use keyboard shortcuts for efficiency
- Save configurations you like
- Name experiments descriptively
- Document interesting findings

### Troubleshooting Tips

- Check terminal size if layout breaks
- Verify Textual version is current
- Read error messages in log panel
- Test with small parameters first

---

## 📈 Experiment Comparison

### Comparison Selection Screen

```
┌─────────────────────────────────────────────────────────┐
│ 📈 Compare Experiments                                   │
│ Select 2 or more experiments to compare. Use SPACE...   │
├─────────────────────────────────────────────────────────┤
│ ┌─────────────────────────────────────────────────────┐ │
│ │ ☑ 2025-12-30_17-33-19_beb22357 | 50p x 10v | beb... │ │
│ │ ☑ 2025-12-30_18-25-34_3e02d790 | 1000p x 25v | 3e..│ │
│ │ ☐ 2025-12-30_17-32-25_ac0efdeb | 100p x 15v | ac... │ │
│ └─────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│ Mode: ▼ Rows=experiments  Format: ▼ Text  Selected: 2  │
│ [◀ Back] [Select All] [Deselect All] [🔄] [📊 Compare] │
└─────────────────────────────────────────────────────────┘
```

### Comparison Modes

| Mode              | Description                                         |
| ----------------- | --------------------------------------------------- |
| **by_experiment** | Rows=experiments, Columns=metrics (VSE, rank, etc.) |
| **by_rule**       | Rows=voting rules, Columns=experiments              |
| **by_metric**     | Rows=metrics, Columns=experiment/rule combinations  |
| **differential**  | Show differences from first experiment as baseline  |

### Comparison Results Screen

```
┌─────────────────────────────────────────────────────────┐
│ 📊 Comparison Results                                    │
│ Comparing 2 experiments | Mode: by_experiment           │
├─────────────────────────────────────────────────────────┤
│ ┌─ Table View ─┬─ Text Output ─┬─ By Rule ─┬─ Diff ─┐  │
│ │ Experiment   │ borda_vse │ irv_vse │ plurality_vse│  │
│ ├──────────────┼───────────┼─────────┼──────────────┤  │
│ │ exp_beb22... │ 0.9423    │ 0.8612  │ 0.8798       │  │
│ │ exp_3e02d... │ 0.9549    │ 0.9359  │ 0.8781       │  │
│ └──────────────┴───────────┴─────────┴──────────────┘  │
├─────────────────────────────────────────────────────────┤
│ [◀ Back] [📋 Copy to Clipboard] [💾 Export CSV]        │
└─────────────────────────────────────────────────────────┘
```

### CLI Comparison Commands

```bash
# Compare experiments by ID or filename
python -m simulator.cli --compare exp1 exp2 exp3

# Compare with specific mode
python -m simulator.cli --compare exp1 exp2 --compare-mode by_rule

# Compare and output as markdown
python -m simulator.cli --compare exp1 exp2 --compare-format markdown

# Compare and output as CSV
python -m simulator.cli --compare exp1 exp2 --compare-format csv

# Interactive comparison mode
python -m simulator.cli --compare-interactive
```

### Differential Mode Example

```
Baseline: beb22357

Comparison: 3e02d790
------------------------------------------------------------
  borda:
    vse: +0.0126 (+1.3%)
    winner_rank: -0.0190 (-11.9%)
  irv:
    vse: +0.0748 (+8.7%)
    winner_rank: -0.1170 (-39.0%)
  plurality:
    vse: -0.0017 (-0.2%)
    winner_rank: +0.0500 (+19.2%)
```

---

## 🌟 Best Practices

1. **Start Simple**: Begin with defaults, then customize
2. **One Variable**: Change one thing at a time
3. **Document**: Note successful configurations
4. **Compare**: Always run multiple rules together
5. **Validate**: Verify results make intuitive sense
6. **Export**: Save important findings immediately
7. **Iterate**: Refine based on initial results
8. **Use Comparison**: Compare experiments to understand parameter effects

---

This GUI makes voting system analysis accessible, interactive, and visually engaging! 🗳️✨
