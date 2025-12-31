# 2D Visual Comparison GUI

Interactive 2D visualization tool for exploring voting simulations in spatial preference models.

## Features

- **2D Spatial Visualization**: View voters and candidates plotted in 2D space
- **Distance Metric Visualization**: See which distance metric each voter uses (color-coded)
  - L1 (Manhattan) - Red
  - L2 (Euclidean) - Teal
  - Cosine - Light Teal
  - Chebyshev - Pink
- **Voting Lines**: Visualize which candidate each voter votes for with lines connecting them
- **Distance Labels**: See the computed distance (using each voter's metric) displayed on the lines
- **Statistics Panel**: View vote counts and percentages for each candidate
  - Shows number of votes per candidate
  - Displays percentage of total votes
  - Highlights the winner in red
  - Special handling for approval voting (shows total approvals)
- **Interactive Controls**: Adjust simulation parameters in real-time:
  - Geometry generation method (uniform, clustered, polarized, 2d)
  - Number of voters (5-50)
  - Number of candidates (2-6)
  - Voting rule (plurality, borda, irv, approval, star, schulze)
  - Distance metric (for homogeneous mode)
  - Heterogeneous distance strategies (center-extreme, radial-steps)

## Usage

Run the GUI:

```bash
python visual_gui.py
```

Or from the parent directory:

```bash
python 2d-visual-comparison/visual_gui.py
```

## Controls

- **Geometry Method**: Choose how voters and candidates are distributed in space
- **Number of Voters**: Slider to adjust voter count (5-50)
- **Number of Candidates**: Slider to adjust candidate count (2-6)
- **Voting Rule**: Select which voting rule to visualize
- **Distance Metric**: Choose the metric for homogeneous distance mode
- **Heterogeneous Distance**: Enable/disable heterogeneous distance strategies
- **Regenerate**: Generate a new random profile
- **Update**: Refresh the visualization with current settings

## Visualization Elements

- **Blue Squares**: Candidates (red square indicates winner)
- **Colored Circles**: Voters (color indicates their distance metric)
- **Colored Lines**: Voting connections from voters to their chosen candidates (color matches voter's metric)
- **Distance Labels**: Numbers on lines showing the computed distance
- **Statistics Panel**: Table showing vote counts and percentages for each candidate, with winner highlighted

## Notes

- This tool works with a single profile (not batch simulations)
- Results are not stored to disk
- The visualization is optimized for 2D space only

