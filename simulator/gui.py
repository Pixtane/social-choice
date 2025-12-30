"""
Textual GUI for the Spatial Voting Simulator.

A rich terminal-based interface for configuring and running
voting simulations with real-time progress updates.
"""

from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Header, Footer, Button, Static, Input, Label,
    SelectionList, RadioSet, RadioButton, Checkbox,
    DataTable, TabbedContent, TabPane, Select, ProgressBar,
    LoadingIndicator, Log, Pretty
)
from textual.screen import Screen, ModalScreen
from textual.binding import Binding
from textual import work
from textual.worker import Worker, WorkerState
from rich.text import Text
from rich.table import Table as RichTable
from rich.panel import Panel
from rich.syntax import Syntax

import asyncio
from typing import Optional, List, Dict, Any
from dataclasses import asdict

from .config import (
    SimulationConfig, GeometryConfig, ManipulationConfig,
    UtilityConfig, VotingRuleConfig, HeterogeneousDistanceConfig,
    AVAILABLE_VOTING_RULES, AVAILABLE_GEOMETRY_METHODS,
    AVAILABLE_MANIPULATION_STRATEGIES, AVAILABLE_HETEROGENEOUS_STRATEGIES,
    AVAILABLE_RADIAL_SCALING, AVAILABLE_DISTANCE_METRICS
)
from .main import run_experiment, ExperimentResult
from .storage import list_experiments, load_experiment
from .comparison import (
    ExperimentComparator, ExperimentSummary, 
    get_comparison_modes, ComparisonResult
)


class WelcomeScreen(Screen):
    """Welcome screen with main menu options."""
    
    CSS = """
    WelcomeScreen {
        align: center middle;
    }
    
    #welcome-container {
        width: 70;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 2;
    }
    
    #title {
        width: 100%;
        content-align: center middle;
        text-style: bold;
        color: $accent;
        padding: 1;
    }
    
    #subtitle {
        width: 100%;
        content-align: center middle;
        color: $text-muted;
        padding-bottom: 1;
    }
    
    Button {
        width: 100%;
        margin: 1;
    }
    """
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        with Container(id="welcome-container"):
            yield Static("🗳️  SPATIAL VOTING SIMULATOR", id="title")
            yield Static("A comprehensive voting systems analysis tool", id="subtitle")
            yield Button("▶ New Simulation", id="new-sim", variant="success")
            yield Button("📊 View Saved Experiments", id="view-saved", variant="primary")
            yield Button("📈 Compare Experiments", id="compare", variant="primary")
            yield Button("⚡ Quick Simulation", id="quick-sim", variant="warning")
            yield Button("❌ Exit", id="exit", variant="error")
        yield Footer()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-sim":
            self.app.push_screen(ConfigurationScreen())
        elif event.button.id == "view-saved":
            self.app.push_screen(SavedExperimentsScreen())
        elif event.button.id == "compare":
            self.app.push_screen(ComparisonSelectScreen())
        elif event.button.id == "quick-sim":
            self.app.push_screen(QuickSimScreen())
        elif event.button.id == "exit":
            self.app.exit()


class ConfigurationScreen(Screen):
    """Screen for configuring a simulation."""
    
    CSS = """
    ConfigurationScreen {
        layout: grid;
        grid-size: 2;
        grid-rows: 1fr;
    }
    
    #config-left {
        height: 100%;
        border-right: solid $primary;
    }
    
    #config-right {
        height: 100%;
        overflow-y: auto;
    }
    
    .config-section {
        border: solid $primary;
        margin: 1;
        padding: 1;
        height: auto;
    }
    
    .section-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    Input {
        margin: 1 0;
    }
    
    #button-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        border-top: solid $primary;
    }
    
    #button-bar Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "run_simulation", "Run"),
    ]
    
    def __init__(self):
        super().__init__()
        # Default values
        self.config_values = {
            'n_profiles': 1000,
            'n_voters': 25,
            'n_candidates': 3,
            'geometry_method': 'uniform',
            'n_dim': 2,
            'selected_rules': ['plurality', 'borda', 'irv'],
            'manipulation_enabled': False,
            'manip_fraction': 0.2,
            'manip_strategy': 'compromise',
            'utility_func': 'gaussian',
            'distance_metric': 'l2',
            'sigma_factor': 0.5,
            'rng_seed': None,
        }
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        # Left side - Basic parameters
        with VerticalScroll(id="config-left"):
            with Container(classes="config-section"):
                yield Label("⚙️ Basic Parameters", classes="section-title")
                yield Label("Number of profiles:")
                yield Input(
                    value="1000",
                    type="integer",
                    id="n_profiles",
                    placeholder="1-100000"
                )
                yield Label("Number of voters:")
                yield Input(
                    value="25",
                    type="integer",
                    id="n_voters",
                    placeholder="3-1000"
                )
                yield Label("Number of candidates:")
                yield Input(
                    value="3",
                    type="integer",
                    id="n_candidates",
                    placeholder="2-20"
                )
            
            with Container(classes="config-section"):
                yield Label("🌐 Spatial Geometry", classes="section-title")
                yield Label("Geometry method:")
                yield Select(
                    [(desc, method) for method, desc in AVAILABLE_GEOMETRY_METHODS.items()],
                    value='uniform',
                    id="geometry_method"
                )
                yield Label("Spatial dimensions:")
                yield Input(
                    value="2",
                    type="integer",
                    id="n_dim",
                    placeholder="1-10"
                )
        
        # Right side - Advanced options
        with VerticalScroll(id="config-right"):
            with Container(classes="config-section"):
                yield Label("🗳️ Voting Rules", classes="section-title")
                
                # Create selection list for voting rules
                ordinal_rules = [
                    (name, name, name in self.config_values['selected_rules'])
                    for name, info in AVAILABLE_VOTING_RULES.items()
                    if info['type'] == 'ordinal'
                ]
                cardinal_rules = [
                    (name, name, name in self.config_values['selected_rules'])
                    for name, info in AVAILABLE_VOTING_RULES.items()
                    if info['type'] == 'cardinal'
                ]
                
                yield SelectionList(
                    *ordinal_rules,
                    *cardinal_rules,
                    id="voting_rules"
                )
            
            with Container(classes="config-section"):
                yield Label("🎯 Strategic Manipulation", classes="section-title")
                yield Checkbox("Enable manipulation", id="manipulation_enabled")
                yield Label("Manipulator fraction (0.0-1.0):")
                yield Input(
                    value="0.2",
                    type="number",
                    id="manip_fraction",
                    placeholder="0.0-1.0"
                )
                yield Label("Strategy:")
                yield Select(
                    [(desc, strategy) for strategy, desc in AVAILABLE_MANIPULATION_STRATEGIES.items()],
                    value='compromise',
                    id="manip_strategy"
                )
            
            with Container(classes="config-section"):
                yield Label("📐 Utility Function", classes="section-title")
                yield Label("Function type:")
                yield Select(
                    [
                        ("Gaussian", "gaussian"),
                        ("Quadratic", "quadratic"),
                        ("Linear", "linear"),
                        ("Exponential", "exponential")
                    ],
                    value='gaussian',
                    id="utility_func"
                )
                yield Label("Distance metric:")
                yield Select(
                    [
                        ("Euclidean (L2)", "l2"),
                        ("Manhattan (L1)", "l1"),
                        ("Cosine", "cosine"),
                        ("Chebyshev", "chebyshev")
                    ],
                    value='l2',
                    id="distance_metric"
                )
                yield Label("Sigma factor (Gaussian):")
                yield Input(
                    value="0.5",
                    type="number",
                    id="sigma_factor",
                    placeholder="0.1-2.0"
                )
            
            with Container(classes="config-section"):
                yield Label("🎲 Random Seed", classes="section-title")
                yield Input(
                    placeholder="Leave empty for random",
                    id="rng_seed"
                )
            
            with Container(classes="config-section"):
                yield Label("🔀 Heterogeneous Distance", classes="section-title")
                yield Checkbox("Enable heterogeneous distance metrics", id="het_enabled")
                yield Label("Strategy:")
                yield Select(
                    [
                        (desc, strategy) 
                        for strategy, desc in AVAILABLE_HETEROGENEOUS_STRATEGIES.items()
                    ],
                    value='center_extreme',
                    id="het_strategy"
                )
                yield Label("── Center-Extreme Settings ──", classes="section-title")
                yield Label("Center metric:")
                yield Select(
                    [
                        (desc, metric) 
                        for metric, desc in AVAILABLE_DISTANCE_METRICS.items()
                    ],
                    value='l2',
                    id="het_center_metric"
                )
                yield Label("Extreme metric:")
                yield Select(
                    [
                        (desc, metric) 
                        for metric, desc in AVAILABLE_DISTANCE_METRICS.items()
                    ],
                    value='cosine',
                    id="het_extreme_metric"
                )
                yield Label("Extreme threshold (0.0-1.0):")
                yield Input(
                    value="0.5",
                    type="number",
                    id="het_threshold",
                    placeholder="0.0=all extreme, 1.0=none extreme"
                )
                yield Label("── Radial Steps Settings ──", classes="section-title")
                yield Label("Radial scaling:")
                yield Select(
                    [
                        (desc, scaling) 
                        for scaling, desc in AVAILABLE_RADIAL_SCALING.items()
                    ],
                    value='linear',
                    id="het_radial_scaling"
                )
                yield Label("Scaling parameter:")
                yield Input(
                    value="2.0",
                    type="number",
                    id="het_scaling_param",
                    placeholder="Base for log/exp scaling"
                )
        
        # Bottom button bar
        with Horizontal(id="button-bar"):
            yield Button("◀ Back", id="back", variant="default")
            yield Button("▶ Run Simulation", id="run", variant="success")
        
        yield Footer()
    
    def action_back(self) -> None:
        self.app.pop_screen()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.app.pop_screen()
        elif event.button.id == "run":
            self.action_run_simulation()
    
    def action_run_simulation(self) -> None:
        """Collect configuration and start simulation."""
        try:
            # Collect values
            n_profiles = int(self.query_one("#n_profiles", Input).value or 1000)
            n_voters = int(self.query_one("#n_voters", Input).value or 25)
            n_candidates = int(self.query_one("#n_candidates", Input).value or 3)
            
            geometry_method = self.query_one("#geometry_method", Select).value
            n_dim = int(self.query_one("#n_dim", Input).value or 2)
            
            selected_rules = list(self.query_one("#voting_rules", SelectionList).selected)
            if not selected_rules:
                self.notify("Please select at least one voting rule", severity="error")
                return
            
            manipulation_enabled = self.query_one("#manipulation_enabled", Checkbox).value
            manip_fraction = float(self.query_one("#manip_fraction", Input).value or 0.2)
            manip_strategy = self.query_one("#manip_strategy", Select).value
            
            utility_func = self.query_one("#utility_func", Select).value
            distance_metric = self.query_one("#distance_metric", Select).value
            sigma_factor = float(self.query_one("#sigma_factor", Input).value or 0.5)
            
            seed_input = self.query_one("#rng_seed", Input).value
            rng_seed = int(seed_input) if seed_input else None
            
            # Collect heterogeneous distance settings
            het_enabled = self.query_one("#het_enabled", Checkbox).value
            het_strategy = self.query_one("#het_strategy", Select).value
            het_center_metric = self.query_one("#het_center_metric", Select).value
            het_extreme_metric = self.query_one("#het_extreme_metric", Select).value
            het_threshold = float(self.query_one("#het_threshold", Input).value or 0.5)
            het_radial_scaling = self.query_one("#het_radial_scaling", Select).value
            het_scaling_param = float(self.query_one("#het_scaling_param", Input).value or 2.0)
            
            # Build heterogeneous distance config
            het_config = HeterogeneousDistanceConfig(
                enabled=het_enabled,
                strategy=het_strategy,
                center_metric=het_center_metric,
                extreme_metric=het_extreme_metric,
                extreme_threshold=het_threshold,
                radial_scaling=het_radial_scaling,
                scaling_parameter=het_scaling_param,
                # Default radial metrics: l1 -> l2 -> chebyshev
                radial_metrics=['l1', 'l2', 'chebyshev']
            )
            
            # Build config
            config = SimulationConfig(
                n_profiles=n_profiles,
                n_voters=n_voters,
                n_candidates=n_candidates,
                voting_rules=selected_rules,
                geometry=GeometryConfig(method=geometry_method, n_dim=n_dim),
                manipulation=ManipulationConfig(
                    enabled=manipulation_enabled,
                    manipulator_fraction=manip_fraction,
                    strategy=manip_strategy
                ),
                utility=UtilityConfig(
                    function=utility_func,
                    distance_metric=distance_metric,
                    sigma_factor=sigma_factor,
                    heterogeneous_distance=het_config
                ),
                rng_seed=rng_seed
            )
            
            # Validate
            config.validate()
            
            # Push to simulation screen
            self.app.push_screen(SimulationRunScreen(config))
            
        except ValueError as e:
            self.notify(f"Invalid input: {e}", severity="error")
        except Exception as e:
            self.notify(f"Error: {e}", severity="error")


class QuickSimScreen(ModalScreen):
    """Modal for quick simulation with minimal configuration."""
    
    CSS = """
    QuickSimScreen {
        align: center middle;
    }
    
    #quick-dialog {
        width: 60;
        height: auto;
        border: thick $accent;
        background: $surface;
        padding: 2;
    }
    
    #quick-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    Input {
        margin: 1 0;
    }
    
    #quick-buttons {
        margin-top: 1;
    }
    
    #quick-buttons Button {
        margin: 0 1;
    }
    """
    
    def compose(self) -> ComposeResult:
        with Container(id="quick-dialog"):
            yield Label("⚡ Quick Simulation", id="quick-title")
            yield Label("Number of profiles:")
            yield Input(value="1000", type="integer", id="quick_profiles")
            yield Label("Number of voters:")
            yield Input(value="25", type="integer", id="quick_voters")
            yield Label("Number of candidates:")
            yield Input(value="3", type="integer", id="quick_candidates")
            
            with Horizontal(id="quick-buttons"):
                yield Button("Cancel", id="cancel", variant="default")
                yield Button("Run", id="run", variant="success")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.app.pop_screen()
        elif event.button.id == "run":
            try:
                n_profiles = int(self.query_one("#quick_profiles", Input).value or 1000)
                n_voters = int(self.query_one("#quick_voters", Input).value or 25)
                n_candidates = int(self.query_one("#quick_candidates", Input).value or 3)
                
                # Use default voting rules
                config = SimulationConfig(
                    n_profiles=n_profiles,
                    n_voters=n_voters,
                    n_candidates=n_candidates,
                    voting_rules=['plurality', 'borda', 'irv', 'approval', 'star', 'schulze'],
                )
                
                self.app.pop_screen()
                self.app.push_screen(SimulationRunScreen(config))
                
            except ValueError as e:
                self.notify(f"Invalid input: {e}", severity="error")


class SimulationRunScreen(Screen):
    """Screen that runs the simulation and shows progress."""
    
    CSS = """
    SimulationRunScreen {
        align: center middle;
    }
    
    #run-container {
        width: 80;
        height: auto;
        border: solid $primary;
        background: $surface;
        padding: 2;
    }
    
    #run-title {
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }
    
    #status {
        margin: 1 0;
        color: $text-muted;
    }
    
    ProgressBar {
        margin: 1 0;
    }
    
    #log-container {
        height: 15;
        border: solid $primary;
        margin: 1 0;
    }
    
    #config-display {
        margin: 1 0;
        color: $text-muted;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    
    def __init__(self, config: SimulationConfig):
        super().__init__()
        self.config = config
        self.result: Optional[ExperimentResult] = None
        self.worker: Optional[Worker] = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="run-container"):
            yield Label("⚙️ Running Simulation", id="run-title")
            
            # Configuration summary
            config_text = Text()
            config_text.append(f"Profiles: {self.config.n_profiles:,} | ")
            config_text.append(f"Voters: {self.config.n_voters} | ")
            config_text.append(f"Candidates: {self.config.n_candidates}\n")
            config_text.append(f"Rules: {', '.join(self.config.voting_rules)}\n")
            config_text.append(f"Geometry: {self.config.geometry.method} ({self.config.geometry.n_dim}D)")
            yield Static(config_text, id="config-display")
            
            yield Static("Initializing...", id="status")
            yield ProgressBar(total=100, show_eta=True)
            
            with Container(id="log-container"):
                yield Log(highlight=True, auto_scroll=True)
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Start the simulation when screen is mounted."""
        self.run_simulation()
    
    @work(exclusive=True, thread=True)
    def run_simulation(self) -> ExperimentResult:
        """Run the simulation in a background thread."""
        log_widget = self.query_one(Log)
        status_widget = self.query_one("#status", Static)
        progress = self.query_one(ProgressBar)
        
        # Update status via call_from_thread (use app.call_from_thread)
        def log(msg: str):
            self.app.call_from_thread(log_widget.write_line, msg)
        
        def update_status(msg: str):
            self.app.call_from_thread(status_widget.update, msg)
        
        def update_progress(value: int):
            self.app.call_from_thread(setattr, progress, "progress", value)
        
        try:
            log("Starting simulation...")
            update_status("Generating preferences...")
            update_progress(10)
            
            # Run simulation with verbose=False to avoid print conflicts
            result = run_experiment(
                self.config,
                save_results=True,
                verbose=False
            )
            
            update_progress(100)
            update_status("Simulation complete!")
            log(f"\n✓ Simulation completed in {result.total_compute_time:.2f}s")
            log(f"Results saved:")
            log(f"  Inputs: {result.inputs_path}")
            log(f"  Results: {result.results_path}")
            
            # Store result
            self.result = result
            
            # Navigate to results screen
            self.app.call_from_thread(self.show_results)
            
            return result
            
        except Exception as e:
            log(f"\n✗ Error: {e}")
            update_status(f"Error: {e}")
            import traceback
            log(traceback.format_exc())
            return None
    
    def show_results(self) -> None:
        """Show results screen."""
        if self.result:
            self.app.push_screen(ResultsScreen(self.result))
    
    def action_cancel(self) -> None:
        """Cancel the simulation."""
        if self.worker and self.worker.state == WorkerState.RUNNING:
            self.worker.cancel()
        self.app.pop_screen()


class ResultsScreen(Screen):
    """Screen to display simulation results."""
    
    CSS = """
    ResultsScreen {
        layout: vertical;
    }
    
    #results-header {
        dock: top;
        height: auto;
        background: $surface;
        border-bottom: solid $primary;
        padding: 1;
    }
    
    #results-title {
        text-style: bold;
        color: $accent;
    }
    
    #results-summary {
        color: $text-muted;
        margin-top: 1;
    }
    
    DataTable {
        height: 100%;
    }
    
    #button-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        border-top: solid $primary;
    }
    
    #button-bar Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("d", "details", "Details"),
    ]
    
    def __init__(self, result: ExperimentResult):
        super().__init__()
        self.result = result
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="results-header"):
            yield Label("📊 Simulation Results", id="results-title")
            
            summary = Text()
            summary.append(f"Profiles: {self.result.config.n_profiles:,} | ")
            summary.append(f"Voters: {self.result.config.n_voters} | ")
            summary.append(f"Candidates: {self.result.config.n_candidates}\n")
            summary.append(f"Time: {self.result.total_compute_time:.2f}s | ")
            summary.append(f"Geometry: {self.result.config.geometry.method}")
            yield Static(summary, id="results-summary")
        
        # Create data table
        table = DataTable()
        table.cursor_type = "row"
        table.zebra_stripes = True
        
        # Add columns
        table.add_column("Rule", key="rule")
        table.add_column("Type", key="type")
        table.add_column("VSE", key="vse")
        table.add_column("VSE σ", key="vse_std")
        table.add_column("1st %", key="first_pct")
        table.add_column("Cycles %", key="cycles")
        table.add_column("CW Eff %", key="cw_eff")
        
        # Add rows
        for rule_name, rule_result in self.result.rule_results.items():
            agg = rule_result.aggregate_metrics
            rule_type = AVAILABLE_VOTING_RULES[rule_name]['type'][:4]
            
            table.add_row(
                rule_name,
                rule_type,
                f"{agg.vse_mean:.3f}",
                f"{agg.vse_std:.3f}",
                f"{agg.winner_rank_1st_pct:.1f}",
                f"{agg.cycle_percentage:.1f}",
                f"{agg.condorcet_efficiency:.1f}",
                key=rule_name
            )
        
        yield table
        
        with Horizontal(id="button-bar"):
            yield Button("◀ Back to Menu", id="back", variant="default")
            yield Button("📈 Detailed View", id="details", variant="primary")
        
        yield Footer()
    
    def action_back(self) -> None:
        # Pop all screens back to welcome
        while len(self.app.screen_stack) > 1:
            self.app.pop_screen()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "details":
            self.action_details()
    
    def action_details(self) -> None:
        """Show detailed results view."""
        self.app.push_screen(DetailedResultsScreen(self.result))


class DetailedResultsScreen(Screen):
    """Detailed results with tabs for different metrics."""
    
    CSS = """
    DetailedResultsScreen {
        layout: vertical;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    
    .metric-container {
        height: 100%;
        overflow-y: auto;
    }
    """
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
    ]
    
    def __init__(self, result: ExperimentResult):
        super().__init__()
        self.result = result
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with TabbedContent():
            # Summary tab
            with TabPane("Summary", id="tab-summary"):
                with VerticalScroll(classes="metric-container"):
                    yield self._create_summary_view()
            
            # Per-rule tabs
            for rule_name, rule_result in self.result.rule_results.items():
                with TabPane(rule_name, id=f"tab-{rule_name}"):
                    with VerticalScroll(classes="metric-container"):
                        yield self._create_rule_view(rule_name, rule_result)
            
            # Configuration tab
            with TabPane("Config", id="tab-config"):
                with VerticalScroll(classes="metric-container"):
                    yield Pretty(self.result.config.to_dict())
        
        yield Footer()
    
    def _create_summary_view(self) -> Container:
        """Create summary statistics view."""
        container = Container()
        
        # Create rich table
        table = RichTable(title="Aggregate Metrics", show_header=True)
        table.add_column("Rule", style="cyan")
        table.add_column("VSE Mean", justify="right")
        table.add_column("VSE Std", justify="right")
        table.add_column("1st Place %", justify="right")
        table.add_column("Cycles %", justify="right")
        table.add_column("CW Efficiency %", justify="right")
        
        for rule_name, rule_result in self.result.rule_results.items():
            agg = rule_result.aggregate_metrics
            table.add_row(
                rule_name,
                f"{agg.vse_mean:.4f}",
                f"{agg.vse_std:.4f}",
                f"{agg.winner_rank_1st_pct:.2f}",
                f"{agg.cycle_percentage:.2f}",
                f"{agg.condorcet_efficiency:.2f}"
            )
        
        return Static(table)
    
    def _create_rule_view(self, rule_name: str, rule_result) -> Container:
        """Create detailed view for a single rule."""
        container = Container()
        
        agg = rule_result.aggregate_metrics
        
        # Create metrics display
        info = Text()
        info.append(f"Voting Rule: {rule_name}\n\n", style="bold cyan")
        info.append("Performance Metrics:\n", style="bold yellow")
        info.append(f"  VSE Mean: {agg.vse_mean:.4f}\n")
        info.append(f"  VSE Std: {agg.vse_std:.4f}\n")
        info.append(f"  VSE Min: {agg.vse_min:.4f}\n")
        info.append(f"  VSE Max: {agg.vse_max:.4f}\n\n")
        
        info.append("Winner Statistics:\n", style="bold yellow")
        info.append(f"  1st Place: {agg.winner_rank_1st_pct:.2f}%\n")
        info.append(f"  2nd Place: {agg.winner_rank_2nd_pct:.2f}%\n")
        info.append(f"  3rd Place: {agg.winner_rank_3rd_pct:.2f}%\n")
        info.append(f"  4th+ Place: {agg.winner_rank_4th_plus_pct:.2f}%\n\n")
        
        info.append("Condorcet Properties:\n", style="bold yellow")
        info.append(f"  CW Exists: {agg.condorcet_winner_exists_pct:.2f}%\n")
        info.append(f"  CW Efficiency: {agg.condorcet_efficiency:.2f}%\n")
        info.append(f"  Cycles: {agg.cycle_percentage:.2f}%\n\n")
        
        info.append(f"Compute Time: {rule_result.compute_time:.3f}s\n", style="dim")
        
        return Static(info)
    
    def action_back(self) -> None:
        self.app.pop_screen()


class SavedExperimentsScreen(Screen):
    """Screen to browse saved experiments."""
    
    CSS = """
    SavedExperimentsScreen {
        layout: vertical;
    }
    
    #saved-header {
        dock: top;
        height: 3;
        background: $surface;
        border-bottom: solid $primary;
        padding: 1;
    }
    
    #saved-title {
        text-style: bold;
        color: $accent;
    }
    
    DataTable {
        height: 100%;
    }
    
    #button-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        border-top: solid $primary;
    }
    
    #button-bar Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("l", "load", "Load"),
        Binding("r", "refresh", "Refresh"),
    ]
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="saved-header"):
            yield Label("📁 Saved Experiments", id="saved-title")
        
        # Create data table
        table = DataTable()
        table.cursor_type = "row"
        table.zebra_stripes = True
        
        # Add columns
        table.add_column("#", key="num")
        table.add_column("Filename", key="filename")
        table.add_column("Profiles", key="profiles")
        table.add_column("Voters", key="voters")
        table.add_column("Candidates", key="candidates")
        table.add_column("ID", key="id")
        
        yield table
        
        with Horizontal(id="button-bar"):
            yield Button("◀ Back", id="back", variant="default")
            yield Button("🔄 Refresh", id="refresh", variant="primary")
            yield Button("📂 Load Selected", id="load", variant="success")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Load experiments when screen is mounted."""
        self.load_experiments()
    
    def load_experiments(self) -> None:
        """Load and display saved experiments."""
        table = self.query_one(DataTable)
        table.clear()
        
        experiments = list_experiments()
        
        if not experiments:
            self.notify("No saved experiments found", severity="warning")
            return
        
        for i, exp in enumerate(experiments, 1):
            table.add_row(
                str(i),
                exp.get('filename', 'N/A')[:40],
                str(exp.get('n_profiles', 'N/A')),
                str(exp.get('n_voters', 'N/A')),
                str(exp.get('n_candidates', 'N/A')),
                exp.get('experiment_id', 'N/A')[:8],
                key=exp.get('filename', str(i))
            )
    
    def action_back(self) -> None:
        self.app.pop_screen()
    
    def action_refresh(self) -> None:
        self.load_experiments()
        self.notify("Experiments list refreshed")
    
    def action_load(self) -> None:
        """Load selected experiment."""
        table = self.query_one(DataTable)
        if table.cursor_row is not None:
            row_key = table.get_row_at(table.cursor_row)
            self.notify(f"Loading experiment: {row_key[1]}", severity="information")
            # TODO: Implement loading experiment details
        else:
            self.notify("No experiment selected", severity="warning")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "refresh":
            self.action_refresh()
        elif event.button.id == "load":
            self.action_load()


class ComparisonSelectScreen(Screen):
    """Screen to select experiments for comparison."""
    
    CSS = """
    ComparisonSelectScreen {
        layout: vertical;
    }
    
    #compare-header {
        dock: top;
        height: auto;
        background: $surface;
        border-bottom: solid $primary;
        padding: 1;
    }
    
    #compare-title {
        text-style: bold;
        color: $accent;
    }
    
    #compare-help {
        color: $text-muted;
        margin-top: 1;
    }
    
    #exp-list-container {
        height: 1fr;
        padding: 1;
    }
    
    SelectionList {
        height: 100%;
        border: solid $primary;
    }
    
    #options-container {
        height: auto;
        padding: 1;
        background: $surface;
    }
    
    #mode-select {
        width: 50;
    }
    
    #format-select {
        width: 30;
    }
    
    #button-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        border-top: solid $primary;
    }
    
    #button-bar Button {
        margin: 0 1;
    }
    
    #selection-count {
        color: $text-muted;
        margin-left: 2;
    }
    """
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("enter", "compare", "Compare"),
        Binding("a", "select_all", "Select All"),
    ]
    
    def __init__(self):
        super().__init__()
        self.experiments: List[Dict] = []
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="compare-header"):
            yield Label("📈 Compare Experiments", id="compare-title")
            yield Static(
                "Select 2 or more experiments to compare. Use SPACE to toggle selection.",
                id="compare-help"
            )
        
        with Container(id="exp-list-container"):
            # Use SelectionList for multi-select
            yield SelectionList[str](id="exp-selection")
        
        with Container(id="options-container"):
            with Horizontal():
                yield Label("Mode: ")
                yield Select[str](
                    [(desc, name) for name, desc in get_comparison_modes().items()],
                    value="by_experiment",
                    id="mode-select"
                )
                yield Label("  Format: ")
                yield Select[str](
                    [("Text", "text"), ("Markdown", "markdown"), ("CSV", "csv")],
                    value="text",
                    id="format-select"
                )
                yield Static("Selected: 0", id="selection-count")
        
        with Horizontal(id="button-bar"):
            yield Button("◀ Back", id="back", variant="default")
            yield Button("Select All", id="select-all", variant="primary")
            yield Button("Deselect All", id="deselect-all", variant="default")
            yield Button("🔄 Refresh", id="refresh", variant="default")
            yield Button("📊 Compare", id="compare", variant="success")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Load experiments when screen is mounted."""
        self.load_experiments()
    
    def load_experiments(self) -> None:
        """Load and display saved experiments."""
        selection_list = self.query_one("#exp-selection", SelectionList)
        selection_list.clear_options()
        
        self.experiments = list_experiments()
        
        if not self.experiments:
            self.notify("No saved experiments found", severity="warning")
            return
        
        for exp in self.experiments:
            filename = exp.get('filename', 'unknown')
            n_profiles = exp.get('n_profiles', '?')
            n_voters = exp.get('n_voters', '?')
            exp_id = exp.get('experiment_id', 'unknown')[:8]
            
            label = f"{filename} | {n_profiles}p x {n_voters}v | {exp_id}"
            selection_list.add_option((label, filename))
        
        self.update_selection_count()
    
    def update_selection_count(self) -> None:
        """Update the selection count display."""
        selection_list = self.query_one("#exp-selection", SelectionList)
        count = len(selection_list.selected)
        count_label = self.query_one("#selection-count", Static)
        count_label.update(f"Selected: {count}")
    
    def on_selection_list_selection_toggled(self, event) -> None:
        """Handle selection change."""
        self.update_selection_count()
    
    def action_back(self) -> None:
        self.app.pop_screen()
    
    def action_select_all(self) -> None:
        """Select all experiments."""
        selection_list = self.query_one("#exp-selection", SelectionList)
        selection_list.select_all()
        self.update_selection_count()
    
    def action_compare(self) -> None:
        """Run comparison on selected experiments."""
        selection_list = self.query_one("#exp-selection", SelectionList)
        selected_values = list(selection_list.selected)
        
        if len(selected_values) < 2:
            self.notify("Please select at least 2 experiments to compare", severity="warning")
            return
        
        # Get selected experiment paths
        selected_paths = []
        for exp in self.experiments:
            if exp.get('filename') in selected_values:
                selected_paths.append(exp['inputs_path'])
        
        # Get comparison options
        mode = self.query_one("#mode-select", Select).value
        output_format = self.query_one("#format-select", Select).value
        
        # Navigate to comparison results screen
        self.app.push_screen(ComparisonResultsScreen(
            selected_paths=selected_paths,
            mode=mode or "by_experiment",
            output_format=output_format or "text"
        ))
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "select-all":
            self.action_select_all()
        elif event.button.id == "deselect-all":
            selection_list = self.query_one("#exp-selection", SelectionList)
            selection_list.deselect_all()
            self.update_selection_count()
        elif event.button.id == "refresh":
            self.load_experiments()
            self.notify("Experiments refreshed")
        elif event.button.id == "compare":
            self.action_compare()


class ComparisonResultsScreen(Screen):
    """Screen to display comparison results."""
    
    CSS = """
    ComparisonResultsScreen {
        layout: vertical;
    }
    
    #compare-results-header {
        dock: top;
        height: auto;
        background: $surface;
        border-bottom: solid $primary;
        padding: 1;
    }
    
    #results-title {
        text-style: bold;
        color: $accent;
    }
    
    #results-subtitle {
        color: $text-muted;
    }
    
    #results-container {
        height: 1fr;
        padding: 1;
    }
    
    TabbedContent {
        height: 100%;
    }
    
    TabPane {
        padding: 1;
    }
    
    .comparison-table {
        height: 100%;
    }
    
    DataTable {
        height: 100%;
    }
    
    #text-output {
        height: 100%;
        border: solid $primary;
        padding: 1;
        overflow-y: auto;
    }
    
    #button-bar {
        dock: bottom;
        height: 3;
        background: $surface;
        border-top: solid $primary;
    }
    
    #button-bar Button {
        margin: 0 1;
    }
    """
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("c", "copy", "Copy"),
    ]
    
    def __init__(
        self, 
        selected_paths: List[str], 
        mode: str = "by_experiment",
        output_format: str = "text"
    ):
        super().__init__()
        self.selected_paths = selected_paths
        self.mode = mode
        self.output_format = output_format
        self.comparator: Optional[ExperimentComparator] = None
        self.result: Optional[ComparisonResult] = None
        self.formatted_output: str = ""
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Container(id="compare-results-header"):
            yield Label("📊 Comparison Results", id="results-title")
            yield Static(
                f"Comparing {len(self.selected_paths)} experiments | Mode: {self.mode}",
                id="results-subtitle"
            )
        
        with Container(id="results-container"):
            with TabbedContent():
                with TabPane("Table View", id="tab-table"):
                    yield DataTable(id="comparison-table", classes="comparison-table")
                
                with TabPane("Text Output", id="tab-text"):
                    yield VerticalScroll(Static("Loading...", id="text-output"))
                
                with TabPane("By Experiment", id="tab-by-exp"):
                    yield DataTable(id="exp-table", classes="comparison-table")
                
                with TabPane("By Rule", id="tab-by-rule"):
                    yield DataTable(id="rule-table", classes="comparison-table")
                
                with TabPane("Differential", id="tab-diff"):
                    yield VerticalScroll(Static("Loading...", id="diff-output"))
        
        with Horizontal(id="button-bar"):
            yield Button("◀ Back", id="back", variant="default")
            yield Button("📋 Copy to Clipboard", id="copy", variant="primary")
            yield Button("💾 Export CSV", id="export-csv", variant="default")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Load comparison data when screen is mounted."""
        self.run_comparison()
    
    @work(exclusive=True, thread=True)
    def run_comparison(self) -> None:
        """Run comparison in background thread."""
        try:
            # Create comparator
            self.comparator = ExperimentComparator.from_paths(self.selected_paths)
            
            # Run comparison for all modes
            self.result = self.comparator.compare(mode=self.mode)
            
            # Get formatted output
            self.formatted_output = self.comparator.format_table(
                self.result, format=self.output_format
            )
            
            # Update UI from main thread
            self.app.call_from_thread(self.populate_tables)
            
        except Exception as e:
            self.app.call_from_thread(
                self.notify, f"Error comparing experiments: {e}", severity="error"
            )
    
    def populate_tables(self) -> None:
        """Populate all comparison tables."""
        if not self.comparator:
            return
        
        # Populate main table based on mode
        main_table = self.query_one("#comparison-table", DataTable)
        self._populate_table_from_mode(main_table, self.mode)
        
        # Populate text output
        text_output = self.query_one("#text-output", Static)
        text_output.update(self.formatted_output)
        
        # Populate by_experiment table
        exp_table = self.query_one("#exp-table", DataTable)
        self._populate_table_from_mode(exp_table, "by_experiment")
        
        # Populate by_rule table  
        rule_table = self.query_one("#rule-table", DataTable)
        self._populate_table_from_mode(rule_table, "by_rule")
        
        # Populate differential view
        diff_output = self.query_one("#diff-output", Static)
        diff_result = self.comparator.compare(mode="differential")
        diff_text = self.comparator.format_table(diff_result, format="text")
        diff_output.update(diff_text)
    
    def _populate_table_from_mode(self, table: DataTable, mode: str) -> None:
        """Populate a DataTable from comparison data."""
        table.clear(columns=True)
        
        result = self.comparator.compare(mode=mode)
        
        if mode == "by_experiment":
            data = result.by_experiment
        elif mode == "by_rule":
            data = result.by_rule
        elif mode == "by_metric":
            data = result.by_metric
        else:
            return
        
        if not data:
            return
        
        # Get all columns
        all_cols = set()
        for row in data.values():
            all_cols.update(row.keys())
        cols = ["Experiment"] + sorted(all_cols)
        
        # Add columns
        for col in cols:
            table.add_column(col, key=col)
        
        # Add rows
        for exp_name, row_data in data.items():
            row_values = [exp_name]
            for col in cols[1:]:
                val = row_data.get(col, "")
                if isinstance(val, float):
                    row_values.append(f"{val:.4f}")
                else:
                    row_values.append(str(val) if val else "-")
            table.add_row(*row_values, key=exp_name)
    
    def action_back(self) -> None:
        self.app.pop_screen()
    
    def action_copy(self) -> None:
        """Copy formatted output to clipboard."""
        # Note: Clipboard access depends on terminal capabilities
        self.notify(f"Output ({len(self.formatted_output)} chars) ready to copy", severity="information")
    
    def export_csv(self) -> None:
        """Export comparison as CSV."""
        if self.comparator and self.result:
            csv_output = self.comparator.format_table(self.result, format="csv")
            # Save to file
            import datetime
            filename = f"comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            with open(filename, 'w') as f:
                f.write(csv_output)
            self.notify(f"Exported to {filename}", severity="information")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "back":
            self.action_back()
        elif event.button.id == "copy":
            self.action_copy()
        elif event.button.id == "export-csv":
            self.export_csv()


class VotingSimulatorApp(App):
    """Main Textual application for voting simulator."""
    
    CSS = """
    Screen {
        background: $background;
    }
    """
    
    TITLE = "Spatial Voting Simulator"
    SUB_TITLE = "by Social Choice Analysis"
    
    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("d", "toggle_dark", "Toggle Dark Mode"),
    ]
    
    def on_mount(self) -> None:
        """Initialize the app."""
        self.theme = "textual-dark"
        self.push_screen(WelcomeScreen())


def main():
    """Entry point for the GUI application."""
    app = VotingSimulatorApp()
    app.run()


if __name__ == "__main__":
    main()

