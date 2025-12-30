"""
Storage module for simulation data.

Handles saving and loading of:
- Input data (NPZ format): positions, utilities, parameters
- Results data (Parquet format): per-profile metrics and analysis
"""

import numpy as np
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import asdict
import json

from .config import SimulationConfig
from .metrics import ProfileMetrics, AggregateMetrics


def get_timestamp() -> str:
    """Get formatted timestamp for filenames."""
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


def get_storage_paths(
    base_dir: Optional[str] = None,
    experiment_id: Optional[str] = None,
    timestamp: Optional[str] = None
) -> tuple:
    """
    Get paths for input and result files.
    
    Args:
        base_dir: Base directory for simulator (default: module directory)
        experiment_id: Unique experiment identifier
        timestamp: Timestamp string (default: current time)
        
    Returns:
        Tuple of (inputs_path, results_path)
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    else:
        base_dir = Path(base_dir)
    
    if timestamp is None:
        timestamp = get_timestamp()
    
    # Create unique filename
    if experiment_id:
        filename_base = f"{timestamp}_{experiment_id}"
    else:
        filename_base = timestamp
    
    inputs_dir = base_dir / "inputs"
    results_dir = base_dir / "results"
    
    # Ensure directories exist
    inputs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    inputs_path = inputs_dir / f"{filename_base}.npz"
    results_path = results_dir / f"{filename_base}.parquet"
    
    return str(inputs_path), str(results_path)


class InputStorage:
    """
    Handles storage of simulation inputs in NPZ format.
    
    Stores:
    - Voter positions (n_profiles, n_voters, n_dim)
    - Candidate positions (n_profiles, n_candidates, n_dim)
    - Utilities (n_profiles, n_voters, n_candidates)
    - Rankings (n_profiles, n_voters, n_candidates)
    - Configuration parameters
    """
    
    @staticmethod
    def save(
        filepath: str,
        voter_positions: np.ndarray,
        candidate_positions: np.ndarray,
        utilities: np.ndarray,
        rankings: np.ndarray,
        config: SimulationConfig,
        manipulated_rankings: Optional[np.ndarray] = None,
        manipulated_utilities: Optional[np.ndarray] = None,
        manipulator_masks: Optional[np.ndarray] = None,
        extra_data: Optional[Dict[str, np.ndarray]] = None
    ) -> str:
        """
        Save simulation inputs to NPZ file.
        
        Args:
            filepath: Path to save file
            voter_positions: Voter position array
            candidate_positions: Candidate position array
            utilities: Utility matrix
            rankings: Rankings matrix
            config: Simulation configuration
            manipulated_rankings: Rankings after manipulation
            manipulated_utilities: Utilities after manipulation
            manipulator_masks: Boolean masks of manipulators
            extra_data: Additional numpy arrays to store
            
        Returns:
            Path to saved file
        """
        # Prepare data dictionary
        data = {
            'voter_positions': voter_positions,
            'candidate_positions': candidate_positions,
            'utilities': utilities,
            'rankings': rankings,
        }
        
        # Add manipulation data if present
        if manipulated_rankings is not None:
            data['manipulated_rankings'] = manipulated_rankings
        if manipulated_utilities is not None:
            data['manipulated_utilities'] = manipulated_utilities
        if manipulator_masks is not None:
            data['manipulator_masks'] = manipulator_masks
        
        # Add extra data
        if extra_data:
            for key, value in extra_data.items():
                if not key.startswith('_'):  # Avoid conflicts with metadata
                    data[key] = value
        
        # Serialize config to JSON string and store as array
        config_json = json.dumps(config.to_dict())
        data['_config_json'] = np.array([config_json], dtype=object)
        
        # Store metadata
        metadata = {
            'experiment_id': config.experiment_id,
            'created_at': config.created_at,
            'n_profiles': config.n_profiles,
            'n_voters': config.n_voters,
            'n_candidates': config.n_candidates,
            'rng_seed': config.rng_seed if config.rng_seed else -1,
        }
        data['_metadata'] = np.array([json.dumps(metadata)], dtype=object)
        
        # Save
        np.savez_compressed(filepath, **data)
        
        return filepath
    
    @staticmethod
    def load(filepath: str) -> Dict[str, Any]:
        """
        Load simulation inputs from NPZ file.
        
        Args:
            filepath: Path to NPZ file
            
        Returns:
            Dictionary with loaded data and config
        """
        with np.load(filepath, allow_pickle=True) as data:
            result = {}
            
            # Load arrays
            for key in data.files:
                if not key.startswith('_'):
                    result[key] = data[key]
            
            # Load config
            if '_config_json' in data.files:
                config_json = str(data['_config_json'][0])
                result['config'] = json.loads(config_json)
            
            # Load metadata
            if '_metadata' in data.files:
                metadata_json = str(data['_metadata'][0])
                result['metadata'] = json.loads(metadata_json)
        
        return result


class ResultStorage:
    """
    Handles storage of simulation results in Parquet or JSON format.
    
    Each row represents one election profile with its metrics.
    Falls back to JSON if pandas is not available.
    """
    
    @staticmethod
    def save(
        filepath: str,
        profile_results: List[Dict[str, Any]],
        config: SimulationConfig,
        aggregate_metrics: Optional[AggregateMetrics] = None,
        compute_time: Optional[float] = None
    ) -> str:
        """
        Save simulation results to Parquet file (or JSON fallback).
        
        Args:
            filepath: Path to save file
            profile_results: List of per-profile result dictionaries
            config: Simulation configuration
            aggregate_metrics: Aggregated metrics across profiles
            compute_time: Total computation time in seconds
            
        Returns:
            Path to saved file
        """
        # Prepare data
        config_dict = config.to_dict()
        agg_dict = asdict(aggregate_metrics) if aggregate_metrics else {}
        
        try:
            import pandas as pd
            
            # Build dataframe from profile results
            df = pd.DataFrame(profile_results)
            
            # Add config columns
            for key, value in config_dict.items():
                if isinstance(value, list):
                    df[f'config_{key}'] = [value] * len(df)
                else:
                    df[f'config_{key}'] = value
            
            # Add aggregate metrics if provided
            for key, value in agg_dict.items():
                df[f'agg_{key}'] = value
            
            # Add compute time
            if compute_time is not None:
                df['compute_time_seconds'] = compute_time
            
            # Add timestamp
            df['saved_at'] = datetime.now().isoformat()
            
            # Save to Parquet
            try:
                df.to_parquet(filepath, index=False, engine='pyarrow')
                return filepath
            except ImportError:
                try:
                    df.to_parquet(filepath, index=False, engine='fastparquet')
                    return filepath
                except ImportError:
                    # Fallback to CSV
                    csv_path = filepath.replace('.parquet', '.csv')
                    df.to_csv(csv_path, index=False)
                    return csv_path
        
        except ImportError:
            # No pandas - save as JSON
            json_path = filepath.replace('.parquet', '.json')
            
            # Prepare JSON-serializable data
            data = {
                'config': config_dict,
                'aggregate_metrics': agg_dict,
                'compute_time_seconds': compute_time,
                'saved_at': datetime.now().isoformat(),
                'profiles': profile_results,
            }
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            return json_path
    
    @staticmethod
    def load(filepath: str):
        """
        Load simulation results from file.
        
        Args:
            filepath: Path to Parquet/JSON/CSV file
            
        Returns:
            pandas DataFrame (if available) or dictionary
        """
        # Check for JSON fallback
        json_path = filepath.replace('.parquet', '.json')
        if os.path.exists(json_path) and not os.path.exists(filepath):
            filepath = json_path
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        
        try:
            import pandas as pd
            return pd.read_parquet(filepath)
        except ImportError:
            # Try CSV
            csv_path = filepath.replace('.parquet', '.csv')
            if os.path.exists(csv_path):
                # Without pandas, return as dict
                import csv
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    return list(reader)
            raise ImportError(
                "pandas is required for Parquet loading. "
                "Install with: pip install pandas pyarrow"
            )
    
    @staticmethod
    def load_as_dict(filepath: str) -> Dict[str, Any]:
        """
        Load results as dictionary.
        
        Args:
            filepath: Path to file
            
        Returns:
            Dictionary representation of results
        """
        # Handle different file types
        if filepath.endswith('.json'):
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    return json.load(f)
        
        if filepath.endswith('.csv'):
            try:
                import pandas as pd
                if os.path.exists(filepath):
                    df = pd.read_csv(filepath)
                    return df.to_dict(orient='list')
            except ImportError:
                # Fallback to csv module
                import csv
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Convert to dict of lists
                        result = {key: [] for key in rows[0].keys()}
                        for row in rows:
                            for key, value in row.items():
                                result[key].append(value)
                        return result
                return {}
        
        # Check for JSON fallback first
        json_path = filepath.replace('.parquet', '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                return json.load(f)
        
        try:
            import pandas as pd
            if os.path.exists(filepath):
                df = pd.read_parquet(filepath)
                return df.to_dict(orient='list')
            
            csv_path = filepath.replace('.parquet', '.csv')
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                return df.to_dict(orient='list')
        except ImportError:
            pass
        
        raise FileNotFoundError(f"No results file found: {filepath}")


def save_experiment(
    config: SimulationConfig,
    voter_positions: np.ndarray,
    candidate_positions: np.ndarray,
    utilities: np.ndarray,
    rankings: np.ndarray,
    profile_results: List[Dict[str, Any]],
    aggregate_metrics: AggregateMetrics,
    compute_time: float,
    base_dir: Optional[str] = None,
    manipulated_rankings: Optional[np.ndarray] = None,
    manipulated_utilities: Optional[np.ndarray] = None,
    manipulator_masks: Optional[np.ndarray] = None
) -> tuple:
    """
    Save complete experiment data.
    
    Convenience function that saves both inputs and results.
    
    Args:
        config: Simulation configuration
        voter_positions: Voter positions
        candidate_positions: Candidate positions
        utilities: Utility matrix
        rankings: Rankings matrix
        profile_results: Per-profile result dictionaries
        aggregate_metrics: Aggregated metrics
        compute_time: Computation time
        base_dir: Base directory for storage
        manipulated_rankings: Optional manipulated rankings
        manipulated_utilities: Optional manipulated utilities
        manipulator_masks: Optional manipulator masks
        
    Returns:
        Tuple of (inputs_path, actual_results_path)
    """
    # Get paths
    inputs_path, results_path = get_storage_paths(
        base_dir=base_dir,
        experiment_id=config.experiment_id,
        timestamp=config.created_at.replace(':', '-').replace('T', '_')[:19]
    )
    
    # Save inputs
    InputStorage.save(
        inputs_path,
        voter_positions,
        candidate_positions,
        utilities,
        rankings,
        config,
        manipulated_rankings=manipulated_rankings,
        manipulated_utilities=manipulated_utilities,
        manipulator_masks=manipulator_masks
    )
    
    # Save results - returns actual path (may be different format)
    actual_results_path = ResultStorage.save(
        results_path,
        profile_results,
        config,
        aggregate_metrics=aggregate_metrics,
        compute_time=compute_time
    )
    
    return inputs_path, actual_results_path


def load_experiment(
    inputs_path: str,
    results_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load complete experiment data.
    
    Args:
        inputs_path: Path to inputs NPZ file
        results_path: Path to results Parquet file (optional)
        
    Returns:
        Dictionary with all experiment data
    """
    # Load inputs
    data = InputStorage.load(inputs_path)
    
    # Load results if path provided
    if results_path and os.path.exists(results_path):
        data['results'] = ResultStorage.load(results_path)
    
    return data


def list_experiments(base_dir: Optional[str] = None) -> List[Dict[str, str]]:
    """
    List all saved experiments.
    
    Args:
        base_dir: Base directory for simulator
        
    Returns:
        List of experiment info dictionaries
    """
    if base_dir is None:
        base_dir = Path(__file__).parent
    else:
        base_dir = Path(base_dir)
    
    inputs_dir = base_dir / "inputs"
    results_dir = base_dir / "results"
    
    experiments = []
    
    if inputs_dir.exists():
        for npz_file in sorted(inputs_dir.glob("*.npz")):
            experiment = {
                'inputs_path': str(npz_file),
                'filename': npz_file.stem,
            }
            
            # Check for matching results file
            parquet_file = results_dir / f"{npz_file.stem}.parquet"
            if parquet_file.exists():
                experiment['results_path'] = str(parquet_file)
            
            # Try to extract metadata
            try:
                data = InputStorage.load(str(npz_file))
                if 'metadata' in data:
                    experiment.update(data['metadata'])
            except Exception:
                pass
            
            experiments.append(experiment)
    
    return experiments

