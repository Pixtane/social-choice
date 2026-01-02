"""
Fix serialization issues in JSON result files.

Converts numpy types and other non-serializable objects to native Python types.
"""

import json
import numpy as np
from pathlib import Path
from typing import Any


def make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types and other non-serializable objects."""
    # Check for numpy types
    if hasattr(obj, 'dtype'):  # numpy array or scalar
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif np.issubdtype(type(obj), np.integer):
            return int(obj)
        elif np.issubdtype(type(obj), np.floating):
            return float(obj)
        elif np.issubdtype(type(obj), np.bool_):
            return bool(obj)
    
    # Handle standard types
    if isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, set):
        return list(obj)
    else:
        return obj


def fix_json_file(filepath: Path):
    """Fix serialization issues in a JSON file."""
    print(f"Processing {filepath.name}...")
    
    try:
        # Try to load
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert to serializable
        data_fixed = make_serializable(data)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data_fixed, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] Fixed {filepath.name}")
        return True
    except json.JSONDecodeError as e:
        print(f"  [ERROR] JSON decode error in {filepath.name}: {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] Error processing {filepath.name}: {e}")
        return False


def main():
    results_dir = Path("heterogenity-simulator/results")
    
    json_files = list(results_dir.glob("*.json"))
    print(f"Found {len(json_files)} JSON files to process\n")
    
    fixed_count = 0
    for filepath in json_files:
        if fix_json_file(filepath):
            fixed_count += 1
    
    print(f"\nFixed {fixed_count}/{len(json_files)} files")


if __name__ == "__main__":
    main()

