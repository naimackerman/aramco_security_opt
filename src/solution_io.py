"""
Solution I/O Module for Saving and Loading Optimization Results.

Enables saving solutions from Exact and Heuristic methods along with
the problem data, allowing visualization without re-running solvers.

Usage:
    # Saving:
    from src.solution_io import save_solution
    save_solution(data, solution, "Balanced", "exact")
    
    # Loading:
    from src.solution_io import load_solution, list_saved_solutions
    solutions = list_saved_solutions()
    data, solution = load_solution("balanced_exact_20231231_143012.json")
    
    # Visualizing:
    python -m src.solution_io --list
    python -m src.solution_io --load balanced_exact_20231231_143012.json
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from .config import RESULTS_DIR

SOLUTIONS_DIR = RESULTS_DIR / "saved_solutions"
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    """JSON Encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _serialize_data(data: Dict) -> Dict:
    """
    Serialize problem data for JSON storage.
    
    Handles nested numpy arrays and special dictionary structures.
    """
    serialized = {}
    
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, dict):
            # Handle nested dicts like t_ijl, F_il, C_ik, etc.
            nested = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    nested[k] = v.tolist()
                elif isinstance(v, dict):
                    # Double nested (e.g., MAXCAP_lk)
                    nested[k] = {kk: vv.tolist() if isinstance(vv, np.ndarray) else vv 
                                for kk, vv in v.items()}
                else:
                    nested[k] = v
            serialized[key] = nested
        elif isinstance(value, (list, tuple)):
            # Handle list of tuples (coordinates)
            serialized[key] = [list(item) if isinstance(item, tuple) else item 
                              for item in value]
        else:
            serialized[key] = value
    
    return serialized


def _deserialize_data(data: Dict) -> Dict:
    """
    Deserialize problem data from JSON storage.
    
    Converts lists back to numpy arrays where appropriate.
    """
    deserialized = {}
    
    # Keys that should be numpy arrays
    numpy_keys = {'S_j', 'D_j', 'alpha_j', 'd_ij'}
    
    # Keys that are dicts with numpy array values
    dict_numpy_keys = {'t_ijl', 'F_il', 'C_ik'}
    
    for key, value in data.items():
        if key in numpy_keys and isinstance(value, list):
            deserialized[key] = np.array(value)
        elif key in dict_numpy_keys and isinstance(value, dict):
            nested = {}
            for k, v in value.items():
                if isinstance(v, list):
                    nested[k] = np.array(v)
                else:
                    nested[k] = v
            deserialized[key] = nested
        elif key == 'd_ij' and isinstance(value, list):
            deserialized[key] = np.array(value)
        elif key in ('coords_I', 'coords_J') and isinstance(value, list):
            # Convert coordinate lists back to tuples
            deserialized[key] = [tuple(coord) for coord in value]
        else:
            deserialized[key] = value
    
    return deserialized


def _serialize_solution(solution: Dict) -> Dict:
    """
    Serialize solution for JSON storage.
    
    Handles:
    - opened: List of facility indices
    - levels: Dict mapping facility index to level
    - assignments: List of lists (per demand site)
    - resources: Dict mapping facility index to {robot, human}
    """
    if solution is None:
        return None
    
    serialized = {}
    
    for key, value in solution.items():
        if key == 'levels':
            # Convert integer keys to strings for JSON
            serialized[key] = {str(k): v for k, v in value.items()}
        elif key == 'resources':
            # Convert integer keys to strings for JSON
            serialized[key] = {str(k): v for k, v in value.items()}
        else:
            serialized[key] = value
    
    return serialized


def _deserialize_solution(solution: Dict) -> Dict:
    """
    Deserialize solution from JSON storage.
    
    Converts string keys back to integers.
    """
    if solution is None:
        return None
    
    deserialized = {}
    
    for key, value in solution.items():
        if key == 'levels' and isinstance(value, dict):
            # Convert string keys back to integers
            deserialized[key] = {int(k): v for k, v in value.items()}
        elif key == 'resources' and isinstance(value, dict):
            # Convert string keys back to integers
            deserialized[key] = {int(k): v for k, v in value.items()}
        else:
            deserialized[key] = value
    
    return deserialized


def save_solution(
    data: Dict,
    solution: Dict,
    scenario: str,
    method: str,
    cost: Optional[float] = None,
    solve_time: Optional[float] = None,
    custom_filename: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> Path:
    """
    Save a solution and its associated data to a JSON file.
    
    Args:
        data: Problem data dictionary from DataGenerator
        solution: Solution dictionary with opened, levels, assignments, resources
        scenario: Scenario name ('Conservative', 'Balanced', 'Future')
        method: Solution method ('exact', 'heuristic')
        cost: Optional objective value
        solve_time: Optional solve time in seconds
        custom_filename: Optional custom filename (without extension)
        metadata: Optional additional metadata to store
        
    Returns:
        Path to saved file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if custom_filename:
        filename = f"{custom_filename}.json"
    else:
        filename = f"{scenario.lower()}_{method.lower()}_{timestamp}.json"
    
    filepath = SOLUTIONS_DIR / filename
    
    # Build save object
    save_obj = {
        'metadata': {
            'scenario': scenario,
            'method': method,
            'cost': cost,
            'solve_time': solve_time,
            'saved_at': datetime.now().isoformat(),
            'num_facilities': data.get('num_I'),
            'num_demand_sites': data.get('num_J'),
            **(metadata or {})
        },
        'data': _serialize_data(data),
        'solution': _serialize_solution(solution)
    }
    
    with open(filepath, 'w') as f:
        json.dump(save_obj, f, indent=2, cls=NumpyEncoder)
    
    print(f"Solution saved to: {filepath}")
    return filepath


def load_solution(filename: str) -> Tuple[Dict, Dict, Dict]:
    """
    Load a solution and its associated data from a JSON file.
    
    Args:
        filename: Filename (with or without .json extension)
        
    Returns:
        Tuple of (data, solution, metadata)
    """
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    filepath = SOLUTIONS_DIR / filename
    
    if not filepath.exists():
        raise FileNotFoundError(f"Solution file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        save_obj = json.load(f)
    
    data = _deserialize_data(save_obj['data'])
    solution = _deserialize_solution(save_obj['solution'])
    metadata = save_obj.get('metadata', {})
    
    return data, solution, metadata


def list_saved_solutions() -> List[Dict]:
    """
    List all saved solutions with their metadata.
    
    Returns:
        List of dicts with filename and metadata for each saved solution
    """
    solutions = []
    
    for filepath in sorted(SOLUTIONS_DIR.glob("*.json")):
        try:
            with open(filepath, 'r') as f:
                save_obj = json.load(f)
            
            solutions.append({
                'filename': filepath.name,
                'scenario': save_obj.get('metadata', {}).get('scenario', 'Unknown'),
                'method': save_obj.get('metadata', {}).get('method', 'Unknown'),
                'cost': save_obj.get('metadata', {}).get('cost'),
                'saved_at': save_obj.get('metadata', {}).get('saved_at', 'Unknown'),
                'num_facilities': save_obj.get('metadata', {}).get('num_facilities'),
                'num_demand_sites': save_obj.get('metadata', {}).get('num_demand_sites'),
            })
        except Exception as e:
            print(f"Warning: Could not read {filepath.name}: {e}")
    
    return solutions


def delete_solution(filename: str) -> bool:
    """
    Delete a saved solution file.
    
    Args:
        filename: Filename (with or without .json extension)
        
    Returns:
        True if deleted, False if not found
    """
    if not filename.endswith('.json'):
        filename = f"{filename}.json"
    
    filepath = SOLUTIONS_DIR / filename
    
    if filepath.exists():
        filepath.unlink()
        print(f"Deleted: {filepath}")
        return True
    else:
        print(f"File not found: {filepath}")
        return False


def visualize_saved_solution(
    filename: str,
    save_format: str = "pdf",
    show: bool = False
) -> str:
    """
    Load and visualize a saved solution.
    
    Args:
        filename: Solution filename to load
        save_format: Output format ('pdf', 'png')
        show: If True, display the plot interactively
        
    Returns:
        Path to saved figure
    """
    from .visualization import plot_solution
    import matplotlib.pyplot as plt
    
    data, solution, metadata = load_solution(filename)
    
    scenario = metadata.get('scenario', 'Unknown')
    method = metadata.get('method', 'Unknown').capitalize()
    
    title = f"{scenario} Scenario - {method} Method"
    
    figure_path = plot_solution(
        data=data,
        opened_facilities=solution['opened'],
        assignments=solution['assignments'],
        title=title,
        save_format=save_format,
        facility_levels=solution.get('levels'),
        resources=solution.get('resources')
    )
    
    if show:
        plt.show()
    
    return figure_path


def main():
    """Command-line interface for solution management."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Manage saved optimization solutions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.solution_io --list                    List all saved solutions
  python -m src.solution_io --load balanced_exact.json  Visualize a solution
  python -m src.solution_io --load balanced_exact.json --format png
  python -m src.solution_io --delete old_solution.json
        """
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help="List all saved solutions"
    )
    
    parser.add_argument(
        '--load',
        type=str,
        help="Load and visualize a saved solution"
    )
    
    parser.add_argument(
        '--delete',
        type=str,
        help="Delete a saved solution"
    )
    
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['pdf', 'png'],
        default='pdf',
        help="Output format for visualization (default: pdf)"
    )
    
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help="Show plot interactively"
    )
    
    args = parser.parse_args()
    
    if args.list:
        solutions = list_saved_solutions()
        if not solutions:
            print("No saved solutions found.")
            print(f"Solutions directory: {SOLUTIONS_DIR}")
        else:
            print(f"\n{'='*80}")
            print(f"Saved Solutions ({len(solutions)} found)")
            print(f"{'='*80}")
            print(f"{'Filename':<45} | {'Scenario':<12} | {'Method':<10} | {'Cost':>15}")
            print(f"{'-'*80}")
            for sol in solutions:
                cost_str = f"${sol['cost']:,.2f}" if sol['cost'] else "N/A"
                print(f"{sol['filename']:<45} | {sol['scenario']:<12} | {sol['method']:<10} | {cost_str:>15}")
            print(f"{'='*80}")
            print(f"Directory: {SOLUTIONS_DIR}")
    
    elif args.load:
        print(f"Loading solution: {args.load}")
        figure_path = visualize_saved_solution(
            args.load,
            save_format=args.format,
            show=args.show
        )
        print(f"Figure saved to: {figure_path}")
    
    elif args.delete:
        delete_solution(args.delete)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
