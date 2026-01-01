"""
Main Entry Point for Human-Robot Co-Dispatch Facility Location Problem (HRCD-FLP) Optimization.

Runs experiments comparing Exact (Gurobi) and Heuristic solvers
across technology scenarios: Conservative, Balanced, Future.

Usage:
    uv run python -m src.main                    # Run all scenarios (synthetic data)
    uv run python -m src.main --real-data        # Use real facility locations
    uv run python -m src.main -s Balanced        # Run single scenario
    uv run python -m src.main --heuristic-only   # Skip Gurobi solver
    uv run python -m src.main --verbose          # Show detailed progress
    uv run python -m src.main --save-solutions   # Save solutions for later visualization
    uv run python -m src.main --candidates 20 --sites 100  Custom problem size
    uv run python -m src.main --load-dataset <path>  # Load pre-generated dataset

To visualize saved solutions without re-running the solver:
    uv run python -m src.solution_io --list      # List saved solutions
    uv run python -m src.solution_io --load <filename>  # Visualize saved solution
"""
import argparse
import time
import json
from pathlib import Path

from .data_gen import DataGenerator
from .large_scale_data_gen import LargeScaleDataGenerator
from .exact_solver import solve_exact, extract_solution
from .heuristic_solver import HeuristicSolver
from .visualization import plot_solution
from .solution_io import save_solution
from .config import RESULTS_DIR
import numpy as np
import pandas as pd

SOLUTIONS_DIR = RESULTS_DIR / "solutions"
SOLUTIONS_DIR.mkdir(parents=True, exist_ok=True)

VALID_SCENARIOS = ['Conservative', 'Balanced', 'Future']


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Human-Robot Co-Dispatch Facility Location Problem (HRCD-FLP) Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python -m src.main                         Run all scenarios (synthetic)
  uv run python -m src.main --real-data             Use real facility locations
  uv run python -m src.main -s Balanced             Run only Balanced scenario
  uv run python -m src.main -s Conservative Future  Run two scenarios
  uv run python -m src.main --heuristic-only        Skip Gurobi (no license needed)
  uv run python -m src.main --verbose               Show detailed search progress
  uv run python -m src.main --candidates 20 --sites 100  Custom problem size
  uv run python -m src.main --save-solutions        Save solutions for later use

To visualize saved solutions:
  uv run python -m src.solution_io --list           List all saved solutions
  uv run python -m src.solution_io --load <file>    Visualize a saved solution
        """
    )
    
    parser.add_argument(
        '-s', '--scenarios',
        nargs='+',
        choices=VALID_SCENARIOS,
        default=VALID_SCENARIOS,
        metavar='SCENARIO',
        help=f"Scenarios to run. Choices: {', '.join(VALID_SCENARIOS)}. Default: all"
    )
    
    parser.add_argument(
        '--heuristic-only',
        action='store_true',
        help="Run only heuristic solver (no Gurobi license required)"
    )
    
    parser.add_argument(
        '--exact-only',
        action='store_true',
        help="Run only exact Gurobi solver"
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help="Show detailed progress for heuristic solver"
    )
    
    parser.add_argument(
        '--candidates',
        type=int,
        default=15,
        help="Number of candidate facility locations (default: 15)"
    )
    
    parser.add_argument(
        '--sites',
        type=int,
        default=50,
        help="Number of demand sites (default: 50)"
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=100,
        help="Maximum local search iterations for heuristic (default: 100)"
    )
    
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help="Skip generating visualization plots"
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help="Custom output filename for results JSON"
    )
    
    parser.add_argument(
        '--real-data',
        action='store_true',
        help="Use real facility location data instead of synthetic"
    )
    
    parser.add_argument(
        '--save-solutions',
        action='store_true',
        help="Save solutions to files for later visualization (without re-running solver)"
    )
    
    parser.add_argument(
        '--load-dataset',
        type=str,
        default=None,
        metavar='PATH',
        help="Load a pre-generated dataset file (from large_scale_data_gen). Overrides --candidates, --sites, --seed, and --real-data."
    )
    
    return parser.parse_args()


def _generate_params_from_loaded_dataset(loaded_gen: LargeScaleDataGenerator, scenario: str = 'Balanced') -> dict:
    """
    Generate optimization parameters from a loaded dataset.
    
    This function converts the LargeScaleDataGenerator data format to the format
    expected by the solvers (same as DataGenerator.generate_params()).
    
    Args:
        loaded_gen: LargeScaleDataGenerator instance with loaded data
        scenario: One of 'Conservative', 'Balanced', or 'Future'
        
    Returns:
        dict: Complete parameter dictionary for optimization model
    """
    num_I = loaded_gen.num_I
    num_J = loaded_gen.num_J
    levels = loaded_gen.levels
    
    # 1. Base Fixed Cost (F_i): Location-based
    base_F_i = np.array([25000 if i < num_I/2 else 20000 for i in range(num_I)])
    
    F_il = {}
    for level in levels:
        multiplier = loaded_gen.level_cost_multipliers[level]
        F_il[level] = base_F_i * multiplier
    
    # 2. Variable Cost (C_ik): Cost per resource type per location
    C_ik = {
        'Robot': np.array([800 if i < num_I/2 else 750 for i in range(num_I)]),
        'Human': np.array([3000 if i < num_I/2 else 2800 for i in range(num_I)])
    }
    
    # 3. Capacity Parameters per resource type per LEVEL
    MAXCAP_lk = {
        'High': {'Robot': 240, 'Human': 40},
        'Medium': {'Robot': 100, 'Human': 20},
        'Low': {'Robot': 40, 'Human': 10}
    }
    MINCAP_lk = {
        'High': {'Robot': 0, 'Human': 20},
        'Medium': {'Robot': 0, 'Human': 10},
        'Low': {'Robot': 0, 'Human': 5}
    }
    
    # 4. Technology Scenario Settings
    robot_cost_mult = 1.0
    alpha_j_scaler = 1.0
    
    if scenario == 'Conservative':
        alpha = 1/3.0
        robot_cost_mult = 1.0
        alpha_j_scaler = 1.0
    elif scenario == 'Balanced':
        alpha = 1/5.0
        robot_cost_mult = 0.9
        alpha_j_scaler = 0.75
    elif scenario == 'Future':
        alpha = 1/10.0
        robot_cost_mult = 0.8
        alpha_j_scaler = 0.50
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
    
    # Apply scenario adjustments
    C_ik['Robot'] = C_ik['Robot'] * robot_cost_mult
    scenario_alpha_j = loaded_gen.alpha_j * alpha_j_scaler
    
    # 5. Response time matrix (t_ijl)
    t_ijl = {}
    for level in levels:
        multiplier = loaded_gen.level_time_multipliers[level]
        t_ijl[level] = loaded_gen.d_ij * multiplier
    
    return {
        # Sets
        'num_I': num_I,
        'num_J': num_J,
        'num_L': len(levels),
        'levels': levels,
        # Response time and SLA
        't_ijl': t_ijl,
        'S_j': loaded_gen.S_j,
        'd_ij': loaded_gen.d_ij,
        # Demand
        'D_j': loaded_gen.D_j,
        'alpha_j': scenario_alpha_j,
        # Costs
        'F_il': F_il,
        'C_ik': C_ik,
        # Capacity
        'MAXCAP_lk': MAXCAP_lk,
        'MINCAP_lk': MINCAP_lk,
        # Technology / Supervision
        'alpha': alpha,
        'level_cost_multipliers': loaded_gen.level_cost_multipliers,
        'level_time_multipliers': loaded_gen.level_time_multipliers,
        # Coordinates for visualization
        'coords_I': loaded_gen.I_coords,
        'coords_J': loaded_gen.J_coords,
        'corridors': loaded_gen.corridors if hasattr(loaded_gen, 'corridors') else [],
        'names_I': [f"Candidate-{i}" for i in range(num_I)],
        'names_J': [f"Demand-{j}" for j in range(num_J)],
        'use_real_data': False
    }


def run_experiment(args):
    """
    Run optimization experiment with given configuration.
    
    Args:
        args: Parsed command-line arguments
    """
    # Data source selection
    loaded_dataset = None
    use_loaded_dataset = args.load_dataset is not None
    
    if use_loaded_dataset:
        # Load pre-generated dataset
        print(f"Loading dataset from: {args.load_dataset}")
        loaded_gen = LargeScaleDataGenerator.load_dataset(args.load_dataset)
        
        # Compute distances if not already computed
        if loaded_gen.d_ij is None:
            print("Computing distance matrices (not included in dataset)...")
            loaded_gen.compute_distances()
        
        loaded_dataset = loaded_gen
        num_candidates = loaded_gen.num_I
        num_sites = loaded_gen.num_J
        seed = loaded_gen.seed
        use_real_data = False # Always False if loading a pre-generated dataset
        levels = loaded_gen.levels
    else:
        # Setup Data Generator (original behavior)
        gen = DataGenerator(
            num_candidates=args.candidates, 
            num_demand_sites=args.sites,
            seed=args.seed,
            use_real_data=args.real_data
        )
        gen.generate_locations()
        num_candidates = args.candidates
        num_sites = args.sites
        seed = args.seed
        use_real_data = args.real_data
        levels = gen.levels
    
    scenarios = args.scenarios
    run_exact = not args.heuristic_only
    run_heuristic = not args.exact_only
    
    # Results storage
    results = []
    
    print("=" * 80)
    print("Human-Robot Co-Dispatch Facility Location Problem (HRCD-FLP) Optimization")
    print("=" * 80)
    if use_loaded_dataset:
        print(f"Dataset: {args.load_dataset}")
    print(f"Configuration: {num_candidates} candidates, {num_sites} demand sites, seed={seed}")
    print(f"Facility Levels: {levels}")
    print(f"Scenarios: {', '.join(scenarios)}")
    print(f"Solvers: {'Exact' if run_exact else ''}{' + ' if run_exact and run_heuristic else ''}{'Heuristic' if run_heuristic else ''}")
    print("=" * 80)
    print(f"{'Scenario':<15} | {'Method':<10} | {'Cost':>15} | {'Time (s)':>10} | {'Gap %':>8} | {'Facilities'}")
    print("-" * 80)
    
    for sc in scenarios:
        # Generate scenario parameters based on data source
        if use_loaded_dataset:
            data = _generate_params_from_loaded_dataset(loaded_dataset, scenario=sc)
        else:
            data = gen.generate_params(scenario=sc)
        
        scenario_result = {'scenario': sc}
        exact_cost = None
        
        # --- 1. Exact Solution ---
        if run_exact:
            try:
                start_time = time.time()
                exact_cost, model = solve_exact(data)
                exact_time = time.time() - start_time
                
                if model:
                    solution = extract_solution(model, data)
                    opened_exact = solution['opened']
                    assign_exact = solution['assignments']
                    levels_exact = solution['levels']
                    
                    if not args.no_plots:
                        plot_solution(data, opened_exact, assign_exact, 
                                     title=f"{sc} Scenario - Exact Method",
                                     facility_levels=levels_exact,
                                     resources=solution['resources'])
                    
                    # Format facility info
                    fac_info = ", ".join([f"{i}({levels_exact[i][0]})" for i in opened_exact])
                    print(f"{sc:<15} | {'Exact':<10} | ${exact_cost:>14,.2f} | {exact_time:>10.3f} | {'N/A':>8} | {fac_info}")
                    
                    scenario_result['exact'] = {
                        'cost': exact_cost,
                        'time': exact_time,
                        'facilities': opened_exact,
                        'levels': levels_exact,
                        'num_facilities': len(opened_exact),
                        'resources': solution['resources'],
                        'total_resources': {
                            'robot': sum(r['robot'] for r in solution['resources'].values()),
                            'human': sum(r['human'] for r in solution['resources'].values())
                        }
                    }
                    
                    # Save solution for later visualization
                    if args.save_solutions:
                        save_solution(
                            data=data,
                            solution=solution,
                            scenario=sc,
                            method='exact',
                            cost=exact_cost,
                            solve_time=exact_time,
                            metadata={
                                'num_candidates': num_candidates,
                                'num_sites': num_sites,
                                'seed': seed,
                                'use_real_data': use_real_data,
                                'loaded_dataset': args.load_dataset
                            }
                        )
                else:
                    print(f"{sc:<15} | {'Exact':<10} | {'INFEASIBLE':>15} | {exact_time:>10.3f} | {'N/A':>8} |")
                    scenario_result['exact'] = None
                    exact_cost = None
            except Exception as e:
                print(f"{sc:<15} | {'Exact':<10} | {'ERROR':>15} | {'N/A':>10} | {'N/A':>8} |")
                if args.verbose:
                    print(f"  Error: {e}")
                scenario_result['exact'] = {'error': str(e)}
        
        # --- 2. Heuristic Solution ---
        if run_heuristic:
            start_time = time.time()
            heur = HeuristicSolver(
                data, 
                max_iterations=args.max_iterations,
                verbose=args.verbose
            )
            heur.constructive_greedy()
            heur_cost = heur.local_search()
            heur_time = time.time() - start_time
            
            solution = heur.get_solution()
            opened_heur = solution['opened']
            assign_heur = solution['assignments']
            levels_heur = solution['levels']
            resources_heur = solution['resources']
            
            if not args.no_plots:
                plot_solution(data, opened_heur, assign_heur, 
                             title=f"{sc} Scenario - Heuristic Method",
                             facility_levels=levels_heur,
                             resources=resources_heur)
            
            # Calculate optimality gap
            if exact_cost and exact_cost > 0:
                gap = (heur_cost - exact_cost) / exact_cost * 100
                gap_str = f"{gap:>7.2f}%"
            else:
                gap = None
                gap_str = "N/A"
            
            # Format facility info
            fac_info = ", ".join([f"{i}({levels_heur[i][0]})" for i in opened_heur])
            print(f"{sc:<15} | {'Heuristic':<10} | ${heur_cost:>14,.2f} | {heur_time:>10.3f} | {gap_str:>8} | {fac_info}")
            
            scenario_result['heuristic'] = {
                'cost': heur_cost,
                'time': heur_time,
                'facilities': opened_heur,
                'levels': levels_heur,
                'num_facilities': len(opened_heur),
                'gap_percent': gap,
                'resources': resources_heur,
                'total_resources': {
                    'robot': sum(r['robot'] for r in resources_heur.values()),
                    'human': sum(r['human'] for r in resources_heur.values())
                }
            }
            
            # Save solution for later visualization
            if args.save_solutions:
                save_solution(
                    data=data,
                    solution=solution,
                    scenario=sc,
                    method='heuristic',
                    cost=heur_cost,
                    solve_time=heur_time,
                    metadata={
                        'num_candidates': num_candidates,
                        'num_sites': num_sites,
                        'seed': seed,
                        'use_real_data': use_real_data,
                        'loaded_dataset': args.load_dataset,
                        'gap_percent': gap
                    }
                )
        
        results.append(scenario_result)
        print("-" * 80)
    
    # Export results to JSON
    if args.output:
        results_file = Path(args.output)
    else:
        results_file = SOLUTIONS_DIR / "experiment_results.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults exported to: {results_file}")

    # Export results to Excel
    excel_file = results_file.with_suffix('.xlsx')
    excel_rows = []
    
    for res in results:
        sc = res['scenario']
        
        # Exact Method
        if res.get('exact'):
            if 'error' in res['exact']:
                excel_rows.append({
                    'Scenario': sc,
                    'Method': 'Exact',
                    'Cost': 'Error',
                    'Time (s)': 'N/A',
                    'Gap (%)': 'N/A',
                    'Facilities': 'N/A',
                    'Total Robots': 'N/A',
                    'Total Humans': 'N/A',
                    'Note': res['exact']['error']
                })
            else:
                exact = res['exact']
                excel_rows.append({
                    'Scenario': sc,
                    'Method': 'Exact',
                    'Cost': exact['cost'],
                    'Time (s)': exact['time'],
                    'Gap (%)': 0.0,
                    'Facilities': exact['num_facilities'],
                    'Total Robots': exact['total_resources']['robot'],
                    'Total Humans': exact['total_resources']['human'],
                    'Note': ''
                })
        
        # Heuristic Method
        if res.get('heuristic'):
            heur = res['heuristic']
            excel_rows.append({
                'Scenario': sc,
                'Method': 'Heuristic',
                'Cost': heur['cost'],
                'Time (s)': heur['time'],
                'Gap (%)': heur.get('gap_percent', 'N/A'),
                'Facilities': heur['num_facilities'],
                'Total Robots': heur['total_resources']['robot'],
                'Total Humans': heur['total_resources']['human'],
                'Note': ''
            })
            
    if excel_rows:
        try:
            df = pd.DataFrame(excel_rows)
            cols = ['Scenario', 'Method', 'Cost', 'Time (s)', 'Gap (%)', 'Facilities', 'Total Robots', 'Total Humans', 'Note']
            cols = [c for c in cols if c in df.columns]
            df = df[cols]
            df.to_excel(excel_file, index=False)
            print(f"Results exported to: {excel_file}")
        except Exception as e:
            print(f"Failed to export Excel: {e}")
    print("=" * 80)
    
    return results


def main():
    """Main entry point."""
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()