# Petro-HRCD-FLP

An Operations Research project implementing the **Human-Robot Co-Dispatch Facility Location Problem (HRCD-FLP)** to optimize security command center locations and resource allocation for critical infrastructure protection, specifically applied to a case study of Saudi Aramco's Dhahran district.

## Abstract

Securing petroleum infrastructure requires balancing autonomous system efficiency with human judgment for threat escalation. This project formulates the HRCD-FLP, a capacitated facility location variant that explicitly incorporates:

- **Tiered infrastructure criticality** with differentiated service level agreements (SLAs).
- **Human-robot co-dispatch** with supervision ratio constraints.
- **Minimum utilization requirements** for command centers.

The solution addresses the gap between classical facility location models (which often assume homogeneous resources) and modern security needs requiring hybrid human-autonomous teams.

## System Overview

The system optimizes three decision layers:

1. **Strategic:** Determining optimal locations and operational levels (High, Medium, Low) for command centers.
2. **Tactical:** Assigning demand sites (petroleum assets) to command centers.
3. **Operational:** Determining the mix of human and robot resources at each facility to meet coverage demands and supervision ratios.

### Technology Scenarios

Based on the parameters defined in `src/data_gen.py`:

| Scenario               | Supervision Ratio ($\alpha$) | Robot Cost Multiplier | Mix Ratio Scaler | Context                                |
| :--------------------- | :----------------------------: | :-------------------: | :--------------: | :------------------------------------- |
| **Conservative** |              1:3              |          1.0          |       1.00       | Early adoption, high human reliance    |
| **Balanced**     |              1:5              |          0.9          |       0.75       | Current technology, balanced mix       |
| **Future**       |              1:10              |          0.8          |       0.50       | High AI maturity, extensive automation |

## Experiment Results

Experimental evaluation on a case study of the Dhahran district (15 candidate locations, 50 demand sites):

| Scenario               | Method    | Facilities | Robots | Humans | Cost ($) | Gap (%) |
| :--------------------- | :-------- | :--------- | :----- | :----- | :------- | :------ |
| **Conservative** | Exact     | 4          | 207    | 149    | 692,450  | --      |
|                        | Heuristic | 9          | 212    | 149    | 734,700  | 6.10    |
| **Balanced**     | Exact     | 4          | 229    | 127    | 610,175  | --      |
|                        | Heuristic | 9          | 234    | 127    | 652,185  | 6.88    |
| **Future**       | Exact     | 3          | 256    | 98     | 508,000  | --      |
|                        | Heuristic | 8          | 261    | 99     | 537,820  | 5.87    |

## Prerequisites

1. **Python 3.11+**
2. **uv**: https://docs.astral.sh/uv/
3. **Gurobi** (optional): https://www.gurobi.com/academia/

## Installation

```bash
cd Petro-HRCD-FLP
uv sync
```

### Gurobi License Configuration

Create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Configure your Gurobi license:

**For WLS License (Cloud):**

```env
GUROBI_LICENSE_TYPE=wls
GUROBI_LICENSE_ID=your_license_id
GUROBI_WLSACCESSID=your_access_id
GUROBI_WLSSECRET=your_secret
```

**For File License (Local):**

```env
GUROBI_LICENSE_TYPE=file
# Optional: GUROBI_LICENSE_FILE=/path/to/gurobi.lic
```

## Usage

```bash
# Run all scenarios (synthetic data)
uv run python -m src.main

# Use real case study locations
uv run python -m src.main --real-data

# Single scenario
uv run python -m src.main -s Balanced

# Heuristic only (no Gurobi needed)
uv run python -m src.main --heuristic-only

# Verbose mode (see local search progress)
uv run python -m src.main -s Balanced --verbose

# Custom problem size (synthetic mode only)
uv run python -m src.main --candidates 20 --sites 100

# Save solutions for later visualization
uv run python -m src.main --save-solutions
```

### CLI Options

| Option                 | Description                                       |
| :--------------------- | :------------------------------------------------ |
| `-s, --scenarios`    | Scenarios to run (Conservative, Balanced, Future) |
| `--real-data`        | Use real case study facility locations           |
| `--heuristic-only`   | Skip Gurobi solver                                |
| `--exact-only`       | Skip heuristic solver                             |
| `-v, --verbose`      | Show detailed search progress                     |
| `--candidates N`     | Number of facility candidates (default: 15)       |
| `--sites N`          | Number of demand sites (default: 50)              |
| `--seed N`           | Random seed (default: 42)                         |
| `--max-iterations N` | Maximum local search iterations (default: 100)    |
| `--no-plots`         | Skip visualizations                               |
| `--save-solutions`   | Save solutions for later visualization            |
| `--output FILE`      | Custom output filename for results JSON           |

### Solution Management

Saved solutions can be managed and visualized without re-running solvers:

```bash
# List all saved solutions
uv run python -m src.solution_io --list

# Visualize a saved solution
uv run python -m src.solution_io --load <filename>

# Compare exact vs heuristic for a scenario
uv run python -m src.solution_io --compare Balanced

# Delete a saved solution
uv run python -m src.solution_io --delete <filename>

# Generate resource assignment bar charts (Exact vs Heuristic)
uv run python -m src.resource_visualization
```

### Large-Scale Data Generation

For generating synthetic datasets at high scale (thousands to tens of thousands of sites) without solving the optimization problem:

```bash
# Generate 100 candidates × 1,000 demand sites (quick test)
uv run python -c "from src.large_scale_data_gen import generate_and_visualize; generate_and_visualize(100, 1000)"

# Generate 1,000 candidates × 10,000 demand sites
uv run python -c "from src.large_scale_data_gen import generate_and_visualize; generate_and_visualize(1000, 10000)"

# Generate with distance matrices (slower, larger files)
uv run python -c "from src.large_scale_data_gen import generate_and_visualize; generate_and_visualize(500, 5000, compute_distances=True)"

# Run solver with a pre-generated dataset
uv run python -m src.main --load-dataset results/datasets/dataset_I100_J1000_seed42.json.gz --heuristic-only -s Balanced
```

**Features:**
- **Constant density scaling**: Area scales proportionally with √(total_sites) to maintain ~0.91 points/km² density
- **Optimized for scale**: Uses vectorized Haversine formula instead of sequential geodesic calculations
- **Batched processing**: Memory-efficient distance computation for very large datasets
- **Data persistence**: Saves datasets as compressed JSON files for later use
- **Visualization-only mode**: Generate and visualize data without solving the optimization problem
- **Solver integration**: Use `--load-dataset` flag in main solver to use pre-generated data

**Output files:**
- `results/datasets/data_preview_I{n}_J{m}.png` - Visualization preview
- `results/datasets/dataset_I{n}_J{m}_seed{s}.json.gz` - Compressed dataset file

## Output Files

- `results/figures/*.pdf` - Network visualization plots (LaTeX-ready)
- `results/figures/*.png` - Network visualization plots (web/preview)
- `results/solutions/experiment_results.json` - Detailed results (JSON)
- `results/solutions/experiment_results.xlsx` - Detailed results (Excel)
- `results/saved_solutions/` - Saved solutions for later visualization

### Converting Figures to PDF

For LaTeX documents, convert existing PNG figures to optimized PDFs:

```bash
uv run python -m src.convert_figures_to_pdf
```

## Project Structure

```
aramco_security_opt/
├── src/
│   ├── config.py                  # Path configuration
│   ├── convert_figures_to_pdf.py  # PNG → PDF converter for LaTeX
│   ├── data_gen.py                # Data generator (synthetic + real)
│   ├── large_scale_data_gen.py    # High-scale data gen (thousands of sites)
│   ├── exact_solver.py            # Gurobi MIP model
│   ├── heuristic_solver.py        # Greedy + Local Search
│   ├── main.py                    # CLI entry point
│   ├── resource_visualization.py  # Resource allocation plots
│   ├── solution_io.py             # Save/load solutions
│   └── visualization.py           # Network plot generation
├── data/                          # Real location data (JSON)
├── results/
│   ├── datasets/                  # Generated large-scale datasets
│   ├── figures/                   # Visualization outputs
│   ├── solutions/                 # Experiment results
│   └── saved_solutions/           # Saved solutions for re-visualization
├── report/
│   ├── main.tex                   # Research paper (LaTeX)
│   └── figures/                   # LaTeX figures
├── .env.example                   # Environment template
└── pyproject.toml                 # Dependencies
```

## Dependencies

Core dependencies managed via `pyproject.toml`:

- **gurobipy** - Gurobi optimizer for exact MIP solving
- **numpy/pandas** - Data manipulation
- **matplotlib** - Visualization
- **contextily** - Geographic basemaps
- **geopy** - Distance calculations
- **adjusttext** - Label adjustment in plots
- **openpyxl** - Excel export
- **python-dotenv** - Environment configuration

## Mathematical Formulation

The HRCD-FLP is a mixed-integer linear program (MILP) that minimizes total infrastructure and operational costs.

### Sets and Indices

- $I$: Set of candidate command center locations, indexed by $i$.
- $J$: Set of demand sites, indexed by $j$.
- $L$: Set of facility levels (High, Medium, Low), indexed by $l$.
- $K$: Set of resource types (Robot, Human), indexed by $k$.

### Parameters

**Costs:**

- $F_{il}$: Fixed construction and overhead cost for facility $i$ at level $l$.
- $C_{ik}$: Unit deployment cost for resource $k$ at location $i$.

**Capacities and Capabilities:**

- $\text{MAXCAP}_{lk}$: Maximum capacity of resource $k$ for a facility at level $l$.
- $\text{MINCAP}_{lk}$: Minimum required resource $k$ for a facility at level $l$ (if built).
- $t_{ijl}$: Response time from facility $i$ at level $l$ to site $j$.
- $S_j$: Service Level Agreement (maximum allowable response time) for site $j$.

**Demand:**

- $D_j$: Total security demand (SCU) at site $j$.
- $\alpha_j$: Site-specific mix ratio required to cover demand.
- $\alpha$: Global supervision ratio (minimum Humans per Robot).

### Decision Variables

- $x_{il} \in \{0, 1\}$: 1 if candidate location $i$ is developed at level $l$.
- $y_{ij} \in \{0, 1\}$: 1 if demand site $j$ is assigned to facility $i$.
- $z_{ik} \in \mathbb{Z}^+$: Number of resources of type $k$ deployed at facility $i$.

### Objective Function

Minimize total cost $Z$:

$$
\min_{x, y, z} Z = \sum_{i \in I} \sum_{l \in L} F_{il} \, x_{il} + \sum_{i \in I} \sum_{k \in K} C_{ik} \, z_{ik}
$$

### Constraints

1. **Facility Configuration:** Each candidate location may host at most one facility.

   $$
   \sum_{l \in L} x_{il} \leq 1, \quad \forall i \in I
   $$
2. **Demand Coverage:** Every demand site must be assigned to at least one command center.

   $$
   \sum_{i \in I} y_{ij} \geq 1, \quad \forall j \in J
   $$
3. **Assignment Feasibility:** Sites can only be assigned to active facilities.

   $$
   y_{ij} \leq \sum_{l \in L} x_{il}, \quad \forall i \in I, \; \forall j \in J
   $$
4. **Service Level Compliance:** Response time must meet SLA thresholds.

   $$
   t_{ijl} \, x_{il} \leq S_j + M(1 - y_{ij}), \quad \forall i \in I, \; \forall j \in J, \; \forall l \in L
   $$
5. **Resource Capacity:** Deployment must respect facility capacity limits (min and max).

   $$
   \sum_{l \in L} \text{MINCAP}_{lk} \, x_{il} \leq z_{ik} \leq \sum_{l \in L} \text{MAXCAP}_{lk} \, x_{il}, \quad \forall i \in I, \; \forall k \in K
   $$
6. **Coverage Satisfaction:** Resources must meet aggregate demand.

   - **Robots:** $z_{i,\text{Robot}} \geq \sum_{j \in J} \frac{D_j}{1 + \alpha_j} \, y_{ij}$
   - **Humans:** $z_{i,\text{Human}} \geq \sum_{j \in J} \frac{D_j \, \alpha_j}{1 + \alpha_j} \, y_{ij}$
7. **Human-in-the-Loop Supervision:** Human contingent must scale with robot fleet size.

   $$
   z_{i,\text{Human}} \geq \alpha \cdot z_{i,\text{Robot}}, \quad \forall i \in I
   $$

## License

Academic research project.
