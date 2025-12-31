"""
Exact Solver Module using Gurobi Optimizer.

Implements the Mixed Integer Programming (MIP) model for the 
Saudi Aramco Security Command Center Location Problem.

Sets:
- I: Candidate command center locations
- J: Demand sites
- L: Command center levels (High, Medium, Low)
- K: Resource types (Robot, Human)

Decision Variables:
- x_il: Binary, 1 if command center at location i with level l is built
- y_ij: Binary, 1 if demand site j is assigned to command center i
- z_ik: Integer, number of resource type k at location i

Parameters:
- F_il: Fixed cost of building command center at location i with level l
- C_ik: Variable cost of resource type k at location i
- t_ijl: Response time from location i with level l to site j
- S_j: SLA (maximum response time) for demand site j
- MAXCAP_lk, MINCAP_lk: Capacity constraints
"""
import os
from dotenv import load_dotenv
import gurobipy as gp
from gurobipy import GRB

# Load environment variables from .env file
load_dotenv()


def solve_exact(data):
    """
    Solve the optimization model using Gurobi.
    
    Args:
        data: Dictionary containing all problem parameters
        
    Returns:
        tuple: (objective_value, model) if optimal, (None, None) otherwise
    
    Environment Variables:
        GUROBI_LICENSE_TYPE: 'wls' for Web License Service, 'file' for license file (default: 'file')
        
        For WLS license (GUROBI_LICENSE_TYPE='wls'):
            GUROBI_LICENSE_ID: Your Gurobi license ID
            GUROBI_WLSACCESSID: WLS access ID
            GUROBI_WLSSECRET: WLS secret key
        
        For file license (GUROBI_LICENSE_TYPE='file' or unset):
            GUROBI_LICENSE_FILE: Path to gurobi.lic file (optional, uses default location if not set)
    """
    license_type = os.environ.get("GUROBI_LICENSE_TYPE", "file").lower()
    
    if license_type == "wls":
        env = gp.Env(empty=True)
        license_id = os.environ.get("GUROBI_LICENSE_ID")
        wls_access_id = os.environ.get("GUROBI_WLSACCESSID")
        wls_secret = os.environ.get("GUROBI_WLSSECRET")
        
        if license_id:
            env.setParam("LicenseID", int(license_id))
        if wls_access_id:
            env.setParam("WLSAccessID", wls_access_id)
        if wls_secret:
            env.setParam("WLSSecret", wls_secret)
        
        env.start()
        model = gp.Model("Aramco_Security_Location", env=env)
    else:
        license_file = os.environ.get("GUROBI_LICENSE_FILE")
        if license_file:
            os.environ["GRB_LICENSE_FILE"] = license_file
        
        model = gp.Model("Aramco_Security_Location")
    
    # Unpack data
    I = range(data['num_I'])
    J = range(data['num_J'])
    L = data['levels']
    t = data['t_ijl']
    S = data['S_j']
    
    # --- Decision Variables ---
    # x_il: 1 if command center at location i with level l is built
    x = model.addVars(I, L, vtype=GRB.BINARY, name="x")
    # y_ij: 1 if demand site j is assigned to center i
    y = model.addVars(I, J, vtype=GRB.BINARY, name="y")
    # z_ik: Number of resources at location i
    z_robot = model.addVars(I, vtype=GRB.INTEGER, name="z_robot")
    z_human = model.addVars(I, vtype=GRB.INTEGER, name="z_human")

    # --- Objective Function (Minimize Total Cost) ---
    # Fixed cost: F_il for each opened facility with level
    fixed_cost = gp.quicksum(
        data['F_il'][l][i] * x[i, l] 
        for i in I for l in L
    )
    # Variable cost: C_ik per resource
    var_cost = gp.quicksum(
        data['C_ik']['Robot'][i] * z_robot[i] + 
        data['C_ik']['Human'][i] * z_human[i] 
        for i in I
    )
    model.setObjective(fixed_cost + var_cost, GRB.MINIMIZE)

    # --- Constraints ---
    
    # 1. Each location can have at most one level (or no facility)
    model.addConstrs(
        (gp.quicksum(x[i, l] for l in L) <= 1 for i in I),
        name="OneLevel"
    )
    
    # 2. Demand Assignment: Each site j must be assigned to at least one center i
    model.addConstrs(
        (gp.quicksum(y[i, j] for i in I) >= 1 for j in J), 
        name="DemandAssign"
    )

    # 3. Logical Link: y_ij <= sum(x_il for l in L) (can only assign if facility is built)
    model.addConstrs(
        (y[i, j] <= gp.quicksum(x[i, l] for l in L) for i in I for j in J), 
        name="Logical"
    )

    # 4. SLA Compliance: Response time must be <= S_j for assigned demand sites
    # If y_ij = 1 and x_il = 1, then t_ijl <= S_j
    # Linearization: t_ijl * x_il <= S_j + M*(1-y_ij) for each i,j,l
    # Simplified: Only enforce SLA when both x_il and y_ij are 1
    M = 100000  # Big-M constant
    for i in I:
        for j in J:
            for l in L:
                # If facility i has level l and serves site j, response time must meet SLA
                model.addConstr(
                    t[l][i][j] * x[i, l] <= S[j] + M * (1 - y[i, j]),
                    name=f"SLA_{i}_{j}_{l}"
                )

    # 5. Maximum Physical Capacity: z_ik <= MAXCAP_lk * x_il for the selected level
    for i in I:
        model.addConstr(
            z_robot[i] <= gp.quicksum(data['MAXCAP_lk'][l]['Robot'] * x[i, l] for l in L),
            name=f"MaxCapRobot_{i}"
        )
        model.addConstr(
            z_human[i] <= gp.quicksum(data['MAXCAP_lk'][l]['Human'] * x[i, l] for l in L),
            name=f"MaxCapHuman_{i}"
        )

    # 6. Minimum Capacity: z_ik >= MINCAP_lk * x_il for the selected level
    for i in I:
        model.addConstr(
            z_robot[i] >= gp.quicksum(data['MINCAP_lk'][l]['Robot'] * x[i, l] for l in L),
            name=f"MinCapRobot_{i}"
        )
        model.addConstr(
            z_human[i] >= gp.quicksum(data['MINCAP_lk'][l]['Human'] * x[i, l] for l in L),
            name=f"MinCapHuman_{i}"
        )

    # 7. SCU Coverage Constraint: Resources must cover demand of assigned sites
    D_j = data['D_j']
    alpha_j = data['alpha_j']
    for i in I:
        # Robots needed: D_j / (1 + alpha_j) for each assigned site
        model.addConstr(
            z_robot[i] >= gp.quicksum((D_j[j] / (1 + alpha_j[j])) * y[i, j] for j in J),
            name=f"SCU_Robot_{i}"
        )
        # Humans needed: D_j * alpha_j / (1 + alpha_j) for each assigned site
        model.addConstr(
            z_human[i] >= gp.quicksum((D_j[j] * alpha_j[j] / (1 + alpha_j[j])) * y[i, j] for j in J),
            name=f"SCU_Human_{i}"
        )

    # 8. Global Supervision Constraint: z_human >= alpha * z_robot
    model.addConstrs(
        (z_human[i] >= data['alpha'] * z_robot[i] for i in I), 
        name="Supervision"
    )

    # Set parameters
    model.setParam(GRB.Param.MIPGap, 0.01)
    model.setParam(GRB.Param.TimeLimit, 60)

    # Solve the model
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        return model.objVal, model
    else:
        return None, None


def extract_solution(model, data):
    """
    Extract solution details from solved Gurobi model.
    
    Args:
        model: Solved Gurobi model
        data: Problem data dictionary
        
    Returns:
        dict: Solution details including opened facilities, levels, and assignments
    """
    if not model:
        return None
    
    num_I = data['num_I']
    num_J = data['num_J']
    levels = data['levels']
    
    # Extract x_il (opened facilities with levels)
    opened = []
    facility_levels = {}
    for i in range(num_I):
        for l in levels:
            var = model.getVarByName(f"x[{i},{l}]")
            if var and var.X > 0.5:
                opened.append(i)
                facility_levels[i] = l
                break
    
    # Extract y_ij (assignments)
    assignments = [[] for _ in range(num_J)]
    for j in range(num_J):
        for i in range(num_I):
            var = model.getVarByName(f"y[{i},{j}]")
            if var and var.X > 0.5:
                assignments[j].append(i)
    
    # Extract resources
    resources = {}
    for i in range(num_I):
        z_r = model.getVarByName(f"z_robot[{i}]")
        z_h = model.getVarByName(f"z_human[{i}]")
        if z_r and z_h:
            resources[i] = {
                'robot': int(z_r.X) if z_r.X > 0 else 0,
                'human': int(z_h.X) if z_h.X > 0 else 0
            }
    
    return {
        'opened': opened,
        'levels': facility_levels,
        'assignments': assignments,
        'resources': resources
    }