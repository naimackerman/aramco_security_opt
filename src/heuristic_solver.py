"""
Heuristic Solver Module for Large-Scale Instances.

Implements a two-stage approach:
1. Constructive Greedy Heuristic for initial feasible solution
2. Local Search Improvement (Shift, Swap, Drop/Open moves)

Sets:
- I: Candidate command center locations
- J: Demand sites
- L: Command center levels (High, Medium, Low)
- K: Resource types (Robot, Human)

Decision Variables:
- x_il: 1 if command center at location i with level l is built
- y_ij: 1 if demand site j is assigned to command center i
"""
import numpy as np
import math


class HeuristicSolver:
    def __init__(self, data, max_iterations=100, verbose=False):
        """
        Initialize the heuristic solver.
        
        Args:
            data: Dictionary containing all problem parameters
            max_iterations: Maximum local search iterations (default: 100)
            verbose: If True, print progress messages
        """
        self.data = data
        self.num_I = data['num_I']
        self.num_J = data['num_J']
        self.levels = data['levels']
        self.num_L = data['num_L']
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Solution state
        self.x = [None] * self.num_I
        self.assignments = [[] for _ in range(self.num_J)]
        self.resources = {i: {'human': 0, 'robot': 0} for i in range(self.num_I)}

    def _can_serve(self, facility_idx, level, demand_site_idx):
        """
        Check if facility at location i with level l can serve demand site j.
        Uses response time t_ijl and SLA S_j.
        """
        t = self.data['t_ijl'][level][facility_idx][demand_site_idx]
        s = self.data['S_j'][demand_site_idx]
        return t <= s

    def _get_feasible_levels(self, facility_idx, demand_site_idx):
        """Get list of feasible levels for serving a demand site from a facility."""
        feasible = []
        for level in self.levels:
            if self._can_serve(facility_idx, level, demand_site_idx):
                feasible.append(level)
        return feasible

    def calculate_resource_mix(self, facility_idx, num_sites_assigned, level=None, site_indices=None):
        """
        Calculate optimal Robot & Human count for a facility with assigned sites.
        Uses D_j (SCU demand) and alpha_j (human/robot mix ratio).
        """
        if num_sites_assigned <= 0:
            return 0, 0
        
        # Get the level for this facility
        if level is None:
            level = self.x[facility_idx]
        if level is None:
            level = 'Low'
        
        # Get capacity constraints for this level
        max_robot = self.data['MAXCAP_lk'][level]['Robot']
        max_human = self.data['MAXCAP_lk'][level]['Human']
        min_robot = self.data['MINCAP_lk'][level]['Robot']
        min_human = self.data['MINCAP_lk'][level]['Human']
        
        # Get site indices if not provided
        if site_indices is None:
            site_indices = self._get_facility_sites(facility_idx)
        
        # Calculate resource needs based on SCU demand and mix ratio
        # For each site j: robots = D_j / (1 + alpha_j), humans = D_j * alpha_j / (1 + alpha_j)
        D_j = self.data['D_j']
        alpha_j = self.data['alpha_j']
        alpha = self.data['alpha']
        
        total_robots = 0
        total_humans = 0
        for j in site_indices:
            d = D_j[j]
            a = alpha_j[j]
            total_robots += d / (1 + a)
            total_humans += d * a / (1 + a)
        
        required_robots = math.ceil(total_robots)
        required_humans = math.ceil(total_humans)
        
        # Apply global supervision constraint: H >= alpha * R
        required_humans = max(required_humans, math.ceil(alpha * required_robots))
        
        # Ensure minimum capacity
        required_humans = max(required_humans, min_human)
        required_robots = max(required_robots, min_robot)
        
        # Check Maximum Capacity constraints
        if required_robots > max_robot:
            return None
        if required_humans > max_human:
            return None
            
        return required_robots, required_humans

    def _get_facility_sites(self, facility_idx):
        """Get list of demand site indices assigned to a facility."""
        return [j for j in range(self.num_J) if facility_idx in self.assignments[j]]

    def _get_num_sites_at_facility(self, facility_idx):
        """Get number of sites assigned to a facility."""
        return sum(1 for j in range(self.num_J) if facility_idx in self.assignments[j])

    def _update_facility_state(self):
        """Update resources state based on current assignments."""
        for i in range(self.num_I):
            num_sites = self._get_num_sites_at_facility(i)
            if num_sites > 0 and self.x[i] is not None:
                res = self.calculate_resource_mix(i, num_sites)
                if res:
                    self.resources[i] = {'robot': res[0], 'human': res[1]}
            else:
                if num_sites == 0:
                    self.x[i] = None
                self.resources[i] = {'robot': 0, 'human': 0}

    def _get_best_level_for_sites(self, facility_idx, site_indices):
        """
        Determine the minimum level needed to serve all sites from a facility.
        Returns None if no level can serve all sites (SLA or capacity).
        """
        # Try levels from lowest (cheapest) to highest (most expensive)
        for level in reversed(self.levels):
            # Check SLA feasibility
            can_serve_all = True
            for j in site_indices:
                if not self._can_serve(facility_idx, level, j):
                    can_serve_all = False
                    break
            
            if not can_serve_all:
                continue
            
            # Check capacity feasibility
            res = self.calculate_resource_mix(facility_idx, len(site_indices), 
                                             level=level, site_indices=site_indices)
            if res is not None:
                return level
        
        return None

    def constructive_greedy(self):
        """
        Stage 1: Constructive Greedy Heuristic
        
        Generate initial feasible solution using greedy strategy.
        """
        if self.verbose:
            print("Stage 1: Constructive Greedy Heuristic")
            
        for j in range(self.num_J):
            best_i = -1
            best_level = None
            min_marginal_cost = float('inf')
            
            for i in range(self.num_I):
                feasible_levels = self._get_feasible_levels(i, j)
                if not feasible_levels:
                    continue
                
                # Get current sites at this facility plus the new site
                current_site_indices = self._get_facility_sites(i)
                new_site_indices = current_site_indices + [j]
                
                # Determine which level to use
                if self.x[i] is not None:
                    current_level = self.x[i]
                    if current_level in feasible_levels:
                        chosen_level = current_level
                    else:
                        # Need to upgrade level - find highest feasible
                        higher_levels = [l for l in feasible_levels if 
                                        self.levels.index(l) <= self.levels.index(current_level)]
                        if higher_levels:
                            chosen_level = higher_levels[0]
                        else:
                            continue
                else:
                    # Facility not open: try to find cheapest feasible level with capacity
                    chosen_level = None
                    for level in reversed(self.levels):
                        if level in feasible_levels:
                            res = self.calculate_resource_mix(i, len(new_site_indices),
                                                             level=level, site_indices=new_site_indices)
                            if res is not None:
                                chosen_level = level
                                break
                    if chosen_level is None:
                        continue
                
                # Check resource feasibility for chosen level
                res = self.calculate_resource_mix(i, len(new_site_indices),
                                                 level=chosen_level, site_indices=new_site_indices)
                if res is None:
                    # Current level can't handle - try upgrading
                    upgraded = False
                    for level in self.levels:
                        if self.levels.index(level) < self.levels.index(chosen_level):
                            res = self.calculate_resource_mix(i, len(new_site_indices),
                                                             level=level, site_indices=new_site_indices)
                            if res is not None:
                                chosen_level = level
                                upgraded = True
                                break
                    if not upgraded:
                        continue
                
                r_new, h_new = res
                
                # Calculate marginal cost
                cost_new = (
                    r_new * self.data['C_ik']['Robot'][i] + 
                    h_new * self.data['C_ik']['Human'][i]
                )
                
                # Add fixed cost if facility not yet opened (or upgrading level)
                if self.x[i] is None:
                    cost_new += self.data['F_il'][chosen_level][i]
                elif self.x[i] != chosen_level:
                    cost_new += (self.data['F_il'][chosen_level][i] - 
                                self.data['F_il'][self.x[i]][i])
                
                if cost_new < min_marginal_cost:
                    min_marginal_cost = cost_new
                    best_i = i
                    best_level = chosen_level
            
            if best_i != -1:
                # Add facility to assignments (first assignment for each site)
                if best_i not in self.assignments[j]:
                    self.assignments[j].append(best_i)
                self.x[best_i] = best_level
                # Update resources
                num_sites = self._get_num_sites_at_facility(best_i)
                res = self.calculate_resource_mix(best_i, num_sites)
                if res:
                    self.resources[best_i] = {'robot': res[0], 'human': res[1]}
        
        # After assignment, optimize levels for each facility
        self._optimize_facility_levels()
        
        if self.verbose:
            print(f"  Initial solution cost: ${self.calculate_total_cost():,.2f}")
            print(f"  Opened facilities: {sum(1 for x in self.x if x is not None)}")
    
    def _optimize_facility_levels(self):
        """Optimize level for each open facility to use cheapest feasible level."""
        for i in range(self.num_I):
            if self.x[i] is None:
                continue
            
            sites = self._get_facility_sites(i)
            if not sites:
                self.x[i] = None
                continue
            
            # Find cheapest level that can serve all sites
            best_level = self._get_best_level_for_sites(i, sites)
            if best_level:
                self.x[i] = best_level
    
    def calculate_total_cost(self):
        """
        Calculate global total cost for current solution.
        """
        total_cost = 0
        facility_sites_count = {i: 0 for i in range(self.num_I)}
        
        # Check that each site has at least one assignment
        for j in range(self.num_J):
            if not self.assignments[j]:
                return float('inf')
            # Count each site for each facility serving it
            for i in self.assignments[j]:
                facility_sites_count[i] += 1
            
        for i in range(self.num_I):
            num_sites = facility_sites_count[i]
            if num_sites > 0 and self.x[i] is not None:
                res = self.calculate_resource_mix(i, num_sites)
                if res is None:
                    return float('inf')
                r, h = res
                
                # Fixed cost for this level
                total_cost += self.data['F_il'][self.x[i]][i]
                # Variable costs
                total_cost += (
                    r * self.data['C_ik']['Robot'][i] + 
                    h * self.data['C_ik']['Human'][i]
                )
                    
        return total_cost

    def _is_valid_assignment(self, facility_idx, demand_site_idx):
        """Check if assignment is valid given current facility level."""
        if self.x[facility_idx] is None:
            return False
        return self._can_serve(facility_idx, self.x[facility_idx], demand_site_idx)

    def _shift_move(self):
        """Shift Move: Try moving demand site j from current center to a different one."""
        current_cost = self.calculate_total_cost()
        
        for j in range(self.num_J):
            if not self.assignments[j]:
                continue
            original_assignments = self.assignments[j].copy()
            original_i = original_assignments[0] if original_assignments else -1
            original_level = self.x[original_i] if original_i >= 0 else None
            
            for k in range(self.num_I):
                if k in original_assignments:
                    continue
                
                # Check if k can serve j
                if self.x[k] is not None:
                    if not self._can_serve(k, self.x[k], j):
                        feasible = self._get_feasible_levels(k, j)
                        if not feasible:
                            continue
                        old_level = self.x[k]
                        self.x[k] = feasible[0]
                else:
                    feasible = self._get_feasible_levels(k, j)
                    if not feasible:
                        continue
                    self.x[k] = feasible[-1]
                
                # Replace primary assignment with k
                self.assignments[j] = [k]
                
                self._optimize_facility_levels()
                new_cost = self.calculate_total_cost()
                
                if new_cost < current_cost:
                    self._update_facility_state()
                    if self.verbose:
                        print(f"  Shift: site {j} from facility {original_i} to {k}, "
                              f"saving ${current_cost - new_cost:,.2f}")
                    return True
                else:
                    self.assignments[j] = original_assignments
                    self._optimize_facility_levels()
                    
        return False

    def _swap_move(self):
        """Swap Move: Exchange assignments of two demand sites between two facilities."""
        current_cost = self.calculate_total_cost()
        
        for j1 in range(self.num_J):
            if not self.assignments[j1]:
                continue
            i1 = self.assignments[j1][0]
            
            for j2 in range(j1 + 1, self.num_J):
                if not self.assignments[j2]:
                    continue
                i2 = self.assignments[j2][0]
                
                if i1 == i2:
                    continue
                
                # Check feasibility of swap
                l1 = self.x[i1]
                l2 = self.x[i2]
                
                if l1 is None or l2 is None:
                    continue
                
                # Check if i1 can serve j2 and i2 can serve j1
                can_swap = (self._can_serve(i1, l1, j2) and 
                           self._can_serve(i2, l2, j1))
                
                if not can_swap:
                    continue
                
                # Store originals
                orig1 = self.assignments[j1].copy()
                orig2 = self.assignments[j2].copy()
                
                # Perform swap
                self.assignments[j1] = [i2]
                self.assignments[j2] = [i1]
                
                self._optimize_facility_levels()
                new_cost = self.calculate_total_cost()
                
                if new_cost < current_cost:
                    self._update_facility_state()
                    if self.verbose:
                        print(f"  Swap: sites ({j1}, {j2}) between facilities ({i1}, {i2}), "
                              f"saving ${current_cost - new_cost:,.2f}")
                    return True
                else:
                    self.assignments[j1] = orig1
                    self.assignments[j2] = orig2
                    self._optimize_facility_levels()
                    
        return False

    def _drop_move(self):
        """Drop Move: Try closing a facility by redistributing its sites to neighbors."""
        current_cost = self.calculate_total_cost()
        
        open_facilities = [i for i in range(self.num_I) if self.x[i] is not None]
        open_facilities.sort(key=lambda i: self._get_num_sites_at_facility(i))
        
        for drop_i in open_facilities:
            sites_at_i = self._get_facility_sites(drop_i)
            if not sites_at_i:
                continue
            
            # Store original assignments for sites served by drop_i
            original_assignments = {j: self.assignments[j].copy() for j in sites_at_i}
            original_level = self.x[drop_i]
            redistribution_possible = True
            
            for j in sites_at_i:
                best_alt = -1
                min_cost_increase = float('inf')
                
                for alt_i in open_facilities:
                    if alt_i == drop_i:
                        continue
                    
                    # Check if alt can serve j
                    if not self._can_serve(alt_i, self.x[alt_i], j):
                        continue
                    
                    current_alt_sites = self._get_num_sites_at_facility(alt_i)
                    new_alt_sites = current_alt_sites + 1
                    
                    res = self.calculate_resource_mix(alt_i, new_alt_sites)
                    if res is None:
                        continue
                    
                    r_new, h_new = res
                    res_old = self.calculate_resource_mix(alt_i, current_alt_sites)
                    r_old, h_old = res_old if res_old else (0, 0)
                    
                    increase = (
                        (r_new - r_old) * self.data['C_ik']['Robot'][alt_i] + 
                        (h_new - h_old) * self.data['C_ik']['Human'][alt_i]
                    )
                    
                    if increase < min_cost_increase:
                        min_cost_increase = increase
                        best_alt = alt_i
                
                if best_alt == -1:
                    redistribution_possible = False
                    break
                else:
                    # Remove drop_i and add best_alt
                    self.assignments[j] = [best_alt]
            
            if redistribution_possible:
                self.x[drop_i] = None
                self._optimize_facility_levels()
                new_cost = self.calculate_total_cost()
                
                if new_cost < current_cost:
                    self._update_facility_state()
                    if self.verbose:
                        print(f"  Drop: closed facility {drop_i}, redistributed {len(sites_at_i)} sites, "
                              f"saving ${current_cost - new_cost:,.2f}")
                    return True
            
            # Revert
            for j in sites_at_i:
                self.assignments[j] = original_assignments[j]
            self.x[drop_i] = original_level
            self._optimize_facility_levels()
                    
        return False

    def _open_move(self):
        """Open Move: Try opening a new facility to reduce travel distances."""
        current_cost = self.calculate_total_cost()
        
        closed_facilities = [i for i in range(self.num_I) if self.x[i] is None]
        
        for new_i in closed_facilities:
            potential_sites = []
            for j in range(self.num_J):
                feasible_levels = self._get_feasible_levels(new_i, j)
                if feasible_levels and self.assignments[j]:
                    current_i = self.assignments[j][0]
                    if current_i >= 0 and self.x[current_i] is not None:
                        current_time = self.data['t_ijl'][self.x[current_i]][current_i][j]
                        new_time = self.data['t_ijl'][feasible_levels[-1]][new_i][j]
                        if new_time < current_time:
                            potential_sites.append((j, current_time - new_time))
            
            if len(potential_sites) < 2:
                continue
            
            potential_sites.sort(key=lambda x: x[1], reverse=True)
            original_assignments = {}
            
            for j, _ in potential_sites[:10]:
                original_assignments[j] = self.assignments[j].copy()
                self.assignments[j] = [new_i]
            
            # Set level for new facility
            sites_for_new = list(original_assignments.keys())
            best_level = self._get_best_level_for_sites(new_i, sites_for_new)
            if best_level:
                self.x[new_i] = best_level
                self._optimize_facility_levels()
                new_cost = self.calculate_total_cost()
                
                if new_cost < current_cost:
                    self._update_facility_state()
                    if self.verbose:
                        print(f"  Open: opened facility {new_i} (level {best_level}) with "
                              f"{len(original_assignments)} sites, saving ${current_cost - new_cost:,.2f}")
                    return True
            
            # Revert
            for j, orig_list in original_assignments.items():
                self.assignments[j] = orig_list
            self.x[new_i] = None
            self._optimize_facility_levels()
                    
        return False

    def local_search(self):
        """
        Stage 2: Local Search Improvement
        """
        if self.verbose:
            print("Stage 2: Local Search Improvement")
        
        iteration = 0
        improving = True
        
        while improving and iteration < self.max_iterations:
            improving = False
            iteration += 1
            
            if self._shift_move():
                improving = True
                continue
            
            if self._swap_move():
                improving = True
                continue
            
            if self._drop_move():
                improving = True
                continue
            
            if self._open_move():
                improving = True
                continue
        
        final_cost = self.calculate_total_cost()
        
        if self.verbose:
            print(f"Local search completed after {iteration} iterations")
            print(f"  Final cost: ${final_cost:,.2f}")
            print(f"  Final facilities: {sum(1 for x in self.x if x is not None)}")
        
        return final_cost
    
    def get_solution(self):
        """Return solution in standard format."""
        opened = [i for i in range(self.num_I) if self.x[i] is not None]
        levels = {i: self.x[i] for i in opened}
        return {
            'opened': opened,
            'levels': levels,
            'assignments': self.assignments,
            'resources': self.resources
        }