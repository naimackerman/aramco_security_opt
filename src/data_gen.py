"""
Data Generator Module for Saudi Aramco Security Optimization.

Supports two modes:
1. Synthetic data: Randomly generated locations around Dhahran
2. Real data: Actual Saudi Aramco facility locations from JSON file
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from geopy.distance import geodesic


class DataGenerator:
    def __init__(self, num_candidates=10, num_demand_sites=50, seed=42, use_real_data=False):
        """
        Initialize the data generator.
        
        Args:
            num_candidates: Number of candidate facility locations (synthetic mode)
            num_demand_sites: Number of demand sites (synthetic mode)
            seed: Random seed for reproducibility
            use_real_data: If True, load real Dhahran location data from JSON
        """
        np.random.seed(seed)
        self.use_real_data = use_real_data
        
        # Dhahran center coordinates (approximately 26.30N, 50.13E for Core Area)
        self.center_lat = 26.30
        self.center_lon = 50.13
        
        if not use_real_data:
            self.num_I = num_candidates
            self.num_J = num_demand_sites
        
        self.levels = ['High', 'Medium', 'Low']
        self.num_L = len(self.levels)
        
        self.level_time_multipliers = {
            'High': 0.5,
            'Medium': 1.0,
            'Low': 1.5
        }
        
        self.level_cost_multipliers = {
            'High': 1.5,
            'Medium': 1.0,
            'Low': 0.5
        }
        
    def _load_real_data(self):
        """Load real location data from JSON file."""
        data_file = Path(__file__).parent.parent / "data" / "raw" / "dhahran_locations.json"
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Real data file not found: {data_file}\n"
                "Please ensure data/raw/dhahran_locations.json exists."
            )
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        return data
        
    def generate_locations(self):
        """Generate or load candidate and demand site locations."""
        
        if self.use_real_data:
            # Load real Dhahran locations
            data = self._load_real_data()
            
            self.I_coords = [(loc['lat'], loc['lon']) for loc in data['candidate_locations']]
            self.J_coords = [(site['lat'], site['lon']) for site in data['demand_sites']]
            
            # Store additional metadata
            self.I_names = [loc['name'] for loc in data['candidate_locations']]
            self.J_names = [site['name'] for site in data['demand_sites']]
            self.J_tiers = [site['tier'] for site in data['demand_sites']]
            
            self.num_I = len(self.I_coords)
            self.num_J = len(self.J_coords)
            
            print(f"Loaded real data: {self.num_I} candidates, {self.num_J} demand sites")
        else:
            # Generate synthetic locations with corridor pattern
            self._generate_corridor_pattern()
            self.I_names = [f"Candidate-{i}" for i in range(self.num_I)]
            self.J_names = [f"Demand-{j}" for j in range(self.num_J)]
    
    def _generate_corridor_pattern(self):
        """Generate demand sites in corridor/pipeline pattern with hubs plus scattered sites."""
        # Generate candidate locations randomly
        def random_coords(n):
            lats = self.center_lat + np.random.uniform(-0.04, 0.04, n)
            lons = self.center_lon + np.random.uniform(-0.04, 0.04, n)
            return list(zip(lats, lons))
        
        self.I_coords = random_coords(self.num_I)
        
        # Split: 70% corridor sites, 30% scattered
        num_corridor_sites = int(self.num_J * 0.7)
        num_scattered = self.num_J - num_corridor_sites
        
        demand_coords = []
        demand_tiers = []
        self.corridors = []
        
        # Number of corridors
        num_corridors = max(2, num_corridor_sites // 12)
        sites_per_corridor = num_corridor_sites // num_corridors
        
        for c in range(num_corridors):
            corridor_sites = []
            
            # Generate two high-critical hub endpoints
            angle = 2 * np.pi * c / num_corridors + np.random.uniform(-0.3, 0.3)
            radius = 0.025 + np.random.uniform(0, 0.015)
            
            hub1_lat = self.center_lat + radius * np.cos(angle)
            hub1_lon = self.center_lon + radius * np.sin(angle)
            
            hub2_lat = self.center_lat + radius * np.cos(angle + np.pi + np.random.uniform(-0.4, 0.4))
            hub2_lon = self.center_lon + radius * np.sin(angle + np.pi + np.random.uniform(-0.4, 0.4))
            
            corridor_points = []
            
            corridor_points.append((0.0, hub1_lat, hub1_lon, 1))
            
            t1 = 0.33
            med1_lat = hub1_lat + t1 * (hub2_lat - hub1_lat) + np.random.uniform(-0.002, 0.002)
            med1_lon = hub1_lon + t1 * (hub2_lon - hub1_lon) + np.random.uniform(-0.002, 0.002)
            corridor_points.append((t1, med1_lat, med1_lon, 2))
            
            t2 = 0.67
            med2_lat = hub1_lat + t2 * (hub2_lat - hub1_lat) + np.random.uniform(-0.002, 0.002)
            med2_lon = hub1_lon + t2 * (hub2_lon - hub1_lon) + np.random.uniform(-0.002, 0.002)
            corridor_points.append((t2, med2_lat, med2_lon, 2))
            
            corridor_points.append((1.0, hub2_lat, hub2_lon, 1))
            
            remaining = sites_per_corridor - 4
            for i in range(max(0, remaining)):
                t = (i + 1) / (remaining + 1)
                if abs(t - 0.33) < 0.03:
                    t = 0.33 - 0.05 if t < 0.33 else 0.33 + 0.05
                if abs(t - 0.67) < 0.03:
                    t = 0.67 - 0.05 if t < 0.67 else 0.67 + 0.05
                t = max(0.03, min(0.97, t))
                
                low_lat = hub1_lat + t * (hub2_lat - hub1_lat)
                low_lon = hub1_lon + t * (hub2_lon - hub1_lon)
                corridor_points.append((t, low_lat, low_lon, 3))
            
            corridor_points.sort(key=lambda x: x[0])
            
            for t, lat, lon, tier in corridor_points:
                idx = len(demand_coords)
                corridor_sites.append(idx)
                demand_coords.append((lat, lon))
                demand_tiers.append(tier)
            
            self.corridors.append(corridor_sites)
        
        # Distribution: high = rare (5%), medium = moderate (25%), low = many (70%)
        for _ in range(num_scattered):
            lat = self.center_lat + np.random.uniform(-0.035, 0.035)
            lon = self.center_lon + np.random.uniform(-0.035, 0.035)
            demand_coords.append((lat, lon))
            demand_tiers.append(np.random.choice([1, 2, 3], p=[0.05, 0.25, 0.70]))
        
        # Fill remaining if needed
        while len(demand_coords) < self.num_J:
            lat = self.center_lat + np.random.uniform(-0.03, 0.03)
            lon = self.center_lon + np.random.uniform(-0.03, 0.03)
            demand_coords.append((lat, lon))
            demand_tiers.append(np.random.choice([1, 2, 3], p=[0.1, 0.3, 0.6]))
        
        # Trim if too many
        self.J_coords = demand_coords[:self.num_J]
        self.J_tiers = demand_tiers[:self.num_J]
        
        # Calculate Base Distance Matrix (d_ij) in kilometers
        self.d_ij = np.zeros((self.num_I, self.num_J))
        for i in range(self.num_I):
            for j in range(self.num_J):
                self.d_ij[i][j] = geodesic(self.I_coords[i], self.J_coords[j]).km
        
        # Calculate Response Time Matrix (t_ijl) in minutes
        self.t_ijl = {}
        for l_idx, level in enumerate(self.levels):
            multiplier = self.level_time_multipliers[level]
            self.t_ijl[level] = self.d_ij * multiplier
        
        # Generate SLA per demand site (S_j)
        self._generate_demand_params()

    def _generate_demand_params(self):
        """Generate demand parameters for each site: SLA, SCU demand, and human/robot mix."""
        # D_j: Demand in SCU (Surveillance Coverage Units) per site
        # alpha_j: Human/robot mix ratio for each site
        #   alpha_j > 1: More humans than robots (high-critical sites)
        #   alpha_j = 1: Equal humans and robots
        #   alpha_j < 1: More robots than humans (low-critical sites)
        
        # Tier-based parameters
        tier_sla = {1: 5.0, 2: 10.0, 3: 15.0} # minutes
        tier_scu_range = {1: (15, 21), 2: (8, 15), 3: (3, 8)} # SCU
        
        if hasattr(self, 'J_tiers'):
            self.S_j = np.array([tier_sla.get(tier, 5.0) for tier in self.J_tiers])
            self.D_j = np.array([np.random.randint(*tier_scu_range.get(tier, (5, 10))) 
                                for tier in self.J_tiers])
            
            alpha_values = []
            for tier in self.J_tiers:
                if tier == 1:
                    val = np.random.uniform(1.01, 2.0)
                elif tier == 2:
                    val = 1.0
                else:
                    val = np.random.uniform(0.1, 0.99)
                alpha_values.append(val)
            self.alpha_j = np.array(alpha_values)
            
        elif not self.use_real_data:
            self.S_j = np.zeros(self.num_J)
            self.D_j = np.zeros(self.num_J, dtype=int)
            self.alpha_j = np.zeros(self.num_J)
            for j in range(self.num_J):
                rand = np.random.rand()
                if rand < 0.2:
                    tier = 1
                    alpha_val = np.random.uniform(1.01, 2.0)
                elif rand < 0.5:
                    tier = 2
                    alpha_val = 1.0
                else:
                    tier = 3
                    alpha_val = np.random.uniform(0.1, 0.99)
                
                self.S_j[j] = tier_sla[tier]
                self.D_j[j] = np.random.randint(*tier_scu_range[tier])
                self.alpha_j[j] = alpha_val

    def generate_params(self, scenario='Balanced'):
        """
        Generate cost, demand, and technology parameters based on scenario.
        
        Args:
            scenario: One of 'Conservative', 'Balanced', or 'Future'
            
        Returns:
            dict: Complete parameter dictionary for optimization model
        """
        # 1. Base Fixed Cost (F_i): Core area locations more expensive
        if self.use_real_data:
            base_F_i = np.array([27000 if i < 5 else 21000 for i in range(self.num_I)])
        else:
            base_F_i = np.array([25000 if i < self.num_I/2 else 20000 for i in range(self.num_I)])
        
        self.F_il = {}
        for level in self.levels:
            multiplier = self.level_cost_multipliers[level]
            self.F_il[level] = base_F_i * multiplier
        
        # 2. Variable Cost (C_ik): Cost per resource type per location
        self.C_ik = {
            'Robot': np.array([800 if i < self.num_I/2 else 750 for i in range(self.num_I)]),
            'Human': np.array([3000 if i < self.num_I/2 else 2800 for i in range(self.num_I)])
        }

        # 3. Capacity Parameters per resource type per LEVEL
        self.MAXCAP_lk = {
            'High': {'Robot': 240, 'Human': 40},
            'Medium': {'Robot': 100, 'Human': 20},
            'Low': {'Robot': 40, 'Human': 10}
        }
        self.MINCAP_lk = {
            'High': {'Robot': 0, 'Human': 20},
            'Medium': {'Robot': 0, 'Human': 10},
            'Low': {'Robot': 0, 'Human': 5}
        }


        # 4. Technology Scenario Settings
        robot_cost_mult = 1.0 # Robot cost reduce as technology advances
        alpha_j_scaler = 1.0 # Human/robot mix ratio reduce (fewer humans per robot) as technology advances
        
        if scenario == 'Conservative':
            self.alpha = 1/3.0
            robot_cost_mult = 1.0
            alpha_j_scaler = 1.0
        elif scenario == 'Balanced':
            self.alpha = 1/5.0
            robot_cost_mult = 0.9
            alpha_j_scaler = 0.75
        elif scenario == 'Future':
            self.alpha = 1/10.0
            robot_cost_mult = 0.8
            alpha_j_scaler = 0.50
            
        self.C_ik['Robot'] = self.C_ik['Robot'] * robot_cost_mult
        
        scenario_alpha_j = self.alpha_j * alpha_j_scaler
            
        return {
            # Sets
            'num_I': self.num_I, 
            'num_J': self.num_J,
            'num_L': self.num_L,
            'levels': self.levels,
            # Response time and SLA
            't_ijl': self.t_ijl,
            'S_j': self.S_j,
            'd_ij': self.d_ij,
            # Demand
            'D_j': self.D_j,
            'alpha_j': scenario_alpha_j,
            # Costs
            'F_il': self.F_il,
            'C_ik': self.C_ik,
            # Capacity
            'MAXCAP_lk': self.MAXCAP_lk,
            'MINCAP_lk': self.MINCAP_lk,
            # Technology / Supervision
            'alpha': self.alpha,
            'level_cost_multipliers': self.level_cost_multipliers,
            'level_time_multipliers': self.level_time_multipliers,
            # Coordinates for visualization
            'coords_I': self.I_coords, 
            'coords_J': self.J_coords,
            'corridors': self.corridors if hasattr(self, 'corridors') else [],
            'names_I': self.I_names if hasattr(self, 'I_names') else None,
            'names_J': self.J_names if hasattr(self, 'J_names') else None,
            'use_real_data': self.use_real_data
        }