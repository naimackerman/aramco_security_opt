"""
Large-Scale Synthetic Data Generator for HRCD-FLP Optimization.

This module provides functionality to:
1. Generate synthetic data at large scale
2. Save generated data to files for later use
3. Visualize generated data

Optimizations for large scale:
- Vectorized distance calculations using NumPy
- Haversine formula for fast distance calculations
- Efficient storage using compressed JSON or Parquet format
"""
import json
import gzip
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
import matplotlib.pyplot as plt
import contextily as ctx

from .config import RESULTS_DIR


# Directory for generated datasets
DATASETS_DIR = RESULTS_DIR / "datasets"
DATASETS_DIR.mkdir(parents=True, exist_ok=True)


class LargeScaleDataGenerator:
    """
    High-performance data generator for large-scale HRCD-FLP instances.
    
    Supports generation of facility/demand sites with optimized distance calculations and data persistence.
    """
    
    def __init__(
        self, 
        num_candidates: int = 100, 
        num_demand_sites: int = 1000, 
        seed: int = 42,
        use_haversine: bool = True,
        batch_size: int = 1000
    ):
        """
        Initialize the large-scale data generator.
        
        Args:
            num_candidates: Number of candidate facility locations
            num_demand_sites: Number of demand sites
            seed: Random seed for reproducibility
            use_haversine: Use fast vectorized Haversine formula (True) or precise geodesic (False)
            batch_size: Batch size for distance calculations
        """
        np.random.seed(seed)
        self.seed = seed
        self.num_I = num_candidates
        self.num_J = num_demand_sites
        self.use_haversine = use_haversine
        self.batch_size = batch_size
        
        # Case study: Dhahran center coordinates
        self.center_lat = 26.30
        self.center_lon = 50.13
        
        # Define spatial spread based on scale
        self._calculate_spatial_spread()
        
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
        
        # Storage for generated data
        self.I_coords = []
        self.J_coords = []
        self.J_tiers = []
        self.corridors = []
        self.d_ij = None
        self.t_ijl = {}
        self.S_j = None
        self.D_j = None
        self.alpha_j = None
        
        print(f"Initialized LargeScaleDataGenerator:")
        print(f"  - Candidates: {self.num_I:,}")
        print(f"  - Demand Sites: {self.num_J:,}")
        print(f"  - Spatial Spread: ±{self.lat_spread:.3f}° lat, ±{self.lon_spread:.3f}° lon")
        print(f"  - Distance Method: {'Haversine (fast)' if use_haversine else 'Geodesic (precise)'}")
        
    def _calculate_spatial_spread(self):
        """
        Calculate appropriate spatial spread based on dataset size.
        
        The goal is to maintain a similar visual density to the original data_gen.py
        which uses ±0.04° for 15 candidates and 50 demand sites (65 total points).
        
        We scale the area proportionally with the number of points to maintain
        constant density (points per km²).
        
        Reference: 0.04° ≈ 4.4 km, so reference area is ~77 km² for 65 points
        Target density: ~0.84 points/km²
        """
        # Reference values from original data_gen.py (15 candidates, 50 demand sites)
        reference_spread = 0.04  # degrees
        reference_total_sites = 65  # 15 + 50
        
        # Current total sites
        total_sites = self.num_I + self.num_J
        
        # Scale spread proportionally with sqrt(total_sites) to maintain density
        # Since area scales with spread², and we want area proportional to num_sites,
        # spread should scale with sqrt(num_sites / reference_sites)
        scale_factor = np.sqrt(total_sites / reference_total_sites)
        
        self.lat_spread = reference_spread * scale_factor
        self.lon_spread = reference_spread * scale_factor
        
        # Calculate approximate area and density for logging
        # 1° latitude ≈ 111 km, 1° longitude at 26°N ≈ 100 km
        area_km2 = (2 * self.lat_spread * 111) * (2 * self.lon_spread * 100)
        density = total_sites / area_km2
        
        print(f"  - Approximate Area: {area_km2:.1f} km²")
        print(f"  - Density: {density:.2f} points/km²")
    
    @staticmethod
    def _haversine_vectorized(lat1: np.ndarray, lon1: np.ndarray, 
                               lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
        """
        Vectorized Haversine formula for fast distance calculation.
        
        Args:
            lat1, lon1: Arrays of source coordinates (shape: N)
            lat2, lon2: Arrays of destination coordinates (shape: M)
            
        Returns:
            Distance matrix in kilometers (shape: N x M)
        """
        R = 6371.0  # Earth's radius in km
        
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Create meshgrid for broadcasting
        lat1_mesh, lat2_mesh = np.meshgrid(lat1_rad, lat2_rad, indexing='ij')
        lon1_mesh, lon2_mesh = np.meshgrid(lon1_rad, lon2_rad, indexing='ij')
        
        dlat = lat2_mesh - lat1_mesh
        dlon = lon2_mesh - lon1_mesh
        
        a = np.sin(dlat / 2)**2 + np.cos(lat1_mesh) * np.cos(lat2_mesh) * np.sin(dlon / 2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _compute_distances_batched(self) -> np.ndarray:
        """
        Compute distance matrix in batches to manage memory.
        
        Returns:
            Distance matrix d_ij of shape (num_I, num_J)
        """
        I_lats = np.array([c[0] for c in self.I_coords])
        I_lons = np.array([c[1] for c in self.I_coords])
        J_lats = np.array([c[0] for c in self.J_coords])
        J_lons = np.array([c[1] for c in self.J_coords])
        
        if self.use_haversine:
            # Fast vectorized computation
            print("  Computing distances using vectorized Haversine...")
            
            # For very large datasets, use batching
            if self.num_I * self.num_J > 10_000_000:
                print(f"    Large dataset detected ({self.num_I * self.num_J:,} pairs), using batching...")
                d_ij = np.zeros((self.num_I, self.num_J))
                
                for i_start in range(0, self.num_I, self.batch_size):
                    i_end = min(i_start + self.batch_size, self.num_I)
                    batch_distances = self._haversine_vectorized(
                        I_lats[i_start:i_end], I_lons[i_start:i_end],
                        J_lats, J_lons
                    )
                    d_ij[i_start:i_end, :] = batch_distances
                    
                    if (i_end) % (self.batch_size * 10) == 0 or i_end == self.num_I:
                        progress = i_end / self.num_I * 100
                        print(f"    Progress: {progress:.1f}% ({i_end:,}/{self.num_I:,} candidates)")
            else:
                d_ij = self._haversine_vectorized(I_lats, I_lons, J_lats, J_lons)
        else:
            # Precise geodesic
            print("  Computing distances using geodesic formula (this may take a while)...")
            from geopy.distance import geodesic
            
            d_ij = np.zeros((self.num_I, self.num_J))
            total_pairs = self.num_I * self.num_J
            computed = 0
            
            for i in range(self.num_I):
                for j in range(self.num_J):
                    d_ij[i][j] = geodesic(self.I_coords[i], self.J_coords[j]).km
                    computed += 1
                    
                if (i + 1) % 100 == 0:
                    progress = computed / total_pairs * 100
                    print(f"    Progress: {progress:.1f}%")
        
        return d_ij
    
    def generate_locations(self):
        """
        Generate candidate and demand site locations at scale.
        
        Uses the SAME algorithm as data_gen.py but with proportional scaling
        to ensure identical results for the same seed when using the same parameters.
        """
        print("\nGenerating locations...")
        
        # Calculate scale factor relative to reference (0.04 degrees for 15+50 sites)
        reference_spread = 0.04
        scale_factor = self.lat_spread / reference_spread
        
        # Generate candidate locations (uniformly distributed)
        print(f"  Generating {self.num_I:,} candidate locations...")
        I_lats = self.center_lat + np.random.uniform(-self.lat_spread, self.lat_spread, self.num_I)
        I_lons = self.center_lon + np.random.uniform(-self.lon_spread, self.lon_spread, self.num_I)
        self.I_coords = list(zip(I_lats, I_lons))
        
        # Generate demand sites with corridor pattern
        print(f"  Generating {self.num_J:,} demand sites with corridor pattern...")
        self._generate_corridor_pattern_scaled(scale_factor)
        
        print(f"  Generated {len(self.I_coords):,} candidates, {len(self.J_coords):,} demand sites")
        
    def _generate_corridor_pattern_scaled(self, scale_factor: float):
        """
        Generate demand sites in corridor/pipeline pattern.
        
        This method uses the EXACT SAME algorithm as data_gen.py._generate_corridor_pattern()
        but with all hardcoded distance values scaled by scale_factor.
        
        This ensures:
        - Identical results when scale_factor = 1.0 (i.e., 15 candidates + 50 demand sites)
        - Proportionally scaled results for larger datasets
        
        Args:
            scale_factor: Ratio of current spread to reference spread
        """
        # Split: 70% corridor sites, 30% scattered (same as data_gen.py)
        num_corridor_sites = int(self.num_J * 0.7)
        num_scattered = self.num_J - num_corridor_sites
        
        demand_coords = []
        demand_tiers = []
        self.corridors = []
        
        # Number of corridors: same formula as data_gen.py
        num_corridors = max(2, num_corridor_sites // 12)
        sites_per_corridor = num_corridor_sites // num_corridors
        
        print(f"  Creating {num_corridors} corridors with ~{sites_per_corridor} sites each...")
        
        for c in range(num_corridors):
            corridor_sites = []
            
            # Generate two high-critical hub endpoints
            angle = 2 * np.pi * c / num_corridors + np.random.uniform(-0.3, 0.3)
            radius = (0.025 + np.random.uniform(0, 0.015)) * scale_factor
            
            hub1_lat = self.center_lat + radius * np.cos(angle)
            hub1_lon = self.center_lon + radius * np.sin(angle)
            
            hub2_lat = self.center_lat + radius * np.cos(angle + np.pi + np.random.uniform(-0.4, 0.4))
            hub2_lon = self.center_lon + radius * np.sin(angle + np.pi + np.random.uniform(-0.4, 0.4))
            
            corridor_points = []
            
            # Hub 1 (High tier) at t=0
            corridor_points.append((0.0, hub1_lat, hub1_lon, 1))
            
            # Medium tier points at t=0.33 and t=0.67
            t1 = 0.33
            med1_lat = hub1_lat + t1 * (hub2_lat - hub1_lat) + np.random.uniform(-0.002, 0.002) * scale_factor
            med1_lon = hub1_lon + t1 * (hub2_lon - hub1_lon) + np.random.uniform(-0.002, 0.002) * scale_factor
            corridor_points.append((t1, med1_lat, med1_lon, 2))
            
            t2 = 0.67
            med2_lat = hub1_lat + t2 * (hub2_lat - hub1_lat) + np.random.uniform(-0.002, 0.002) * scale_factor
            med2_lon = hub1_lon + t2 * (hub2_lon - hub1_lon) + np.random.uniform(-0.002, 0.002) * scale_factor
            corridor_points.append((t2, med2_lat, med2_lon, 2))
            
            # Hub 2 (High tier) at t=1
            corridor_points.append((1.0, hub2_lat, hub2_lon, 1))
            
            # Fill remaining sites along corridor (Low tier)
            remaining = sites_per_corridor - 4
            for i in range(max(0, remaining)):
                t = (i + 1) / (remaining + 1)
                # Avoid overlap with medium tier points
                if abs(t - 0.33) < 0.03:
                    t = 0.33 - 0.05 if t < 0.33 else 0.33 + 0.05
                if abs(t - 0.67) < 0.03:
                    t = 0.67 - 0.05 if t < 0.67 else 0.67 + 0.05
                t = max(0.03, min(0.97, t))
                
                # No offset for low-tier sites
                low_lat = hub1_lat + t * (hub2_lat - hub1_lat)
                low_lon = hub1_lon + t * (hub2_lon - hub1_lon)
                corridor_points.append((t, low_lat, low_lon, 3))
            
            # Sort by position along corridor
            corridor_points.sort(key=lambda x: x[0])
            
            # Add to global lists
            for t, lat, lon, tier in corridor_points:
                idx = len(demand_coords)
                corridor_sites.append(idx)
                demand_coords.append((lat, lon))
                demand_tiers.append(tier)
            
            self.corridors.append(corridor_sites)
        
        # Generate scattered sites
        # Scale the spread: ±0.035 -> scaled
        print(f"  Generating {num_scattered:,} scattered demand sites...")
        scattered_spread = 0.035 * scale_factor
        for _ in range(num_scattered):
            lat = self.center_lat + np.random.uniform(-scattered_spread, scattered_spread)
            lon = self.center_lon + np.random.uniform(-scattered_spread, scattered_spread)
            demand_coords.append((lat, lon))
            demand_tiers.append(np.random.choice([1, 2, 3], p=[0.05, 0.25, 0.70]))
        
        # Fill remaining if needed
        fill_spread = 0.03 * scale_factor
        while len(demand_coords) < self.num_J:
            lat = self.center_lat + np.random.uniform(-fill_spread, fill_spread)
            lon = self.center_lon + np.random.uniform(-fill_spread, fill_spread)
            demand_coords.append((lat, lon))
            demand_tiers.append(np.random.choice([1, 2, 3], p=[0.1, 0.3, 0.6]))
        
        # Trim if too many
        self.J_coords = demand_coords[:self.num_J]
        self.J_tiers = demand_tiers[:self.num_J]
    
    def generate_demand_params(self):
        """
        Generate demand parameters for each site: SLA, SCU demand, and human/robot mix.
        
        Uses the EXACT SAME sequential algorithm as data_gen.py._generate_demand_params()
        to ensure identical results with the same seed.
        """
        print("\nGenerating demand parameters...")
        
        # Tier-based parameters
        tier_sla = {1: 5.0, 2: 10.0, 3: 15.0}  # minutes
        tier_scu_range = {1: (15, 21), 2: (8, 15), 3: (3, 8)}  # SCU
        
        # Generate SLA (deterministic based on tier, no random calls)
        self.S_j = np.array([tier_sla.get(tier, 5.0) for tier in self.J_tiers])
        
        # Generate D_j sequentially
        self.D_j = np.array([np.random.randint(*tier_scu_range.get(tier, (5, 10))) 
                            for tier in self.J_tiers])
        
        # Generate alpha_j sequentially
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
        
        print(f"  SLA range: [{self.S_j.min():.1f}, {self.S_j.max():.1f}] minutes")
        print(f"  SCU range: [{self.D_j.min()}, {self.D_j.max()}]")
        print(f"  Alpha range: [{self.alpha_j.min():.2f}, {self.alpha_j.max():.2f}]")
    
    def compute_distances(self):
        """Compute distance and response time matrices."""
        print("\nComputing distance matrices...")
        
        self.d_ij = self._compute_distances_batched()
        
        print(f"  Distance range: [{self.d_ij.min():.2f}, {self.d_ij.max():.2f}] km")
        
        # Compute response time matrix
        self.t_ijl = {}
        for level in self.levels:
            multiplier = self.level_time_multipliers[level]
            self.t_ijl[level] = self.d_ij * multiplier
        
        print(f"  Response time matrices computed for {len(self.levels)} levels")
    
    def generate_all(self, compute_distances: bool = True) -> Dict[str, Any]:
        """
        Generate all data components.
        
        Args:
            compute_distances: Whether to compute distance matrices (can be skipped for visualization-only)
            
        Returns:
            Dictionary containing all generated data
        """
        import time
        start_time = time.time()
        
        self.generate_locations()
        self.generate_demand_params()
        
        if compute_distances:
            self.compute_distances()
        
        elapsed = time.time() - start_time
        print(f"\nData generation completed in {elapsed:.2f} seconds")
        
        return self.get_data_dict()
    
    def get_data_dict(self, include_distances: bool = True) -> Dict[str, Any]:
        """
        Get all data as a dictionary.
        
        Args:
            include_distances: Whether to include distance matrices (large for big datasets)
            
        Returns:
            Dictionary with all data
        """
        data = {
            'metadata': {
                'num_candidates': self.num_I,
                'num_demand_sites': self.num_J,
                'seed': self.seed,
                'center_lat': self.center_lat,
                'center_lon': self.center_lon,
                'lat_spread': self.lat_spread,
                'lon_spread': self.lon_spread,
                'generated_at': datetime.now().isoformat(),
            },
            'num_I': self.num_I,
            'num_J': self.num_J,
            'num_L': self.num_L,
            'levels': self.levels,
            'coords_I': self.I_coords,
            'coords_J': self.J_coords,
            'J_tiers': [int(t) for t in self.J_tiers],
            'corridors': self.corridors,
            'S_j': self.S_j.tolist() if self.S_j is not None else None,
            'D_j': self.D_j.tolist() if self.D_j is not None else None,
            'alpha_j': self.alpha_j.tolist() if self.alpha_j is not None else None,
            'level_time_multipliers': self.level_time_multipliers,
            'level_cost_multipliers': self.level_cost_multipliers,
        }
        
        if include_distances and self.d_ij is not None:
            data['d_ij'] = self.d_ij.tolist()
            data['t_ijl'] = {level: matrix.tolist() for level, matrix in self.t_ijl.items()}
        
        return data
    
    def save_dataset(
        self, 
        filename: Optional[str] = None, 
        include_distances: bool = True,
        compress: bool = True
    ) -> str:
        """
        Save generated dataset to file.
        
        Args:
            filename: Custom filename (auto-generated if None)
            include_distances: Whether to include distance matrices
            compress: Whether to use gzip compression
            
        Returns:
            Path to saved file
        """
        if filename is None:
            filename = f"dataset_I{self.num_I}_J{self.num_J}_seed{self.seed}"
        
        data = self.get_data_dict(include_distances=include_distances)
        
        if compress:
            filepath = DATASETS_DIR / f"{filename}.json.gz"
            with gzip.open(filepath, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
            print(f"Dataset saved (compressed): {filepath}")
        else:
            filepath = DATASETS_DIR / f"{filename}.json"
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Dataset saved: {filepath}")
        
        # Print file size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        
        return str(filepath)
    
    @classmethod
    def load_dataset(cls, filepath: str) -> 'LargeScaleDataGenerator':
        """
        Load a dataset from file.
        
        Args:
            filepath: Path to the dataset file
            
        Returns:
            LargeScaleDataGenerator instance with loaded data
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.gz':
            with gzip.open(filepath, 'rt', encoding='utf-8') as f:
                data = json.load(f)
        else:
            with open(filepath, 'r') as f:
                data = json.load(f)
        
        # Create instance (this will calculate default spread)
        instance = cls(
            num_candidates=data['num_I'],
            num_demand_sites=data['num_J'],
            seed=data['metadata'].get('seed', 42)
        )
        
        # Override spread with saved values from metadata if available
        if 'lat_spread' in data['metadata']:
            instance.lat_spread = data['metadata']['lat_spread']
        if 'lon_spread' in data['metadata']:
            instance.lon_spread = data['metadata']['lon_spread']
        
        # Load data
        instance.I_coords = [tuple(c) for c in data['coords_I']]
        instance.J_coords = [tuple(c) for c in data['coords_J']]
        instance.J_tiers = data['J_tiers']
        instance.corridors = data['corridors']
        instance.S_j = np.array(data['S_j']) if data.get('S_j') else None
        instance.D_j = np.array(data['D_j']) if data.get('D_j') else None
        instance.alpha_j = np.array(data['alpha_j']) if data.get('alpha_j') else None
        
        if 'd_ij' in data:
            instance.d_ij = np.array(data['d_ij'])
            instance.t_ijl = {level: np.array(matrix) for level, matrix in data['t_ijl'].items()}
        
        print(f"Dataset loaded: {filepath}")
        print(f"  Candidates: {instance.num_I:,}")
        print(f"  Demand Sites: {instance.num_J:,}")
        
        return instance


def visualize_data_only(
    data: Dict[str, Any],
    title: str = "Generated Data",
    save_path: Optional[str] = None,
    show_corridors: bool = True,
    max_sites_to_label: int = 100,
    figsize: Tuple[int, int] = (16, 14)
) -> str:
    """
    Visualize generated data WITHOUT solving the optimization problem.
    
    Args:
        data: Dictionary containing coords_I, coords_J, corridors, D_j, etc.
        title: Title for the plot
        save_path: Path to save figure (auto-generated if None)
        show_corridors: Whether to draw corridor lines
        max_sites_to_label: Maximum number of sites to add labels for (for readability)
        figsize: Figure size
        
    Returns:
        Path to saved figure
    """
    print(f"\nVisualizing data: {data['num_I']:,} candidates, {data['num_J']:,} demand sites")
    
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Convert coordinates
    coords_I = np.array(data['coords_I'])
    coords_J = np.array(data['coords_J'])
    
    # Calculate bounds
    all_lats = np.concatenate([coords_I[:, 0], coords_J[:, 0]])
    all_lons = np.concatenate([coords_I[:, 1], coords_J[:, 1]])
    
    lat_margin = (all_lats.max() - all_lats.min()) * 0.15
    lon_margin = (all_lons.max() - all_lons.min()) * 0.15
    
    ax.set_xlim(all_lons.min() - lon_margin, all_lons.max() + lon_margin)
    ax.set_ylim(all_lats.min() - lat_margin, all_lats.max() + lat_margin)
    
    # Colors
    CC_COLOR = '#2196F3'  # Blue for candidates
    SITE_COLORS = {
        1: '#D32F2F',  # Red - High critical
        2: '#FF9800',  # Orange - Medium
        3: '#4CAF50',  # Green - Low
    }
    
    # 1. Draw corridor lines
    if show_corridors and 'corridors' in data and data['corridors']:
        print(f"  Drawing {len(data['corridors'])} corridors...")
        for corridor in data['corridors']:
            if len(corridor) >= 2:
                for k in range(len(corridor) - 1):
                    j1, j2 = corridor[k], corridor[k + 1]
                    if j1 < len(coords_J) and j2 < len(coords_J):
                        ax.plot([coords_J[j1, 1], coords_J[j2, 1]], 
                                [coords_J[j1, 0], coords_J[j2, 0]], 
                                c='#9E9E9E', linestyle=':', alpha=0.5, linewidth=1.0, zorder=1)
    
    # 2. Plot demand sites by tier
    J_tiers = np.array(data['J_tiers'])
    
    # Size by D_j if available
    if 'D_j' in data and data['D_j'] is not None:
        D_j = np.array(data['D_j'])
        d_min, d_max = D_j.min(), D_j.max()
        if d_max > d_min:
            d_normalized = (D_j - d_min) / (d_max - d_min)
        else:
            d_normalized = np.ones_like(D_j) * 0.5
        sizes = 20 + d_normalized * 80
    else:
        sizes = np.ones(len(coords_J)) * 40
    
    for tier, color, label in [(1, SITE_COLORS[1], 'High-Critical'),
                                (2, SITE_COLORS[2], 'Medium-Critical'),
                                (3, SITE_COLORS[3], 'Low-Critical')]:
        mask = J_tiers == tier
        if mask.any():
            ax.scatter(coords_J[mask, 1], coords_J[mask, 0],
                      c=color, marker='o', s=sizes[mask],
                      label=f'{label} ({mask.sum():,})',
                      alpha=0.7, edgecolors='black', linewidths=0.5, zorder=3)
    
    # 3. Plot candidate locations
    ax.scatter(coords_I[:, 1], coords_I[:, 0],
              c=CC_COLOR, marker='s', s=80,
              label=f'Candidates ({len(coords_I):,})',
              alpha=0.9, edgecolors='white', linewidths=1.5, zorder=4)
    
    # 4. Add basemap
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", 
                       source=ctx.providers.CartoDB.PositronNoLabels,
                       zoom='auto', alpha=0.5)
    except Exception as e:
        print(f"  Note: Could not load basemap: {e}")
        ax.grid(True, color='#CCCCCC', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_axis_off()
    
    # 5. Title and legend
    ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    
    # 6. Stats box
    stats_lines = [
        f"Scale: {data['num_I']:,} × {data['num_J']:,}",
        f"Total Sites: {data['num_I'] + data['num_J']:,}",
        f"Corridors: {len(data.get('corridors', []))}",
    ]
    if 'D_j' in data and data['D_j'] is not None:
        D_j = np.array(data['D_j'])
        stats_lines.append(f"Total SCU: {D_j.sum():,}")
    
    stats_text = "\n".join(stats_lines)
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=14,
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                    edgecolor='gray', alpha=0.95))
    
    plt.tight_layout()
    
    # Save figure
    if save_path is None:
        filename = f"data_preview_I{data['num_I']}_J{data['num_J']}.png"
        save_path = DATASETS_DIR / filename
    else:
        save_path = Path(save_path)
    
    plt.savefig(save_path, format='png', bbox_inches='tight', 
               facecolor='white', dpi=150)
    plt.close()
    
    print(f"Visualization saved: {save_path}")
    return str(save_path)


def generate_and_visualize(
    num_candidates: int = 100,
    num_demand_sites: int = 1000,
    seed: int = 42,
    save_dataset: bool = True,
    compute_distances: bool = False,
    title: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """
    Convenience function to generate data and visualize without solving.
    
    Args:
        num_candidates: Number of candidate facility locations
        num_demand_sites: Number of demand sites
        seed: Random seed
        save_dataset: Whether to save the dataset to file
        compute_distances: Whether to compute distance matrices (slow for large datasets)
        title: Custom title for visualization
        
    Returns:
        Tuple of (visualization_path, dataset_path or None)
    """
    print("=" * 60)
    print(f"LARGE-SCALE DATA GENERATION")
    print("=" * 60)
    
    # Generate data
    generator = LargeScaleDataGenerator(
        num_candidates=num_candidates,
        num_demand_sites=num_demand_sites,
        seed=seed
    )
    
    data = generator.generate_all(compute_distances=compute_distances)
    
    # Visualize
    if title is None:
        title = f"Synthetic Data: {num_candidates:,} Candidates × {num_demand_sites:,} Sites"
    
    viz_path = visualize_data_only(data, title=title)
    
    # Save dataset
    dataset_path = None
    if save_dataset:
        dataset_path = generator.save_dataset(include_distances=compute_distances)
    
    print("=" * 60)
    print("GENERATION COMPLETE")
    print(f"  Visualization: {viz_path}")
    if dataset_path:
        print(f"  Dataset: {dataset_path}")
    print("=" * 60)
    
    return viz_path, dataset_path


# CLI interface
def main():
    """Command-line interface for large-scale data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate and visualize large-scale HRCD-FLP synthetic data'
    )
    parser.add_argument(
        '-c', '--candidates', 
        type=int, 
        default=100,
        help='Number of candidate facility locations (default: 100)'
    )
    parser.add_argument(
        '-d', '--demand-sites', 
        type=int, 
        default=1000,
        help='Number of demand sites (default: 1000)'
    )
    parser.add_argument(
        '-s', '--seed', 
        type=int, 
        default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--no-save', 
        action='store_true',
        help='Do not save dataset to file'
    )
    parser.add_argument(
        '--compute-distances', 
        action='store_true',
        help='Compute distance matrices (slow for large datasets)'
    )
    parser.add_argument(
        '--load', 
        type=str,
        help='Load and visualize an existing dataset file'
    )
    parser.add_argument(
        '-t', '--title', 
        type=str,
        help='Custom title for visualization'
    )
    
    args = parser.parse_args()
    
    if args.load:
        # Load existing dataset
        generator = LargeScaleDataGenerator.load_dataset(args.load)
        data = generator.get_data_dict(include_distances=False)
        title = args.title or f"Dataset: {generator.num_I:,} × {generator.num_J:,}"
        visualize_data_only(data, title=title)
    else:
        # Generate new data
        generate_and_visualize(
            num_candidates=args.candidates,
            num_demand_sites=args.demand_sites,
            seed=args.seed,
            save_dataset=not args.no_save,
            compute_distances=args.compute_distances,
            title=args.title
        )


if __name__ == "__main__":
    main()
