"""
Visualization Module for Solution Plotting.

Creates geographic network plots showing:
- Command center locations (active vs unused)
- Demand site locations (sized by SLA)
- Assignment connections between centers and demand sites
- Monochrome black and white theme for academic papers
"""
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
from .config import FIGURES_DIR

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42


def plot_solution(data, opened_facilities, assignments, title="Optimization Result", 
                  save_format="pdf", facility_levels=None, resources=None):
    """
    Visualize the location map and assignments on a geographic basemap.
    
    Args:
        data: Dictionary containing coordinate and demand data
        opened_facilities: List of opened facility indices (x_i = 1)
        assignments: Mapping of demand site j to facility i (list format: assignments[j] = i)
        title: Plot title string
        save_format: Output format - 'pdf' (recommended for LaTeX) or 'png'
        facility_levels: Optional dict mapping facility index to level
        resources: Optional dict mapping facility index to {'robot': n, 'human': m}
        
    Returns:
        str: Path to saved figure
    """
    coords_I = np.array(data['coords_I'])  # (lat, lon)
    coords_J = np.array(data['coords_J'])  # (lat, lon)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    all_lats = np.concatenate([coords_I[:, 0], coords_J[:, 0]])
    all_lons = np.concatenate([coords_I[:, 1], coords_J[:, 1]])
    
    lat_margin = (all_lats.max() - all_lats.min()) * 0.15
    lon_margin = (all_lons.max() - all_lons.min()) * 0.15
    
    ax.set_xlim(all_lons.min() - lon_margin, all_lons.max() + lon_margin)
    ax.set_ylim(all_lats.min() - lat_margin, all_lats.max() + lat_margin)
    
    COLOR_BLACK = '#000000'
    COLOR_DARK_GRAY = '#404040'
    COLOR_MEDIUM_GRAY = '#808080'
    COLOR_LIGHT_GRAY = '#C0C0C0'
    COLOR_WHITE = '#FFFFFF'
    
    # Command Center colors (blue gradient by level)
    CC_COLORS = {
        'High': '#1A237E',    # Dark blue (highest level)
        'Medium': '#3F51B5',  # Medium blue
        'Low': '#90CAF9'      # Light blue (lowest level)
    }
    
    # Demand site colors by criticality (red gradient - dark to light)
    SITE_COLORS = {
        'high': '#8B0000',    # Dark red (high-critical)
        'medium': '#D32F2F',  # Medium red (standard)
        'low': '#EF9A9A'      # Light red (low-critical)
    }
    
    def get_site_criticality(j):
        """Determine site criticality based on D_j (SCU demand)."""
        if 'D_j' in data:
            d = data['D_j'][j]
            # Match tier_scu_range: (15-20) high, (8-14) medium, (3-7) low
            if d >= 15:
                return 'high'
            elif d >= 8:
                return 'medium'
            else:
                return 'low'
        return 'medium'
    
    # 0. Draw Corridor Lines (dotted lines connecting sites in each corridor)
    if 'corridors' in data and data['corridors']:
        for corridor in data['corridors']:
            if len(corridor) >= 2:
                for k in range(len(corridor) - 1):
                    j1, j2 = corridor[k], corridor[k + 1]
                    if j1 < len(coords_J) and j2 < len(coords_J):
                        ax.plot([coords_J[j1, 1], coords_J[j2, 1]], 
                                [coords_J[j1, 0], coords_J[j2, 0]], 
                                c='#808080', linestyle=':', alpha=0.7, linewidth=2.0, zorder=1)
    
    # 1. Draw Assignment Lines (draw first so they appear behind points)
    for j, assigned in enumerate(assignments):
        if isinstance(assigned, list):
            facilities = assigned
        else:
            facilities = [assigned] if assigned != -1 else []
        
        # Use site criticality color for assignment line
        site_crit = get_site_criticality(j)
        line_color = SITE_COLORS.get(site_crit, COLOR_MEDIUM_GRAY)
        
        for i in facilities:
            if i >= 0:
                start_point = coords_I[i]
                end_point = coords_J[j]
                ax.plot([start_point[1], end_point[1]], 
                        [start_point[0], end_point[0]], 
                        c=line_color, linestyle='-', alpha=0.6, linewidth=1.0, zorder=2)
    
    # 2. Plot All Candidate Locations (Light gray = Not Built)
    ax.scatter(coords_I[:, 1], coords_I[:, 0], 
               c=COLOR_LIGHT_GRAY, marker='s', s=120, 
               label='Unused Candidate', edgecolors=COLOR_DARK_GRAY, linewidths=1.5, zorder=4)

    # 3. Plot Opened Command Centers (blue gradient by level)
    if len(opened_facilities) > 0:
        for idx in opened_facilities:
            level = facility_levels.get(idx, 'Medium') if facility_levels else 'Medium'
            cc_color = CC_COLORS.get(level, CC_COLORS['Medium'])
            
            ax.scatter(coords_I[idx, 1], coords_I[idx, 0], 
                       c=cc_color, marker='s', s=200, 
                       edgecolors=COLOR_WHITE, linewidths=2, zorder=6)
            
            # Label with level and resources
            level_short = level[0] if level else 'M'
            text_color = COLOR_WHITE
            
            # Build label text
            if resources and idx in resources:
                r = resources[idx].get('robot', 0)
                h = resources[idx].get('human', 0)
                label_text = f"CC-{idx}({level_short})\nR:{r} H:{h}"
            else:
                label_text = f"CC-{idx}({level_short})"
            
            ax.annotate(label_text, 
                       xy=(coords_I[idx, 1], coords_I[idx, 0]),
                       xytext=(0, 12), textcoords='offset points',
                       fontsize=8, ha='center', fontweight='bold',
                       color=text_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=cc_color, 
                                edgecolor=COLOR_WHITE, alpha=0.95),
                       zorder=7)
        
        # Add legend entries for CC levels
        ax.scatter([], [], c=CC_COLORS['High'], marker='s', s=100, label='High Level CC',
                  edgecolors=COLOR_WHITE, linewidths=1.5)
        ax.scatter([], [], c=CC_COLORS['Medium'], marker='s', s=100, label='Medium Level CC',
                  edgecolors=COLOR_WHITE, linewidths=1.5)
        ax.scatter([], [], c=CC_COLORS['Low'], marker='s', s=100, label='Low Level CC',
                  edgecolors=COLOR_WHITE, linewidths=1.5)

    # 4. Plot Demand Sites (circles, colored by criticality)
    high_crit_j = [j for j in range(len(coords_J)) if get_site_criticality(j) == 'high']
    med_crit_j = [j for j in range(len(coords_J)) if get_site_criticality(j) == 'medium']
    low_crit_j = [j for j in range(len(coords_J)) if get_site_criticality(j) == 'low']
    
    # Plot each group with different color
    if high_crit_j:
        ax.scatter(coords_J[high_crit_j, 1], coords_J[high_crit_j, 0], 
                   c=SITE_COLORS['high'], marker='o', s=100, 
                   label='High-Critical Site', alpha=0.9, edgecolors=COLOR_BLACK, linewidths=1.5, zorder=5)
    if med_crit_j:
        ax.scatter(coords_J[med_crit_j, 1], coords_J[med_crit_j, 0], 
                   c=SITE_COLORS['medium'], marker='o', s=80, 
                   label='Standard Site', alpha=0.9, edgecolors=COLOR_BLACK, linewidths=1.5, zorder=5)
    if low_crit_j:
        ax.scatter(coords_J[low_crit_j, 1], coords_J[low_crit_j, 0], 
                   c=SITE_COLORS['low'], marker='o', s=60, 
                   label='Low-Critical Site', alpha=0.9, edgecolors=COLOR_BLACK, linewidths=1.5, zorder=5)
    
    # 5. Add SCU demand labels on demand sites
    if 'D_j' in data:
        for j in range(len(coords_J)):
            d = int(data['D_j'][j])
            ax.annotate(f"{d}", 
                       xy=(coords_J[j, 1], coords_J[j, 0]),
                       xytext=(0, -12), textcoords='offset points',
                       fontsize=7, ha='center', fontweight='bold',
                       color=COLOR_BLACK,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor=COLOR_WHITE, 
                                edgecolor=COLOR_DARK_GRAY, alpha=0.8),
                       zorder=8)

    # 6. Add basemap with better fallback options
    try:
        ctx.add_basemap(ax, crs="EPSG:4326", 
                       source=ctx.providers.CartoDB.PositronNoLabels,
                       zoom=13, alpha=0.6)
    except Exception:
        try:
            ctx.add_basemap(ax, crs="EPSG:4326", 
                           source=ctx.providers.OpenStreetMap.Mapnik,
                           zoom=13, alpha=0.5)
        except Exception:
            ax.set_facecolor('#F5F5F5')
            ax.grid(True, color=COLOR_MEDIUM_GRAY, linestyle='-', linewidth=0.5, alpha=0.5)
            print("  Note: Could not load basemap tiles. Using grid background.")

    ax.set_axis_off()
    
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95)
    legend.get_frame().set_facecolor(COLOR_WHITE)
    legend.get_frame().set_edgecolor(COLOR_DARK_GRAY)
    
    # Calculate total resources
    total_robots = 0
    total_humans = 0
    if resources:
        for idx in opened_facilities:
            if idx in resources:
                total_robots += resources[idx].get('robot', 0)
                total_humans += resources[idx].get('human', 0)
    
    # Add summary stats as text box
    num_open = len(opened_facilities)
    num_sites = len(coords_J)
    stats_text = (f"Open Facilities: {num_open}/{len(coords_I)}\n"
                  f"Demand Sites: {num_sites}\n"
                  f"Total Robots: {total_robots}\n"
                  f"Total Humans: {total_humans}")
    ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='left',
           color=COLOR_BLACK,
           bbox=dict(boxstyle='round,pad=0.5', facecolor=COLOR_WHITE, 
                    edgecolor=COLOR_DARK_GRAY, alpha=0.95))
    
    plt.tight_layout()
    
    filename_base = f"result_{title.replace(' ', '_').lower()}"
    
    if save_format.lower() == "pdf":
        filename = f"{filename_base}.pdf"
        filepath = FIGURES_DIR / filename
        plt.savefig(filepath, format='pdf', bbox_inches='tight', 
                   facecolor='white', dpi=150)
        print(f"PDF visualization saved to: {filepath}")
    else:
        filename = f"{filename_base}.png"
        filepath = FIGURES_DIR / filename
        plt.savefig(filepath, format='png', bbox_inches='tight', 
                   facecolor='white', dpi=300)
        print(f"PNG visualization saved to: {filepath}")
    
    plt.close()
    
    return str(filepath)


def regenerate_all_figures_as_pdf():
    """
    Utility function to regenerate existing PNG figures as PDFs.
    This is useful for converting existing results to LaTeX-friendly format.
    """
    import os
    from pathlib import Path
    
    png_files = list(FIGURES_DIR.glob("*.png"))
    
    if not png_files:
        print("No PNG files found in figures directory.")
        return []
    
    pdf_paths = []
    for png_path in png_files:
        pdf_filename = png_path.stem + ".pdf"
        pdf_path = FIGURES_DIR / pdf_filename
        
        try:
            from PIL import Image
            img = Image.open(png_path)
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            img.save(pdf_path, 'PDF', resolution=150, optimize=True)
            
            png_size = png_path.stat().st_size / (1024 * 1024)
            pdf_size = pdf_path.stat().st_size / (1024 * 1024)
            
            print(f"Converted: {png_path.name} ({png_size:.2f}MB) -> {pdf_filename} ({pdf_size:.2f}MB)")
            pdf_paths.append(str(pdf_path))
            
        except Exception as e:
            print(f"Error converting {png_path.name}: {e}")
    
    return pdf_paths