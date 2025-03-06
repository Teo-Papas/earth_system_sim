"""
Script for creating comprehensive visualizations of Earth system simulation results.
"""

import numpy as np
from pathlib import Path
import sys
import yaml
import argparse
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.visualization.physical_vis import PhysicalSystemVisualizer
from src.visualization.biosphere_vis import BiosphereVisualizer
from src.visualization.geosphere_vis import GeosphereVisualizer

def load_simulation_data(filepath: str) -> dict:
    """
    Load simulation results from NPZ file.
    
    Args:
        filepath: Path to simulation results file
        
    Returns:
        Dictionary of simulation data
    """
    data = np.load(filepath)
    return {key: data[key] for key in data.files}

def create_visualizations(
    data: dict,
    config: dict,
    output_dir: str,
    make_animations: bool = True
):
    """
    Create comprehensive visualizations of simulation results.
    
    Args:
        data: Dictionary of simulation data
        config: Visualization configuration
        output_dir: Output directory for visualizations
        make_animations: Whether to create animations
    """
    # Create visualizers
    phys_vis = PhysicalSystemVisualizer(data, config, output_dir)
    bio_vis = BiosphereVisualizer(data, config, output_dir)
    geo_vis = GeosphereVisualizer(data, config, output_dir)
    
    # Extract time points
    times = np.array([t['physical'] for t in data['times']])
    
    print("Creating physical system visualizations...")
    
    # Physical system visualizations
    # Temperature field plot and animation
    fig, _ = phys_vis.plot_temperature_field(
        data['physical_states'][..., 0]  # Temperature channel
    )
    phys_vis.save_figure(fig, 'temperature_field.png')
    
    if make_animations:
        anim = phys_vis.create_wind_field_animation(
            data['physical_states'][..., 3],  # U wind
            data['physical_states'][..., 4]   # V wind
        )
        phys_vis.save_animation(anim, 'wind_field.gif')
    
    # Interactive temperature evolution
    fig = phys_vis.create_interactive_temperature_plot(
        data['physical_states'][..., 0],
        times
    )
    phys_vis.save_interactive_figure(fig, 'temperature_evolution.html')
    
    print("Creating biosphere visualizations...")
    
    # Biosphere visualizations
    # Vegetation distribution plot and animation
    fig, _ = bio_vis.plot_vegetation_distribution(
        data['biosphere_states'][..., 0]  # Vegetation channel
    )
    bio_vis.save_figure(fig, 'vegetation_distribution.png')
    
    if make_animations:
        anim = bio_vis.create_vegetation_animation(
            data['biosphere_states'][..., 0]
        )
        bio_vis.save_animation(anim, 'vegetation_evolution.gif')
    
    # Carbon flux time series
    fig, _ = bio_vis.plot_carbon_flux_timeseries(
        data['biosphere_states'][..., 2],  # Carbon flux channel
        times
    )
    bio_vis.save_figure(fig, 'carbon_flux.png')
    
    # Interactive ecosystem dashboard
    fig = bio_vis.create_interactive_ecosystem_dashboard(
        data['biosphere_states'][..., 0],  # Vegetation
        data['biosphere_states'][..., 2],  # Carbon flux
        data['biosphere_states'][..., 1],  # Soil moisture
        times
    )
    bio_vis.save_interactive_figure(fig, 'ecosystem_dashboard.html')
    
    print("Creating geosphere visualizations...")
    
    # Geosphere visualizations
    # 3D topography plot
    fig, _ = geo_vis.plot_topography_3d(
        data['geosphere_states'][..., 0]  # Topography channel
    )
    geo_vis.save_figure(fig, 'topography_3d.png')
    
    if make_animations:
        anim = geo_vis.create_erosion_animation(
            data['geosphere_states'][..., 0],  # Topography
            data['geosphere_states'][..., 2]   # Erosion rate
        )
        geo_vis.save_animation(anim, 'erosion_process.gif')
    
    # Interactive topography viewer
    fig = geo_vis.create_interactive_topography_viewer(
        data['geosphere_states'][..., 0],
        times
    )
    geo_vis.save_interactive_figure(fig, 'topography_viewer.html')
    
    # Long-term trends
    mean_elevation = np.mean(
        data['geosphere_states'][..., 0],
        axis=(1, 2)
    )
    erosion_total = np.cumsum(
        np.mean(data['geosphere_states'][..., 2], axis=(1, 2))
    )
    
    fig, _ = geo_vis.plot_long_term_trends(
        mean_elevation,
        erosion_total,
        times
    )
    geo_vis.save_figure(fig, 'geological_trends.png')
    
    print("Creating cross-component visualizations...")
    
    # Cross-component correlations
    fig, _ = bio_vis.create_spatial_correlation_plot(
        data['biosphere_states'][..., 0],    # Vegetation
        data['physical_states'][..., 0],     # Temperature
        data['physical_states'][..., 2]      # Humidity
    )
    bio_vis.save_figure(fig, 'ecosystem_climate_correlations.png')
    
    print("Visualization complete!")

def main():
    parser = argparse.ArgumentParser(
        description='Create visualizations for Earth system simulation results'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to simulation results (.npz file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to visualization config file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--no-animations',
        action='store_true',
        help='Skip creating animations'
    )
    
    args = parser.parse_args()
    
    # Load data and config
    data = load_simulation_data(args.data)
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create visualizations
    create_visualizations(
        data,
        config,
        args.output_dir,
        not args.no_animations
    )

if __name__ == "__main__":
    main()