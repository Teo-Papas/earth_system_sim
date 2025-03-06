"""
Integration tests for Earth system simulation.
Verifies connections and interactions between all components.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys
import yaml
import tempfile
import os

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.physical import PINN
from src.models.biosphere import BiospherePolicy
from src.models.geosphere import GeospherePolicy
from src.integration.temporal_sync import TemporalSynchronizer, create_default_timescales
from src.integration.data_flow import DataFlowManager
from src.visualization.physical_vis import PhysicalSystemVisualizer
from src.visualization.biosphere_vis import BiosphereVisualizer
from src.visualization.geosphere_vis import GeosphereVisualizer
from scripts.run_simulation import EarthSystemSimulation

@pytest.fixture
def config():
    """Load configuration for testing."""
    config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

@pytest.fixture
def device():
    """Get compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def sim(config, device):
    """Create simulation instance."""
    return EarthSystemSimulation(
        config_path=str(Path(__file__).parent.parent / 'config' / 'model_config.yaml'),
        device=device
    )

def test_component_initialization(sim, device):
    """Test that all components initialize correctly."""
    # Check physical system
    assert isinstance(sim.physical, PINN)
    assert next(sim.physical.parameters()).device == device
    
    # Check biosphere system
    assert isinstance(sim.biosphere, BiospherePolicy)
    assert next(sim.biosphere.parameters()).device == device
    
    # Check geosphere system
    assert isinstance(sim.geosphere, GeospherePolicy)
    assert next(sim.geosphere.parameters()).device == device
    
    # Check integration components
    assert isinstance(sim.synchronizer, TemporalSynchronizer)
    assert isinstance(sim.data_flow, DataFlowManager)

def test_data_flow(sim, device):
    """Test data flow between components."""
    # Initialize test states
    physical_state = torch.randn(
        1, sim.config['physical_system']['input_dim'],
        sim.config['grid_height'], sim.config['grid_width'],
        device=device
    )
    
    biosphere_state = torch.randn(
        1, sim.config['biosphere']['state_dim'],
        device=device
    )
    
    geosphere_state = torch.randn(
        1, sim.config['geosphere']['state_dim'],
        device=device
    )
    
    # Test data flow updates
    sim.data_flow.update_state('physical', physical_state)
    sim.data_flow.update_state('biosphere', biosphere_state)
    sim.data_flow.update_state('geosphere', geosphere_state)
    
    # Verify state transformations
    bio_input = sim.data_flow.get_state_for_component('physical', 'biosphere')
    assert bio_input is not None
    assert bio_input.device == device
    
    geo_input = sim.data_flow.get_state_for_component('physical', 'geosphere')
    assert geo_input is not None
    assert geo_input.device == device

def test_temporal_synchronization(sim):
    """Test temporal synchronization between components."""
    # Run multiple timesteps
    for _ in range(100):
        updates = sim.synchronizer.step()
        
        # Physical system should update most frequently
        assert updates['physical']
        
        # Check relative timescales
        phys_time = sim.synchronizer.current_times['physical']
        bio_time = sim.synchronizer.current_times['biosphere']
        geo_time = sim.synchronizer.current_times['geosphere']
        
        assert bio_time <= phys_time
        assert geo_time <= bio_time

def test_conservation_laws(sim, device):
    """Test physical conservation laws."""
    # Create test input
    batch_size = 4
    seq_len = 5
    input_state = torch.randn(
        batch_size, seq_len,
        sim.config['physical_system']['input_dim'],
        sim.config['grid_height'],
        sim.config['grid_width'],
        device=device
    )
    
    # Get PINN predictions
    predictions, physics_losses = sim.physical(input_state)
    
    # Check conservation losses
    assert 'mass' in physics_losses
    assert 'energy' in physics_losses
    assert 'momentum' in physics_losses
    
    # Losses should be non-negative
    for loss in physics_losses.values():
        assert loss >= 0

def test_full_simulation_step(sim, device):
    """Test complete simulation step with all components."""
    # Initialize states
    physical_state, biosphere_state, geosphere_state = sim._initialize_states()
    
    # Run one simulation step
    new_physical, new_bio, new_geo = sim.run_timestep(
        physical_state,
        biosphere_state,
        geosphere_state
    )
    
    # Check outputs
    assert new_physical.shape == physical_state.shape
    assert new_bio.shape == biosphere_state.shape
    assert new_geo.shape == geosphere_state.shape
    
    # Check state validation
    assert sim.data_flow.validate_state('physical', new_physical)
    assert sim.data_flow.validate_state('biosphere', new_bio)
    assert sim.data_flow.validate_state('geosphere', new_geo)

def test_visualization_pipeline():
    """Test visualization pipeline with sample data."""
    # Create temporary directory for test outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample data
        data = {
            'physical_states': np.random.randn(10, 32, 32, 5),
            'biosphere_states': np.random.randn(10, 5),
            'geosphere_states': np.random.randn(10, 3),
            'times': [{'physical': t, 'biosphere': t, 'geosphere': t} 
                     for t in range(10)]
        }
        
        # Save sample data
        data_path = os.path.join(tmpdir, 'test_data.npz')
        np.savez(data_path, **data)
        
        # Load visualization config
        config_path = Path(__file__).parent.parent / 'config' / 'visualization_config.yaml'
        with open(config_path, 'r') as f:
            viz_config = yaml.safe_load(f)
        
        # Create visualizers
        phys_vis = PhysicalSystemVisualizer(data, viz_config, tmpdir)
        bio_vis = BiosphereVisualizer(data, viz_config, tmpdir)
        geo_vis = GeosphereVisualizer(data, viz_config, tmpdir)
        
        # Test basic plotting
        fig, _ = phys_vis.plot_temperature_field(data['physical_states'][..., 0])
        assert fig is not None
        
        fig, _ = bio_vis.plot_vegetation_distribution(data['biosphere_states'][..., 0])
        assert fig is not None
        
        fig, _ = geo_vis.plot_topography_3d(data['geosphere_states'][..., 0])
        assert fig is not None
        
        # Check output files
        output_files = os.listdir(tmpdir)
        assert len(output_files) > 0

def test_end_to_end_simulation(sim):
    """Test end-to-end simulation for a few timesteps."""
    # Run short simulation
    trajectory = sim.run_simulation(num_steps=10, save_frequency=5)
    
    # Check trajectory contents
    assert 'physical' in trajectory
    assert 'biosphere' in trajectory
    assert 'geosphere' in trajectory
    assert 'times' in trajectory
    
    # Check trajectory lengths
    assert len(trajectory['physical']) == 2  # Saved twice (at step 5 and 10)
    assert len(trajectory['biosphere']) == 2
    assert len(trajectory['geosphere']) == 2
    assert len(trajectory['times']) == 2

if __name__ == '__main__':
    pytest.main([__file__, '-v'])