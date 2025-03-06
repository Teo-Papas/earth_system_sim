"""
Basic tests for Earth system simulation components.
"""

import torch
import pytest
import sys
from pathlib import Path
import yaml

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.physical.conv_lstm import ConvLSTM
from src.models.physical.pinn_module import PINN
from src.models.biosphere.policy_network import BiospherePolicy
from src.models.geosphere.slow_policy import GeospherePolicy
from src.integration.temporal_sync import TemporalSynchronizer, create_default_timescales
from src.integration.data_flow import DataFlowManager

@pytest.fixture
def device():
    """Fixture for compute device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def config():
    """Fixture for configuration."""
    config_path = Path(__file__).parent.parent / 'config' / 'model_config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def test_convlstm(device):
    """Test ConvLSTM basic functionality."""
    # Test parameters
    batch_size = 4
    seq_len = 10
    input_dim = 5
    hidden_dims = [16, 16]
    kernel_size = 3
    height, width = 32, 32
    
    # Create model
    model = ConvLSTM(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        kernel_size=kernel_size,
        num_layers=len(hidden_dims),
        batch_first=True
    ).to(device)
    
    # Create dummy input
    x = torch.randn(batch_size, seq_len, input_dim, height, width, device=device)
    
    # Forward pass
    output, hidden_states = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_len, hidden_dims[-1], height, width)
    assert len(hidden_states) == len(hidden_dims)

def test_pinn(device, config):
    """Test PINN basic functionality."""
    # Create model
    model = PINN(
        input_dim=config['physical_system']['input_dim'],
        hidden_dims=config['physical_system']['hidden_dims'],
        kernel_size=config['physical_system']['kernel_size'],
        num_layers=config['physical_system']['num_layers']
    ).to(device)
    
    # Create dummy input
    batch_size = 4
    seq_len = 5
    height, width = config['grid_height'], config['grid_width']
    x = torch.randn(
        batch_size, seq_len,
        config['physical_system']['input_dim'],
        height, width,
        device=device
    )
    
    # Forward pass
    predictions, physics_losses = model(x)
    
    # Check outputs
    assert predictions.shape[0] == batch_size
    assert all(isinstance(loss, torch.Tensor) for loss in physics_losses.values())

def test_biosphere_policy(device, config):
    """Test BiospherePolicy basic functionality."""
    # Create model
    model = BiospherePolicy(
        state_dim=config['biosphere']['state_dim'],
        action_dim=config['biosphere']['action_dim'],
        hidden_dims=config['biosphere']['hidden_dims']
    ).to(device)
    
    # Create dummy input
    batch_size = 4
    state = torch.randn(batch_size, config['biosphere']['state_dim'], device=device)
    
    # Test action sampling
    action, value, dist = model.act(state)
    
    # Check outputs
    assert action.shape == (batch_size, config['biosphere']['action_dim'])
    assert value.shape == (batch_size, 1)
    assert isinstance(dist, torch.distributions.Distribution)

def test_geosphere_policy(device, config):
    """Test GeospherePolicy basic functionality."""
    # Create model
    model = GeospherePolicy(
        state_dim=config['geosphere']['state_dim'],
        action_dim=config['geosphere']['action_dim'],
        hidden_dim=config['geosphere']['hidden_dim']
    ).to(device)
    
    # Create dummy input
    batch_size = 4
    seq_len = 3
    state = torch.randn(
        batch_size, seq_len,
        config['geosphere']['state_dim'],
        device=device
    )
    
    # Forward pass
    dist, value, hidden = model(state)
    
    # Check outputs
    assert isinstance(dist, torch.distributions.Distribution)
    assert value.shape == (batch_size, seq_len, 1)
    assert isinstance(hidden, tuple) and len(hidden) == 2

def test_temporal_synchronizer():
    """Test TemporalSynchronizer basic functionality."""
    # Create synchronizer
    timescales = create_default_timescales()
    state_dims = {'physical': 5, 'biosphere': 3, 'geosphere': 2}
    sync = TemporalSynchronizer(timescales, state_dims)
    
    # Run a few steps
    for _ in range(10):
        updates = sync.step()
        # Physical should update every step
        assert updates['physical']
        
    # Check relative times
    assert sync.get_relative_time('physical', 'biosphere') > 0
    assert sync.get_relative_time('physical', 'geosphere') > 0

def test_data_flow_manager(device, config):
    """Test DataFlowManager basic functionality."""
    # Create manager
    manager = DataFlowManager(config, device)
    
    # Create dummy states
    physical_state = torch.randn(
        1, config['physical_system']['input_dim'],
        config['grid_height'], config['grid_width'],
        device=device
    )
    biosphere_state = torch.randn(
        1, config['biosphere']['state_dim'],
        device=device
    )
    geosphere_state = torch.randn(
        1, config['geosphere']['state_dim'],
        device=device
    )
    
    # Test state updates and transformations
    manager.update_state('physical', physical_state)
    manager.update_state('biosphere', biosphere_state)
    manager.update_state('geosphere', geosphere_state)
    
    # Test getting transformed states
    bio_input = manager.get_state_for_component('physical', 'biosphere')
    assert bio_input is not None
    
    # Test state validation
    assert manager.validate_state('physical', physical_state)
    assert manager.validate_state('biosphere', biosphere_state)
    assert manager.validate_state('geosphere', geosphere_state)

def test_integration(device, config):
    """Test basic integration of all components."""
    from scripts.run_simulation import EarthSystemSimulation
    
    # Create simulation
    sim = EarthSystemSimulation(
        config_path=str(Path(__file__).parent.parent / 'config' / 'model_config.yaml'),
        device=device
    )
    
    # Run a few steps
    trajectory = sim.run_simulation(num_steps=10, save_frequency=5)
    
    # Check trajectory contents
    assert 'physical' in trajectory
    assert 'biosphere' in trajectory
    assert 'geosphere' in trajectory
    assert 'times' in trajectory
    
    # Check that states were saved
    assert len(trajectory['physical']) > 0
    assert len(trajectory['biosphere']) > 0
    assert len(trajectory['geosphere']) > 0
    assert len(trajectory['times']) > 0

if __name__ == '__main__':
    pytest.main([__file__])