"""
Main script for running the Earth system simulation with all components.
"""

import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Tuple, Optional, Union, Any
import yaml

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.physical import PINN
from src.models.biosphere import BiospherePolicy
from src.models.geosphere import GeospherePolicy
from src.integration.temporal_sync import TemporalSynchronizer, create_default_timescales
from src.integration.data_flow import DataFlowManager

class EarthSystemSimulation:
    """
    Main simulation class that coordinates all components of the Earth system.
    """
    def __init__(
        self,
        config_path: str,
        device: torch.device = None
    ):
        """
        Initialize the Earth system simulation.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._initialize_components()
        
        # Initialize integration components
        self._initialize_integration()
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
        
    def _initialize_components(self):
        """Initialize all system components."""
        # Physical system (PINN)
        self.physical = PINN(
            input_dim=self.config['physical_system']['input_dim'],
            hidden_dims=self.config['physical_system']['hidden_dims'],
            kernel_size=self.config['physical_system']['kernel_size'],
            num_layers=self.config['physical_system']['num_layers']
        ).to(self.device)
        
        # Biosphere system
        self.biosphere = BiospherePolicy(
            state_dim=self.config['biosphere']['state_dim'],
            action_dim=self.config['biosphere']['action_dim'],
            hidden_dims=self.config['biosphere']['hidden_dims']
        ).to(self.device)
        
        # Geosphere system
        self.geosphere = GeospherePolicy(
            state_dim=self.config['geosphere']['state_dim'],
            action_dim=self.config['geosphere']['action_dim'],
            hidden_dim=self.config['geosphere']['hidden_dim']
        ).to(self.device)
        
    def _initialize_integration(self):
        """Initialize integration components."""
        # Temporal synchronization
        self.timescales = create_default_timescales()
        self.synchronizer = TemporalSynchronizer(
            timescales=self.timescales,
            state_dims={
                'physical': self.config['physical_system']['input_dim'],
                'biosphere': self.config['biosphere']['state_dim'],
                'geosphere': self.config['geosphere']['state_dim']
            },
            device=self.device
        )
        
        # Data flow management
        self.data_flow = DataFlowManager(
            component_configs=self.config,
            device=self.device
        )
        
    def _initialize_states(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize states for all components."""
        with torch.no_grad():
            # Physical state initialization
            physical_state = torch.zeros(
                1,
                self.config['physical_system']['input_dim'],
                self.config['grid_height'],
                self.config['grid_width'],
                device=self.device
            )
            
            # Initialize with configuration values
            mean = self.config['simulation']['initial_conditions']
            physical_state[:, 0] = mean['pressure_mean']  # density
            physical_state[:, 1] = mean['temperature_mean']  # temperature
            physical_state[:, 2] = mean['pressure_mean']  # pressure
            physical_state[:, 3] = mean['wind_speed_mean']  # u velocity
            physical_state[:, 4] = mean['wind_speed_mean']  # v velocity
            
            # Add some random variation
            physical_state += torch.randn_like(physical_state) * 0.1
            
            # Biosphere state
            biosphere_state = torch.zeros(
                1,
                self.config['biosphere']['state_dim'],
                device=self.device
            )
            biosphere_state[0, 0] = mean['vegetation_cover_mean']
            biosphere_state[0, 1] = mean['soil_moisture_mean']
            
            # Geosphere state
            geosphere_state = torch.zeros(
                1,
                self.config['geosphere']['state_dim'],
                device=self.device
            )
            geosphere_state[0, 0] = mean['elevation_mean']
            
            # Update state buffers
            self.data_flow.update_state('physical', physical_state)
            self.data_flow.update_state('biosphere', biosphere_state)
            self.data_flow.update_state('geosphere', geosphere_state)
        
        return physical_state, biosphere_state, geosphere_state
    
    @torch.no_grad()
    def run_timestep(
        self,
        physical_state: torch.Tensor,
        biosphere_state: torch.Tensor,
        geosphere_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one timestep of the simulation.
        """
        # Check which components should update
        updates = self.synchronizer.step()
        
        # Physical system update (every timestep)
        if updates['physical']:
            # Prepare input for PINN
            physical_input = physical_state.unsqueeze(1)  # [batch, seq=1, channels, height, width]
            
            # Update physical state using PINN
            physical_pred, _ = self.physical(physical_input)
            physical_state = physical_pred.squeeze(1)  # Remove sequence dimension
            
            # Get feedback from other components
            physical_feedback = self.data_flow.compute_feedback('physical')
            if physical_feedback is not None:
                physical_state = physical_state + physical_feedback
                
            # Validate and update state
            if self.data_flow.validate_state('physical', physical_state):
                self.data_flow.update_state('physical', physical_state)
        
        # Biosphere update
        if updates['biosphere']:
            # Get relevant physical state information
            bio_input = self.data_flow.get_state_for_component('physical', 'biosphere')
            
            if bio_input is not None:
                # Sample action from policy
                bio_action = self.biosphere.act(
                    torch.cat([biosphere_state, bio_input], dim=-1)
                )[0]
                
                # Update biosphere state
                biosphere_state = biosphere_state + bio_action
                
                # Validate and update state
                if self.data_flow.validate_state('biosphere', biosphere_state):
                    self.data_flow.update_state('biosphere', biosphere_state)
        
        # Geosphere update (least frequent)
        if updates['geosphere']:
            # Get relevant physical state information
            geo_input = self.data_flow.get_state_for_component('physical', 'geosphere')
            
            if geo_input is not None:
                # Sample action from policy
                geo_action = self.geosphere.act(
                    torch.cat([geosphere_state, geo_input], dim=-1)
                )[0]
                
                # Update geosphere state
                geosphere_state = geosphere_state + geo_action
                
                # Validate and update state
                if self.data_flow.validate_state('geosphere', geosphere_state):
                    self.data_flow.update_state('geosphere', geosphere_state)
        
        return physical_state, biosphere_state, geosphere_state
    
    def run_simulation(
        self,
        num_steps: int,
        save_frequency: int = 100
    ) -> Dict[str, List]:
        """
        Run the full simulation for a specified number of steps.
        """
        # Initialize states
        physical_state, biosphere_state, geosphere_state = self._initialize_states()
        
        # Storage for trajectory
        trajectory = {
            'physical': [],
            'biosphere': [],
            'geosphere': [],
            'times': []
        }
        
        # Save initial states
        with torch.no_grad():
            trajectory['physical'].append(physical_state.cpu().numpy())
            trajectory['biosphere'].append(biosphere_state.cpu().numpy())
            trajectory['geosphere'].append(geosphere_state.cpu().numpy())
            trajectory['times'].append(self.synchronizer.current_times)
        
        for step in range(num_steps):
            # Run one timestep
            physical_state, biosphere_state, geosphere_state = self.run_timestep(
                physical_state, biosphere_state, geosphere_state
            )
            
            # Save states periodically
            if step % save_frequency == 0:
                with torch.no_grad():
                    trajectory['physical'].append(physical_state.cpu().numpy())
                    trajectory['biosphere'].append(biosphere_state.cpu().numpy())
                    trajectory['geosphere'].append(geosphere_state.cpu().numpy())
                    trajectory['times'].append(self.synchronizer.current_times)
                
                print(f"Step {step}/{num_steps}")
                
        return trajectory

def main():
    """Main function to run the simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Earth system simulation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of timesteps to simulate')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='How often to save states')
    parser.add_argument('--output', type=str, default='simulation_output.npz',
                       help='Output file path')
    
    args = parser.parse_args()
    
    # Initialize and run simulation
    sim = EarthSystemSimulation(args.config)
    trajectory = sim.run_simulation(args.steps, args.save_freq)
    
    # Save results
    np.savez(
        args.output,
        physical_states=np.array(trajectory['physical']),
        biosphere_states=np.array(trajectory['biosphere']),
        geosphere_states=np.array(trajectory['geosphere']),
        times=trajectory['times']
    )
    
    print(f"Simulation complete. Results saved to {args.output}")

if __name__ == "__main__":
    main()