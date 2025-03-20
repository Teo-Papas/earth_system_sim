"""
Main script for running the Earth system simulation with all components.
"""

import torch
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List
import yaml

# Add src to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.physical import PINN
from src.models.biosphere import BiospherePolicy
from src.models.geosphere import GeospherePolicy
from src.integration.temporal_sync import TemporalSynchronizer, create_default_timescales
from src.integration.data_flow import DataFlowManager

def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Safely convert a PyTorch tensor to numpy array.
    
    Args:
        tensor: Input tensor
        
    Returns:
        Numpy array
    """
    return tensor.detach().cpu().numpy()

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
        
        Args:
            config_path: Path to configuration file
            device: Compute device to use
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
            state_dim=self.config['biosphere']['state_dim'],  # Original state dimensions
            action_dim=self.config['biosphere']['action_dim'],
            hidden_dims=self.config['biosphere']['hidden_dims']
        ).to(self.device)
        
        # Geosphere system
        self.geosphere = GeospherePolicy(
            state_dim=self.config['geosphere']['state_dim'],  # Original state dimensions
            action_dim=self.config['geosphere']['action_dim'],
            hidden_dim=self.config['geosphere']['hidden_dim']
        ).to(self.device)

    def _initialize_states(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize system states from config."""
        print("\nDEBUG - Initializing States:")
        print(f"Physical input dim: {self.config['physical_system']['input_dim']}")
        print(f"Biosphere state dim: {self.config['biosphere']['state_dim']}")
        print(f"Geosphere state dim: {self.config['geosphere']['state_dim']}")
        
        physical_state = torch.zeros(
            (1, self.config['physical_system']['input_dim']),
            device=self.device
        )
        print(f"Initialized physical state shape: {physical_state.shape}")
        
        biosphere_state = torch.zeros(
            (1, self.config['biosphere']['state_dim']),
            device=self.device
        )
        print(f"Initialized biosphere state shape: {biosphere_state.shape}")
        
        geosphere_state = torch.zeros(
            (1, self.config['geosphere']['state_dim']),
            device=self.device
        )
        print(f"Initialized geosphere state shape: {geosphere_state.shape}")
        
        # Print policy network input dimensions
        print("\nDEBUG - Policy Network Dimensions:")
        print(f"Biosphere first layer input dim: {self.biosphere.actor[0].weight.shape[1]}")
        print(f"Geosphere first layer input dim: {self.geosphere.actor[0].weight.shape[1]}")
        
        return physical_state, biosphere_state, geosphere_state
        
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
            component_configs={
                'physical': self.config['physical_system'],
                'biosphere': self.config['biosphere'],
                'geosphere': self.config['geosphere']
            },
            device=self.device
        )
        
    def run_timestep(
        self,
        physical_state: torch.Tensor,
        biosphere_state: torch.Tensor,
        geosphere_state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run one timestep of the simulation.
        
        Args:
            physical_state: Current physical system state
            biosphere_state: Current biosphere state
            geosphere_state: Current geosphere state
            
        Returns:
            Tuple of updated states
        """
        # Check which components should update
        updates = self.synchronizer.step()
        
        # Physical system update (every timestep)
        if updates['physical']:
            # Prepare input for PINN (add sequence dimension)
            print("\nDEBUG - Physical Update:")
            print(f"Physical state shape: {physical_state.shape}")
            physical_input = physical_state.unsqueeze(1)
            print(f"Physical input shape after unsqueeze: {physical_input.shape}")
            
            # Update physical state using PINN
            with torch.no_grad():
                physical_pred, _ = self.physical(physical_input)
                physical_state = physical_pred.squeeze(1)
                print(f"Physical state shape after PINN: {physical_state.shape}")
            
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
            print("\nDEBUG - Biosphere Update:")
            print(f"Biosphere state shape: {biosphere_state.shape}")
            bio_input = self.data_flow.get_state_for_component('physical', 'biosphere')
            
            if bio_input is not None:
                print(f"Bio input shape from physical: {bio_input.shape}")
                print(f"Biosphere actor first layer weight shape: {self.biosphere.actor[0].weight.shape}")
                
                # Process physical input first
                with torch.no_grad():
                    print("DEBUG - Processing physical input...")
                    try:
                        # Project physical input to biosphere state space
                        bio_feedback = torch.tanh(bio_input @ torch.randn(bio_input.shape[1], biosphere_state.shape[1], device=self.device))
                        # Update state with physical influence
                        biosphere_state = biosphere_state + 0.1 * bio_feedback
                        print(f"Updated biosphere state shape: {biosphere_state.shape}")
                        
                        # Now get policy action using only biosphere state
                        bio_action = self.biosphere.act(biosphere_state)[0]
                        print(f"Bio action shape: {bio_action.shape}")
                    except Exception as e:
                        print(f"Error in biosphere update:")
                        print(f"bio_input shape: {bio_input.shape}")
                        print(f"biosphere_state shape: {biosphere_state.shape}")
                        print(f"actor weight shape: {self.biosphere.actor[0].weight.shape}")
                        print(f"Error: {str(e)}")
                        raise
                
                # Update biosphere state
                biosphere_state = biosphere_state + bio_action
                
                # Validate and update state
                if self.data_flow.validate_state('biosphere', biosphere_state):
                    self.data_flow.update_state('biosphere', biosphere_state)
        
        # Geosphere update (least frequent)
        if updates['geosphere']:
            # Get relevant physical state information
            print("\nDEBUG - Geosphere Update:")
            print(f"Geosphere state shape: {geosphere_state.shape}")
            geo_input = self.data_flow.get_state_for_component('physical', 'geosphere')
            
            if geo_input is not None:
                print(f"Geo input shape from physical: {geo_input.shape}")
                print(f"Geosphere actor first layer weight shape: {self.geosphere.actor[0].weight.shape}")
                
                # Process physical input first
                with torch.no_grad():
                    print("DEBUG - Processing physical input...")
                    try:
                        # Project physical input to geosphere state space
                        geo_feedback = torch.tanh(geo_input @ torch.randn(geo_input.shape[1], geosphere_state.shape[1], device=self.device))
                        # Update state with physical influence
                        geosphere_state = geosphere_state + 0.1 * geo_feedback
                        print(f"Updated geosphere state shape: {geosphere_state.shape}")
                        
                        # Now get policy action using only geosphere state
                        geo_action = self.geosphere.act(geosphere_state)[0]
                        print(f"Geo action shape: {geo_action.shape}")
                    except Exception as e:
                        print(f"Error in geosphere update:")
                        print(f"geo_input shape: {geo_input.shape}")
                        print(f"geosphere_state shape: {geosphere_state.shape}")
                        print(f"actor weight shape: {self.geosphere.actor[0].weight.shape}")
                        print(f"Error: {str(e)}")
                        raise
                
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
        
        Args:
            num_steps: Number of timesteps to simulate
            save_frequency: How often to save states
            
        Returns:
            Dictionary containing simulation trajectory
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
            trajectory['physical'].append(tensor_to_numpy(physical_state))
            trajectory['biosphere'].append(tensor_to_numpy(biosphere_state))
            trajectory['geosphere'].append(tensor_to_numpy(geosphere_state))
            trajectory['times'].append(self.synchronizer.current_times)
        
        for step in range(num_steps):
            # Run one timestep
            physical_state, biosphere_state, geosphere_state = self.run_timestep(
                physical_state, biosphere_state, geosphere_state
            )
            
            # Save states periodically
            if step % save_frequency == 0:
                with torch.no_grad():
                    trajectory['physical'].append(tensor_to_numpy(physical_state))
                    trajectory['biosphere'].append(tensor_to_numpy(biosphere_state))
                    trajectory['geosphere'].append(tensor_to_numpy(geosphere_state))
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
    
    # Run simulation with gradient disabled
    with torch.no_grad():
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