"""
Temporal synchronization module for coordinating different timescales
between physical, biosphere, and geosphere components.
"""

import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import numpy as np

@dataclass
class TimeScale:
    """Data class for defining a component's timescale."""
    name: str
    dt: float  # timestep in hours
    update_frequency: int  # steps between updates
    
    def __post_init__(self):
        """Validate timescale parameters."""
        if self.dt <= 0:
            raise ValueError(f"Timestep must be positive, got {self.dt}")
        if self.update_frequency <= 0:
            raise ValueError(f"Update frequency must be positive, got {self.update_frequency}")

class StateBuffer:
    """
    Circular buffer for storing and interpolating state histories
    across different timescales.
    """
    def __init__(
        self,
        max_size: int,
        state_dim: int,
        device: torch.device
    ):
        """
        Initialize the state buffer.
        
        Args:
            max_size: Maximum number of states to store
            state_dim: Dimension of state vectors
            device: Device to store tensors on
        """
        self.max_size = max_size
        self.state_dim = state_dim
        self.device = device
        
        # Initialize buffer with zeros
        self.states = torch.zeros(
            (max_size, state_dim),
            dtype=torch.float32,
            device=device
        )
        self.timestamps = torch.zeros(
            max_size,
            dtype=torch.float32,
            device=device
        )
        
        self.position = 0
        self.size = 0
        
    def push(
        self,
        state: torch.Tensor,
        timestamp: float
    ):
        """
        Add a new state to the buffer.
        
        Args:
            state: State vector to store
            timestamp: Time of the state
        """
        self.states[self.position] = state
        self.timestamps[self.position] = timestamp
        
        self.position = (self.position + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
    def interpolate(
        self,
        target_time: float,
        method: str = 'linear'
    ) -> torch.Tensor:
        """
        Interpolate state at a specific time.
        
        Args:
            target_time: Time to interpolate at
            method: Interpolation method ('linear' or 'nearest')
            
        Returns:
            Interpolated state vector
        """
        if self.size < 2:
            raise ValueError("Need at least 2 states for interpolation")
            
        # Find surrounding timestamps
        times = self.timestamps[:self.size]
        if target_time < times.min() or target_time > times.max():
            raise ValueError(f"Target time {target_time} outside buffer range")
            
        if method == 'nearest':
            idx = torch.abs(times - target_time).argmin()
            return self.states[idx]
            
        elif method == 'linear':
            # Find indices of surrounding timestamps
            next_idx = torch.where(times > target_time)[0][0]
            prev_idx = next_idx - 1
            
            # Compute interpolation weights
            t0, t1 = times[prev_idx], times[next_idx]
            w1 = (target_time - t0) / (t1 - t0)
            w0 = 1 - w1
            
            # Interpolate
            return w0 * self.states[prev_idx] + w1 * self.states[next_idx]
        
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

class TemporalSynchronizer:
    """
    Coordinates the execution and data exchange between components
    operating at different timescales.
    """
    def __init__(
        self,
        timescales: Dict[str, TimeScale],
        state_dims: Dict[str, int],
        buffer_size: int = 1000,
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize the synchronizer.
        
        Args:
            timescales: Dictionary of component timescales
            state_dims: Dictionary of state dimensions for each component
            buffer_size: Size of state buffers
            device: Device to store tensors on
        """
        self.timescales = timescales
        self.state_dims = state_dims
        self.device = device
        
        # Create state buffers for each component
        self.buffers = {
            name: StateBuffer(buffer_size, dim, device)
            for name, dim in state_dims.items()
        }
        
        # Track current time for each component
        self.current_times = {name: 0.0 for name in timescales}
        
        # Initialize update counters
        self.step_counters = {name: 0 for name in timescales}
        
    def should_update(self, component: str) -> bool:
        """
        Check if a component should be updated.
        
        Args:
            component: Name of the component
            
        Returns:
            True if component should be updated
        """
        return (self.step_counters[component] %
                self.timescales[component].update_frequency) == 0
                
    def step(self) -> Dict[str, bool]:
        """
        Advance all components by their respective timesteps.
        
        Returns:
            Dictionary indicating which components should update
        """
        updates = {}
        
        for name, timescale in self.timescales.items():
            # Increment step counter
            self.step_counters[name] += 1
            
            # Update time
            self.current_times[name] += timescale.dt
            
            # Check if component should update
            updates[name] = self.should_update(name)
            
        return updates
    
    def get_interpolated_state(
        self,
        component: str,
        target_time: Optional[float] = None,
        method: str = 'linear'
    ) -> torch.Tensor:
        """
        Get interpolated state for a component at a specific time.
        
        Args:
            component: Name of the component
            target_time: Time to interpolate at (default: current time)
            method: Interpolation method
            
        Returns:
            Interpolated state vector
        """
        if target_time is None:
            target_time = self.current_times[component]
            
        return self.buffers[component].interpolate(target_time, method)
    
    def update_state(
        self,
        component: str,
        state: torch.Tensor
    ):
        """
        Update the state of a component.
        
        Args:
            component: Name of the component
            state: New state vector
        """
        self.buffers[component].push(
            state,
            self.current_times[component]
        )
        
    def get_relative_time(self, reference: str, target: str) -> float:
        """
        Get the time difference between two components.
        
        Args:
            reference: Name of reference component
            target: Name of target component
            
        Returns:
            Time difference in hours
        """
        return (self.current_times[target] -
                self.current_times[reference])

def create_default_timescales() -> Dict[str, TimeScale]:
    """
    Create default timescales for the three main components.
    
    Returns:
        Dictionary of default timescales
    """
    return {
        'physical': TimeScale(
            name='physical',
            dt=1.0,        # 1 hour timestep
            update_frequency=1
        ),
        'biosphere': TimeScale(
            name='biosphere',
            dt=24.0,       # 1 day timestep
            update_frequency=24
        ),
        'geosphere': TimeScale(
            name='geosphere',
            dt=720.0,      # 30 day timestep
            update_frequency=720
        )
    }