"""
Data flow management module for handling interactions between
physical, biosphere, and geosphere components.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

@dataclass
class DataMapping:
    """Defines a data mapping between components."""
    source: str
    target: str
    source_vars: List[str]
    target_vars: List[str]
    transform: Optional[callable] = None

class StateAdapter:
    """
    Adapter for transforming state representations between different components.
    Handles variable selection, rescaling, and any necessary transformations.
    """
    def __init__(
        self,
        mappings: List[DataMapping],
        variable_indices: Dict[str, Dict[str, int]],
        scale_factors: Dict[str, Dict[str, float]] = None
    ):
        """
        Initialize the state adapter.
        
        Args:
            mappings: List of data mappings between components
            variable_indices: Dictionary of variable indices for each component
            scale_factors: Optional scaling factors for variables
        """
        self.mappings = mappings
        self.variable_indices = variable_indices
        self.scale_factors = scale_factors or {}
        
    def transform_state(
        self,
        source: str,
        target: str,
        state: torch.Tensor
    ) -> torch.Tensor:
        """
        Transform state from source component format to target component format.
        
        Args:
            source: Source component name
            target: Target component name
            state: State tensor to transform
            
        Returns:
            Transformed state tensor
        """
        # Find relevant mappings
        relevant_mappings = [
            m for m in self.mappings
            if m.source == source and m.target == target
        ]
        
        if not relevant_mappings:
            raise ValueError(f"No mapping found from {source} to {target}")
            
        transformed_states = []
        
        for mapping in relevant_mappings:
            # Extract relevant variables
            source_indices = [
                self.variable_indices[source][var]
                for var in mapping.source_vars
            ]
            selected_state = state[..., source_indices]
            
            # Apply scaling if defined
            if source in self.scale_factors:
                scale = torch.tensor([
                    self.scale_factors[source].get(var, 1.0)
                    for var in mapping.source_vars
                ], device=state.device)
                selected_state = selected_state * scale
            
            # Apply custom transform if defined
            if mapping.transform is not None:
                selected_state = mapping.transform(selected_state)
                
            transformed_states.append(selected_state)
            
        # Concatenate all transformed states
        return torch.cat(transformed_states, dim=-1)

class DataFlowManager:
    """
    Manages data flow and interactions between Earth system components.
    Handles state transformations, feedback mechanisms, and data validation.
    """
    def __init__(
        self,
        component_configs: Dict[str, Dict[str, Any]],
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        """
        Initialize the data flow manager.
        
        Args:
            component_configs: Configuration for each component
            device: Device to use for computations
        """
        self.device = device
        self.component_configs = component_configs
        
        # Initialize state adapters
        self.initialize_adapters()
        
        # Create state buffers for each component
        self.state_buffers = {
            name: {'current': None, 'previous': None}
            for name in component_configs.keys()
        }
        
    def initialize_adapters(self):
        """Initialize state adapters for component interactions."""
        # Define variable indices for each component
        variable_indices = {
            'physical': {
                'temperature': 0,
                'pressure': 1,
                'humidity': 2,
                'wind_u': 3,
                'wind_v': 4
            },
            'biosphere': {
                'vegetation': 0,
                'soil_moisture': 1,
                'carbon_flux': 2
            },
            'geosphere': {
                'topography': 0,
                'soil_type': 1,
                'erosion_rate': 2
            }
        }
        
        # Define data mappings between components
        mappings = [
            # Physical to Biosphere
            DataMapping(
                source='physical',
                target='biosphere',
                source_vars=['temperature', 'humidity'],
                target_vars=['temperature_bio', 'humidity_bio']
            ),
            # Biosphere to Physical
            DataMapping(
                source='biosphere',
                target='physical',
                source_vars=['vegetation'],
                target_vars=['surface_roughness'],
                transform=lambda x: torch.exp(x * 0.1)  # Example transformation
            ),
            # Physical to Geosphere
            DataMapping(
                source='physical',
                target='geosphere',
                source_vars=['temperature', 'pressure'],
                target_vars=['temp_geo', 'pressure_geo']
            ),
            # Geosphere to Physical
            DataMapping(
                source='geosphere',
                target='physical',
                source_vars=['topography'],
                target_vars=['surface_height']
            )
        ]
        
        # Define scaling factors
        scale_factors = {
            'physical': {
                'temperature': 1/300.0,  # Normalize around typical temperature
                'pressure': 1/101325.0,  # Normalize around 1 atm
                'humidity': 1.0          # Already normalized
            },
            'biosphere': {
                'vegetation': 1.0,       # Already normalized
                'soil_moisture': 1.0     # Already normalized
            },
            'geosphere': {
                'topography': 1/1000.0,  # Normalize elevation in km
                'erosion_rate': 1.0      # Already normalized
            }
        }
        
        self.adapter = StateAdapter(
            mappings,
            variable_indices,
            scale_factors
        )
        
    def update_state(
        self,
        component: str,
        new_state: torch.Tensor
    ):
        """
        Update the state of a component.
        
        Args:
            component: Name of the component
            new_state: New state tensor
        """
        if component not in self.state_buffers:
            raise ValueError(f"Unknown component: {component}")
            
        self.state_buffers[component]['previous'] = self.state_buffers[component]['current']
        self.state_buffers[component]['current'] = new_state.to(self.device)
        
    def get_state_for_component(
        self,
        source: str,
        target: str
    ) -> torch.Tensor:
        """
        Get transformed state from source component for target component.
        
        Args:
            source: Source component name
            target: Target component name
            
        Returns:
            Transformed state tensor
        """
        source_state = self.state_buffers[source]['current']
        if source_state is None:
            raise ValueError(f"No current state for component: {source}")
            
        return self.adapter.transform_state(source, target, source_state)
        
    def compute_feedback(
        self,
        component: str
    ) -> torch.Tensor:
        """
        Compute feedback effects from other components.
        
        Args:
            component: Name of the component to compute feedback for
            
        Returns:
            Tensor of feedback effects
        """
        feedbacks = []
        
        # Get relevant components that influence the target component
        influences = {
            'physical': ['biosphere', 'geosphere'],
            'biosphere': ['physical'],
            'geosphere': ['physical']
        }
        
        for source in influences.get(component, []):
            if self.state_buffers[source]['current'] is not None:
                feedback = self.get_state_for_component(source, component)
                feedbacks.append(feedback)
                
        if not feedbacks:
            return None
            
        # Combine feedbacks (simple sum for now, could be more sophisticated)
        return torch.stack(feedbacks).sum(dim=0)
        
    def validate_state(
        self,
        component: str,
        state: torch.Tensor
    ) -> bool:
        """
        Validate state values for a component.
        
        Args:
            component: Name of the component
            state: State tensor to validate
            
        Returns:
            True if state is valid
        """
        # Define valid ranges for each component
        valid_ranges = {
            'physical': {
                'min': torch.tensor([180.0, 0.0, 0.0, -100.0, -100.0]),  # K, Pa, %, m/s
                'max': torch.tensor([330.0, 110000.0, 1.0, 100.0, 100.0])
            },
            'biosphere': {
                'min': torch.tensor([0.0, 0.0, -1.0]),  # normalized units
                'max': torch.tensor([1.0, 1.0, 1.0])
            },
            'geosphere': {
                'min': torch.tensor([-1000.0, 0.0, 0.0]),  # m, normalized units
                'max': torch.tensor([9000.0, 1.0, 1.0])
            }
        }
        
        if component not in valid_ranges:
            raise ValueError(f"No validation ranges defined for component: {component}")
            
        ranges = valid_ranges[component]
        min_valid = (state >= ranges['min'].to(state.device)).all()
        max_valid = (state <= ranges['max'].to(state.device)).all()
        
        return min_valid and max_valid