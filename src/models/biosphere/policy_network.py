"""
Biosphere policy network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class BiospherePolicy(nn.Module):
    """Policy network for biosphere system."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        """
        Initialize biosphere policy.
        
        Args:
            state_dim: Input state dimension (vegetation, moisture, temp, pressure)
            action_dim: Output action dimension (growth, water consumption)
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        if state_dim != 4:
            raise ValueError(
                f"BiospherePolicy expects state_dim=4 (vegetation, moisture, temp, pressure), "
                f"got {state_dim}"
            )
            
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create actor network (policy)
        self.actor = self._build_network(state_dim, action_dim, hidden_dims)
        
        # Create critic network (value function)
        self.critic = self._build_network(state_dim, 1, hidden_dims)
        
        # Initialize action log standard deviation
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Register buffers for state normalization
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        
    def _build_network(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int]
    ) -> nn.Sequential:
        """Build a neural network with the given architecture."""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
            
        layers.append(nn.Linear(current_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _validate_input(self, state: torch.Tensor):
        """Validate input state tensor."""
        if state.dim() not in [1, 2]:
            raise ValueError(
                f"Expected state to have 1 or 2 dimensions, got {state.dim()}"
            )
            
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state to have {self.state_dim} features, got {state.shape[-1]}\n"
                f"State shape: {state.shape}"
            )
    
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state using running statistics."""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def update_normalization(self, state: torch.Tensor):
        """Update state normalization statistics."""
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            self.state_mean = 0.99 * self.state_mean + 0.01 * state.mean(0)
            self.state_std = 0.99 * self.state_std + 0.01 * state.std(0)
    
    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.distributions.Distribution]]:
        """
        Sample action from policy.
        
        Args:
            state: Input state tensor [batch_size, state_dim]
            deterministic: If True, return mean action
            
        Returns:
            Tuple of (action, value, action distribution)
        """
        # Input validation
        self._validate_input(state)
        
        # Ensure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Normalize state
        state = self.normalize_state(state)
        
        # Get action mean and create distribution
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        # Get state value
        value = self.critic(state)
        
        # Sample or take mean action
        if deterministic:
            action = action_mean
        else:
            action = action_dist.sample()
            
        return action, value, action_dist
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        
        Args:
            states: Input state tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, action_dim]
            
        Returns:
            Tuple of (log_probs, values, entropy)
        """
        # Input validation
        self._validate_input(states)
        
        # Normalize states
        states = self.normalize_state(states)
        
        # Get action distribution parameters
        action_mean = self.actor(states)
        action_std = torch.exp(self.action_log_std)
        action_dist = torch.distributions.Normal(action_mean, action_std)
        
        # Compute log probabilities
        log_probs = action_dist.log_prob(actions).sum(-1)
        
        # Get state values
        values = self.critic(states)
        
        # Compute entropy
        entropy = action_dist.entropy().mean()
        
        return log_probs, values, entropy
    
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        old_log_probs: torch.Tensor,
        clip_range: float = 0.2
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update policy using PPO algorithm.
        
        Args:
            states: State tensor [batch_size, state_dim]
            actions: Action tensor [batch_size, action_dim]
            advantages: Advantage tensor [batch_size]
            returns: Return tensor [batch_size]
            old_log_probs: Old log probabilities [batch_size]
            clip_range: PPO clip range
            
        Returns:
            Tuple of (policy_loss, value_loss, entropy)
        """
        # Update normalization statistics
        self.update_normalization(states)
        
        # Evaluate actions
        log_probs, values, entropy = self.evaluate_actions(states, actions)
        
        # Compute value loss
        value_loss = F.mse_loss(values.squeeze(), returns)
        
        # Compute policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss1 = advantages * ratio
        policy_loss2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(policy_loss1, policy_loss2).mean()
        
        return policy_loss, value_loss, entropy