"""
Biosphere policy network implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional

class BiospherePolicy(nn.Module):
    """Policy network for the biosphere system."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        """
        Initialize biosphere policy.
        
        Args:
            state_dim: Input state dimension
            action_dim: Output action dimension
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create actor network (policy)
        actor_layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
        actor_layers.append(nn.Linear(current_dim, action_dim))
        
        self.actor = nn.Sequential(*actor_layers)
        
        # Create critic network (value function)
        critic_layers = []
        current_dim = state_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.ReLU()
            ])
            current_dim = hidden_dim
        critic_layers.append(nn.Linear(current_dim, 1))
        
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize action log standard deviation
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Set requires_grad=False for state normalization
        self.register_buffer('state_mean', torch.zeros(state_dim))
        self.register_buffer('state_std', torch.ones(state_dim))
        
    def normalize_state(self, state: torch.Tensor) -> torch.Tensor:
        """Normalize state using running statistics."""
        return (state - self.state_mean) / (self.state_std + 1e-8)
    
    def update_normalization(self, state: torch.Tensor):
        """Update state normalization statistics."""
        with torch.no_grad():
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
        # Ensure state has correct dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        # Validate state dimensions
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"State has incorrect dimension. Expected {self.state_dim}, "
                f"got {state.shape[-1]}. Full shape: {state.shape}"
            )
            
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