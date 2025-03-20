"""
Policy network for biosphere system using actor-critic architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional


class PolicyNetwork(nn.Module):
    """Base policy network implementation."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int]
    ):
        """
        Initialize policy network.
        
        Args:
            state_dim: Input state dimension
            action_dim: Output action dimension
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Create actor network
        actor_layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        actor_layers.append(nn.Linear(in_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Create critic network
        critic_layers = []
        in_dim = state_dim
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU()
            ])
            in_dim = hidden_dim
        critic_layers.append(nn.Linear(in_dim, 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # Action log standard deviation (learnable)
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))
        
    def forward(
        self,
        state: torch.Tensor
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass through policy network.
        
        Args:
            state: Input state tensor
            
        Returns:
            Tuple of (action distribution, state value)
        """
        # Validate input dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)  # Add batch dimension
            
        if state.shape[-1] != self.state_dim:
            raise ValueError(
                f"Expected state dimension {self.state_dim}, got {state.shape[-1]}"
            )
            
        # Compute action mean and create distribution
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std)
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        
        # Compute state value
        state_value = self.critic(state)
        
        return action_distribution, state_value
        
    def evaluate_actions(
        self,
        state: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        
        Args:
            state: Input state tensor
            actions: Actions to evaluate
            
        Returns:
            Tuple of (action log probs, state values, entropy)
        """
        action_distribution, state_value = self.forward(state)
        
        action_log_probs = action_distribution.log_prob(actions).sum(-1)
        entropy = action_distribution.entropy().mean()
        
        return action_log_probs, state_value, entropy


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
            state_dim: Input state dimension
            action_dim: Output action dimension
            hidden_dims: List of hidden layer dimensions
        """
        super().__init__()
        
        # Create core policy network
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=hidden_dims
        )
        
        # Initialize buffers for experience collection
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.returns = []
        self.advantages = []
        
    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        """
        Sample action from policy.
        
        Args:
            state: Input state tensor
            deterministic: If True, use mean action
            
        Returns:
            Tuple of (action, value, action distribution)
        """
        # Ensure state has correct dimensions
        if state.dim() == 1:
            state = state.unsqueeze(0)
            
        # Get expected state shape
        expected_dim = self.policy.state_dim
        actual_dim = state.shape[-1]
        
        if actual_dim != expected_dim:
            raise ValueError(
                f"State dimension mismatch. Expected {expected_dim}, got {actual_dim}. "
                f"State shape: {state.shape}"
            )
            
        # Forward pass through policy
        action_distribution, value = self.policy(state)
        
        # Sample action
        if deterministic:
            action = action_distribution.mean
        else:
            action = action_distribution.sample()
        
        return action, value, action_distribution
    
    def update(
        self,
        optimizer: torch.optim.Optimizer,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
        clip_param: float = 0.2
    ) -> Tuple[float, float, float]:
        """
        Update policy using PPO.
        
        Args:
            optimizer: Optimizer to use
            states: State tensor
            actions: Action tensor
            returns: Return tensor
            advantages: Advantage tensor
            clip_param: PPO clipping parameter
            
        Returns:
            Tuple of (policy loss, value loss, entropy)
        """
        # Get action log probs and values
        action_log_probs, values, entropy = self.policy.evaluate_actions(states, actions)
        
        # Compute value loss
        value_loss = F.mse_loss(values, returns)
        
        # Compute policy loss
        ratio = torch.exp(action_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return policy_loss.item(), value_loss.item(), entropy.item()
    
    def store_experience(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        value: torch.Tensor
    ):
        """Store experience for training."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
    
    def clear_experience(self):
        """Clear stored experience."""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.returns.clear()
        self.advantages.clear()