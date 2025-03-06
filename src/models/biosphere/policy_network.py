"""
Policy Gradient Network implementation for the biosphere system.
This module handles ecosystem dynamics using reinforcement learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import numpy as np

class BiospherePolicy(nn.Module):
    """
    Policy network for biosphere dynamics using policy gradient methods.
    Implements both actor and critic networks for potential A2C implementation.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [128, 64],
        std_init: float = 0.5,
        learn_std: bool = True
    ):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space (combined atmospheric and biological features)
            action_dim: Dimension of action space (ecosystem parameters to adjust)
            hidden_dims: List of hidden layer dimensions
            std_init: Initial standard deviation for action distribution
            learn_std: Whether to learn the standard deviation or keep it fixed
        """
        super(BiospherePolicy, self).__init__()
        
        # Build actor network (policy)
        actor_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            actor_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)  # Helps with training stability
            ])
            prev_dim = hidden_dim
            
        # Final layer for action mean
        actor_layers.append(nn.Linear(prev_dim, action_dim))
        self.actor = nn.Sequential(*actor_layers)
        
        # Standard deviation for action distribution
        if learn_std:
            self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(std_init))
        else:
            self.register_buffer('action_log_std', torch.ones(1, action_dim) * np.log(std_init))
        
        # Build critic network (value function)
        critic_layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            critic_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
            
        critic_layers.append(nn.Linear(prev_dim, 1))  # Value function output
        self.critic = nn.Sequential(*critic_layers)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(self, state: torch.Tensor) -> Tuple[torch.distributions.Distribution, torch.Tensor]:
        """
        Forward pass computing action distribution and value estimate.
        
        Args:
            state: Current state tensor
            
        Returns:
            Tuple of (action distribution, value estimate)
        """
        # Compute action mean and create distribution
        action_mean = self.actor(state)
        action_std = torch.exp(self.action_log_std)
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        
        # Compute value estimate
        value = self.critic(state)
        
        return action_distribution, value
    
    def evaluate_actions(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for given states.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Tuple of (log probability, entropy, value)
        """
        action_distribution, value = self.forward(state)
        log_prob = action_distribution.log_prob(action).sum(-1)
        entropy = action_distribution.entropy().mean()
        
        return log_prob, entropy, value
    
    @torch.no_grad()
    def act(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.distributions.Distribution]]:
        """
        Sample an action from the policy.
        
        Args:
            state: Current state tensor
            deterministic: If True, return mean action instead of sampling
            
        Returns:
            Tuple of (action, value, action distribution)
        """
        action_distribution, value = self.forward(state)
        
        if deterministic:
            action = action_distribution.mean
        else:
            action = action_distribution.sample()
            
        return action, value, action_distribution

class BiosphereRLTrainer:
    """
    Trainer class for the biosphere policy using PPO-style updates.
    """
    def __init__(
        self,
        policy: BiospherePolicy,
        lr: float = 3e-4,
        eps: float = 1e-5,
        clip_range: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5
    ):
        """
        Initialize the trainer.
        
        Args:
            policy: BiospherePolicy instance
            lr: Learning rate
            eps: Epsilon for Adam optimizer
            clip_range: PPO clipping range
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.policy = policy
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=eps)
        
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        clip_range: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Update policy using PPO algorithm.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            old_log_probs: Log probabilities of actions under old policy
            advantages: Computed advantages
            returns: Computed returns
            clip_range: Optional custom clip range
            
        Returns:
            Dictionary of training metrics
        """
        clip_range = clip_range or self.clip_range
        
        # Evaluate actions under current policy
        log_probs, entropy, values = self.policy.evaluate_actions(states, actions)
        
        # Policy loss
        ratio = torch.exp(log_probs - old_log_probs)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
        
        # Value loss
        value_pred_clipped = values + (returns - values).clamp(-clip_range, clip_range)
        value_loss_1 = (values - returns).pow(2)
        value_loss_2 = (value_pred_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss_1, value_loss_2).mean()
        
        # Total loss
        loss = (
            policy_loss +
            self.value_loss_coef * value_loss -
            self.entropy_coef * entropy
        )
        
        # Update policy
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': loss.item(),
            'approx_kl': 0.5 * torch.mean((old_log_probs - log_probs) ** 2).item()
        }

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    Args:
        rewards: Tensor of rewards
        values: Tensor of value estimates
        next_values: Tensor of next state value estimates
        dones: Tensor indicating episode termination
        gamma: Discount factor
        gae_lambda: GAE lambda parameter
        
    Returns:
        Tuple of (advantages, returns)
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = next_values
        else:
            next_value = values[t + 1]
            
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        advantages[t] = last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        
    returns = advantages + values
    
    return advantages, returns