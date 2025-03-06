"""
Slow-paced Policy Gradient implementation for the geosphere system.
This module handles geological processes using episodic reinforcement learning
with extended temporal horizons.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional, List
import numpy as np

class GeosphereMemory:
    """
    Memory buffer for storing geosphere transitions over long time periods.
    Implements a priority-based sampling mechanism where more recent experiences
    have higher priority but older experiences are still maintained.
    """
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        device: torch.device
    ):
        """
        Initialize the memory buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            device: Device to store tensors on
        """
        self.capacity = capacity
        self.device = device
        
        # Pre-allocate memory
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((capacity, action_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros(capacity, dtype=torch.float32, device=device)
        self.next_states = torch.zeros((capacity, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros(capacity, dtype=torch.bool, device=device)
        
        self.position = 0
        self.size = 0
        
    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool
    ):
        """
        Store a transition in memory.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(
        self,
        batch_size: int,
        sequence_length: int
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of transitions with temporal coherence.
        
        Args:
            batch_size: Number of sequences to sample
            sequence_length: Length of each sequence
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        # Sample starting indices
        start_indices = torch.randint(0, self.size - sequence_length, (batch_size,))
        
        # Gather sequences
        batch_states = torch.stack([
            self.states[i:i + sequence_length] for i in start_indices
        ])
        batch_actions = torch.stack([
            self.actions[i:i + sequence_length] for i in start_indices
        ])
        batch_rewards = torch.stack([
            self.rewards[i:i + sequence_length] for i in start_indices
        ])
        batch_next_states = torch.stack([
            self.next_states[i:i + sequence_length] for i in start_indices
        ])
        batch_dones = torch.stack([
            self.dones[i:i + sequence_length] for i in start_indices
        ])
        
        return (
            batch_states,
            batch_actions,
            batch_rewards,
            batch_next_states,
            batch_dones
        )

class GeospherePolicy(nn.Module):
    """
    Policy network for geosphere processes using slow-paced learning.
    Implements a recurrent policy to capture long-term dependencies.
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        std_init: float = 0.1
    ):
        """
        Initialize the policy network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            std_init: Initial standard deviation for action distribution
        """
        super(GeospherePolicy, self).__init__()
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Learnable std deviation
        self.action_log_std = nn.Parameter(torch.ones(1, action_dim) * np.log(std_init))
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize network weights."""
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def forward(
        self,
        state: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.distributions.Distribution, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass computing action distribution and value estimate.
        
        Args:
            state: State tensor (batch_size, sequence_length, state_dim)
            hidden_states: Optional initial hidden states for LSTM
            
        Returns:
            Tuple of (action distribution, value, next hidden states)
        """
        # Process sequence through LSTM
        lstm_out, hidden_states = self.lstm(state, hidden_states)
        
        # Compute action parameters and value
        action_mean = self.actor(lstm_out)
        action_std = torch.exp(self.action_log_std)
        value = self.critic(lstm_out)
        
        # Create action distribution
        action_distribution = torch.distributions.Normal(action_mean, action_std)
        
        return action_distribution, value, hidden_states

class GeosphereTrainer:
    """
    Trainer for the geosphere policy using PPO with importance sampling
    across extended time horizons.
    """
    def __init__(
        self,
        policy: GeospherePolicy,
        lr: float = 1e-4,
        eps: float = 1e-5,
        clip_range: float = 0.1,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4
    ):
        """
        Initialize the trainer.
        
        Args:
            policy: GeospherePolicy instance
            lr: Learning rate
            eps: Epsilon for Adam optimizer
            clip_range: PPO clipping range
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of epochs per update
        """
        self.policy = policy
        self.clip_range = clip_range
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr, eps=eps)
        
    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        hidden_states: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Dict[str, float]:
        """
        Update policy using PPO with importance sampling.
        
        Args:
            states: Batch of state sequences
            actions: Batch of action sequences
            old_log_probs: Log probabilities under old policy
            advantages: Computed advantages
            returns: Computed returns
            hidden_states: Optional initial hidden states
            
        Returns:
            Dictionary of training metrics
        """
        metrics = []
        
        for _ in range(self.num_epochs):
            # Forward pass
            action_distribution, values, _ = self.policy(states, hidden_states)
            log_probs = action_distribution.log_prob(actions).sum(-1)
            entropy = action_distribution.entropy().mean()
            
            # Policy loss with clipping
            ratio = torch.exp(log_probs - old_log_probs)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * torch.clamp(
                ratio,
                1.0 - self.clip_range,
                1.0 + self.clip_range
            )
            policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
            
            # Value loss
            value_pred_clipped = values + (returns - values).clamp(
                -self.clip_range,
                self.clip_range
            )
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
            
            metrics.append({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'entropy': entropy.item(),
                'total_loss': loss.item(),
                'approx_kl': 0.5 * torch.mean((old_log_probs - log_probs) ** 2).item()
            })
        
        # Average metrics across epochs
        return {k: np.mean([m[k] for m in metrics]) for k in metrics[0].keys()}