
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RegimeAwareDQN(nn.Module):
    """
    Deep Q-Network with regime-aware state representation
    Based on Bai et al. (2025) findings for adaptive RL in financial markets
    """
    
    def __init__(self, state_size: int, action_size: int, regime_size: int = 4):
        super(RegimeAwareDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.regime_size = regime_size
        
        # Regime encoder
        self.regime_encoder = nn.Sequential(
            nn.Linear(regime_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8)
        )
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64)
        )
        
        # Combined layers
        self.combined = nn.Sequential(
            nn.Linear(64 + 8, 128),  # state + regime features
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, state, regime_vector):
        state_features = self.state_encoder(state)
        regime_features = self.regime_encoder(regime_vector)
        combined = torch.cat([state_features, regime_features], dim=1)
        return self.combined(combined)

class RegimeAwareTrader:
    """
    Reinforcement Learning trader with built-in regime detection
    Implements regime-sensitive reward functions as recommended in research
    """
    
    def __init__(self, state_size: int, action_size: int = 3, learning_rate: float = 0.001):
        self.state_size = state_size
        self.action_size = action_size  # 0: Hold, 1: Buy, 2: Sell
        self.memory = deque(maxlen=10000)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # discount factor
        
        # Initialize networks
        self.q_network = RegimeAwareDQN(state_size, action_size)
        self.target_network = RegimeAwareDQN(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, regime_vector, action, reward, next_state, next_regime_vector, done):
        """Store experience in replay buffer"""
        self.memory.append((state, regime_vector, action, reward, next_state, next_regime_vector, done))
        
    def act(self, state, regime_vector):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        regime_tensor = torch.FloatTensor(regime_vector).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor, regime_tensor)
            
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self, batch_size: int = 32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        regime_vectors = torch.FloatTensor([e[1] for e in batch])
        actions = torch.LongTensor([e[2] for e in batch])
        rewards = torch.FloatTensor([e[3] for e in batch])
        next_states = torch.FloatTensor([e[4] for e in batch])
        next_regime_vectors = torch.FloatTensor([e[5] for e in batch])
        dones = torch.BoolTensor([e[6] for e in batch])
        
        current_q_values = self.q_network(states, regime_vectors).gather(1, actions.unsqueeze(1))
        
        with torch.no_grad():
            next_q_values = self.target_network(next_states, next_regime_vectors).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def calculate_regime_aware_reward(self, action, price_change, regime_profile, volatility):
        """
        Calculate reward based on regime characteristics
        Higher penalties in high-stress regimes, bonuses for regime-appropriate actions
        """
        base_reward = price_change if action == 1 else (-price_change if action == 2 else 0)
        
        # Regime-specific adjustments
        stress_score = regime_profile.get('market_stress', 50)
        vol_score = regime_profile.get('volatility_score', 50)
        
        # Penalty for trading in high-stress conditions
        if stress_score > 80 and action != 0:  # Non-hold action in crisis
            base_reward *= 0.5
            
        # Bonus for holding in high volatility
        if vol_score > 70 and action == 0:
            base_reward += 0.001  # Small holding bonus
            
        # Transaction cost (regime-dependent)
        if action != 0:
            transaction_cost = 0.001 * (1 + stress_score / 100)
            base_reward -= transaction_cost
            
        return base_reward

def train_rl_agent(price_data: pd.Series, regime_data: pd.DataFrame, episodes: int = 1000):
    """
    Train the regime-aware RL agent
    """
    logger.info("🤖 Training Regime-Aware RL Agent")
    
    state_size = 10  # lookback window
    agent = RegimeAwareTrader(state_size)
    
    # Prepare data
    returns = price_data.pct_change().dropna()
    
    for episode in range(episodes):
        total_reward = 0
        position = 0  # 0: no position, 1: long, -1: short
        
        for i in range(state_size, len(returns) - 1):
            # Current state (price features)
            state = returns.iloc[i-state_size:i].values
            
            # Current regime vector
            regime_row = regime_data.iloc[i]
            regime_vector = [
                regime_row.get('volatility_score', 50) / 100,
                regime_row.get('trend_score', 50) / 100,
                regime_row.get('market_stress', 50) / 100,
                regime_row.get('regime_stability', 50) / 100
            ]
            
            # Choose and execute action
            action = agent.act(state, regime_vector)
            
            # Calculate reward
            next_return = returns.iloc[i + 1]
            reward = agent.calculate_regime_aware_reward(
                action, next_return, 
                regime_data.iloc[i].to_dict(), 
                abs(next_return)
            )
            
            # Next state
            next_state = returns.iloc[i-state_size+1:i+1].values
            next_regime_vector = [
                regime_data.iloc[i+1].get('volatility_score', 50) / 100,
                regime_data.iloc[i+1].get('trend_score', 50) / 100,
                regime_data.iloc[i+1].get('market_stress', 50) / 100,
                regime_data.iloc[i+1].get('regime_stability', 50) / 100
            ]
            
            done = i == len(returns) - 2
            
            # Store experience
            agent.remember(state, regime_vector, action, reward, next_state, next_regime_vector, done)
            
            # Train
            if len(agent.memory) > 32:
                agent.replay()
                
            total_reward += reward
            
        # Update target network periodically
        if episode % 100 == 0:
            agent.update_target_network()
            logger.info(f"Episode {episode}, Total Reward: {total_reward:.4f}, Epsilon: {agent.epsilon:.4f}")
    
    logger.info("✅ RL Agent training complete")
    return agent
