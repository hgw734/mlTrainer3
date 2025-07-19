
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class MAMLFinancialPredictor(nn.Module):
    """
    Model-Agnostic Meta-Learning for financial prediction across markets
    Based on research recommendations for cross-market generalization
    """
    
    def __init__(self, input_size: int, hidden_size: int = 64, output_size: int = 1):
        super(MAMLFinancialPredictor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        
    def forward(self, x):
        return self.network(x)
    
    def clone(self):
        """Create a deep copy of the model"""
        cloned = MAMLFinancialPredictor(
            self.network[0].in_features,
            self.network[2].in_features,
            self.network[-1].out_features
        )
        cloned.load_state_dict(self.state_dict())
        return cloned

class MetaLearningFramework:
    """
    Meta-learning framework for financial prediction across different market regimes
    Implements transfer learning for regime-adaptive modeling
    """
    
    def __init__(self, input_size: int, lr_inner: float = 0.01, lr_outer: float = 0.001):
        self.model = MAMLFinancialPredictor(input_size)
        self.lr_inner = lr_inner  # Task-specific learning rate
        self.lr_outer = lr_outer  # Meta-learning rate
        self.meta_optimizer = optim.Adam(self.model.parameters(), lr=lr_outer)
        
    def inner_update(self, model, loss, create_graph=True):
        """Perform one gradient step for task-specific adaptation"""
        grads = torch.autograd.grad(loss, model.parameters(), create_graph=create_graph)
        updated_params = []
        
        for param, grad in zip(model.parameters(), grads):
            updated_params.append(param - self.lr_inner * grad)
            
        # Create new model with updated parameters
        updated_model = model.clone()
        for param, updated_param in zip(updated_model.parameters(), updated_params):
            param.data = updated_param.data
            
        return updated_model
    
    def meta_train_step(self, support_tasks: List[Tuple], query_tasks: List[Tuple]):
        """
        Perform one meta-training step
        
        Args:
            support_tasks: List of (X_support, y_support) for adaptation
            query_tasks: List of (X_query, y_query) for meta-update
        """
        meta_loss = 0.0
        
        for (X_support, y_support), (X_query, y_query) in zip(support_tasks, query_tasks):
            # Convert to tensors
            X_support = torch.FloatTensor(X_support)
            y_support = torch.FloatTensor(y_support).unsqueeze(1)
            X_query = torch.FloatTensor(X_query)
            y_query = torch.FloatTensor(y_query).unsqueeze(1)
            
            # Clone model for task-specific adaptation
            task_model = self.model.clone()
            
            # Inner loop: adapt to support set
            support_pred = task_model(X_support)
            support_loss = nn.MSELoss()(support_pred, y_support)
            
            # Update model parameters
            adapted_model = self.inner_update(task_model, support_loss)
            
            # Evaluate on query set
            query_pred = adapted_model(X_query)
            query_loss = nn.MSELoss()(query_pred, y_query)
            
            meta_loss += query_loss
        
        # Meta-update
        meta_loss = meta_loss / len(support_tasks)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt_to_new_market(self, X_new: np.ndarray, y_new: np.ndarray, 
                           adaptation_steps: int = 5) -> nn.Module:
        """
        Quickly adapt the meta-learned model to a new market/regime
        
        Args:
            X_new: New market features
            y_new: New market targets
            adaptation_steps: Number of gradient steps for adaptation
        """
        logger.info(f"🔄 Adapting meta-model to new market with {adaptation_steps} steps")
        
        # Clone the meta-model
        adapted_model = self.model.clone()
        optimizer = optim.SGD(adapted_model.parameters(), lr=self.lr_inner)
        
        X_tensor = torch.FloatTensor(X_new)
        y_tensor = torch.FloatTensor(y_new).unsqueeze(1)
        
        # Fine-tune on new market data
        for step in range(adaptation_steps):
            pred = adapted_model(X_tensor)
            loss = nn.MSELoss()(pred, y_tensor)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if step % 2 == 0:
                logger.info(f"  Adaptation step {step}, Loss: {loss.item():.6f}")
        
        logger.info("✅ Market adaptation complete")
        return adapted_model

def create_regime_tasks(price_data: Dict[str, pd.Series], 
                       regime_data: Dict[str, pd.DataFrame]) -> Tuple[List, List]:
    """
    Create meta-learning tasks from different market regimes
    Each task represents a different market condition or time period
    """
    logger.info("📚 Creating meta-learning tasks from regime data")
    
    support_tasks = []
    query_tasks = []
    
    for symbol, prices in price_data.items():
        if symbol not in regime_data:
            continue
            
        returns = prices.pct_change().dropna()
        regimes = regime_data[symbol]
        
        # Create tasks based on different regime periods
        regime_periods = []
        current_regime = None
        start_idx = 0
        
        for i, row in regimes.iterrows():
            regime_vector = [
                row.get('volatility_score', 50),
                row.get('trend_score', 50),
                row.get('market_stress', 50),
                row.get('regime_stability', 50)
            ]
            
            # Detect regime changes (simplified)
            if current_regime is None:
                current_regime = regime_vector
                start_idx = i
            elif np.linalg.norm(np.array(regime_vector) - np.array(current_regime)) > 20:
                # Regime change detected
                if i - start_idx > 30:  # Minimum period length
                    regime_periods.append((start_idx, i, current_regime))
                current_regime = regime_vector
                start_idx = i
        
        # Add final period
        if len(regimes) - start_idx > 30:
            regime_periods.append((start_idx, len(regimes), current_regime))
        
        # Create tasks from regime periods
        for start, end, regime_vector in regime_periods:
            period_returns = returns.iloc[start:end]
            if len(period_returns) < 20:
                continue
                
            # Create features (simple moving averages and regime info)
            features = []
            targets = []
            
            for i in range(10, len(period_returns) - 1):
                # Price features
                price_features = period_returns.iloc[i-10:i].values
                # Regime features
                regime_features = np.array(regime_vector) / 100
                # Combined features
                combined_features = np.concatenate([price_features, regime_features])
                
                features.append(combined_features)
                targets.append(period_returns.iloc[i+1])
            
            if len(features) > 10:
                # Split into support and query sets
                split_idx = len(features) // 2
                support_tasks.append((features[:split_idx], targets[:split_idx]))
                query_tasks.append((features[split_idx:], targets[split_idx:]))
    
    logger.info(f"✅ Created {len(support_tasks)} meta-learning tasks")
    return support_tasks, query_tasks

def train_meta_learning_model(price_data: Dict[str, pd.Series], 
                             regime_data: Dict[str, pd.DataFrame],
                             episodes: int = 100) -> MetaLearningFramework:
    """
    Train the meta-learning model across different markets and regimes
    """
    logger.info("🧠 Training Meta-Learning Model for Cross-Market Adaptation")
    
    # Create tasks from regime data
    support_tasks, query_tasks = create_regime_tasks(price_data, regime_data)
    
    if not support_tasks:
        logger.error("❌ No valid tasks created for meta-learning")
        return None
    
    # Initialize meta-learning framework
    input_size = len(support_tasks[0][0][0])  # Feature dimension
    meta_learner = MetaLearningFramework(input_size)
    
    # Training loop
    for episode in range(episodes):
        # Sample a batch of tasks
        batch_size = min(8, len(support_tasks))
        task_indices = np.random.choice(len(support_tasks), batch_size, replace=False)
        
        batch_support = [support_tasks[i] for i in task_indices]
        batch_query = [query_tasks[i] for i in task_indices]
        
        # Meta-training step
        meta_loss = meta_learner.meta_train_step(batch_support, batch_query)
        
        if episode % 10 == 0:
            logger.info(f"Episode {episode}, Meta-Loss: {meta_loss:.6f}")
    
    logger.info("✅ Meta-learning training complete")
    return meta_learner
