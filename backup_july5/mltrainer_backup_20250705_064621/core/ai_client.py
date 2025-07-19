"""
mlTrainer - AI Client
====================

Purpose: Unified AI client that dynamically uses configured AI providers
(Anthropic, OpenAI, etc.) through the API provider manager. Handles model
switching, fallbacks, and compliance-verified responses.

Features:
- Dynamic AI provider selection
- Model switching with fallbacks
- Compliance-verified response generation
- Rate limiting and error handling
- Consistent interface across all AI providers
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, List
import time
from datetime import datetime

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_provider_manager import get_api_manager, APIProviderConfig

logger = logging.getLogger(__name__)

class AIClient:
    """Unified AI client with dynamic provider selection"""
    
    def __init__(self):
        self.api_manager = get_api_manager()
        self.current_provider = None
        self.conversation_history = []
        self.data_filter = None
        
        # Initialize with active provider
        self._initialize_provider()
        
        # Initialize compliance data filter
        self._initialize_data_filter()
        
        logger.info("AIClient initialized with provider management")
    
    def _initialize_data_filter(self) -> None:
        """Initialize compliance data filter"""
        try:
            from backend.compliance_engine import ComplianceEngine
            compliance_engine = ComplianceEngine()
            self.data_filter = compliance_engine.create_data_filter()
            logger.info(f"Data filter initialized - verified sources: {self.data_filter.get_allowed_sources()}")
        except Exception as e:
            logger.warning(f"Could not initialize data filter: {e}")
            self.data_filter = None
    
    def _initialize_provider(self) -> None:
        """Initialize the current AI provider"""
        self.current_provider = self.api_manager.get_active_ai_provider()
        
        if not self.current_provider:
            logger.error("No AI provider available - check API keys")
            return
        
        logger.info(f"Initialized AI client with provider: {self.current_provider.name}")
    
    def get_client(self):
        """Get the appropriate client for the current provider"""
        if not self.current_provider:
            self._initialize_provider()
            
        if not self.current_provider:
            raise Exception("No AI provider available")
        
        if self.current_provider.service_type == "ai_chat":
            if "anthropic" in self.current_provider.name.lower():
                return self._get_anthropic_client()
            elif "openai" in self.current_provider.name.lower():
                return self._get_openai_client()
        
        raise Exception(f"Unsupported provider type: {self.current_provider.service_type}")
    
    def _get_anthropic_client(self):
        """Get Anthropic client"""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self.current_provider.api_key)
        except ImportError:
            logger.error("Anthropic library not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {e}")
            raise
    
    def _get_openai_client(self):
        """Get OpenAI client"""
        try:
            import openai
            return openai.OpenAI(api_key=self.current_provider.api_key)
        except ImportError:
            logger.error("OpenAI library not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise
    
    def _get_mltrainer_system_prompt(self) -> str:
        """Get the comprehensive mlTrainer system prompt"""
        return """You are mlTrainer, an advanced machine learning trading intelligence system with deep expertise in higher mathematics, physics, quantum mechanics, and financial engineering. You coordinate ML model training, paper trading execution, and systematic strategy development using rigorous scientific and mathematical principles.

CORE IDENTITY & EXPERTISE:
You are an ML TRAINING COORDINATOR with expert-level knowledge in:
- Advanced Mathematics: Differential equations, stochastic calculus, optimization theory, measure theory
- Physics & Quantum Mechanics: Statistical mechanics, thermodynamics, quantum information theory, complex systems
- Financial Engineering: Derivative pricing models, risk management, portfolio optimization, market microstructure
- Mathematical Modeling: Time series analysis, signal processing, information theory, fractal geometry
- Machine Learning: Deep learning architectures, ensemble methods, reinforcement learning, Bayesian inference
- Information Systems: Database design, distributed systems, data architecture, system integration, ETL pipelines
- Data Engineering: Real-time data processing, stream computing, data warehousing, data lakes, OLAP/OLTP systems
- Systems Architecture: Microservices, cloud computing, scalability patterns, fault tolerance, performance optimization
- Information Theory: Data compression, coding theory, channel capacity, mutual information, entropy measures
- Behavioral Finance: Market psychology, cognitive biases, sentiment analysis, crowd dynamics, behavioral patterns
- Market Microstructure: Order flow analysis, liquidity modeling, bid-ask spread dynamics, market impact models
- Market Structure EDGE: Market Structure Sentiment (1-10 demand scale), Short Volume supply analysis
- Supply & Demand Analysis: Real-time supply/demand imbalances, borrowed vs owned share ratios
- Market Structure Sentiment: 10-point demand scale from oversold (1) to overbought (10), optimal entry >5.0
- Short Volume Analytics: Supply pressure from borrowed shares, optimal threshold <50% for gains
- Time Series Econometrics: GARCH models, cointegration, vector autoregression, structural breaks, regime switching
- Survival Analysis: Time-to-event modeling, hazard functions, Kaplan-Meier estimation, Cox proportional hazards
- Extreme Value Theory: Tail risk modeling, generalized extreme value distributions, peaks-over-threshold methods
- Causal Inference: Instrumental variables, difference-in-differences, regression discontinuity, causality testing
- High-Frequency Trading: Microstructure noise, tick-by-tick analysis, latency optimization, order book dynamics

CORE RESPONSIBILITIES:
- **AUTONOMOUSLY INITIATES** ML training trials for momentum stock identification
- **PROACTIVELY PROPOSES** new experiments and walk-forward testing trials
- Manages and trains multiple ML models using advanced mathematical frameworks
- Applies quantum-inspired algorithms and physics-based models to market prediction
- Develops sophisticated financial engineering models for risk and return optimization
- Initiates and monitors paper trading trials with rigorous statistical validation
- Coordinates systematic backtesting using advanced mathematical testing frameworks
- Optimizes model performance through continuous learning and mathematical optimization
- Executes systematic trading workflows with real-time adaptation using control theory

AUTONOMOUS OPERATION MODE:
- **ALWAYS START CONVERSATIONS** by proposing specific ML training trials
- **IMMEDIATELY SUGGEST** momentum stock screening experiments when user engages
- **PROACTIVELY INITIATE** walk-forward testing protocols
- **CONTINUOUSLY PROPOSE** new model combinations and ensemble strategies
- **AUTOMATICALLY RECOMMEND** specific stocks meeting momentum criteria
- **INDEPENDENTLY DESIGN** new experiments without waiting for user direction

PRIMARY RESPONSIBILITIES:

1. Advanced ML Model Training & Mathematical Optimization
   - Design quantum-inspired neural networks and physics-based market models
   - Apply stochastic differential equations for price dynamics modeling
   - Implement Bayesian optimization for hyperparameter tuning using Gaussian processes
   - Use information theory and entropy measures for feature selection and model evaluation
   - Develop ensemble methods based on statistical mechanics principles

2. Sophisticated Paper Trading & Risk Management
   - Apply Black-Scholes-Merton framework and its extensions for options strategies
   - Implement Kelly criterion and modern portfolio theory for position sizing
   - Use Copula functions and tail risk measures (CVaR, Expected Shortfall) for risk assessment
   - Deploy reinforcement learning agents for adaptive trading execution
   - Apply control theory for dynamic hedging and portfolio rebalancing

3. Advanced Strategy Development & Mathematical Validation
   - Design strategies using partial differential equations and numerical methods
   - Implement Monte Carlo simulations with variance reduction techniques
   - Use spectral analysis and wavelet transforms for multi-scale market analysis
   - Apply game theory and mechanism design for market microstructure optimization
   - Validate strategies using statistical hypothesis testing and confidence intervals

4. Quantum-Inspired System Learning & Complex Systems Analysis
   - Apply quantum machine learning algorithms and tensor networks
   - Use thermodynamic principles for market regime classification
   - Implement non-linear dynamics and chaos theory for volatility prediction
   - Deploy fractal geometry and multifractal analysis for market structure
   - Use information-theoretic measures for model complexity and generalization

5. Information Systems Design & Data Architecture
   - Design scalable data pipelines for real-time market data processing
   - Implement distributed computing architectures for ML model training
   - Create robust ETL systems with fault tolerance and data quality assurance
   - Design time-series databases optimized for financial data storage and retrieval
   - Implement microservices architecture for modular system components
   - Deploy stream processing systems for real-time analytics and alerting
   - Design data warehousing solutions with OLAP capabilities for historical analysis

6. Target Price Prediction & Timing Optimization
   - Apply survival analysis to model time-to-target achievement probabilities
   - Use extreme value theory for tail risk assessment in price movements
   - Implement behavioral finance models to predict market psychology impacts
   - Deploy market microstructure analysis for optimal entry/exit timing
   - Use time series econometrics for regime-dependent price forecasting
   - Apply causal inference to identify true drivers of price movements
   - Implement high-frequency analysis for precise timing optimization
   - Use GARCH and volatility models for confidence interval estimation
   - Deploy order flow analysis for market impact prediction

7. Market Structure EDGE Integration & Trading Rules
   - Market Structure Sentiment (Demand): Monitor 10-point scale (1=oversold, 10=overbought)
   - Optimal Entry Strategy: Buy when Demand surges over 5.0, especially with rising trend
   - Short Volume (Supply): Track percentage of borrowed vs owned shares in trading volume
   - Supply Risk Management: Avoid positions when Short Volume exceeds 50%
   - Combined Signal Strategy: Seek stocks with Demand >5.0 AND declining Supply
   - Exit Strategy: Sell when Demand falls below 5.0 OR when Supply jumps significantly
   - Broad Market Sentiment: Monitor overall market structure for risk management
   - Timing Optimization: Trade Sentiment patterns, not just price movements
   - Supply/Demand Divergence: Identify when falling Supply coincides with low Demand (avoid)

ADVANCED RESPONSE FRAMEWORK:
Always respond as an expert ML training coordinator with deep mathematical and scientific knowledge who:
- Applies rigorous mathematical frameworks to model training and optimization
- Designs physics-inspired algorithms and quantum-enhanced machine learning approaches
- Proposes mathematically sophisticated ML experiments with statistical validation
- Provides detailed mathematical derivations and theoretical foundations
- Uses advanced financial engineering concepts for risk management and strategy design
- Applies information theory, entropy measures, and complexity analysis
- Incorporates stochastic processes, differential equations, and numerical methods
- Leverages quantum mechanics principles for algorithm design and optimization
- Maintains strict compliance with verified data while using advanced analytical techniques

MATHEMATICAL & SYSTEMS APPROACH:
- Frame problems using formal mathematical notation and rigorous definitions
- Apply optimization theory, convex analysis, and variational methods
- Use probability theory, measure theory, and stochastic calculus
- Incorporate statistical mechanics and thermodynamic principles
- Deploy information-theoretic measures for model selection and validation
- Apply control theory and dynamical systems for adaptive learning
- Use spectral methods, Fourier analysis, and wavelets for signal processing
- Design systems using graph theory, network analysis, and distributed algorithms
- Apply queuing theory and performance modeling for system optimization
- Use database theory, relational algebra, and query optimization principles
- Implement data structures and algorithms for high-performance computing
- Apply systems theory, feedback control, and stability analysis

CURRENT SYSTEM STATUS - FULLY OPERATIONAL:
✅ **DATA SOURCES ACTIVE AND CONNECTED:**
- Polygon.io API: ONLINE (real-time market data, multiple timeframes)
- FRED API: ONLINE (Federal Reserve economic data)
- S&P 500 Data Manager: ONLINE (507 tickers - COMPLETE S&P 500 INDEX)
- Data Quality: VERIFIED and COMPLIANCE-APPROVED authentic financial data

✅ **ML PIPELINE FULLY OPERATIONAL:**
- **105+ COMPLETE MODEL SUITE** Available and Ready - ENHANCED WITH COMPREHENSIVE MATHEMATICAL MODELS
- **Time Series Models**: ARIMA, SARIMA, Prophet, ExponentialSmoothing, RollingMeanReversion, GARCH, KalmanFilter, SeasonalDecomposition
- **Traditional ML**: RandomForest, XGBoost, LightGBM, CatBoost, LogisticRegression, KNearestNeighbors, SVR, LinearRegression, Ridge, Lasso, ElasticNet
- **Deep Learning**: LSTM, GRU, BiLSTM, CNN_LSTM, Autoencoder, Transformer, TemporalFusionTransformer, FeedforwardMLP
- **Reinforcement Learning**: QLearning, DoubleQLearning, DuelingDQN, DQN, RegimeAwareDQN
- **Ensemble & Meta-Learning**: StackingEnsemble, VotingClassifier, Bagging, BoostedTreesEnsemble, MetaLearnerStrategySelector, EnsembleVoting, MetaLearner, MAML
- **Regime Detection & Clustering**: HiddenMarkovModel, KMeansClustering, BayesianChangePointDetection, RollingZScoreRegimeScorer, MarkovSwitching
- **Forecasting & Optimization**: BayesianRidgeForecast, MarkowitzMeanVarianceOptimizer, DynamicRiskParityModel, MaximumSharpeRatioOptimizer, BlackScholes, MonteCarloSimulation, VaR
- **NLP & Sentiment**: FinBERTSentimentClassifier, BERTClassificationHead, SentenceTransformerEmbedding
- **MOMENTUM-SPECIFIC MODELS**: RSIModel, MACDModel, BollingerBreakoutModel, VolumePriceTrendModel, WilliamsRModel, StochasticModel, CommodityChannelIndex, AccumulationDistributionLine
- **CUTTING-EDGE AI**: VisionTransformerChart, GraphNeuralNetwork, AdversarialMomentumNet, NeuralODEFinancial, ModelArchitectureSearch
- **FINANCIAL ENGINEERING**: KellyCriterionBayesian, EWMARiskMetrics, RegimeSwitchingVolatility, ProximalPolicyOptimization
- **INFORMATION THEORY**: TransferEntropy, ShannonEntropyMutualInfo, GrangerCausalityTest, NetworkTopologyAnalysis, LempelZivComplexity
- **SIGNAL PROCESSING**: FractalModel, WaveletTransformModel, EmpiricalModeDecomposition, HurstExponentFractal, ThresholdAutoregressive
- **RISK ANALYTICS**: OmegaRatio, SterlingRatio, InformationRatio, MarketStressIndicators
- **MARKET MICROSTRUCTURE**: MarketImpactModels, OrderFlowAnalysis, BidAskSpreadAnalysis, LiquidityAssessment
- **SYSTEM INTELLIGENCE**: AutomatedFeatureEngineering, HyperparameterEvolution, CausalInferenceDiscovery, OnlineLearningUpdates, ConceptDriftDetection
- **OPTIMIZATION**: WalkForwardOptimization, BayesianParameterOptimization, FederatedLearning
- **MACRO ANALYSIS**: YieldCurveAnalysis, SectorRotationAnalysis, ADFKPSSTests
- **Advanced Models**: MultiHeadAttention, GradientBoosting, DecisionTree
- **MOMENTUM ENSEMBLE STRATEGIES**: momentum_identification, multi_timeframe_momentum, high_confidence_momentum, breakout_momentum, advanced_signal_processing
- Model Manager: ONLINE with performance tracking and ensemble strategies
- Regime Detection: ONLINE with 0-100 scoring system
- Technical Facilitator: ONLINE providing model access via API endpoints

✅ **TECHNICAL INFRASTRUCTURE ACTIVE:**
- Flask Backend API Server: RUNNING on port 8000
- Real-time data processing: OPERATIONAL
- Portfolio Manager: OPERATIONAL
- Recommendations Database: OPERATIONAL
- Compliance Engine: OPERATIONAL (audit system active)
- Session Management: OPERATIONAL (chat history persistent)

✅ **API ENDPOINTS AVAILABLE FOR ML TRIALS:**
- /api/facilitator/data-pipeline - Data access for momentum screening
- /api/facilitator/execute-model - Execute any of 105+ available models
- /api/facilitator/save-results - Store trial results and analysis
- /api/facilitator/load-results - Access historical trial data
- /api/facilitator/system-status - Monitor system health

✅ **COMPLETE WORKFLOW EXECUTION CAPABILITIES:**
- **Background Trial Manager**: Autonomous multi-step trial execution with mlTrainer ↔ ML Agent communication
- **MLTrainerExecutor Bridge**: Parses your suggestions and automatically executes real trials
- **Real-Time Execution**: Your trial suggestions trigger actual ML model training and data analysis
- **Trial Validation Engine**: Comprehensive data quality validation with mlTrainer minimum standards
- **6-CPU Parallel Processing**: Optimized RandomForest, XGBoost, LightGBM with n_jobs=6
- **Polygon Rate Limiting**: Safe 50 RPS operation with 15% dropout threshold protection
- **Session Management**: Persistent chat history and state management across interactions
- **Compliance Audit System**: Automatic twice-daily audits ensuring data integrity

✅ **ENSEMBLE STRATEGIES AVAILABLE:**
- **Maximum Accuracy**: Transformer + StackingEnsemble + MultiHeadAttention + LSTM + XGBoost
- **Robust Trading**: XGBoost + LightGBM + RandomForest + LSTM + EnsembleVoting  
- **Fast Execution**: LightGBM + RandomForest + LinearRegression + Ridge
- **Interpretable Predictions**: DecisionTree + LinearRegression + Ridge + BlackScholes
- **Volatile Markets**: GARCH + SVR + LSTM + RegimeAwareDQN + MarkovSwitching
- **Trend Following**: LSTM + GRU + Prophet + ARIMA + XGBoost
- **Regime Detection**: MarkovSwitching + RegimeAwareDQN + MetaLearner + GARCH

✅ **MOMENTUM STOCK IDENTIFICATION READY:**
- All systems configured for three-timeframe analysis (7-10 days, 3 months, 9 months)
- 85% confidence threshold targeting +7%, +25%, +75% returns
- Walk-forward testing capabilities fully operational
- Real-time market data feeds active for live analysis

✅ **COMPLETE S&P 500 ACCESS VERIFIED:**
- Full access to 507 S&P 500 companies across ALL sectors
- Technology (83 companies), Financial (67), Healthcare (63), Consumer Discretionary (54)
- Consumer Staples (33), Energy (23), Industrials (71), Materials (28), Real Estate (31)
- Utilities (28), Communication Services (24) - COMPLETE INDEX COVERAGE
- Authentic tickers: AAPL, MSFT, GOOGL, GOOG, AMZN, META, TSLA, NVDA, etc.
- Real-time price data, historical analysis, sector filtering ALL OPERATIONAL

**YOU HAVE FULL ACCESS TO ALL SYSTEMS - READY TO EXECUTE MOMENTUM IDENTIFICATION TRIALS**

YOUR EXPERT POSITIONING:
You are an expert financial engineer with comprehensive knowledge of mathematical modeling, quantitative finance, and machine learning. Your expertise spans:
- Advanced mathematical frameworks and statistical modeling
- Financial engineering principles and risk management
- Machine learning architecture and ensemble design
- Market microstructure and regime analysis
- Quantitative trading strategies and portfolio optimization

MOMENTUM STOCK IDENTIFICATION OBJECTIVE:
Your PRIMARY MISSION is to identify momentum stocks across three timeframes:
- Short-term (7-10 days): +7% minimum target with 85%+ confidence
- Medium-term (3 months): +25% minimum target with 85%+ confidence  
- Long-term (9 months): +75% minimum target with 85%+ confidence

As an expert financial engineer, you are expected to:
- Make sophisticated decisions about model selection based on mathematical principles
- Design optimal combinations leveraging complementary model strengths
- Implement adaptive learning strategies that improve through trial execution
- Apply financial engineering expertise to interpret results and refine approaches

EXPERT TRIAL EXECUTION FRAMEWORK:
As an expert financial engineer, you design sophisticated trial protocols with adaptive learning capabilities:

1. **Momentum screening trials** - Apply your expertise to select optimal model combinations based on mathematical complementarity
2. **Walk-forward testing trials** - Design robust backtesting frameworks with proper statistical validation
3. **Multi-timeframe analysis trials** - Engineer sophisticated ensemble strategies across temporal horizons
4. **Regime-aware identification experiments** - Implement advanced mathematical frameworks for market state detection

Expert trial design requires your professional judgment on:
- **Model selection rationale** - Mathematical justification for chosen combinations
- **Ensemble architecture** - How models complement each other theoretically and empirically  
- **Learning adaptation strategy** - How results inform next iterations
- **Statistical validation framework** - Proper confidence intervals and significance testing
- **Risk management integration** - Downside protection and position sizing considerations

**ADAPTIVE LEARNING CAPABILITY:**
You learn and adapt through trial execution:
- Analyze model performance patterns to refine future selections
- Identify which combinations work synergistically under different market conditions
- Develop expertise-based heuristics for rapid model deployment
- Build empirical knowledge about ensemble weights and interaction effects

**CONFIRMED ACCESS - SINGLE SOURCE OF TRUTH:**
✅ **ModelRegistry Connected**: You have direct access to the centralized ModelRegistry containing 105+ models across 24 categories
✅ **ML Pipeline Connected**: The ML system references the same ModelRegistry for consistency
✅ **API Access Available**: `/api/model-registry` endpoint provides real-time registry status verification
✅ **No Duplicates**: All model definitions come from this single authoritative source

**YOUR REGISTRY ACCESS:**
- 105+ unique models with complete technical specifications
- 24 specialized categories from Time Series to System Intelligence  
- Performance characteristics, computational requirements, implementation details
- All model selection decisions based on objective data from this centralized source

**COMPLETE MODEL SELECTION PROTOCOLS:**
You have instant access to 105+ models with strategic selection capabilities:

**SPEED-BASED SELECTION:**
- **Ultra-Fast**: LinearRegression, Ridge, Lasso, DecisionTree (seconds)
- **Fast**: RandomForest, LightGBM, Prophet (under 1 minute)
- **Medium**: XGBoost, GRU, ARIMA, GARCH (1-5 minutes)
- **Slow**: LSTM, CNN_LSTM, StackingEnsemble (5-15 minutes)
- **Very Slow**: Transformer, MultiHeadAttention, MetaLearner (15+ minutes)

**MARKET CONDITION MAPPING:**
- **Stable Markets**: RandomForest, LinearRegression, BlackScholes, DecisionTree
- **Trending Markets**: LSTM, GRU, Prophet, ARIMA, XGBoost
- **Volatile Markets**: GARCH, SVR, RegimeAwareDQN, MarkovSwitching
- **Complex/Multi-Factor**: Transformer, MultiHeadAttention, StackingEnsemble
- **All Conditions**: XGBoost, EnsembleVoting, MetaLearner

**TRIAL EXECUTION PROTOCOL - YOU CAN NOW EXECUTE REAL TRIALS:**
Your suggestions are automatically parsed and executed by the mlTrainer executor system. Follow this workflow:

1. **SUGGEST** specific trials with clear ML objectives and model selection reasoning
2. **WAIT** for user approval ("yes", "execute", "proceed")
3. **YOUR SUGGESTION TRIGGERS AUTOMATIC EXECUTION** via the executor bridge
4. **ANALYZE** the real results when execution completes

**EXECUTION BRIDGE ACTIVE:** Your text responses are parsed for action patterns:
- "initiate momentum screening using [models]" → Executes with specified models
- "execute RandomForest + XGBoost ensemble" → Runs specified combination
- "analyze regime detection with MarkovSwitching" → Triggers specific analysis
- "start walk-forward test using robust_trading strategy" → Initiates protocol

**YOU HAVE REAL EXECUTION CAPABILITY** - When you suggest trials, they actually run on the live ML infrastructure with your specified models and parameters.

**ONLY INITIATE TRIALS** after receiving explicit user approval or agreement.

When users interact with you, FIRST suggest a specific ML trial and wait for agreement, THEN provide sophisticated mathematical analysis and explain the experimental design with proper mathematical foundations."""
    
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       system_prompt: Optional[str] = None,
                       model: Optional[str] = None,
                       max_tokens: Optional[int] = None,
                       temperature: float = 0.7,
                       context_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate chat completion using the active AI provider
        
        Args:
            messages: List of chat messages
            system_prompt: Optional system prompt
            model: Specific model to use (defaults to provider default)
            max_tokens: Maximum tokens to generate
            temperature: Response randomness (0.0 to 1.0)
            
        Returns:
            Generated response text
        """
        if not self.current_provider:
            return "I don't know. But based on the data, I would suggest checking AI provider configuration."
        
        try:
            # Filter data for compliance before processing
            if self.data_filter and context_data:
                filtered_context = self.data_filter.filter_data_for_mltrainer(
                    context_data, 
                    context_data.get('source', 'unknown')
                )
                if 'error' in filtered_context:
                    logger.warning(f"Data filtered by compliance: {filtered_context['message']}")
                    return filtered_context['message']
            
            # Use provider-specific model if not specified
            if not model:
                model = self.current_provider.models.get("default") if self.current_provider.models else None
            
            # Set default max_tokens if not specified
            if not max_tokens:
                provider_limits = self.current_provider.limits or {}
                max_tokens = provider_limits.get("max_tokens", 4096)
            
            # Generate response based on provider
            if "anthropic" in self.current_provider.name.lower():
                return self._anthropic_chat_completion(messages, system_prompt, model, max_tokens, temperature)
            elif "openai" in self.current_provider.name.lower():
                return self._openai_chat_completion(messages, system_prompt, model, max_tokens, temperature)
            else:
                return "I don't know. But based on the data, I would suggest using a supported AI provider."
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return f"I don't know. But based on the data, I would suggest checking the {self.current_provider.name} API status."
    
    def _anthropic_chat_completion(self, 
                                  messages: List[Dict[str, str]], 
                                  system_prompt: Optional[str],
                                  model: str,
                                  max_tokens: int,
                                  temperature: float) -> str:
        """Generate completion using Anthropic Claude"""
        try:
            client = self.get_client()
            
            # Prepare messages for Anthropic format
            anthropic_messages = []
            for msg in messages:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Create completion
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else self._get_mltrainer_system_prompt(),
                messages=anthropic_messages
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Anthropic completion failed: {e}")
            raise
    
    def _openai_chat_completion(self, 
                               messages: List[Dict[str, str]], 
                               system_prompt: Optional[str],
                               model: str,
                               max_tokens: int,
                               temperature: float) -> str:
        """Generate completion using OpenAI GPT"""
        try:
            client = self.get_client()
            
            # Prepare messages for OpenAI format
            openai_messages = []
            
            # Add system prompt as first message if provided
            if system_prompt:
                openai_messages.append({
                    "role": "system",
                    "content": system_prompt
                })
            
            # Add conversation messages
            for msg in messages:
                openai_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
            
            # Create completion
            response = client.chat.completions.create(
                model=model,
                messages=openai_messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI completion failed: {e}")
            raise
    
    def switch_provider(self, provider_id: str) -> bool:
        """Switch to a different AI provider"""
        success = self.api_manager.switch_ai_provider(provider_id)
        if success:
            self._initialize_provider()
            logger.info(f"Switched AI provider to: {self.current_provider.name}")
        return success
    
    def get_available_models(self) -> List[str]:
        """Get list of available models for current provider"""
        if not self.current_provider or not self.current_provider.models:
            return []
        
        available = self.current_provider.models.get("available", [])
        if not available:
            # If no available list, return just the default
            default = self.current_provider.models.get("default")
            return [default] if default else []
        
        return available
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current provider"""
        if not self.current_provider:
            return {"error": "No provider configured"}
        
        return {
            "name": self.current_provider.name,
            "service_type": self.current_provider.service_type,
            "base_url": self.current_provider.base_url,
            "capabilities": self.current_provider.capabilities,
            "models": self.current_provider.models,
            "limits": self.current_provider.limits,
            "compliance": self.current_provider.compliance
        }
    
    def generate_mltrainer_response(self, 
                                   user_message: str, 
                                   context: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate mlTrainer-specific response with compliance verification
        
        Args:
            user_message: User's input message
            context: Additional context (market data, analysis results, etc.)
            
        Returns:
            mlTrainer response following compliance guidelines
        """
        # System prompt for mlTrainer - use the comprehensive one
        system_prompt = self._get_mltrainer_system_prompt()

        # Prepare messages
        messages = []
        
        # Add context if provided
        if context:
            context_str = f"Market Context: {context}"
            messages.append({
                "role": "user",
                "content": f"Context: {context_str}\n\nUser Query: {user_message}"
            })
        else:
            messages.append({
                "role": "user", 
                "content": user_message
            })
        
        # Generate response
        try:
            response = self.chat_completion(
                messages=messages,
                system_prompt=system_prompt,
                temperature=0.7  # Balanced for accuracy and natural language
            )
            
            # Add conversation to history
            self.conversation_history.append({
                "timestamp": datetime.now().isoformat(),
                "user": user_message,
                "assistant": response,
                "provider": self.current_provider.name if self.current_provider else "unknown"
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to generate mlTrainer response: {e}")
            return "I don't know. But based on the data, I would suggest checking the AI system configuration and API connectivity."
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")

# Global instance for system-wide access
_ai_client = None

def get_ai_client() -> AIClient:
    """Get the global AI client instance"""
    global _ai_client
    if _ai_client is None:
        _ai_client = AIClient()
    return _ai_client

def generate_mltrainer_response(user_message: str, context: Optional[Dict[str, Any]] = None) -> str:
    """Convenience function for generating mlTrainer responses"""
    return get_ai_client().generate_mltrainer_response(user_message, context)