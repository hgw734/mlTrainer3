"""
mlTrainer - Flask Backend API Server
===================================

Purpose: Backend API server for handling real-time data processing,
ML model inference, and compliance verification for the mlTrainer system.

This server runs alongside the Streamlit frontend to provide:
- Real-time data ingestion from multiple sources
- ML model serving and inference
- Compliance validation and filtering
- Portfolio management and recommendations
"""

from flask import Flask, jsonify, request
import os
import sys
from datetime import datetime
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backend.data_sources import DataSourceManager
from backend.compliance_engine import ComplianceEngine
from core.ml_pipeline import MLPipeline
from core.regime_detector import RegimeDetector
from data.portfolio_manager import PortfolioManager
from data.recommendations_db import RecommendationsDB

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize core components
data_manager = DataSourceManager()
compliance_engine = ComplianceEngine()
ml_pipeline = MLPipeline()
regime_detector = RegimeDetector()
portfolio_manager = PortfolioManager()
recommendations_db = RecommendationsDB()

@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "data_sources": data_manager.check_connections(),
                "compliance": compliance_engine.is_active(),
                "ml_pipeline": ml_pipeline.is_ready(),
                "regime_detector": regime_detector.is_initialized()
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 500

@app.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """Get current stock recommendations"""
    try:
        if not compliance_engine.check_compliance_status()["is_compliant"]:
            return jsonify({
                "error": "Compliance mode disabled",
                "message": "I don't know. But based on the data, I would suggest enabling compliance mode first."
            }), 403
        
        # Get live recommendations
        recommendations = recommendations_db.get_active_recommendations()
        
        # Validate data compliance
        validated_recommendations = []
        for rec in recommendations:
            if compliance_engine.validate_recommendation(rec):
                validated_recommendations.append(rec)
        
        return jsonify({
            "recommendations": validated_recommendations,
            "count": len(validated_recommendations),
            "timestamp": datetime.now().isoformat(),
            "compliance_verified": True
        })
        
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        return jsonify({
            "error": "Recommendation retrieval failed",
            "message": "I don't know. But based on the data, I would suggest checking system connectivity."
        }), 500

@app.route('/api/regime-analysis', methods=['GET'])
def get_regime_analysis():
    """Get current market regime analysis"""
    try:
        # Get latest market data
        market_data = data_manager.get_regime_indicators()
        
        if not market_data:
            return jsonify({
                "error": "Insufficient data",
                "message": "I don't know. But based on the data, I would suggest waiting for market data updates."
            }), 400
        
        # Perform regime detection
        regime_analysis = regime_detector.analyze_current_regime(market_data)
        
        # Validate compliance
        if compliance_engine.validate_regime_data(regime_analysis):
            return jsonify({
                "regime_analysis": regime_analysis,
                "timestamp": datetime.now().isoformat(),
                "compliance_verified": True
            })
        else:
            return jsonify({
                "error": "Data validation failed",
                "message": "I don't know. But based on the data, I would suggest verifying data sources."
            }), 400
            
    except Exception as e:
        logger.error(f"Regime analysis failed: {e}")
        return jsonify({
            "error": "Regime analysis failed",
            "message": "I don't know. But based on the data, I would suggest checking system status."
        }), 500

@app.route('/api/portfolio', methods=['GET'])
def get_portfolio():
    """Get current portfolio holdings"""
    try:
        portfolio = portfolio_manager.get_current_portfolio()
        return jsonify({
            "portfolio": portfolio,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Portfolio retrieval failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/add', methods=['POST'])
def add_to_portfolio():
    """Add recommendation to portfolio"""
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({"error": "Missing ticker symbol"}), 400
        
        # Validate recommendation exists and is compliant
        recommendation = recommendations_db.get_recommendation(data['ticker'])
        if not recommendation:
            return jsonify({
                "error": "Recommendation not found",
                "message": "I don't know. But based on the data, I would suggest selecting from active recommendations."
            }), 404
        
        if not compliance_engine.validate_recommendation(recommendation):
            return jsonify({
                "error": "Compliance validation failed",
                "message": "I don't know. But based on the data, I would suggest waiting for data verification."
            }), 400
        
        # Add to portfolio
        result = portfolio_manager.add_holding(recommendation)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to add to portfolio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/portfolio/remove', methods=['POST'])
def remove_from_portfolio():
    """Remove holding from portfolio"""
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({"error": "Missing ticker symbol"}), 400
        
        result = portfolio_manager.remove_holding(data['ticker'])
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to remove from portfolio: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get active alerts and notifications"""
    try:
        from core.notification_system import NotificationSystem
        notification_system = NotificationSystem()
        
        alerts = notification_system.get_active_alerts()
        return jsonify({
            "alerts": alerts,
            "count": len(alerts),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Failed to get alerts: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/model-status', methods=['GET'])
def get_model_status():
    """Get ML model status and performance"""
    try:
        model_status = ml_pipeline.get_model_status()
        return jsonify({
            "models": model_status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Model status retrieval failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/compliance/toggle', methods=['POST'])
def toggle_compliance():
    """Toggle compliance mode"""
    try:
        data = request.get_json()
        enabled = data.get('enabled', True)
        
        result = compliance_engine.set_compliance_mode(enabled)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Compliance toggle failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
