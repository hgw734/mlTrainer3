import logging

logger = logging.getLogger(__name__)


"""
Unified mlTrainer FastAPI Backend
==================================

Provides REST API and WebSocket endpoints for the unified system.
"""

from fastapi import FastAPI, WebSocket, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import asyncio
import json
from datetime import datetime

# Import unified components
from core.unified_executor import get_unified_executor
from core.enhanced_background_manager import get_enhanced_background_manager
from core.autonomous_loop import get_autonomous_loop
from mltrainer_claude_integration import MLTrainerClaude
from goal_system import GoalSystem
from utils.unified_memory import get_unified_memory

# Import authentication and metrics
from backend.auth import get_auth_manager, get_current_user, require_admin, UserCreate, UserLogin, Token, User
from backend.metrics_exporter import MetricsMiddleware, metrics_endpoint, metrics_update_loop

# Initialize FastAPI app
app = FastAPI(
    title="mlTrainer Unified API",
    description="API for the unified mlTrainer system with compliance and 140+ models",
    version="1.0.0",
)

# Add metrics middleware
app.add_middleware(MetricsMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class ChatMessage(BaseModel):
    message: str
    include_history: bool = True
    history_limit: int = 10


    class GoalUpdate(BaseModel):
        goal: str
        compliance_requirements: Optional[List[str]] = None


        class TrialApproval(BaseModel):
            trial_id: str
            approved: bool


            class AutonomousGoal(BaseModel):
                goal: str
                context: Optional[Dict[str, Any]] = None
                max_iterations: int = 10


                class ModelTrainingRequest(BaseModel):
                    model_id: str
                    symbol: str = "AAPL"
                    data_source: str = "polygon"
                    parameters: Optional[Dict[str, Any]] = None


                    # WebSocket manager
                    class ConnectionManager:
                        def __init__(self):
                            self.active_connections: List[WebSocket] = []

                            async def connect(self, websocket: WebSocket):
                                await websocket.accept()
                                self.active_connections.append(websocket)

                                def disconnect(self, websocket: WebSocket):
                                    self.active_connections.remove(websocket)

                                async def broadcast(self, message: dict):
                                    for connection in self.active_connections:
                                        try:
                                            await connection.send_json(message)
                                        except:
                                            # Connection might be closed
                                            pass


manager = ConnectionManager()


# Dependency to get components
def get_components():
    return {
        "executor": get_unified_executor(),
        "background_manager": get_enhanced_background_manager(),
        "autonomous_loop": get_autonomous_loop(),
        "claude": MLTrainerClaude(),
        "goal_system": GoalSystem(),
        "memory": get_unified_memory(),
    }


# Authentication Endpoints
@app.post("/auth/register", response_model=User)
async def register(user: UserCreate):
    """Register a new user"""
    auth_manager = get_auth_manager()
    new_user = auth_manager.create_user(user)

    if not new_user:
        raise HTTPException(status_code=400, detail="Username or email already exists")

    return new_user


@app.post("/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """Login and get access token"""
    auth_manager = get_auth_manager()
    user = auth_manager.authenticate_user(credentials.username, credentials.password)

    if not user:
        raise HTTPException(status_code=401, detail="Invalid username or password")

    return auth_manager.create_tokens(user)


@app.post("/auth/refresh")
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    auth_manager = get_auth_manager()
    new_token = auth_manager.refresh_access_token(refresh_token)

    if not new_token:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    return {"access_token": new_token}


@app.get("/auth/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return current_user


@app.post("/auth/logout")
async def logout(refresh_token: str, current_user: User = Depends(get_current_user)):
    """Logout and invalidate refresh token"""
    auth_manager = get_auth_manager()
    success = auth_manager.logout(current_user.user_id, refresh_token)
    return {"success": success}


# Metrics Endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return await metrics_endpoint()


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with system information"""
    components = get_components()
    return {
        "service": "mlTrainer Unified API",
        "status": "operational",
        "features": {
            "models_available": components["executor"].get_execution_summary()["registered_actions"],
            "current_goal": components["goal_system"].get_current_goal(),
            "memory_stats": components["memory"].get_memory_stats(),
        },
    }


@app.post("/api/chat")
async def chat(
    message: ChatMessage, components: dict = Depends(get_components), current_user: User = Depends(get_current_user)
):
    """Process chat message through mlTrainer"""
    try:
        # Get conversation history if requested
        history = []
        if message.include_history:
            recent_messages = components["memory"].get_recent_context(limit=message.history_limit)
            history = [{"role": m["role"], "content": m["content"]} for m in recent_messages]

        # Get current goal for context
        current_goal = components["goal_system"].get_current_goal()

        # Get response from mlTrainer
        response = components["claude"].get_response_with_goal(message.message, current_goal.get("goal", ""), history)

        # Parse for executable actions
        parsed = components["executor"].parse_mltrainer_response(response)

        # Save to memory
        components["memory"].add_message("user", message.message, goal_context=current_goal)
        components["memory"].add_message("assistant", response, executable=parsed["executable"])

        # Create trial if executable
        trial_id = None
        if parsed["executable"]:
            trial_id = components["background_manager"].start_trial(response, auto_approve=False)

        return {
            "response": response,
            "executable": parsed["executable"],
            "parsed": parsed,
            "trial_id": trial_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/goal")
async def get_goal(components: dict = Depends(get_components)):
    """Get current system goal"""
    return components["goal_system"].get_current_goal()


@app.post("/api/goal")
async def update_goal(goal_update: GoalUpdate, components: dict = Depends(get_components)):
    """Update system goal"""
    try:
        # Update goal
        components["goal_system"].set_goal(goal_update.goal)

        # Log to memory
        components["memory"].add_goal_change(
            old_goal=components["goal_system"].get_current_goal().get("goal", ""),
            new_goal=goal_update.goal,
            reason="API update",
        )

        return {"success": True, "goal": components["goal_system"].get_current_goal()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/trials")
async def get_trials(components: dict = Depends(get_components)):
    """Get all trials"""
    return components["background_manager"].get_all_trials()


@app.get("/api/trials/{trial_id}")
async def get_trial(trial_id: str, components: dict = Depends(get_components)):
    """Get specific trial status"""
    status = components["background_manager"].get_trial_status(trial_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail="Trial not found")
    return status


@app.post("/api/trials/approve")
async def approve_trial(approval: TrialApproval, components: dict = Depends(get_components)):
    """Approve or reject a trial"""
    if approval.approved:
        success = components["background_manager"].approve_trial(approval.trial_id)
        return {"success": success, "message": "Trial approved" if success else "Failed to approve"}
    else:
        success = components["background_manager"].cancel_trial(approval.trial_id)
        return {"success": success, "message": "Trial cancelled" if success else "Failed to cancel"}


@app.post("/api/models/train")
async def train_model(
    request: ModelTrainingRequest,
    components: dict = Depends(get_components),
    current_user: User = Depends(get_current_user),
):
    """Direct model training endpoint"""
    try:
        result = components["executor"].execute_ml_model_training(
            request.model_id, symbol=request.symbol, data_source=request.data_source, **(request.parameters or {})
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def get_models(components: dict = Depends(get_components)):
    """Get available models"""
    executor = components["executor"]
    ml_models = [name.replace("train_", "") for name in list(executor.registered_actions.keys()) if name.startswith("train_")]
    financial_models = [
        name.replace("calculate_", "") for name in list(executor.registered_actions.keys()) if name.startswith("calculate_")
    ]

    return {
        "ml_models": ml_models,
        "financial_models": financial_models,
        "special_actions": ["momentum_screening", "regime_detection", "portfolio_optimization"],
        "total": len(executor.registered_actions),
    }


@app.post("/api/autonomous/start")
async def start_autonomous(
    goal: AutonomousGoal, components: dict = Depends(get_components), current_user: User = Depends(get_current_user)
):
    """Start an autonomous session"""
    try:
        session_id = await components["autonomous_loop"].start_autonomous_session(goal.goal, goal.context)
        return {"session_id": session_id, "status": "started", "goal": goal.goal}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/autonomous/{session_id}")
async def get_autonomous_status(session_id: str, components: dict = Depends(get_components)):
    """Get autonomous session status"""
    status = components["autonomous_loop"].get_session_status(session_id)
    if "error" in status:
        raise HTTPException(status_code=404, detail="Session not found")
    return status


@app.post("/api/autonomous/{session_id}/stop")
async def stop_autonomous(session_id: str, components: dict = Depends(get_components)):
    """Stop an autonomous session"""
    success = await components["autonomous_loop"].stop_session(session_id)
    return {"success": success}


@app.get("/api/memory/search")
async def search_memory(topic: str, limit: int = 10, components: dict = Depends(get_components)):
    """Search memory by topic"""
    results = components["memory"].search_by_topic(topic, limit)
    return {"topic": topic, "results": results, "count": len(results)}


@app.get("/api/memory/stats")
async def memory_stats(components: dict = Depends(get_components)):
    """Get memory statistics"""
    return components["memory"].get_memory_stats()


@app.get("/api/compliance/history")
async def compliance_history(event_type: Optional[str] = None, components: dict = Depends(get_components)):
    """Get compliance event history"""
    return components["memory"].get_compliance_history(event_type)


# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial connection message
        await websocket.send_json(
            {"type": "connection", "status": "connected", "timestamp": datetime.now().isoformat()}
        )

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()

            # Echo back for now
            await websocket.send_json({"type": "echo", "data": data, "timestamp": datetime.now().isoformat()})

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(websocket)


# Background task to broadcast trial updates
async def broadcast_trial_updates():
    """Periodically broadcast trial status updates"""
    components = get_components()
    while True:
        await asyncio.sleep(5)  # Every 5 seconds

        # Get active trials
        trials = components["background_manager"].get_all_trials()
        active_trials = [t for t in trials if t.get("status") not in ["completed", "failed", "cancelled"]]

        if active_trials:
            await manager.broadcast(
                {"type": "trial_update", "active_trials": active_trials, "timestamp": datetime.now().isoformat()}
            )


# Start background tasks
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(broadcast_trial_updates())
    asyncio.create_task(metrics_update_loop())


# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
