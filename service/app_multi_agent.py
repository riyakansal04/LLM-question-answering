#!/usr/bin/env python


import os
import sys
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.multi_agent import CricketInferenceOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferRequest(BaseModel):
    question: str
    max_tokens: int = 512
    temperature: float = 0.7


class InferResponse(BaseModel):
    question: str
    answer: str
    latency_ms: float
    context_facts: int
    status: str = "success"


class HealthResponse(BaseModel):
    status: str
    message: str

orchestrator: Optional[CricketInferenceOrchestrator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    
    global orchestrator
    
    # STARTUP
    logger.info("="*80)
    logger.info("Starting IPL Cricket Q&A Service with Multi-Agent Orchestration")
    logger.info("="*80)
    
    try:
        # Configuration from environment
        hf_token = os.getenv('HF_TOKEN')
        adapter_repo = os.getenv('ADAPTER_REPO', 'riyakl09/ipl-cricket-sft')
        adapter_path = os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft')
        data_path = os.getenv('DATA_PATH', 'data/ipl_qa.json')
        model_name = os.getenv('BASE_MODEL', 'Qwen/Qwen2-0.5B-Instruct')
        
        logger.info(f"Configuration:")
        logger.info(f"  Model: {model_name}")
        logger.info(f"  Adapter Repo (HF): {adapter_repo}")
        logger.info(f"  Adapter Path (Local): {adapter_path}")
        logger.info(f"  Data Path: {data_path}")
        logger.info(f"  HF Token: {'✓ Configured' if hf_token else '✗ Not configured'}")
        logger.info(f"  Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        logger.info("")
        
        # Initialize orchestrator
        logger.info("Initializing multi-agent orchestration system...")
        orchestrator = CricketInferenceOrchestrator(
            data_path=data_path,
            model_name=model_name,
            adapter_repo=adapter_repo,
            adapter_path=adapter_path,
            hf_token=hf_token,
        )
        
        logger.info("✓ Multi-agent orchestrator initialized successfully")
        logger.info("✓ Ready to accept requests")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Failed to initialize orchestrator: {e}")
        raise
    
    # Yield control to FastAPI
    yield
    
    # SHUTDOWN
    logger.info("Shutting down service...")
    if orchestrator:
        orchestrator = None
        logger.info("✓ Orchestrator shutdown complete")


app = FastAPI(
    title="IPL Cricket Q&A Service",
    description="Multi-agent orchestrated inference service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/infer", response_model=InferResponse)
async def infer(request: InferRequest) -> InferResponse:
   
    if orchestrator is None:
        logger.error("Orchestrator not initialized")
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Multi-agent orchestrator not initialized."
        )
    
    try:
        logger.info(f"Processing inference request: {request.question[:60]}...")
        
        # Measure total latency
        start_time = time.time()
        
        # Run multi-agent pipeline
        result = orchestrator.answer_question(request.question)
        
        elapsed_time = time.time() - start_time
        
        logger.info(
            f"✓ Inference complete in {elapsed_time:.2f}s "
            f"({result['facts_count']} facts retrieved)"
        )
        
        return InferResponse(
            question=request.question,
            answer=result['answer'],
            latency_ms=elapsed_time * 1000,
            context_facts=result['facts_count'],
            status="success",
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {str(e)}"
        )


@app.get("/healthz", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    
    status = "healthy" if orchestrator is not None else "unhealthy"
    
    return HealthResponse(
        status=status,
        message="IPL Cricket Q&A Service is operational"
    )


@app.get("/readyz", response_model=HealthResponse)
async def readiness_check() -> HealthResponse:
    
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Service not ready. Orchestrator not initialized."
        )
    
    return HealthResponse(
        status="ready",
        message="IPL Cricket Q&A Service is ready to accept requests"
    )


@app.get("/", response_model=Dict[str, str])
async def root() -> Dict[str, str]:
    
    return {
        "service": "IPL Cricket Q&A",
        "version": "1.0.0",
        "endpoints": {
            "/infer": "POST - Run inference with multi-agent orchestration",
            "/healthz": "GET - Health check",
            "/readyz": "GET - Readiness check",
            "/docs": "GET - Interactive API documentation (Swagger UI)",
            "/redoc": "GET - ReDoc documentation",
        }
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler."""
    logger.error(f"HTTP Exception: {exc.status_code} - {exc.detail}")
    return {
        "error": True,
        "status_code": exc.status_code,
        "detail": exc.detail,
    }

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )
