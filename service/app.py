
import logging
import os
import time
from typing import Optional
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from dotenv import load_dotenv
from huggingface_hub import list_repo_files

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class InferRequest(BaseModel):

    question: str = Field(..., description="IPL cricket question to answer", min_length=3)
    max_length: int = Field(512, description="Maximum response length", ge=50, le=2048)
    temperature: float = Field(0.7, description="Sampling temperature", ge=0.0, le=2.0)


class InferResponse(BaseModel):
    """Response payload for inference endpoint."""
    question: str = Field(..., description="The asked question")
    answer: str = Field(..., description="Generated answer")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    tokens_generated: int = Field(..., description="Number of tokens generated")


class ModelManager:

    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = None
        self.is_loaded = False
    
    def load_model(
        self,
        model_name: str,
        adapter_repo: Optional[str] = None,
        adapter_path: Optional[str] = None,
        hf_token: Optional[str] = None,
    ) -> None:
  
        if self.is_loaded:
            logger.info("Model already loaded in cache")
            return
        
        logger.info(f"Loading model: {model_name}")
        
        # Determine device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            token=hf_token,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info("Loading base model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=True,
            token=hf_token,
            low_cpu_mem_usage=True,
        )
        
        # Load LoRA adapters - try HF repo first, then local path
        adapter_loaded = False
        
        # Try HuggingFace repository
        if adapter_repo:
            logger.info(f"Loading LoRA adapters from HF repo: {adapter_repo}...")
            
            # Debug: List all files in the repo
            try:
                files = list_repo_files(adapter_repo, repo_type='model', token=hf_token)
                logger.info(f"Files in {adapter_repo}:")
                for f in sorted(files):
                    logger.info(f"  - {f}")
            except Exception as e:
                logger.warning(f"Could not list files: {e}")
            
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_repo,
                    is_trainable=False,
                    token=hf_token,
                )
                logger.info("✓ LoRA adapters loaded from HuggingFace Hub")
                adapter_loaded = True
            except Exception as e:
                logger.warning(f"Failed to load from HF repo: {e}")
                logger.info("Falling back to local adapter path...")
        
        # Fall back to local adapter path
        if not adapter_loaded and adapter_path and os.path.exists(adapter_path):
            logger.info(f"Loading LoRA adapters from {adapter_path}...")
            try:
                self.model = PeftModel.from_pretrained(
                    self.model,
                    adapter_path,
                    is_trainable=False,
                )
                logger.info("✓ LoRA adapters loaded from local path")
                adapter_loaded = True
            except Exception as e:
                logger.error(f"✗ Failed to load adapters: {e}")
                logger.warning("Continuing with base model only.")
        elif not adapter_loaded and adapter_path:
            logger.warning(f"Local adapter path not found: {adapter_path}")
        
        if not adapter_loaded and not adapter_repo:
            logger.info("Using base model only (no adapters specified)")

        
        self.model.eval()
        self.is_loaded = True
        
        # Log model info
        param_count = sum(p.numel() for p in self.model.parameters()) / 1e9
        logger.info(f"✓ Model loaded ({param_count:.2f}B parameters)")
    
    def infer(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
    ) -> tuple:
        """Run inference and return (text, tokens_generated, latency_ms)."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded")
        
        start_time = time.time()
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )
        
        # Metrics
        latency_ms = (time.time() - start_time) * 1000
        tokens_generated = outputs.shape[1] - inputs['input_ids'].shape[1]
        
        return generated_text, tokens_generated, latency_ms


# Global model manager
model_manager = ModelManager()


@asynccontextmanager
async def lifespan(app: FastAPI):

    # Startup
    logger.info("=" * 70)
    logger.info("Starting IPL Cricket Q&A Service")
    logger.info("=" * 70)
    
    # Get config from .env (already loaded at module level)
    hf_token = os.getenv('HF_TOKEN')
    base_model = os.getenv('BASE_MODEL', 'Qwen/Qwen2-0.5B-Instruct')
    
    # HuggingFace repo with adapter files
    adapter_repo = os.getenv('ADAPTER_REPO', 'riyakl09/ipl-cricket-sft')
    adapter_path_env = os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft')
    
    # Resolve local adapter path (support both relative and absolute paths)
    if not os.path.isabs(adapter_path_env):
        adapter_path = os.path.join(os.getcwd(), adapter_path_env)
    else:
        adapter_path = adapter_path_env
    
    logger.info(f"Base model: {base_model}")
    logger.info(f"Adapter repo (HF): {adapter_repo}")
    logger.info(f"Adapter path (local): {adapter_path}")
    logger.info(f"Local adapter exists: {os.path.exists(adapter_path)}")
    logger.info(f"HF_TOKEN configured: {bool(hf_token)}")
    
    try:
        model_manager.load_model(
            model_name=base_model,
            adapter_repo=adapter_repo,
            adapter_path=adapter_path,
            hf_token=hf_token,
        )
        logger.info("✓ Service startup complete")
    except Exception as e:
        logger.error(f"✗ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down service...")
    if model_manager.model is not None:
        del model_manager.model
        del model_manager.tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


# Create FastAPI app
app = FastAPI(
    title='IPL Cricket QA Service',
    description='Fine-tuned Qwen2-0.5B on IPL cricket Q&A',
    version='1.0.0',
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', tags=['info'])
async def root() -> dict:
    """Root endpoint with service info."""
    return {
        'service': 'IPL Cricket QA',
        'version': '1.0.0',
        'model_loaded': model_manager.is_loaded,
        'endpoints': {
            'healthz': '/healthz',
            'readyz': '/readyz',
            'infer': '/infer',
            'docs': '/docs',
        }
    }


@app.get('/healthz', tags=['health'])
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        'status': 'ok',
        'service': 'ipl-cricket-qa',
        'model_loaded': model_manager.is_loaded,
    }


@app.get('/readyz', tags=['health'])
async def readiness_check() -> dict:
    """Readiness check endpoint."""
    return {
        'ready': model_manager.is_loaded,
    }


@app.post('/infer', response_model=InferResponse, tags=['inference'])
async def answer_query(request: InferRequest) -> InferResponse:
    """Answer an IPL cricket question."""
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail='Model not loaded. Service not ready.'
        )
    
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail='Question cannot be empty'
        )
    
    try:
        # Format prompt
        prompt = f"""You are an expert on IPL cricket. Answer the following question:

Question: {request.question}

Answer:"""
        
        logger.info(f"Processing: {request.question[:60]}...")
        answer, tokens, latency = model_manager.infer(
            prompt=prompt,
            max_new_tokens=request.max_length,
            temperature=request.temperature,
        )
        
        # Extract answer
        answer_text = answer.split("Answer:")[-1].strip()
        
        logger.info(f"✓ Done ({latency:.1f}ms, {tokens} tokens)")
        
        return InferResponse(
            question=request.question,
            answer=answer_text,
            latency_ms=round(latency, 2),
            tokens_generated=tokens,
        )
    
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f'Inference failed: {str(e)}'
        )


if __name__ == '__main__':
    import uvicorn
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', '8000'))
    
    logger.info(f"Starting server on http://{host}:{port}")
    logger.info(f"Docs available at http://{host}:{port}/docs")
    
    uvicorn.run(
        'app:app',
        host=host,
        port=port,
        reload=False,
        log_level='info',
    )
