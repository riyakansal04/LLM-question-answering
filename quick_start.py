#!/usr/bin/env python

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def option1_standalone():
    
    print("\n" + "="*80)
    print("OPTION 1: Standalone Multi-Agent System")
    print("="*80 + "\n")
    
    from agents.multi_agent import CricketInferenceOrchestrator
    
    # Initialize
    orchestrator = CricketInferenceOrchestrator(
        data_path='data/ipl_qa.json',
        model_name='Qwen/Qwen2-0.5B-Instruct',
        adapter_repo=os.getenv('ADAPTER_REPO', 'riyakl09/ipl-cricket-sft'),
        adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
        hf_token=os.getenv('HF_TOKEN'),
    )
    
    # Example questions
    questions = [
        "Who was the leading run-scorer in IPL 2023?",
        "Which team won the IPL championship?",
        "What was the highest individual score?",
    ]
    
    # Process questions
    for question in questions:
        print(f"\n{'─'*80}")
        print(f"Q: {question}")
        print('─'*80)
        
        result = orchestrator.answer_question(question)
        
        print(f"\nA: {result['answer']}")
        print(f"\nMetadata:")
        print(f"  - Facts retrieved: {result['facts_count']}")
        print(f"  - Dataset size: {result['dataset_size']}")


# OPTION 2: FastAPI Service

def option2_fastapi():
    
    print("\n" + "="*80)
    print("OPTION 2: FastAPI Service with Multi-Agent")
    print("="*80 + "\n")
    
    import uvicorn
    from service.app_multi_agent import app
    
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    
    print(f"Starting FastAPI service on {host}:{port}")
    print(f"\nAvailable endpoints:")
    print(f"  POST http://{host}:{port}/infer")
    print(f"  GET  http://{host}:{port}/healthz")
    print(f"  GET  http://{host}:{port}/readyz")
    print(f"  GET  http://{host}:{port}/docs (Swagger UI)")
    print()
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )

# OPTION 3: Python API (For Integration)

def option3_api():
    print("\n" + "="*80)
    print("OPTION 3: Python API for Integration")
    print("="*80 + "\n")
    
    from agents.multi_agent import CricketInferenceOrchestrator
    
    def get_orchestrator():
        """Factory function to create orchestrator."""
        return CricketInferenceOrchestrator(
            data_path='data/ipl_qa.json',
            model_name='Qwen/Qwen2-0.5B-Instruct',
            adapter_repo=os.getenv('ADAPTER_REPO', 'riyakl09/ipl-cricket-sft'),
            adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
            hf_token=os.getenv('HF_TOKEN'),
        )
    
    # Example usage
    orchestrator = get_orchestrator()
    
    # In your application:
    question = "Who was the leading run-scorer?"
    result = orchestrator.answer_question(question)
    
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer']}")
    print(f"\nYou can integrate this into your application like:")
    print("""
    from quick_start import get_orchestrator
    
    orchestrator = get_orchestrator()  # Create once at startup
    result = orchestrator.answer_question(user_question)
    print(result['answer'])
    """)


# Docker Usage

def option4_docker():
   
    print("\n" + "="*80)
    print("OPTION 4: Docker Deployment")
    print("="*80 + "\n")
    
    print("""
# Build Docker image
docker build -t ipl-qa-service -f service/Dockerfile .

# Run container
docker run --env HF_TOKEN=$HF_TOKEN \\
           --env ADAPTER_REPO=riyakl09/ipl-cricket-sft \\
           --env ADAPTER_PATH=/app/models/ipl-cricket-sft \\
           -p 8000:8000 \\
           ipl-qa-service

# Test the service
curl -X POST http://localhost:8000/infer \\
  -H "Content-Type: application/json" \\
  -d '{"question": "Who was the leading run-scorer in IPL?"}'

# Check health
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz

# View API docs
open http://localhost:8000/docs
    """)


# Main


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-Agent Cricket Q&A System - Quick Start'
    )
    parser.add_argument(
        '--mode',
        choices=['standalone', 'fastapi', 'api', 'docker'],
        default='standalone',
        help='Mode to run (default: standalone)'
    )
    
    args = parser.parse_args()
    
    # Ensure environment is set up
    if not os.getenv('HF_TOKEN'):
        print("\n⚠️  WARNING: HF_TOKEN environment variable not set")
        print("   The system will try to load adapters from local path")
        print("   Set HF_TOKEN for HuggingFace Hub access")
        print()
    
    if args.mode == 'standalone':
        option1_standalone()
    elif args.mode == 'fastapi':
        option2_fastapi()
    elif args.mode == 'api':
        option3_api()
    elif args.mode == 'docker':
        option4_docker()
    else:
        print(f"Unknown mode: {args.mode}")
        print("Use --mode [standalone|fastapi|api|docker]")
