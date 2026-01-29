#!/usr/bin/env python

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_model(model_id: str):
    """Download model and tokenizer to cache."""
    logger.info(f"Downloading model: {model_id}")
    logger.info("This may take a few minutes...")
    
    # Download tokenizer
    logger.info("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
        resume_download=True,
    )
    logger.info("✓ Tokenizer downloaded")
    
    # Download model
    logger.info("Downloading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        resume_download=True,
        low_cpu_mem_usage=True,
    )
    logger.info("✓ Model downloaded")
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"✓ Model size: {param_count:.2f}B parameters")
    logger.info(f"✓ Model cached and ready for training!")
    
    return model, tokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download model for training')
    parser.add_argument(
        '--model',
        type=str,
        default='Qwen/Qwen2-0.5B-Instruct',
        help='Model ID to download'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MODEL DOWNLOAD")
    print("=" * 60)
    print(f"Model: {args.model}")
    print("=" * 60 + "\n")
    
    download_model(args.model)
    
    print("\n" + "=" * 60)
    print("✓ DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nYou can now run training:")
    print("python scripts/train_sft.py --model qwen2-0.5b")
    print("=" * 60 + "\n")
