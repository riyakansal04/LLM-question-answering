#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model_and_adapters(base_model: str, adapter_path: str, hf_token: str = None):
    logger.info(f"Loading base model: {base_model}")
    logger.info(f"Loading adapters from: {adapter_path}")
    
    # Check if adapter path exists
    if not os.path.exists(adapter_path):
        logger.error(f"Adapter path not found: {adapter_path}")
        sys.exit(1)
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
        token=hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
        device_map=device,
        trust_remote_code=True,
        token=hf_token,
        low_cpu_mem_usage=True,
    )
    
    # Load LoRA adapters
    logger.info(f"Loading LoRA adapters from {adapter_path}...")
    model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
    )
    
    model.eval()
    
    # Log model info
    if hasattr(model, 'print_trainable_parameters'):
        logger.info("Trainable parameters:")
        model.print_trainable_parameters()
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"✓ Model loaded ({param_count:.2f}B parameters)\n")
    
    return model, tokenizer, device


def generate_answer(model, tokenizer, device, question: str, max_length: int = 256) -> tuple:
    # Format prompt
    prompt = f"""You are an expert on IPL cricket. Answer the following question:

Question: {question}

Answer:"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        truncation=True,
        max_length=1024,
    ).to(device)
    
    # Generate
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = (time.time() - start_time) * 1000
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = generated_text.split("Answer:")[-1].strip()
    
    return answer, latency, outputs.shape[1] - inputs['input_ids'].shape[1]


def test_sample_questions(model, tokenizer, device):
    sample_questions = [
        "What is IPL?",
        "Who won the first IPL?",
        "Which team has won the most IPL titles?",
        "Who is the leading run scorer in IPL history?",
        "How many teams participate in IPL?",
    ]
    
    logger.info("Testing with sample questions:\n")
    logger.info("=" * 80)
    
    for i, question in enumerate(sample_questions, 1):
        logger.info(f"\nQuestion {i}: {question}")
        logger.info("-" * 80)
        
        try:
            answer, latency, tokens = generate_answer(model, tokenizer, device, question)
            logger.info(f"Answer: {answer[:200]}...")
            logger.info(f"Latency: {latency:.2f}ms | Tokens: {tokens}")
        except Exception as e:
            logger.error(f"Error: {e}")
    
    logger.info("\n" + "=" * 80)


def interactive_mode(model, tokenizer, device):
    logger.info("\n" + "=" * 80)
    logger.info("Interactive IPL Cricket Q&A Mode")
    logger.info("Type 'quit' or 'exit' to stop")
    logger.info("=" * 80 + "\n")
    
    while True:
        try:
            question = input("Question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                logger.info("Exiting...")
                break
            
            if not question:
                continue
            
            logger.info("Generating answer...")
            answer, latency, tokens = generate_answer(model, tokenizer, device, question)
            
            print(f"\nAnswer: {answer}")
            print(f"Latency: {latency:.2f}ms | Tokens: {tokens}\n")
        
        except KeyboardInterrupt:
            logger.info("\nExiting...")
            break
        except Exception as e:
            logger.error(f"Error: {e}")




def main():
    parser = argparse.ArgumentParser(
        description='Test fine-tuned IPL Cricket QA model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/test_finetuned_model.py --base-model Qwen/Qwen2-0.5B-Instruct --adapter-path models/ipl-cricket-sft
  python scripts/test_finetuned_model.py --base-model Qwen/Qwen2-0.5B-Instruct --adapter-path models/ipl-cricket-sft --interactive
  python scripts/test_finetuned_model.py --base-model Qwen/Qwen2-0.5B-Instruct --adapter-path models/ipl-cricket-sft --sample
        """
    )
    
    parser.add_argument(
        '--base-model',
        type=str,
        default='Qwen/Qwen2-0.5B-Instruct',
        help='Base model ID (default: Qwen/Qwen2-0.5B-Instruct)'
    )
    
    parser.add_argument(
        '--adapter-path',
        type=str,
        default='models/ipl-cricket-sft',
        help='Path to LoRA adapters (default: models/ipl-cricket-sft)'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='Test with sample questions'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Start interactive Q&A session'
    )
    
    parser.add_argument(
        '--question',
        type=str,
        help='Single question to answer'
    )
    
    parser.add_argument(
        '--max-length',
        type=int,
        default=256,
        help='Maximum answer length (default: 256)'
    )
    
    args = parser.parse_args()
    
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    
    # Load model and adapters
    logger.info("=" * 80)
    logger.info("IPL Cricket QA - Fine-Tuned Model Test")
    logger.info("=" * 80 + "\n")
    
    try:
        model, tokenizer, device = load_model_and_adapters(
            args.base_model,
            args.adapter_path,
            hf_token
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Run tests based on arguments
    if args.sample:
        test_sample_questions(model, tokenizer, device)
    elif args.interactive:
        interactive_mode(model, tokenizer, device)
    elif args.question:
        logger.info(f"Question: {args.question}")
        logger.info("-" * 80)
        try:
            answer, latency, tokens = generate_answer(
                model, tokenizer, device, args.question, args.max_length
            )
            logger.info(f"Answer: {answer}")
            logger.info(f"Latency: {latency:.2f}ms | Tokens: {tokens}")
        except Exception as e:
            logger.error(f"Error: {e}")
    else:
        # Default: test with sample questions
        logger.info("No mode specified. Running sample tests...")
        test_sample_questions(model, tokenizer, device)
    
    logger.info("\n✓ Test completed")


if __name__ == '__main__':
    main()
