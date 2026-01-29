#!/usr/bin/env python
import argparse
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model recommendations for different use cases
RECOMMENDED_MODELS = {
    'qwen2-0.5b': {
        'model_id': 'Qwen/Qwen2-0.5B-Instruct',
        'description': 'Fastest CPU training (~15-20 min), good quality',
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    },
    'tinyllama': {
        'model_id': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
        'description': 'Fast CPU training (~20-30 min), good quality',
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    },
    'qwen2-1.5b': {
        'model_id': 'Qwen/Qwen2-1.5B-Instruct',
        'description': 'Medium CPU training (~30-40 min), very good quality',
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    },
    'phi-2': {
        'model_id': 'microsoft/phi-2',
        'description': 'Slower CPU training (~45-60 min), excellent quality',
        'target_modules': ['q_proj', 'v_proj', 'dense'],
    },
}


def load_ipl_dataset(dataset_path: str) -> List[Dict[str, str]]:
    
    logger.info(f"Loading dataset from {dataset_path}")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} Q&A pairs")
    return data


def format_instruction_examples(qa_data: List[Dict[str, str]], tokenizer) -> List[Dict[str, str]]:
    
    formatted = []
    
    for item in qa_data:
        messages = [
            {"role": "system", "content": "You are an expert on IPL (Indian Premier League) cricket. Answer questions accurately and concisely based on the 2022 season data."},
            {"role": "user", "content": item['question']},
            {"role": "assistant", "content": item['answer']}
        ]
        
        # Try using chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            try:
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                text = f"### System:\nYou are an expert on IPL cricket.\n\n### User:\n{item['question']}\n\n### Assistant:\n{item['answer']}"
        else:
            text = f"### System:\nYou are an expert on IPL cricket.\n\n### User:\n{item['question']}\n\n### Assistant:\n{item['answer']}"
        
        formatted.append({'text': text})
    
    logger.info(f"✓ Formatted {len(formatted)} examples")
    return formatted


def prepare_dataset(qa_data: List[Dict[str, str]], tokenizer, test_size: float = 0.15) -> DatasetDict:
    
    logger.info("Preparing train/eval splits...")
    
    # Format examples with chat template
    formatted_data = format_instruction_examples(qa_data, tokenizer)
    
    # Create dataset
    dataset = Dataset.from_dict({'text': [item['text'] for item in formatted_data]})
    
    # Split train/test
    split_dataset = dataset.train_test_split(test_size=test_size, seed=42, shuffle=True)
    
    logger.info(f"✓ Train set: {len(split_dataset['train'])} examples")
    logger.info(f"✓ Test set: {len(split_dataset['test'])} examples")
    
    return DatasetDict({
        'train': split_dataset['train'],
        'eval': split_dataset['test']
    })


def setup_model_and_tokenizer(
    model_name: str,
    use_qlora: bool = False
) -> tuple:
    
    logger.info(f"\nLoading model: {model_name}")
    
    # Check device availability
    has_cuda = torch.cuda.is_available()
    device_type = 'GPU' if has_cuda else 'CPU'
    logger.info(f"Device: {device_type}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side='right'
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    if use_qlora and has_cuda:
        logger.info("Configuring 4-bit QLoRA quantization (GPU mode)")
        
        # 4-bit quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            trust_remote_code=True,
        )
    else:
        if use_qlora and not has_cuda:
            logger.warning("QLoRA requires GPU. Falling back to CPU mode with FP32")
        logger.info("Loading model in FP32 (CPU-optimized mode)")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map='cpu',
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
    
    # Disable cache for training
    model.config.use_cache = False
    if hasattr(model.config, 'pretraining_tp'):
        model.config.pretraining_tp = 1
    
    param_count = sum(p.numel() for p in model.parameters()) / 1e9
    logger.info(f"✓ Model loaded: {param_count:.2f}B parameters\n")
    
    return model, tokenizer


def setup_lora_config(model_name: str, rank: int = 8) -> LoraConfig:
   
    logger.info("Setting up LoRA configuration...")
    
    # Determine target modules based on model
    model_lower = model_name.lower()
    if 'qwen' in model_lower:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    elif 'phi' in model_lower:
        target_modules = ['q_proj', 'v_proj', 'dense']
    elif 'llama' in model_lower or 'tinyllama' in model_lower:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    else:
        target_modules = ['q_proj', 'v_proj']  # Default
    
    lora_config = LoraConfig(
        r=rank,  # Rank of adaptation matrices (8 for CPU efficiency)
        lora_alpha=16,  # LoRA scaling factor (2x rank)
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM',
        target_modules=target_modules,
    )
    
    logger.info(f"✓ LoRA config: rank={rank}, alpha={lora_config.lora_alpha}")
    logger.info(f"✓ Target modules: {target_modules}")
    
    return lora_config


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    num_train_epochs: int = 3,
    per_device_batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_rank: int = 8,
) -> tuple:
   
    import time
    
    logger.info("\n" + "=" * 60)
    logger.info("STARTING SFT TRAINING")
    logger.info("=" * 60)
    
    # Configure LoRA (trainer will wrap the base model)
    model_name = model.config._name_or_path if hasattr(model.config, '_name_or_path') else 'unknown'
    lora_config = setup_lora_config(model_name, rank=lora_rank)
    
    # Estimate training time
    total_steps = (len(train_dataset) // per_device_batch_size // 4) * num_train_epochs  # 4 = gradient_accumulation_steps
    logger.info(f"\nEstimated training steps: {total_steps}")
    logger.info(f"Training will take approximately 15-45 minutes on CPU\n")
    
    # Check device
    has_cuda = torch.cuda.is_available()
    
    # Training arguments optimized for CPU
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=4,  # Simulate larger batch
        warmup_ratio=0.1,
        weight_decay=0.01,
        learning_rate=learning_rate,
        logging_dir=f'{output_dir}/logs',
        logging_steps=10,
        eval_strategy='epoch',
        save_strategy='epoch',
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
        bf16=False,  # Disable for CPU compatibility
        fp16=False,  # Disable for CPU
        max_grad_norm=0.3,
        dataloader_num_workers=0,  # Single worker for CPU
        seed=42,
        report_to='none',  # Disable wandb/tensorboard
         # Fix for TRL bug - library expects this parameter
    )
    
    # SFT Trainer
    logger.info("Initializing SFT Trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
    )

    if hasattr(trainer.model, 'print_trainable_parameters'):
        logger.info("\nTrainable parameters:")
        trainer.model.print_trainable_parameters()
    
    logger.info("\n" + "=" * 60)
    logger.info("Training started...")
    logger.info("=" * 60 + "\n")
    
    start_time = time.time()
    train_result = trainer.train()
    training_time = time.time() - start_time
    
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETED!")
    logger.info("=" * 60)
    logger.info(f"Training time: {training_time / 60:.1f} minutes")
    logger.info(f"Final train loss: {train_result.training_loss:.4f}")
    
    # Run final evaluation
    logger.info("\nRunning final evaluation...")
    eval_results = trainer.evaluate()
    logger.info(f"Evaluation loss: {eval_results['eval_loss']:.4f}")
    
    # Save training metrics
    metrics = {
        'training_time_minutes': training_time / 60,
        'train_loss': train_result.training_loss,
        'eval_loss': eval_results['eval_loss'],
        'num_epochs': num_train_epochs,
        'train_samples': len(train_dataset),
        'eval_samples': len(eval_dataset),
    }
    
    import json
    with open(f'{output_dir}/training_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info(f"\n✓ Training metrics saved to {output_dir}/training_metrics.json")
    
    return model, trainer


def save_adapters(model, output_dir: str) -> None:
    
    logger.info(f"Saving adapters to {output_dir}")
    
    adapter_save_path = os.path.join(output_dir, 'lora_adapters')
    os.makedirs(adapter_save_path, exist_ok=True)
    
    model.save_pretrained(adapter_save_path)
    logger.info(f"Adapters saved to {adapter_save_path}")


def push_to_hub(
    output_dir: str,
    repo_name: str,
    hf_token: str = None,
) -> None:
    
    if not hf_token:
        hf_token = os.getenv('HF_TOKEN')
    
    if not hf_token:
        logger.warning("HF_TOKEN not found. Skipping push to hub.")
        return
    
    logger.info(f"Pushing adapters to {repo_name}")
    
    try:
        from huggingface_hub import HfApi
        
        api = HfApi(token=hf_token)
        adapter_path = os.path.join(output_dir, 'lora_adapters')
        
        api.upload_folder(
            folder_path=adapter_path,
            repo_id=repo_name,
            repo_type='model',
            commit_message='Upload fine-tuned IPL adapters',
        )
        
        logger.info(f"Successfully pushed to {repo_name}")
    except Exception as e:
        logger.error(f"Failed to push to hub: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune SLMs for IPL Cricket Q&A using SFT (Supervised Fine-Tuning)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Recommended models for CPU training:
  qwen2-0.5b    - Fastest (~15-20 min), good quality
  tinyllama     - Fast (~20-30 min), good quality  
  qwen2-1.5b    - Medium (~30-40 min), very good quality
  phi-2         - Slower (~45-60 min), excellent quality

Example usage:
  python scripts/train_sft.py --model qwen2-0.5b --epochs 3
  python scripts/train_sft.py --custom-model Qwen/Qwen2-0.5B-Instruct
        """
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen2-0.5b',
        choices=list(RECOMMENDED_MODELS.keys()),
        help='Recommended model to use (default: qwen2-0.5b)'
    )
    parser.add_argument(
        '--custom-model',
        type=str,
        default=None,
        help='Custom model ID from HuggingFace (overrides --model)'
    )
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List recommended models and exit'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='data/ipl_qa.json',
        help='Path to IPL Q&A dataset (default: data/ipl_qa.json)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/ipl-cricket-sft',
        help='Output directory for trained adapters (default: models/ipl-cricket-sft)'
    )
    parser.add_argument(
        '--use-qlora',
        action='store_true',
        help='Use 4-bit QLoRA quantization (requires GPU, ignored on CPU)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=3,
        help='Number of training epochs (default: 3)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=2,
        help='Batch size per device (default: 2 for CPU)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=2e-4,
        help='Learning rate (default: 2e-4)'
    )
    parser.add_argument(
        '--lora-rank',
        type=int,
        default=8,
        help='LoRA rank (default: 8 for CPU efficiency)'
    )
    parser.add_argument(
        '--push-to-hub',
        action='store_true',
        help='Push adapters to HuggingFace hub'
    )
    parser.add_argument(
        '--hf-repo',
        type=str,
        default=None,
        help='HuggingFace repo name (e.g., username/ipl-cricket-adapters)'
    )
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("\n" + "=" * 70)
        print("RECOMMENDED MODELS FOR CPU-OPTIMIZED SFT TRAINING")
        print("=" * 70)
        for key, info in RECOMMENDED_MODELS.items():
            print(f"\n🔹 {key}")
            print(f"   Model: {info['model_id']}")
            print(f"   Description: {info['description']}")
        print("\n" + "=" * 70)
        print("Usage: python scripts/train_sft.py --model qwen2-0.5b --epochs 3")
        print("=" * 70 + "\n")
        return
    
    # Determine model ID
    if args.custom_model:
        model_id = args.custom_model
        logger.info(f"Using custom model: {model_id}")
    else:
        model_info = RECOMMENDED_MODELS[args.model]
        model_id = model_info['model_id']
        logger.info(f"\n{'=' * 70}")
        logger.info("IPL CRICKET Q&A - SFT TRAINING")
        logger.info("=" * 70)
        logger.info(f"Selected model: {args.model}")
        logger.info(f"Model ID: {model_id}")
        logger.info(f"Description: {model_info['description']}")
        logger.info(f"Dataset: {args.dataset}")
        logger.info(f"Output: {args.output_dir}")
        logger.info("=" * 70)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    qa_data = load_ipl_dataset(args.dataset)
    
    # Setup model and tokenizer FIRST (needed for prepare_dataset)
    model, tokenizer = setup_model_and_tokenizer(
        model_id,
        use_qlora=args.use_qlora
    )
    
    # Prepare datasets (needs tokenizer)
    dataset_dict = prepare_dataset(qa_data, tokenizer)
    
    # Train model
    trained_model, trainer = train_model(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict['eval'],
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
    )
    
    # Save adapters
    logger.info(f"\nSaving LoRA adapters to {args.output_dir}...")
    trained_model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"✓ Adapters and tokenizer saved")
    
    # Push to hub if requested
    if args.push_to_hub and args.hf_repo:
        push_to_hub(args.output_dir, args.hf_repo)
    
    logger.info("\n" + "=" * 70)
    logger.info("✓ SFT FINE-TUNING COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info(f"\nTo use your fine-tuned model:")
    logger.info(f"1. Load base model: {model_id}")
    logger.info(f"2. Load LoRA adapters from: {args.output_dir}")
    logger.info(f"3. Run: python scripts/test_finetuned_model.py --base-model {model_id} --adapter-path {args.output_dir}")
    logger.info("=" * 70 + "\n")


if __name__ == '__main__':
    main()
