#!/usr/bin/env python
import os
import sys
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def check_adapter_files(adapter_path: str) -> bool:
    """Check if all required adapter files exist."""
    logger.info("\n" + "=" * 80)
    logger.info("1. CHECKING ADAPTER FILES")
    logger.info("=" * 80)
    
    required_files = [
        'adapter_config.json',
        'adapter_model.safetensors',
        'tokenizer.json',
        'tokenizer_config.json',
    ]
    
    all_exist = True
    for file in required_files:
        path = os.path.join(adapter_path, file)
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        size = f"({os.path.getsize(path) / 1024:.1f}KB)" if exists else "(missing)"
        logger.info(f"  {status} {file:30s} {size}")
        all_exist = all_exist and exists
    
    if all_exist:
        logger.info("\n✓ All required adapter files present!")
        return True
    else:
        logger.error("\n✗ Some adapter files missing!")
        return False


def check_adapter_config(adapter_path: str) -> dict:
 
    logger.info("\n" + "=" * 80)
    logger.info("2. CHECKING ADAPTER CONFIGURATION")
    logger.info("=" * 80)
    
    config_path = os.path.join(adapter_path, 'adapter_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        logger.info(f"  r (LoRA rank):        {config.get('r', 'N/A')}")
        logger.info(f"  lora_alpha:           {config.get('lora_alpha', 'N/A')}")
        logger.info(f"  lora_dropout:         {config.get('lora_dropout', 'N/A')}")
        logger.info(f"  task_type:            {config.get('task_type', 'N/A')}")
        logger.info(f"  target_modules:       {config.get('target_modules', 'N/A')}")
        logger.info(f"  bias:                 {config.get('bias', 'N/A')}")
        
        logger.info("\n✓ Adapter configuration loaded successfully!")
        return config
    
    except Exception as e:
        logger.error(f"\n✗ Error reading adapter config: {e}")
        return {}


def load_and_check_model(base_model: str, adapter_path: str) -> tuple:

    logger.info("\n" + "=" * 80)
    logger.info("3. LOADING MODEL AND ADAPTERS")
    logger.info("=" * 80)
    
    # Load tokenizer
    logger.info("  Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model,
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    logger.info("  ✓ Tokenizer loaded")
    
    # Load base model
    logger.info("  Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float32,
        device_map='cpu',
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    logger.info("  ✓ Base model loaded")
    
    # Load adapters
    logger.info("  Loading LoRA adapters...")
    try:
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=False,
        )
        logger.info("  ✓ LoRA adapters loaded successfully!")
    except Exception as e:
        logger.error(f"  ✗ Failed to load adapters: {e}")
        return None, None
    
    model.eval()
    return model, tokenizer


def check_trainable_params(model) -> bool:
    logger.info("\n" + "=" * 80)
    logger.info("4. CHECKING TRAINABLE PARAMETERS")
    logger.info("=" * 80)
    
    if hasattr(model, 'print_trainable_parameters'):
        logger.info("  ")
        model.print_trainable_parameters()
        logger.info("  ")
        logger.info("✓ Trainable parameters show LoRA was applied!")
        return True
    else:
        logger.warning("  Could not call print_trainable_parameters()")
        return False


def check_model_type(model) -> bool:
    logger.info("\n" + "=" * 80)
    logger.info("5. CHECKING MODEL TYPE")
    logger.info("=" * 80)
    
    model_type = type(model).__name__
    logger.info(f"  Model type: {model_type}")
    
    if 'Peft' in model_type or 'peft' in str(type(model)):
        logger.info("  ✓ Model is PEFT model with adapters!")
        return True
    else:
        logger.warning("  ⚠ Model might not be PEFT model")
        return False


def check_adapter_weights(model) -> bool:
    logger.info("\n" + "=" * 80)
    logger.info("6. CHECKING ADAPTER WEIGHTS")
    logger.info("=" * 80)
    
    adapter_weights_found = False
    
    for name, param in model.named_parameters():
        if 'lora' in name.lower():
            adapter_weights_found = True
            logger.info(f"  ✓ Found LoRA weight: {name}")
            logger.info(f"    Shape: {param.shape}, Requires grad: {param.requires_grad}")
            break
    
    if adapter_weights_found:
        logger.info("\n✓ LoRA adapter weights are present in model!")
        return True
    else:
        logger.warning("\n⚠ No LoRA weights found in model!")
        return False


def test_inference(model, tokenizer, device='cpu') -> bool:
    logger.info("\n" + "=" * 80)
    logger.info("7. TESTING INFERENCE")
    logger.info("=" * 80)
    
    test_question = "What is IPL?"
    
    logger.info(f"  Test question: {test_question}")
    logger.info(f"  Generating answer...")
    
    try:
        prompt = f"""You are an expert on IPL cricket. Answer the following question:

Question: {test_question}

Answer:"""
        
        inputs = tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        ).to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("Answer:")[-1].strip()
        
        logger.info(f"  Answer: {answer[:100]}...")
        logger.info("\n✓ Inference test successful!")
        return True
    
    except Exception as e:
        logger.error(f"\n✗ Inference test failed: {e}")
        return False


def generate_report(checks: dict) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("FINAL DIAGNOSIS REPORT")
    logger.info("=" * 80)
    
    all_passed = all(checks.values())
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status:8s} - {check_name}")
    
    logger.info("\n" + "-" * 80)
    
    if all_passed:
        logger.info("✓✓✓ ALL CHECKS PASSED! ✓✓✓")
        logger.info("\nYour LoRA adapters are loaded perfectly!")
        logger.info("Model is ready for inference.")
    else:
        failed = [name for name, passed in checks.items() if not passed]
        logger.error(f"⚠ {len(failed)} check(s) failed:")
        for name in failed:
            logger.error(f"  - {name}")
        logger.error("\nPlease check the issues above.")


def main():
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "  IPL CRICKET QA - ADAPTER VERIFICATION DIAGNOSTIC".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "=" * 78 + "╝")
    
    base_model = "Qwen/Qwen2-0.5B-Instruct"
    adapter_path = "models/ipl-cricket-sft"
    
    logger.info(f"\n📋 Configuration:")
    logger.info(f"  Base Model: {base_model}")
    logger.info(f"  Adapter Path: {adapter_path}")
    logger.info(f"  Device: CPU")
    
    checks = {}
    
    # Run checks
    checks['Adapter Files Present'] = check_adapter_files(adapter_path)
    config = check_adapter_config(adapter_path)
    
    model, tokenizer = load_and_check_model(base_model, adapter_path)
    if model is None:
        logger.error("\n✗ Failed to load model. Cannot continue.")
        sys.exit(1)
    
    checks['Model Type is PEFT'] = check_model_type(model)
    checks['Trainable Parameters'] = check_trainable_params(model)
    checks['Adapter Weights Present'] = check_adapter_weights(model)
    checks['Inference Test'] = test_inference(model, tokenizer)
    
    # Generate report
    generate_report(checks)
    
    logger.info("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
