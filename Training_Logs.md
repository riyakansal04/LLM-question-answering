python scripts/train_sft.py --model qwen2-0.5b --epochs 5 --batch-size 2 --lora-rank 8 --output-dir models/ipl-cricket-sft-5epochs
2026-01-29 19:25:57,542 - INFO - 
======================================================================
2026-01-29 19:25:57,542 - INFO - IPL CRICKET Q&A - SFT TRAINING
2026-01-29 19:25:57,543 - INFO - ======================================================================
2026-01-29 19:25:57,543 - INFO - Selected model: qwen2-0.5b
2026-01-29 19:25:57,543 - INFO - Model ID: Qwen/Qwen2-0.5B-Instruct
2026-01-29 19:25:57,543 - INFO - Description: Fastest CPU training (~15-20 min), good quality
2026-01-29 19:25:57,543 - INFO - Dataset: data/ipl_qa.json
2026-01-29 19:25:57,543 - INFO - Output: models/ipl-cricket-sft-5epochs
2026-01-29 19:25:57,543 - INFO - ======================================================================   
2026-01-29 19:25:57,544 - INFO - Loading dataset from data/ipl_qa.json
2026-01-29 19:25:57,546 - INFO - Loaded 805 Q&A pairs
2026-01-29 19:25:57,546 - INFO -
Loading model: Qwen/Qwen2-0.5B-Instruct
2026-01-29 19:25:57,546 - INFO - Device: CPU
2026-01-29 19:25:58,261 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:25:58,303 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
2026-01-29 19:25:58,561 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:25:58,610 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/tokenizer_config.json "HTTP/1.1 200 OK"       
2026-01-29 19:25:58,863 - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-0.5B-Instruct/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-01-29 19:25:59,117 - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-0.5B-Instruct/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-01-29 19:26:00,098 - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-0.5B-Instruct "HTTP/1.1 200 OK"
2026-01-29 19:26:00,099 - INFO - Loading model in FP32 (CPU-optimized mode)
2026-01-29 19:26:00,349 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:26:00,389 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
`torch_dtype` is deprecated! Use `dtype` instead!
2026-01-29 19:26:00,655 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:26:00,694 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
Loading weights: 100%|█████████| 290/290 [00:00<00:00, 460.26it/s, Materializing param=model.norm.weight]
2026-01-29 19:26:02,004 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/generation_config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:26:02,046 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/generation_config.json "HTTP/1.1 200 OK"      
2026-01-29 19:26:02,302 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/custom_generate/generate.py "HTTP/1.1 404 Not Found"
2026-01-29 19:26:02,306 - INFO - ✓ Model loaded: 0.49B parameters

2026-01-29 19:26:02,306 - INFO - Preparing train/eval splits...
2026-01-29 19:26:02,363 - INFO - ✓ Formatted 805 examples
2026-01-29 19:26:02,406 - INFO - ✓ Train set: 684 examples
2026-01-29 19:26:02,407 - INFO - ✓ Test set: 121 examples
2026-01-29 19:26:02,407 - INFO -
============================================================
2026-01-29 19:26:02,407 - INFO - STARTING SFT TRAINING
2026-01-29 19:26:02,407 - INFO - ============================================================
2026-01-29 19:26:02,407 - INFO - Setting up LoRA configuration...
2026-01-29 19:26:02,408 - INFO - ✓ LoRA config: rank=8, alpha=16
2026-01-29 19:26:02,408 - INFO - ✓ Target modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj']
2026-01-29 19:26:02,408 - INFO -
Estimated training steps: 425
2026-01-29 19:26:02,408 - INFO - Training will take approximately 15-45 minutes on CPU

warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
`logging_dir` is deprecated and will be removed in v5.2. Please set `TENSORBOARD_LOGGING_DIR` instead.    
2026-01-29 19:26:02,414 - INFO - Initializing SFT Trainer...
Warning: Passing `TrainingArguments` to `SFTTrainer` is deprecated in Transformers v5+. Please
warmup_ratio is deprecated and will be removed in v5.2. Use `warmup_steps` instead.
`logging_dir` is deprecated and will be removed in v5.2. Please set `TENSORBOARD_LOGGING_DIR` instead.    
2026-01-29 19:26:02,677 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/processor_config.json "HTTP/1.1 404 Not Found"
2026-01-29 19:26:02,937 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/preprocessor_config.json "HTTP/1.1 404 Not Found"
2026-01-29 19:26:03,197 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/video_preprocessor_config.json "HTTP/1.1 404 Not Found"
2026-01-29 19:26:03,506 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/preprocessor_config.json "HTTP/1.1 404 Not Found"
2026-01-29 19:26:03,755 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:26:03,797 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/tokenizer_config.json "HTTP/1.1 200 OK"       
2026-01-29 19:26:04,056 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:26:04,095 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
2026-01-29 19:26:04,359 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:26:04,398 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
2026-01-29 19:26:04,652 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/tokenizer_config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:26:04,693 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/tokenizer_config.json "HTTP/1.1 200 OK"       
2026-01-29 19:26:04,959 - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-0.5B-Instruct/tree/main/additional_chat_templates?recursive=false&expand=false "HTTP/1.1 404 Not Found"
2026-01-29 19:26:05,228 - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-0.5B-Instruct/tree/main?recursive=true&expand=false "HTTP/1.1 200 OK"
2026-01-29 19:26:06,541 - INFO - HTTP Request: GET https://huggingface.co/api/models/Qwen/Qwen2-0.5B-Instruct "HTTP/1.1 200 OK"
Adding EOS to train dataset: 100%|███████████████████████████| 684/684 [00:00<00:00, 12436.56 examples/s]
Tokenizing train dataset: 100%|███████████████████████████████| 684/684 [00:00<00:00, 1867.38 examples/s]
Truncating train dataset: 100%|██████████████████████████████| 684/684 [00:00<00:00, 95138.58 examples/s] 
Adding EOS to eval dataset: 100%|█████████████████████████████| 121/121 [00:00<00:00, 6196.94 examples/s] 
Tokenizing eval dataset: 100%|████████████████████████████████| 121/121 [00:00<00:00, 2412.39 examples/s]
Truncating eval dataset: 100%|███████████████████████████████| 121/121 [00:00<00:00, 68140.55 examples/s] 
2026-01-29 19:26:11,661 - INFO - 
Trainable parameters:
trainable params: 1,081,344 || all params: 495,114,112 || trainable%: 0.2184
2026-01-29 19:26:11,664 - INFO -
============================================================
2026-01-29 19:26:11,664 - INFO - Training started...
2026-01-29 19:26:11,664 - INFO - ============================================================

The tokenizer has new PAD/BOS/EOS tokens that differ from the model config and generation config. The model config and generation config were aligned accordingly, being updated with rning_rate': '8.837e-05', 'entropy': '1.733', 'num_tokens': '1.389e+04', 'mean_token_accuracy': '0.577', 'epoch': '0.2339'}
  5%|███▏                                                             | 21/430 [05:58<1:54:0  5%|▎    | 22/430 [06:14<1:53:08, 16.64s/it]                                                 6%|▎    | 26/430 [07:15<1:44:25, 1  6%|▏  | 27/430 [07:31<1:46:12, 15.81s/it] {'loss': '1.315', 'grad_norm': '2.15', 'learning_rate': '0.0001349', 'entropy': '1.547', 'num_tokens': '2.094e+04', 'mean_token_accuracy': '0.6896', 'epoch': '0.3509'}
{'loss': '0.8866', 'grad_norm': '1.671', 'learning_rate': '0.0001814', 'entropy': '1.02', 'num_tokens': '2.789e+04', 'mean_token_accuracy': '0.807', 'epoch': '0.4678'}
{'loss': '0.611', 'grad_norm': '1.027', 'learning_rate': '0.0001969', 'entropy': '0.6724', 'num_tokens': '3.502e+04', 'mean_token_accuracy': '0.8476', 'epoch': '0.5848'}
{'loss': '0.5159', 'grad_norm': '1.545', 'learning_rate': '0.0001917', 'entropy': '0.5435', 'num_tokens': '4.201e+04', 'mean_token_accuracy': '0.8721', 'epoch': '0.7018'}
{'loss': '0.469', 'grad_norm': '1.249', 'learning_rate': '0.0001866', 'entropy': '0.4935', 'num_tokens': '4.91e+04', 'mean_token_accuracy': '0.8783', 'epoch': '0.8187'}
{'loss': '0.4116', 'grad_norm': '1.209', 'learning_rate': '0.0001814', 'entropy': '0.4544', 'num_tokens': '5.608e+04', 'mean_token_accuracy': '0.8932', 'epoch': '0.9357'}
{'eval_loss': '0.3904', 'eval_runtime': '105.9', 'eval_samples_per_second': '1.143', 'eval_steps_per_second': '0.576', 'eval_entropy': '0.4444', 'eval_num_tokens': '5.994e+04', 'eval_mean_token_accuracy': '0.8924', 'epoch': '1'}
 20%|██████████                                        | 86/430 [25:34<1:19:26, 13.86s/it]2026-01-29 19:51:47,768 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:51:48,046 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
2026-01-29 19:51:48,509 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 19:51:48,766 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
C:\Users\Hp\Documents\New folder\llm-uestion-answering--main\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
{'loss': '0.3749', 'grad_norm': '0.9655', 'learning_rate': '0.0001762', 'entropy': '0.419', 'num_tokens': '6.276e+04', 'mean_token_accuracy': '0.8969', 'epoch': '1.047'}
{'loss': '0.3473', 'grad_norm': '1.111', 'learning_rate': '0.0001711', 'entropy': '0.39', 'num_tokens': '6.964e+04', 'mean_token_accuracy': '0.905', 'epoch': '1.164'}
{'loss': '0.3545', 'grad_norm': '1.053', 'learning_rate': '0.0001659', 'entropy': '0.3905', 'num_tokens': '7.67e+04', 'mean_token_accuracy': '0.8984', 'epoch': '1.281'}
{'loss': '0.3291', 'grad_norm': '1.199', 'learning_rate': '0.0001607', 'entropy': '0.3823', 'num_tokens': '8.376e+04', 'mean_token_accuracy': '0.9054', 'epoch': '1.398'}
{'loss': '0.2833', 'grad_norm': '1.022', 'learning_rate': '0.0001556', 'entropy': '0.3279', 'num_tokens': '9.069e+04', 'mean_token_accuracy': '0.9131', 'epoch': '1.515'}
{'loss': '0.3074', 'grad_norm': '0.744', 'learning_rate': '0.0001504', 'entropy': '0.3486', 'num_tokens': '9.778e+04', 'mean_token_accuracy': '0.9087', 'epoch': '1.632'}
{'loss': '0.3062', 'grad_norm': '1.07', 'learning_rate': '0.0001452', 'entropy': '0.3393', 'num_tokens': '1.048e+05', 'mean_token_accuracy': '0.9102', 'epoch': '1.749'}
{'loss': '0.3153', 'grad_norm': '1.523', 'learning_rate': '0.0001401', 'entropy': '0.3374', 'num_tokens': '1.119e+05', 'mean_token_accuracy': '0.9089', 'epoch': '1.865'}
{'loss': '0.274', 'grad_norm': '1.494', 'learning_rate': '0.0001349', 'entropy': '0.3083', 'num_tokens': '1.188e+05', 'mean_token_accuracy': '0.9176', 'epoch': '1.982'}
{'eval_loss': '0.2899', 'eval_runtime': '99.47', 'eval_samples_per_second': '1.216', 'eval_steps_per_second': '0.613', 'eval_entropy': '0.317', 'eval_num_tokens': '1.199e+05', 'eval_mean_token_accuracy': '0.9111', 'epoch': '2'}
 40%|████████████████████▍                              | 172/430 [50:49<57:54, 13.47s/it]2026-01-29 20:17:02,762 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 20:17:03,020 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
2026-01-29 20:17:03,487 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 20:17:03,744 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
C:\Users\Hp\Documents\New folder\llm-uestion-answering--main\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
{'loss': '0.2429', 'grad_norm': '1.157', 'learning_rate': '0.0001297', 'entropy': '0.2821', 'num_tokens': '1.254e+05', 'mean_token_accuracy': '0.9259', 'epoch': '2.094'}
{'loss': '0.266', 'grad_norm': '1.036', 'learning_rate': '0.0001245', 'entropy': '0.3093', 'num_tokens': '1.324e+05', 'mean_token_accuracy': '0.9178', 'epoch': '2.211'}
{'loss': '0.2718', 'grad_norm': '1.37', 'learning_rate': '0.0001194', 'entropy': '0.3135', 'num_tokens': '1.394e+05', 'mean_token_accuracy': '0.9171', 'epoch': '2.327'}
{'loss': '0.2532', 'grad_norm': '1.23', 'learning_rate': '0.0001142', 'entropy': '0.2912', 'num_tokens': '1.465e+05', 'mean_token_accuracy': '0.917', 'epoch': '2.444'}
{'loss': '0.256', 'grad_norm': '0.8626', 'learning_rate': '0.000109', 'entropy': '0.2886', 'num_tokens': '1.535e+05', 'mean_token_accuracy': '0.922', 'epoch': '2.561'}
{'loss': '0.2432', 'grad_norm': '0.9095', 'learning_rate': '0.0001039', 'entropy': '0.2743', 'num_tokens': '1.605e+05', 'mean_token_accuracy': '0.9277', 'epoch': '2.678'}
{'loss': '0.2505', 'grad_norm': '0.8931', 'learning_rate': '9.871e-05', 'entropy': '0.2868', 'num_tokens': '1.676e+05', 'mean_token_accuracy': '0.9217', 'epoch': '2.795'}
{'loss': '0.2415', 'grad_norm': '1.48', 'learning_rate': '9.354e-05', 'entropy': '0.2667', 'num_tokens': '1.745e+05', 'mean_token_accuracy': '0.9246', 'epoch': '2.912'}
{'eval_loss': '0.2628', 'eval_runtime': '108.4', 'eval_samples_per_second': '1.117', 'eval_steps_per_second': '0.563', 'eval_entropy': '0.2951', 'eval_num_tokens': '1.798e+05', 'eval_mean_token_accuracy': '0.917', 'epoch': '3'}
 60%|█████████████████████████████▍                   | 258/430 [1:16:38<40:55, 14.28s/it]2026-01-29 20:42:51,639 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK"
2026-01-29 20:42:52,373 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 20:42:52,629 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK" 
C:\Users\Hp\Documents\New folder\llm-uestion-answering--main\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
{'loss': '0.235', 'grad_norm': '1.187', 'learning_rate': '8.837e-05', 'entropy': '0.2579', 'num_tokens': '1.812e+05', 'mean_token_accuracy': '0.9281', 'epoch': '3.023'}
{'loss': '0.2373', 'grad_norm': '1.256', 'learning_rate': '8.32e-05', 'entropy': '0.2865', 'num_tokens': '1.883e+05', 'mean_token_accuracy': '0.9239', 'epoch': '3.14'}
{'loss': '0.2369', 'grad_norm': '1.036', 'learning_rate': '7.804e-05', 'entropy': '0.2642', 'num_tokens': '1.952e+05', 'mean_token_accuracy': '0.9243', 'epoch': '3.257'}
{'loss': '0.2282', 'grad_norm': '0.7822', 'learning_rate': '7.287e-05', 'entropy': '0.2616', 'num_tokens': '2.023e+05', 'mean_token_accuracy': '0.9268', 'epoch': '3.374'}
{'loss': '0.2301', 'grad_norm': '0.9754', 'learning_rate': '6.77e-05', 'entropy': '0.2665', 'num_tokens': '2.093e+05', 'mean_token_accuracy': '0.9246', 'epoch': '3.491'}
{'loss': '0.229', 'grad_norm': '1.172', 'learning_rate': '6.253e-05', 'entropy': '0.2689', 'num_tokens': '2.163e+05', 'mean_token_accuracy': '0.9296', 'epoch': '3.608'}
{'loss': '0.2354', 'grad_norm': '0.9802', 'learning_rate': '5.736e-05', 'entropy': '0.2582', 'num_tokens': '2.233e+05', 'mean_token_accuracy': '0.924', 'epoch': '3.725'}
{'loss': '0.2332', 'grad_norm': '1.049', 'learning_rate': '5.22e-05', 'entropy': '0.2673', 'num_tokens': '2.304e+05', 'mean_token_accuracy': '0.9216', 'epoch': '3.842'}
{'loss': '0.2248', 'grad_norm': '1.025', 'learning_rate': '4.703e-05', 'entropy': '0.2571', 'num_tokens': '2.374e+05', 'mean_token_accuracy': '0.9261', 'epoch': '3.959'}
{'eval_loss': '0.2587', 'eval_runtime': '103.5', 'eval_samples_per_second': '1.169', 'eval_steps_per_second': '0.589', 'eval_entropy': '0.2661', 'eval_num_tokens': '2.398e+05', 'eval_mean_token_accuracy': '0.9194', 'epoch': '4'}
 80%|████████████████████████████████████████████▊           | 344/430 [1:41:32<20:28, 14.28s/it]2026-01-29 21:07:55,653 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 21:07:55,919 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK" 
2026-01-29 21:07:56,405 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 21:07:56,674 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK" 
C:\Users\Hp\Documents\New folder\llm-uestion-answering--main\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
{'loss': '0.2148', 'grad_norm': '1.072', 'learning_rate': '4.186e-05', 'entropy': '0.2486', 'num_tokens': '2.439e+05', 'mean_token_accuracy': '0.9304', 'epoch': '4.07'}
{'loss': '0.2137', 'grad_norm': '1.02', 'learning_rate': '3.669e-05', 'entropy': '0.2446', 'num_tokens': '2.509e+05', 'mean_token_accuracy': '0.9297', 'epoch': '4.187'}
{'loss': '0.2092', 'grad_norm': '1.273', 'learning_rate': '3.152e-05', 'entropy': '0.2378', 'num_tokens': '2.578e+05', 'mean_token_accuracy': '0.932', 'epoch': '4.304'}
{'loss': '0.2138', 'grad_norm': '1.145', 'learning_rate': '2.636e-05', 'entropy': '0.2561', 'num_tokens': '2.648e+05', 'mean_token_accuracy': '0.931', 'epoch': '4.421'}
{'loss': '0.2348', 'grad_norm': '0.8827', 'learning_rate': '2.119e-05', 'entropy': '0.2788', 'num_tokens': '2.72e+05', 'mean_token_accuracy': '0.9256', 'epoch': '4.538'}
{'loss': '0.2073', 'grad_norm': '0.8622', 'learning_rate': '1.602e-05', 'entropy': '0.251', 'num_tokens': '2.789e+05', 'mean_token_accuracy': '0.9309', 'epoch': '4.655'}
{'loss': '0.222', 'grad_norm': '1.009', 'learning_rate': '1.085e-05', 'entropy': '0.2443', 'num_tokens': '2.859e+05', 'mean_token_accuracy': '0.9288', 'epoch': '4.772'}
{'loss': '0.2301', 'grad_norm': '1.397', 'learning_rate': '5.685e-06', 'entropy': '0.2626', 'num_tokens': '2.93e+05', 'mean_token_accuracy': '0.9238', 'epoch': '4.889'}
{'loss': '0.2058', 'grad_norm': '1.256', 'learning_rate': '5.168e-07', 'entropy': '0.2428', 'num_tokens': '2.997e+05', 'mean_token_accuracy': '0.9337', 'epoch': '5'}
{'eval_loss': '0.2528', 'eval_runtime': '93.1', 'eval_samples_per_second': '1.3', 'eval_steps_per_second': '0.655', 'eval_entropy': '0.2645', 'eval_num_tokens': '2.997e+05', 'eval_mean_token_accuracy': '0.9199', 'epoch': '5'}
100%|████████████████████████████████████████████████████████| 430/430 [2:04:23<00:00, 12.64s/it]2026-01-29 21:30:36,034 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 21:30:36,078 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK" 
2026-01-29 21:30:36,336 - INFO - HTTP Request: HEAD https://huggingface.co/Qwen/Qwen2-0.5B-Instruct/resolve/main/config.json "HTTP/1.1 307 Temporary Redirect"
2026-01-29 21:30:36,379 - INFO - HTTP Request: HEAD https://huggingface.co/api/resolve-cache/models/Qwen/Qwen2-0.5B-Instruct/c540970f9e29518b1d8f06ab8b24cba66ad77b6d/config.json "HTTP/1.1 200 OK" 
{'train_runtime': '7465', 'train_samples_per_second': '0.458', 'train_steps_per_second': '0.058', 'train_loss': '0.4132', 'epoch': '5'}
100%|████████████████████████████████████████████████████████| 430/430 [2:04:24<00:00, 17.36s/it] 
2026-01-29 21:30:36,582 - INFO -
============================================================
2026-01-29 21:30:36,582 - INFO - TRAINING COMPLETED!
2026-01-29 21:30:36,582 - INFO - ============================================================     
2026-01-29 21:30:36,582 - INFO - Training time: 124.4 minutes
2026-01-29 21:30:36,583 - INFO - Final train loss: 0.4132
2026-01-29 21:30:36,583 - INFO -
Running final evaluation...
C:\Users\Hp\Documents\New folder\llm-uestion-answering--main\venv\Lib\site-packages\torch\utils\data\dataloader.py:775: UserWarning: 'pin_memory' argument is set as true but no accelerator is found, then device pinned memory won't be used.
  super().__init__(loader)
100%|████████████████████████████████████████████████████████████| 61/61 [01:24<00:00,  1.38s/it]
2026-01-29 21:32:02,094 - INFO - Evaluation loss: 0.2528
Saving LoRA adapters to models/ipl-cricket-sft-5epochs...
Saving LoRA adapters to models/ipl-cricket-sft-5epochs...
Saving LoRA adapters to models/ipl-cricket-sft-5epochs...
Writing model shards: 100%|████████████████████████████████████████| 1/1 [00:04<00:00,  4.62s/it]
2026-01-29 21:32:07,028 - INFO - ✓ Adapters and tokenizer saved
2026-01-29 21:32:07,028 - INFO -
======================================================================
2026-01-29 21:32:07,028 - INFO - ✓ SFT FINE-TUNING COMPLETED SUCCESSFULLY!
2026-01-29 21:32:07,029 - INFO - ======================================================================
2026-01-29 21:32:07,029 - INFO -
To use your fine-tuned model:
2026-01-29 21:32:07,029 - INFO - 1. Load base model: Qwen/Qwen2-0.5B-Instruct
2026-01-29 21:32:07,029 - INFO - 2. Load LoRA adapters from: models/ipl-cricket-sft-5epochs
2026-01-29 21:32:07,029 - INFO - 3. Run: python scripts/test_finetuned_model.py --base-model Qwen/Qwen2-0.5B-Instruct --adapter-path models/ipl-cricket-sft-5epochs       
2026-01-29 21:32:07,030 - INFO - ======================================================================



