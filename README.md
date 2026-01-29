# IPL Cricket Q&A System

Production-ready system for answering IPL cricket statistics questions using a fine-tuned language model, FastAPI inference service, and Kubernetes deployment.

## Overview

**Base Model:** Qwen/Qwen2-0.5B-Instruct  
**Fine-tuning:** LoRA (Low-Rank Adaptation)  
**Dataset:** 805 Q&A pairs (684 train / 121 validation)  
**Deployment:** FastAPI with multi-agent architecture  
**Infrastructure:** Kubernetes with Helm charts

## Project Structure

```
.
├── data/ipl_qa.json
├── scripts/
│   ├── train_sft.py
│   └── test_finetuned_model.py
├── service/
│   ├── app.py
│   ├── app_multi_agent.py
│   └── Dockerfile
├── agents/multi_agent.py
├── deploy/helm/
├── tests/
│   ├── test_inference.py
│   └── test_multi_agent.py
├── requirements.txt
└── MODEL_MONITORING.md
```

## Quick Start

```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run service
export HF_TOKEN="hf_your_token_here"
python quick_start.py --mode fastapi
```

Service available at http://localhost:8000

## API Endpoints

**POST /infer** - Answer cricket questions  
Request: `{"question": "Who won IPL 2023?"}`  
Response: `{"answer": "...", "latency_ms": 1234}`

**GET /healthz** - Service health check  
**GET /readyz** - Model readiness probe  
**GET /docs** - Interactive API documentation

## Model Training

```bash
export HF_TOKEN="hf_your_token"
python scripts/train_sft.py \
  --model qwen2-0.5b \
  --epochs 5 \
  --batch-size 2 \
  --lora-rank 8 \
  --output-dir models/ipl-cricket-sft
```

**Training time:** ~30 minutes on CPU  
**Final validation loss:** 0.2528  
**Memory usage:** ~2GB

**LoRA Configuration:**
- Rank: 8
- Alpha: 16
- Target modules: q_proj, v_proj

## Multi-Agent Architecture

**RetrieverAgent:** Searches Q&A dataset for relevant facts  
**AnalystAgent:** Generates answers using fine-tuned model with retrieved context

This separation improves accuracy and reduces hallucinations.

## Deployment

### Docker

```bash
docker build -t ipl-qa-service -f service/Dockerfile .
docker run -e HF_TOKEN="hf_..." -p 8000:8000 ipl-qa-service
```

### Kubernetes

```bash
kubectl create secret generic hf-secret --from-literal=token="hf_..."
helm install cricket-inference deploy/helm/ --namespace cricket
kubectl port-forward -n cricket svc/cricket-inference 8000:8000
```

**Helm Configuration (values.yaml):**

```yaml
replicaCount: 2
image:
  repository: youracr.azurecr.io/cricket-inference
  tag: "1.0.0"
resources:
  requests:
    cpu: "1000m"
    memory: "2Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"
autoscaling:
  enabled: true
  minReplicas: 1
  maxReplicas: 3
```

## Testing

```bash
pytest tests/ -v
pytest tests/test_inference.py -v
pytest tests/ --cov=service --cov-report=html
```

## Monitoring

**Request Logging:**
```json
{
  "timestamp": "2025-01-29T10:30:45.123Z",
  "request_id": "uuid",
  "question": "Who won IPL 2023?",
  "latency_ms": 1234,
  "tokens_generated": 87,
  "status": "success"
}
```

**Metrics to Track:**
- P50/P95/P99 latency
- Error rate
- Model readiness state

See MODEL_MONITORING.md for detailed monitoring strategy.

## Design Decisions

**LoRA over Full Fine-Tuning**  
90% memory reduction with no quality loss. Enables CPU training.

**Multi-Agent Architecture**  
Separates deterministic retrieval from LLM reasoning for better accuracy.

**CPU-Only Inference**  
Cost-effective with acceptable latency (target under 5 seconds).

**Qwen2-0.5B Base Model**  
Optimal size and quality tradeoff. 500M parameters, instruction-tuned.

## Troubleshooting

**Model not loading:**
```bash
echo $HF_TOKEN  # Verify token is set
huggingface-cli repo-info your-org/private-ipl-adapters
```

**Out of memory:**
- Reduce batch_size from 4 to 2
- Enable gradient accumulation

**Slow inference:**
- Check CPU usage
- Verify model is cached in memory
- Deploy multiple replicas

## Accuracy Scaling Strategy

**Model Capacity:**  
Scale to Qwen2-1.5B or 7B with QLoRA. Higher LoRA rank if GPU memory allows.

**Data and Training:**  
Expand Q&A coverage across multiple IPL seasons. Clean and normalize answers. Tune hyperparameters (learning rate, rank/alpha, dropout, epochs).

**Retrieval and Tooling:**  
Add embedding-based retrieval. Improve RetrieverAgent to compute stats from raw tables. Enrich prompts with verified facts.

**Inference-Time Improvements:**  
Use longer context windows. Constrained decoding for numeric answers. Reranking with self-consistency or verifier.

**Evaluation Loop:**  
Continuous evaluation on fixed eval set plus new curated questions. Weekly human review. Feedback-driven retraining.

## Candidate Decisions

**Base Model:** SLM - Qwen/Qwen2-0.5B-Instruct (CPU-friendly, fast inference)  
**Fine-tuning Method:** LoRA (efficient, minimal memory, CPU-compatible)  
**Training Setup:** CPU (single machine)  
**HF Private Repo:** riyakl09/ipl-cricket-sft  
**Inference Approach:** Load model at startup, attach LoRA adapters, cache in memory  
**Multi-Agent Design:** RetrieverAgent (facts) → AnalystAgent (reasoning)  
**Observability:** Per-request latency logs, token counts, P50/P95/P99 tracking

## Deliverables

- scripts/train_sft.py
- service/app.py and Dockerfile
- agents/multi_agent.py
- deploy/helm/
- tests/test_inference.py
- MODEL_MONITORING.md
- README.md
