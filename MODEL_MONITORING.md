Model monitoring · MD
Copy

# Model Monitoring and Production Observability

Monitoring strategy for the Cricket Inference System in production, including accuracy tracking, quality assurance, and drift detection.

## Monitoring Architecture

The monitoring system has three key components:

1. **In-Service Metrics**: Collected at inference time in FastAPI service
2. **Offline Evaluation**: Periodic batch evaluation against held-out eval set
3. **User Feedback Loop**: Structured collection of user feedback
4. **Drift Detection**: Statistical monitoring for distribution shifts

## Accuracy Metrics

### Primary Metrics

**Answer Relevance Score (0-1)**  
Measures whether the generated answer directly addresses the user's question.  
Target: Greater than 0.85  
Measurement: Semantic similarity using BERT embeddings and cosine similarity

**Factual Accuracy (0-1)**  
Percentage of claims in the answer that are factually correct.  
Target: Greater than 0.90  
Measurement: Human evaluation against cricket statistics database

**Answer Completeness (0-1)**  
Does the answer address all aspects of the question?  
Target: Greater than 0.80  
Scoring: Full answer (1.0), Partial answer (0.5), No answer (0.0)

**Hallucination Rate (0-1)**  
Percentage of false or made-up facts in the answer.  
Target: Less than 0.05  
Method: Fact-checking against IPL database, named entity matching, temporal consistency checks

**Latency (milliseconds)**  
Time to generate answer from question.  
Target: Less than 5000ms for CPU inference  
Tracking: P50, P95, P99 percentiles

**Token Efficiency**  
Average tokens generated per question.  
Target: 50-300 tokens per answer

## Evaluation Sets

### Eval Set (Offline Evaluation)

**Purpose:** Periodic evaluation of model quality using held-out split from training

**Composition:**
- 121 Q&A pairs (15% of 805 total examples) from data/ipl_qa.json
- Mix of player stats, team results, and season-level facts

**Structure:**
```json
{
  "question": "Who was the leading run-scorer in IPL 2023?",
  "golden_answer": "Virat Kohli with 714 runs",
  "acceptable_variations": ["Virat Kohli scored 714 runs"],
  "question_type": "individual_stats",
  "difficulty": "easy",
  "domain": "batting"
}
```

**Evaluation Frequency:**
- Daily: Automated on 121-example eval set
- Weekly: Human review of 10% sample
- Monthly: Comprehensive human evaluation

### Training/Validation Tracking

**Purpose:** Monitor model during training and fine-tuning

**Size:**
- Train: 684 examples
- Eval: 121 examples
- Total: 805 examples

**Metrics Tracked:**
- Training loss
- Eval loss
- Mean token accuracy
- Optional: Exact match and semantic similarity on eval set

### Production Evaluation Set

**Purpose:** Continuous monitoring of live inference

**Sampling Strategy:**
- Random 1% of all inference requests
- Stratified sampling by question type
- All error cases (100%)

**Data Collection:**
```python
{
  "timestamp": "2025-01-29T10:30:00Z",
  "question": "...",
  "generated_answer": "...",
  "context_used": "...",
  "latency_ms": 1234,
  "tokens_generated": 87,
  "model_version": "qwen2-0.5b-ipl-lora-v1",
  "user_feedback": null,
  "marked_for_review": false
}
```

## Feedback Loop

### Implicit Feedback

**Collection Method:** Capture user behavior patterns
- Answer viewing time
- Question reformulation attempts
- Follow-up questions
- Bounce rate

**Implementation:**
```python
@app.post('/feedback')
async def collect_feedback(
    request_id: str,
    helpful: bool,
    rating: Optional[int] = None,
    comment: Optional[str] = None,
):
    db.feedback.insert({
        'request_id': request_id,
        'helpful': helpful,
        'rating': rating,
        'comment': comment,
        'timestamp': datetime.utcnow(),
    })
    return {'status': 'feedback_recorded'}
```

### Explicit Feedback

**Collection Method:** Post-inference survey
- Rate answer relevance (1-5)
- Flag inaccurate information
- Suggest corrections
- Report missing information

**UI Prompt:**
```
"Was this answer helpful?" [Yes] [No]
[Optional] Tell us why: [Text field]
[Optional] Correct answer: [Text field]
```

### Feedback Loop Integration

**Process:**
1. Collect negative feedback (marked as unhelpful)
2. Validate against golden answers
3. Add high-quality corrections to retraining set
4. Fine-tune model on corrected examples (monthly cadence)

## Drift Detection

### Data Drift (Input Distribution)

**Definition:** Detecting when user questions diverge from training distribution

**Implementation:**
```python
from sklearn.neighbors import LocalOutlierFactor

class InputDriftDetector:
    def __init__(self, baseline_embeddings, threshold=0.8):
        self.lof = LocalOutlierFactor(n_neighbors=20)
        self.baseline = baseline_embeddings
        self.lof.fit(baseline_embeddings)
        self.threshold = threshold
    
    def detect_drift(self, new_question):
        embedding = self.get_embedding(new_question)
        score = self.lof.decision_function(embedding.reshape(1, -1))
        
        if score > self.threshold:
            logger.warning(f"Input drift detected: {new_question}")
            return True
        return False
```

**Metrics:**
- Mean cosine distance between daily questions and baseline
- Isolation Forest anomaly score
- Embedding variance over time

**Thresholds:**
- Alert if 10% of daily questions are anomalous
- Alert if mean distance increases by more than 20% week-over-week

### Model Drift (Output Distribution)

**Definition:** Detecting when model outputs change unexpectedly

**Tracked Metrics:**
- Average answer length (word count)
- Token generation consistency
- Hallucination rate trends
- Quality metric distributions
- Percentage of answers that cite retrieved facts

**Implementation:**
```python
class OutputDriftDetector:
    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.answer_lengths = deque(maxlen=window_size)
        self.token_counts = deque(maxlen=window_size)
        self.hallucination_rates = deque(maxlen=window_size)
    
    def detect_drift(self):
        baseline_mean = np.mean(list(self.answer_lengths)[:500])
        recent_mean = np.mean(list(self.answer_lengths)[500:])
        
        ks_stat, p_value = kstest(
            list(self.answer_lengths)[500:],
            'norm',
            args=(baseline_mean, np.std(list(self.answer_lengths)[:500]))
        )
        
        if p_value < 0.05:
            logger.warning(f"Output drift detected: p={p_value:.4f}")
            return True
        return False
```

### Performance Drift

**Definition:** Accuracy metrics degrading over time

**Monitoring:**
```python
daily_metrics = {
    "date": "2025-01-29",
    "accuracy": 0.92,
    "relevance": 0.88,
    "latency_p50_ms": 1234,
    "latency_p99_ms": 3456,
    "hallucination_rate": 0.03,
    "total_inferences": 5000,
}

baseline_accuracy = 0.90
if daily_metrics['accuracy'] < baseline_accuracy * 0.95:
    alert("Model accuracy degradation detected")
```

**Root Cause Analysis:**
- Check for new question types
- Monitor data quality changes
- Verify model version consistency
- Check for upstream data issues

## Performance Monitoring

### Latency Monitoring

**Metrics:**
- P50 Latency: Target less than 2s
- P95 Latency: Target less than 4s
- P99 Latency: Target less than 6s
- Error Rate: Target less than 0.1%

### Resource Utilization

**CPU Monitoring:**  
Target: 60-80% average CPU usage  
Alert: Greater than 90% for more than 5 minutes

**Memory Monitoring:**  
Target: Keep model in memory cache  
Alert: Memory pressure greater than 85%  
Metric: Cache hit rate for repeated questions

### Throughput

**Target:** 5-15 requests per second (CPU-only, Qwen2-0.5B + LoRA)  
**Scaling:** Auto-scale replicas if P95 latency greater than 4s

## Alert Thresholds

### Severity 1 (Critical)
- Model unavailable or not loading
- Error rate greater than 10%
- P99 latency greater than 10 seconds

### Severity 2 (High)
- Accuracy drop greater than 10% from baseline
- Hallucination rate greater than 15%
- Memory usage greater than 95%

### Severity 3 (Medium)
- P95 latency greater than 6 seconds
- Drift detected in input or output
- Cold start latency greater than 3 seconds

### Severity 4 (Low)
- Data drift detected (anomalous questions)
- Feedback volume greater than 20% negative
- Model version inconsistency

## Logging Strategy

### Structured Logging Format

```json
{
  "timestamp": "2025-01-29T10:30:45.123Z",
  "level": "INFO",
  "service": "cricket-inference",
  "request_id": "uuid",
  "log_type": "inference",
  "question": "Who won IPL 2023?",
  "model_version": "qwen2-0.5b-ipl-lora-v1",
  "latency_ms": 1234,
  "tokens_generated": 87,
  "answer_length": 156,
  "context_facts_used": 3,
  "status": "success",
  "metrics": {
    "relevance_score": null,
    "accuracy_score": null,
    "hallucination_detected": false
  }
}
```

### Log Levels

- DEBUG: Model loading, tokenization details
- INFO: Successful inferences, cache hits
- WARNING: Drift detection, hallucinations, slow responses
- ERROR: Failed inferences, missing adapters
- CRITICAL: Service crashes, model load failures

### Log Retention

- Real-time: 7 days (hot storage)
- Archive: 90 days (warm storage)
- Compliance: 1 year (cold storage)

## Deployment Checklist

Before deploying to production:

- Establish baseline metrics from dev/staging (accuracy, latency)
- Set up Prometheus scraping for /metrics endpoint
- Configure Grafana dashboards
- Set up alert rules in Alertmanager
- Deploy ELK stack for log aggregation
- Create feedback collection endpoint
- Implement drift detection monitoring
- Schedule weekly evaluation runs
- Document on-call procedures
- Set up A/B testing infrastructure for model updates

## Continuous Improvement

### Monthly Retraining

```bash
# Aggregate feedback and failures
python scripts/collect_feedback.py --output-dir data/feedback --days 30

# Create new training data from corrections
python scripts/create_training_set.py \
    --qa-file data/ipl_qa.json \
    --feedback-file data/feedback/corrections.json \
    --output data/ipl_qa_v2.json

# Retrain model
python scripts/train_sft.py \
  --model qwen2-0.5b \
  --epochs 5 \
  --batch-size 2 \
  --lora-rank 8 \
  --output-dir models/ipl-cricket-sft-5epochs

# Evaluate on test set
python scripts/evaluate.py \
    --model-repo your-org/private-ipl-adapters \
    --test-file data/test_set.json \
    --output results/eval_v2.json

# Deploy new version
helm upgrade cricket-inference deploy/helm/ --set image.tag=v1.1.0
```

### Model Version Management

Track model versions with metadata:

```yaml
versions:
  - version: "1.0.0"
    date: 2025-01-29
    accuracy: 0.92
    training_set_size: 684
    adapter_repo: your-org/private-ipl-adapters
    deployment: production
    notes: "Initial release with LoRA fine-tuning"
  
  - version: "1.1.0"
    date: 2025-02-28
    accuracy: 0.94
    training_set_size: 804
    adapter_repo: your-org/private-ipl-adapters
    deployment: staging
    notes: "Improved accuracy with feedback-driven retraining"
```

## References

- HuggingFace Model Evaluation
- ML Monitoring Best Practices
- Drift Detection Techniques
- Cricket Statistics Database