#!/usr/bin/env python
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CricketStats:
    query: str
    stats: Dict[str, Any]
    formatted_context: str


class IPLDataLoader:
    
    def __init__(self, data_path: str = 'data/ipl_qa.json'):
        """
        Initialize data loader.
        
        Args:
            data_path: Path to IPL Q&A JSON file
        """
        self.data_path = data_path
        self.qa_data = []
        self.df = None
        self._load_data()
    
    def _load_data(self) -> None:
        try:
            with open(self.data_path, 'r', encoding='utf-8') as f:
                self.qa_data = json.load(f)
            
            # Create DataFrame for efficient querying
            self.df = pd.DataFrame(self.qa_data)
            logger.info(f"Loaded {len(self.qa_data)} Q&A pairs from {self.data_path}")
        except FileNotFoundError:
            logger.warning(f"Data file not found: {self.data_path}. Using empty dataset.")
            self.qa_data = []
            self.df = pd.DataFrame(columns=['question', 'answer'])
    
    def search_related_facts(self, query: str, limit: int = 3) -> List[Dict[str, str]]:
        if self.df.empty:
            return []
        
        # Simple keyword matching (case-insensitive)
        query_lower = query.lower()
        keywords = query_lower.split()
        
        # Score each Q&A pair
        scores = []
        for idx, row in self.df.iterrows():
            question_lower = row['question'].lower()
            answer_lower = row['answer'].lower()
            
            # Count keyword matches
            score = sum(
                question_lower.count(kw) + answer_lower.count(kw)
                for kw in keywords
            )
            scores.append((idx, score))
        
        # Sort by score and return top results
        top_results = sorted(scores, key=lambda x: x[1], reverse=True)[:limit]
        
        results = []
        for idx, score in top_results:
            if score > 0:
                results.append({
                    'question': self.df.iloc[idx]['question'],
                    'answer': self.df.iloc[idx]['answer'],
                    'relevance': score
                })
        
        return results


class RetrieverAgent(ABC):
    
    @abstractmethod
    def retrieve_context(self, query: str) -> CricketStats:
        pass


class IPLRetrieverAgent(RetrieverAgent):
    
    def __init__(self, data_loader: IPLDataLoader):
        self.data_loader = data_loader
        logger.info("RetrieverAgent initialized")
    
    def retrieve_context(self, query: str) -> CricketStats:
        logger.info(f"Retrieving context for: {query}")
        
        # Search for related facts
        related_facts = self.data_loader.search_related_facts(query)
        
        # Build statistics dictionary
        stats = {
            'related_facts_count': len(related_facts),
            'related_facts': related_facts,
            'dataset_size': len(self.data_loader.qa_data),
        }
        
        # Format context for LLM
        context_lines = [
            "### IPL Cricket Context:",
            f"Found {len(related_facts)} related facts from the IPL dataset.",
        ]
        
        if related_facts:
            context_lines.append("\n### Related Information:")
            for i, fact in enumerate(related_facts, 1):
                context_lines.append(f"\n{i}. Q: {fact['question']}")
                context_lines.append(f"   A: {fact['answer'][:200]}...")
        
        formatted_context = '\n'.join(context_lines)
        
        logger.info(f"Retrieved {len(related_facts)} related facts")
        
        return CricketStats(
            query=query,
            stats=stats,
            formatted_context=formatted_context,
        )


class AnalystAgent:
    
    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2-0.5B-Instruct',
        adapter_repo: Optional[str] = None,
        adapter_path: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        self.model_name = model_name
        self.adapter_repo = adapter_repo
        self.adapter_path = adapter_path
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self.device = 'cpu' 
        logger.info(f"AnalystAgent initialized with model: {model_name}")
    
    def load_model(self) -> None:
        if self.model is not None:
            logger.info("Model already loaded")
            return
        
        logger.info(f"Loading model: {self.model_name}")
        
        import time
        max_retries = 3
        retry_delay = 2
        
        # Load tokenizer with retry logic
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading tokenizer (attempt {attempt + 1}/{max_retries})...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    token=self.hf_token,
                    timeout=60,
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("✓ Tokenizer loaded successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Tokenizer load failed (attempt {attempt + 1}): {e}")
                    logger.info(f"Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to load tokenizer after {max_retries} attempts: {e}")
                    raise
        
        # Load base model
        model_kwargs = {
            'dtype': torch.float32,
            'trust_remote_code': True,
            'token': self.hf_token,
        }
        
        # Only use device_map for non-CPU devices
        if self.device != 'cpu':
            model_kwargs['device_map'] = self.device
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        ).to(self.device)
        
        # Load adapters - try HF repo first, then local path
        if self.adapter_repo or self.adapter_path:
            logger.info(f"Loading LoRA adapters from repo: {self.adapter_repo} or path: {self.adapter_path}")
            try:
                if self.adapter_repo:
                    # Try loading from HuggingFace repo (preferred for production)
                    logger.info(f"Attempting to load from HF repo: {self.adapter_repo}")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        self.adapter_repo,
                        is_trainable=False,
                        token=self.hf_token,
                    )
                    logger.info("✓ Adapters loaded from HuggingFace Hub")
                elif self.adapter_path:
                    # Fallback to local path
                    logger.info(f"Loading from local path: {self.adapter_path}")
                    self.model = PeftModel.from_pretrained(
                        self.model,
                        self.adapter_path,
                        is_trainable=False,
                    )
                    logger.info("✓ Adapters loaded from local path")
            except Exception as e:
                logger.error(f"Failed to load adapters: {e}")
                if self.adapter_path:
                    logger.info(f"Falling back to local adapters at {self.adapter_path}")
                    try:
                        self.model = PeftModel.from_pretrained(
                            self.model,
                            self.adapter_path,
                            is_trainable=False,
                        )
                        logger.info("✓ Adapters loaded from local fallback")
                    except Exception as e2:
                        logger.warning(f"Could not load adapters from fallback: {e2}. Using base model only.")
                else:
                    logger.warning("No adapters loaded. Using base model only.")
        
        self.model.eval()
        logger.info("✓ Model ready for inference")
    
    def analyze(
        self,
        query: str,
        context: CricketStats,
        max_tokens: int = 512,
    ) -> str:
        if self.model is None:
            self.load_model()
        
        logger.info(f"Analyzing query with {len(context.stats.get('related_facts', []))} context facts")
        
        # Build comprehensive prompt
        prompt = f"""### Instruction:
Answer the following IPL cricket question accurately using the provided context and your knowledge.

### Question:
{query}

{context.formatted_context}

### Answer:
"""
        
        logger.info(f"Prompt length: {len(prompt)} chars")
        
        # Generate response
        inputs = self.tokenizer(
            prompt,
            return_tensors='pt',
            truncation=True,
            max_length=1024,
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_response.split("### Answer:")[-1].strip()
        
        return answer


class CricketInferenceOrchestrator:
    
    def __init__(
        self,
        data_path: str = 'data/ipl_qa.json',
        model_name: str = 'Qwen/Qwen2-0.5B-Instruct',
        adapter_repo: Optional[str] = None,
        adapter_path: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):

        self.data_loader = IPLDataLoader(data_path)
        self.retriever = IPLRetrieverAgent(self.data_loader)
        self.analyst = AnalystAgent(
            model_name=model_name,
            adapter_repo=adapter_repo,
            adapter_path=adapter_path,
            hf_token=hf_token,
        )
        logger.info("Orchestrator initialized with RetrieverAgent and AnalystAgent")
    
    def answer_question(self, question: str) -> Dict[str, Any]:
    
        logger.info(f"Processing question: {question}")
        
        # Step 1: Retrieve context
        logger.info("Step 1: RetrieverAgent - Retrieving cricket context")
        context = self.retriever.retrieve_context(question)
        
        # Step 2: Analyze with LLM
        logger.info("Step 2: AnalystAgent - Analyzing with fine-tuned model")
        answer = self.analyst.analyze(question, context)
        
        # Step 3: Return structured result
        result = {
            'question': question,
            'answer': answer,
            'retrieved_facts': context.stats.get('related_facts', []),
            'facts_count': context.stats.get('related_facts_count', 0),
            'dataset_size': context.stats.get('dataset_size', 0),
        }
        
        logger.info("Multi-agent pipeline completed successfully")
        return result


def main():

    import sys
    import time
    
    # Initialize orchestrator with your fine-tuned model
    hf_token = os.getenv('HF_TOKEN')
    adapter_repo = os.getenv('ADAPTER_REPO', 'riyakl09/ipl-cricket-sft')
    adapter_path = os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft')
    
    orchestrator = CricketInferenceOrchestrator(
        data_path='data/ipl_qa.json',
        model_name='Qwen/Qwen2-0.5B-Instruct',
        adapter_repo=adapter_repo,
        adapter_path=adapter_path,
        hf_token=hf_token,
    )
    
    # Example questions
    questions = [
        "Who was the leading run-scorer in IPL 2023?",
        "What was RCB's performance in IPL 2023?",
        "Which team won the IPL 2023 championship?",
    ]
    
    for question in questions:
        print(f"\n{'='*80}")
        print(f"Question: {question}")
        print('='*80)
        
        start_time = time.time()
        result = orchestrator.answer_question(question)
        elapsed = time.time() - start_time
        
        print(f"\nAnswer: {result['answer']}")
        print(f"\nContext: Retrieved {result['facts_count']} related facts")
        print(f"Inference Time: {elapsed:.2f}s")
        
        if result['retrieved_facts']:
            print("\nRelated Facts:")
            for fact in result['retrieved_facts']:
                print(f"  - {fact['question'][:60]}...")


if __name__ == '__main__':
    main()
