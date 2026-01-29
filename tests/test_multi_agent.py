#!/usr/bin/env python
"""
Integration tests for multi-agent orchestration system.

Tests:
  - RetrieverAgent: Data loading and context retrieval
  - AnalystAgent: Model loading and inference
  - CricketInferenceOrchestrator: End-to-end pipeline
"""

import os
import sys
import pytest
import time
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.multi_agent import (
    IPLDataLoader,
    IPLRetrieverAgent,
    AnalystAgent,
    CricketInferenceOrchestrator,
    CricketStats,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestIPLDataLoader:
    """Test IPL data loading functionality."""
    
    def test_data_loader_initialization(self):
        """Test data loader initializes correctly."""
        loader = IPLDataLoader('data/ipl_qa.json')
        assert loader.df is not None
        assert len(loader.qa_data) > 0
        logger.info(f"✓ Loaded {len(loader.qa_data)} Q&A pairs")
    
    def test_search_related_facts(self):
        """Test related facts search."""
        loader = IPLDataLoader('data/ipl_qa.json')
        results = loader.search_related_facts('IPL runs', limit=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3
        
        if results:
            assert 'question' in results[0]
            assert 'answer' in results[0]
            assert 'relevance' in results[0]
            logger.info(f"✓ Found {len(results)} related facts")


class TestRetrieverAgent:
    """Test RetrieverAgent functionality."""
    
    def test_retriever_initialization(self):
        """Test retriever initializes correctly."""
        loader = IPLDataLoader('data/ipl_qa.json')
        retriever = IPLRetrieverAgent(loader)
        
        assert retriever.data_loader is not None
        logger.info("✓ RetrieverAgent initialized")
    
    def test_retrieve_context(self):
        """Test context retrieval."""
        loader = IPLDataLoader('data/ipl_qa.json')
        retriever = IPLRetrieverAgent(loader)
        
        query = "Who won IPL 2023?"
        context = retriever.retrieve_context(query)
        
        assert isinstance(context, CricketStats)
        assert context.query == query
        assert 'related_facts' in context.stats
        assert 'dataset_size' in context.stats
        assert len(context.formatted_context) > 0
        
        logger.info(f"✓ Retrieved context for query: {query}")
        logger.info(f"  - Facts found: {context.stats['related_facts_count']}")
        logger.info(f"  - Dataset size: {context.stats['dataset_size']}")


class TestAnalystAgent:
    """Test AnalystAgent functionality."""
    
    def test_analyst_initialization(self):
        """Test analyst agent initializes."""
        analyst = AnalystAgent(
            model_name='Qwen/Qwen2-0.5B-Instruct',
            adapter_repo=os.getenv('ADAPTER_REPO'),
            adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
            hf_token=os.getenv('HF_TOKEN'),
        )
        
        assert analyst.model_name == 'Qwen/Qwen2-0.5B-Instruct'
        logger.info("✓ AnalystAgent initialized")
    
    @pytest.mark.slow
    def test_model_loading(self):
        """Test model loading (slow test)."""
        analyst = AnalystAgent(
            model_name='Qwen/Qwen2-0.5B-Instruct',
            adapter_repo=os.getenv('ADAPTER_REPO'),
            adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
            hf_token=os.getenv('HF_TOKEN'),
        )
        
        analyst.load_model()
        
        assert analyst.model is not None
        assert analyst.tokenizer is not None
        logger.info("✓ Model loaded successfully")
    
    @pytest.mark.slow
    def test_inference(self):
        """Test inference with analyst agent (slow test)."""
        analyst = AnalystAgent(
            model_name='Qwen/Qwen2-0.5B-Instruct',
            adapter_repo=os.getenv('ADAPTER_REPO'),
            adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
            hf_token=os.getenv('HF_TOKEN'),
        )
        
        loader = IPLDataLoader('data/ipl_qa.json')
        retriever = IPLRetrieverAgent(loader)
        
        query = "Who won IPL 2023?"
        context = retriever.retrieve_context(query)
        
        start_time = time.time()
        answer = analyst.analyze(query, context, max_tokens=256)
        elapsed = time.time() - start_time
        
        assert isinstance(answer, str)
        assert len(answer) > 0
        
        logger.info(f"✓ Generated answer in {elapsed:.2f}s")
        logger.info(f"  Answer: {answer[:100]}...")


class TestOrchestrator:
    """Test CricketInferenceOrchestrator end-to-end."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        orchestrator = CricketInferenceOrchestrator(
            data_path='data/ipl_qa.json',
            model_name='Qwen/Qwen2-0.5B-Instruct',
            adapter_repo=os.getenv('ADAPTER_REPO'),
            adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
            hf_token=os.getenv('HF_TOKEN'),
        )
        
        assert orchestrator.data_loader is not None
        assert orchestrator.retriever is not None
        assert orchestrator.analyst is not None
        logger.info("✓ CricketInferenceOrchestrator initialized")
    
    @pytest.mark.slow
    def test_end_to_end_pipeline(self):
        """Test complete end-to-end pipeline (slow test)."""
        orchestrator = CricketInferenceOrchestrator(
            data_path='data/ipl_qa.json',
            model_name='Qwen/Qwen2-0.5B-Instruct',
            adapter_repo=os.getenv('ADAPTER_REPO'),
            adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
            hf_token=os.getenv('HF_TOKEN'),
        )
        
        query = "Who was the leading run-scorer?"
        
        start_time = time.time()
        result = orchestrator.answer_question(query)
        elapsed = time.time() - start_time
        
        # Verify result structure
        assert 'question' in result
        assert 'answer' in result
        assert 'retrieved_facts' in result
        assert 'facts_count' in result
        assert 'dataset_size' in result
        
        assert result['question'] == query
        assert isinstance(result['answer'], str)
        assert len(result['answer']) > 0
        assert isinstance(result['retrieved_facts'], list)
        
        logger.info("✓ End-to-end pipeline completed successfully")
        logger.info(f"  Query: {query}")
        logger.info(f"  Answer: {result['answer'][:100]}...")
        logger.info(f"  Facts retrieved: {result['facts_count']}")
        logger.info(f"  Total time: {elapsed:.2f}s")
        
        return result


# Performance benchmarking
class TestPerformance:
    """Performance and latency tests."""
    
    @pytest.mark.slow
    def test_inference_latency(self):
        """Test inference latency."""
        orchestrator = CricketInferenceOrchestrator(
            data_path='data/ipl_qa.json',
            model_name='Qwen/Qwen2-0.5B-Instruct',
            adapter_repo=os.getenv('ADAPTER_REPO'),
            adapter_path=os.getenv('ADAPTER_PATH', 'models/ipl-cricket-sft'),
            hf_token=os.getenv('HF_TOKEN'),
        )
        
        queries = [
            "Who won IPL?",
            "Best batsman?",
            "Highest score?",
        ]
        
        latencies = []
        for query in queries:
            start = time.time()
            orchestrator.answer_question(query)
            elapsed = time.time() - start
            latencies.append(elapsed)
            logger.info(f"Query latency: {elapsed:.2f}s")
        
        avg_latency = sum(latencies) / len(latencies)
        logger.info(f"✓ Average latency: {avg_latency:.2f}s")
        
        # Assert reasonable latency (adjust based on your hardware)
        assert avg_latency < 30  # Should complete in reasonable time


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-m', 'not slow',  # Skip slow tests by default
    ])
