#!/usr/bin/env python


import asyncio
import json
import sys
from typing import Optional

import httpx


async def test_query(
    question: str,
    server_url: str = "http://localhost:8000",
    max_length: int = 512,
    temperature: float = 0.7,
) -> None:
    async with httpx.AsyncClient() as client:
        # Check if server is ready
        print(f"Checking server health at {server_url}...")
        try:
            health = await client.get(f"{server_url}/health")
            if health.status_code != 200:
                print(f"✗ Server not healthy: {health.text}")
                return
            print(f"✓ Server is healthy")
        except Exception as e:
            print(f"✗ Cannot reach server: {e}")
            print(f"  Make sure the server is running: python -m uvicorn service.app:app --reload")
            return
        
        # Send query
        print(f"\nQuestion: {question}")
        print(f"Max length: {max_length}, Temperature: {temperature}")
        print("-" * 70)
        
        payload = {
            "question": question,
            "max_length": max_length,
            "temperature": temperature,
        }
        
        try:
            response = await client.post(
                f"{server_url}/query",
                json=payload,
                timeout=120.0,
            )
            
            if response.status_code != 200:
                print(f"✗ Error: {response.status_code}")
                print(response.text)
                return
            
            result = response.json()
            print(f"\nAnswer:\n{result['answer']}")
            print(f"\nMetrics:")
            print(f"  Latency: {result['latency_ms']:.2f}ms")
            print(f"  Tokens: {result['tokens_generated']}")
        
        except Exception as e:
            print(f"✗ Request failed: {e}")


async def test_health(server_url: str = "http://localhost:8000") -> None:
    async with httpx.AsyncClient() as client:
        print("Testing health checks...\n")
        
        # Health check
        health = await client.get(f"{server_url}/health")
        print(f"GET /health: {health.status_code}")
        print(f"  {json.dumps(health.json(), indent=2)}")
        
        # Readiness check
        ready = await client.get(f"{server_url}/ready")
        print(f"\nGET /ready: {ready.status_code}")
        print(f"  {json.dumps(ready.json(), indent=2)}")


async def main():
    if len(sys.argv) < 2:
        print("IPL Cricket QA API Test Script")
        print("-" * 70)
        print("Usage:")
        print("  python test_api.py <question>")
        print("  python test_api.py --health")
        print("  python test_api.py --server http://localhost:8000 <question>")
        print("\nExamples:")
        print('  python test_api.py "What is IPL?"')
        print('  python test_api.py --server http://192.168.1.100:8000 "Who won IPL 2024?"')
        return
    
    server_url = "http://localhost:8000"
    question = None
    test_only_health = False
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "--server" and i + 1 < len(sys.argv):
            server_url = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == "--health":
            test_only_health = True
            i += 1
        else:
            question = sys.argv[i]
            i += 1
    
    if test_only_health:
        await test_health(server_url)
    elif question:
        await test_query(question, server_url)
    else:
        print("No question provided!")


if __name__ == '__main__':
    asyncio.run(main())
