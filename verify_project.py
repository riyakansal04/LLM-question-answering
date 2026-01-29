#!/usr/bin/env python


import os
import sys
import json
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def check_file_exists(path: str, description: str) -> bool:
    
    exists = os.path.isfile(path)
    status = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
    print(f"{status} {description:50} {path}")
    return exists


def check_dir_exists(path: str, description: str) -> bool:
 
    exists = os.path.isdir(path)
    status = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
    print(f"{status} {description:50} {path}")
    return exists


def check_file_contains(path: str, text: str) -> bool:
   
    try:
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        return text in content
    except:
        return False


def verify_project_structure():
   
    print(f"\n{BLUE}=== PROJECT STRUCTURE VERIFICATION ==={RESET}\n")
    
    checks = [
        # Core files
        ("data/ipl_qa.json", "IPL Q&A dataset"),
        ("scripts/train_sft.py", "Fine-tuning script"),
        ("service/app.py", "FastAPI inference service"),
        ("service/Dockerfile", "Docker container definition"),
        ("agents/multi_agent.py", "Multi-agent orchestration"),
        ("tests/test_inference.py", "Integration tests"),
        
        # Configuration & Deployment
        ("deploy/helm/Chart.yaml", "Helm chart metadata"),
        ("deploy/helm/values.yaml", "Helm configuration"),
        ("deploy/helm/templates.yaml", "Kubernetes manifests"),
        ("deploy/helm/templates/_helpers.tpl", "Helm helpers"),
        
        # Documentation
        ("README.md", "Main documentation"),
        ("MODEL_MONITORING.md", "Monitoring strategy"),
        ("DEPLOYMENT.md", "Deployment guide"),
        ("QUICK_REFERENCE.md", "Quick reference"),
        ("COMPLETION_SUMMARY.md", "Completion summary"),
        
        # Python packaging
        ("requirements.txt", "Python dependencies"),
        ("pytest.ini", "Pytest configuration"),
        (".gitignore", "Git ignore file"),
        ("__init__.py", "Package init"),
    ]
    
    print(f"{BLUE}Files:{RESET}")
    file_count = sum(1 for path, desc in checks if check_file_exists(path, desc))
    
    print(f"\n{BLUE}Directories:{RESET}")
    dir_checks = [
        ("data", "Data directory"),
        ("scripts", "Scripts directory"),
        ("service", "Service directory"),
        ("agents", "Agents directory"),
        ("tests", "Tests directory"),
        ("deploy", "Deployment directory"),
        ("deploy/helm", "Helm directory"),
        ("deploy/helm/templates", "Helm templates"),
    ]
    
    dir_count = sum(1 for path, desc in dir_checks if check_dir_exists(path, desc))
    
    return file_count, dir_count


def verify_file_content():

    print(f"\n{BLUE}=== FILE CONTENT VERIFICATION ==={RESET}\n")
    
    checks = [
        ("scripts/train_sft.py", "QLoRA", "QLoRA quantization support"),
        ("scripts/train_sft.py", "push_to_hub", "HuggingFace Hub integration"),
        ("service/app.py", "FastAPI", "FastAPI framework"),
        ("service/app.py", "/infer", "Inference endpoint"),
        ("service/app.py", "/healthz", "Health check endpoint"),
        ("service/app.py", "/readyz", "Readiness endpoint"),
        ("agents/multi_agent.py", "RetrieverAgent", "RetrieverAgent class"),
        ("agents/multi_agent.py", "AnalystAgent", "AnalystAgent class"),
        ("deploy/helm/values.yaml", "replicaCount", "Helm configuration"),
        ("README.md", "Phi-3-Mini", "Model documentation"),
        ("MODEL_MONITORING.md", "accuracy", "Monitoring documentation"),
    ]
    
    content_count = 0
    for file_path, search_text, description in checks:
        exists = check_file_contains(file_path, search_text)
        status = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
        print(f"{status} {description:40} ({file_path})")
        if exists:
            content_count += 1
    
    return content_count


def verify_dependencies():

    print(f"\n{BLUE}=== DEPENDENCY VERIFICATION ==={RESET}\n")
    
    essential = [
        "transformers",
        "datasets",
        "peft",
        "torch",
        "accelerate",
        "huggingface_hub",
        "bitsandbytes",
        "fastapi",
        "uvicorn",
        "pydantic",
        "pandas",
        "pytest",
    ]
    
    try:
        with open("requirements.txt", "r") as f:
            requirements = f.read().lower()
        
        found_count = 0
        for package in essential:
            found = package.lower() in requirements
            status = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
            print(f"{status} {package:20}")
            if found:
                found_count += 1
        
        return found_count
    except:
        print(f"{RED}Could not read requirements.txt{RESET}")
        return 0


def verify_data():
 
    print(f"\n{BLUE}=== DATA VERIFICATION ==={RESET}\n")
    
    try:
        with open("data/ipl_qa.json", "r") as f:
            data = json.load(f)
        
        print(f"✓ JSON format valid")
        print(f"✓ Total Q&A pairs: {len(data)}")
        
        if isinstance(data, list) and len(data) > 0:
            first = data[0]
            has_question = "question" in first
            has_answer = "answer" in first
            
            status_q = f"{GREEN}✓{RESET}" if has_question else f"{RED}✗{RESET}"
            status_a = f"{GREEN}✓{RESET}" if has_answer else f"{RED}✗{RESET}"
            
            print(f"{status_q} Questions present")
            print(f"{status_a} Answers present")
            
            return len(data) >= 10
        
        return False
    except Exception as e:
        print(f"{RED}Error reading data: {e}{RESET}")
        return False


def verify_docker():
   
    print(f"\n{BLUE}=== DOCKERFILE VERIFICATION ==={RESET}\n")
    
    try:
        with open("service/Dockerfile", "r") as f:
            dockerfile = f.read()
        
        checks = [
            ("multi-stage", "Multi-stage build"),
            ("python:3.11-slim", "Slim base image"),
            ("appuser", "Non-root user"),
            ("HEALTHCHECK", "Health check"),
            ("8000", "Port 8000"),
            ("fastapi", "FastAPI"),
        ]
        
        count = 0
        for text, description in checks:
            found = text in dockerfile
            status = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
            print(f"{status} {description:30}")
            if found:
                count += 1
        
        return count
    except Exception as e:
        print(f"{RED}Error reading Dockerfile: {e}{RESET}")
        return 0


def verify_tests():
  
    print(f"\n{BLUE}=== TEST VERIFICATION ==={RESET}\n")
    
    try:
        with open("tests/test_inference.py", "r") as f:
            tests = f.read()
        
        checks = [
            ("TestHealthEndpoints", "Health endpoint tests"),
            ("TestInferenceEndpoint", "Inference endpoint tests"),
            ("TestErrorHandling", "Error handling tests"),
            ("TestRequestValidation", "Request validation tests"),
            ("def test_", "Test functions"),
        ]
        
        count = 0
        for text, description in checks:
            found = text in tests
            status = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
            print(f"{status} {description:35}")
            if found:
                count += 1
        
        # Count test functions
        import re
        test_count = len(re.findall(r'def test_\w+', tests))
        print(f"✓ Total test functions: {test_count}")
        
        return count
    except Exception as e:
        print(f"{RED}Error reading tests: {e}{RESET}")
        return 0


def main():
   
    print(f"\n{BLUE}╔════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BLUE}║   CRICKET INFERENCE SYSTEM - VERIFICATION SCRIPT          ║{RESET}")
    print(f"{BLUE}╚════════════════════════════════════════════════════════════╝{RESET}")
    
    # Run all checks
    files_ok, dirs_ok = verify_project_structure()
    content_ok = verify_file_content()
    deps_ok = verify_dependencies()
    data_ok = verify_data()
    docker_ok = verify_docker()
    tests_ok = verify_tests()
    
    # Summary
    print(f"\n{BLUE}=== VERIFICATION SUMMARY ==={RESET}\n")
    
    total_checks = [
        ("Project Structure", files_ok > 15, f"{files_ok} files checked"),
        ("File Content", content_ok > 8, f"{content_ok} content checks passed"),
        ("Dependencies", deps_ok > 10, f"{deps_ok} dependencies found"),
        ("Data Quality", data_ok, "IPL Q&A dataset valid"),
        ("Dockerfile", docker_ok > 4, f"{docker_ok} Dockerfile checks passed"),
        ("Test Suite", tests_ok > 3, f"{tests_ok} test classes found"),
    ]
    
    passed = 0
    for name, check, details in total_checks:
        status = f"{GREEN}✓{RESET}" if check else f"{RED}✗{RESET}"
        print(f"{status} {name:30} {details}")
        if check:
            passed += 1
    
    print(f"\n{BLUE}Overall Status: {passed}/{len(total_checks)} checks passed{RESET}\n")
    
    if passed == len(total_checks):
        print(f"{GREEN}✓ PROJECT VERIFICATION SUCCESSFUL!{RESET}")
        print(f"{GREEN}All components are properly set up and ready.{RESET}\n")
        return 0
    else:
        print(f"{YELLOW}⚠ Some checks failed. Please review the output above.{RESET}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
