#!/usr/bin/env python3
from huggingface_hub import login, whoami, list_repo_files

HF_TOKEN = 'token-here-replace-with-your-own-token'

print('Testing token...')
login(token=HF_TOKEN)

# Check who we're logged in as
try:
    user = whoami()
    print(f'✓ Logged in as: {user["name"]}')
except Exception as e:
    print(f'✗ Error: {e}')

# Try to list files in the repo
print('\nTesting repo access: riyakl09/ipl-cricket-sft')
try:
    files = list_repo_files('riyakl09/ipl-cricket-sft', repo_type='model', token=HF_TOKEN)
    print(f'✓ SUCCESS! Found {len(files)} files in repo:')
    for f in sorted(files)[:15]:
        print(f'  - {f}')
except Exception as e:
    print(f'✗ FAILED: {e}')
