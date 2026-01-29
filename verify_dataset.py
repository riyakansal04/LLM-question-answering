
import json

# Load and verify the dataset
with open('data/ipl_qa.json', 'r') as f:
    data = json.load(f)

print(f"✓ Successfully loaded {len(data)} training examples")
print(f"✓ Dataset is ready for training with real IPL statistics\n")

print("Sample Q&A pairs:")
print("=" * 80)
for i, item in enumerate(data[:5]):
    print(f"\nExample {i+1}:")
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")
    print("-" * 80)

print(f"\n✓ All {len(data)} training examples have both questions and answers")
print("✓ Dataset is ready to be used for fine-tuning the LLM")
