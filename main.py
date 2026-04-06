import ollama
import json
import time

# Settings
MODEL = "llama3.2"        # fixed: all caps to be consistent
NUM_EXAMPLES = 100
OUTPUT_FILE = "output.jsonl"
TOPIC = "School Students dataset"

# System prompt — ask for ONE example per call, not all 100 at once
SYSTEM_PROMPT = f"""You are a synthetic data generator for {TOPIC}.
Generate ONE realistic student record.
Respond ONLY with valid JSON. No extra text, no markdown, no explanation.
Use exactly this format:
{{
  "name": "student full name",
  "age": 14,
  "grade": "9th",
  "favorite_subject": "Mathematics"
}}"""

def generate_one(example_number):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate unique example number {example_number}."}
        ]
    )
    return response["message"]["content"]


def parse_response(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# Main loop
dataset = []

print(f"Generating {NUM_EXAMPLES} student records...")
print("-" * 50)

for i in range(1, NUM_EXAMPLES + 1):
    try:
        raw = generate_one(i)
        item = parse_response(raw)
        dataset.append(item)
        print(f"✓ {i}/{NUM_EXAMPLES}: {item['name']}, Age: {item['age']}, Grade: {item['grade']}")
    except json.JSONDecodeError:
        print(f"✗ {i} — JSON parse failed, skipping")
    except Exception as e:
        print(f"✗ {i} — Error: {e}")
    time.sleep(0.5)

# Save to file
with open(OUTPUT_FILE, "w") as f:
    for item in dataset:
        f.write(json.dumps(item) + "\n")

print("-" * 50)
print(f"Done! Saved {len(dataset)} records to {OUTPUT_FILE}")