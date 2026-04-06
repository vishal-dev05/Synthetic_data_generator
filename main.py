import ollama
import json
import time

# Settings
MODEL = "llama3.2"        
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

#Genrates dataset  
def generate_one(example_number):
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Generate unique example number {example_number}."}
        ]
    )
    return response["message"]["content"]

# Refines the data for correct json format, in case the model adds extra text or markdown

def parse_response(text):
    text = text.strip()
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


# Main loop
def generate_dataset(num_examples):
    dataset = []

    print(f"Generating {num_examples} student records...")
    print("-" * 50)

    for i in range(1, num_examples + 1):
        try:
            raw = generate_one(i)
            item = parse_response(raw)
            dataset.append(item)

            print(f"✓ {i}/{num_examples}: {item['name']}, Age: {item['age']}, Grade: {item['grade']}")

        except json.JSONDecodeError:
            print(f"✗ {i} — JSON parse failed, skipping")
        except Exception as e:
            print(f"✗ {i} — Error: {e}")

        time.sleep(0.5)

    print("-" * 50)
    print(f"Done! Generated {len(dataset)} records")

    return dataset

def save_to_file(dataset, filename=OUTPUT_FILE):
    with open(filename, "w") as f:
        for item in dataset:
            f.write(json.dumps(item) + "\n")

    print(f"Saved {len(dataset)} records to {filename}")
    

      
      #UI 
import gradio as gr
def ui_generate(num):
    data = generate_dataset(int(num))
    save_to_file(data)
    return data[:5]   # show only first 5 records

interface = gr.Interface(
    fn=ui_generate,
    inputs=gr.Number(label="Number of Student Records"),
    outputs="text",
    title="Synthetic Student Dataset Generator",
    description="Enter the number of student records you want to generate. The records will be saved to output.jsonl."
)

if __name__ == "__main__":
    interface.launch()