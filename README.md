# Synthetic Data Generator 

This project generates synthetic student datasets using LLM (Ollama + Llama3).

## Features
- Generates structured JSON data
- Handles JSON parsing errors
- Saves data in JSONL format
- Easily customizable schema

## Example Output
```json
{"name": "Rahul Sharma", "age": 15, "grade": "10th", "favorite_subject": "Math"}


## Setup
1. Install [Ollama](https://ollama.com) and run `ollama pull llama3.2`
2. Install uv: `pip install uv`
3. Install dependencies: `uv add ollama pandas`


How to Run
Install Ollama
Pull model:
ollama run llama3.2

Install dependencies:
pip install uv
uv sync 

Run:
uv run  main.py

Use Cases
ML training data
Testing pipelines
Data augmentation

Author
Vishal Rathod