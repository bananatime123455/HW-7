# distilgpt2_sft_cpu.py
# CPU-only supervised fine-tuning (SFT) of DistilGPT2 on a small text dataset,
# then text generation from given prompts. We will only implement:
#   1) get_params()                -> return generation hyperparameters
#   2) the short write loop in generate_cpu() to create generated_distilgpt.txt
#
# Install:
#   Follow Section 3.2 of the handout (requirements.txt).
#
# Timing choices for ~10–15 minutes on CPU:
# - Use data.csv with ~3000 lines
# - Tokenize with max_length=128
# - Train for max_steps=600
# - Disable mid-run eval, save only at final step
# Note - Please check the references provided in the writeup for more details on DistilGPT2, Hugging Face Transformers and Trainer API

import os
from typing import Dict
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
)

# MODEL CONFIGURATION
MODEL_ID = "distilgpt2"         # Pre-trained model from Hugging Face
                                # DistilGPT2 = smaller, faster version of GPT-2
                                # Good for learning and CPU training

OUT_DIR = "distilgpt2_sft_cpu"  # Directory where our fine-tuned model will be saved
                                # After training, this folder contains:
                                # - Fine-tuned model weights
                                # - Tokenizer configuration  
                                # - Training configuration

# GENERATION PROMPTS
# These prompts will be used to test our fine-tuned model
# We generate 3 different responses for each prompt to see variety
PROMPTS = [
    "Who is Michael Jordan and what sport is he known for?",
    "What are Ancient Egyptian deities?", 
    "The Himalayas: tectonic origin, highest peaks, and impact on regional climate. Two sentences."
]

def load_text_dataset(path: str) -> DatasetDict:
    """
    Load a CSV or JSON/JSONL file and ensure a single 'text' column for SFT.

    What this does:
    - Reads the file as a Hugging Face Dataset with a 'train' split.
    - If the file lacks a 'text' column, merges all string fields into 'text'.
    - Creates a tiny 'test' split (5%) for flexibility, though eval is disabled in this config.

    Expected input:
    - CSV with a 'text' column, or JSON/JSONL where each record contains 'text'.
    - If not, it will create 'text' by concatenating string fields.

    Returns:
    - DatasetDict with 'train' and 'test', each having a 'text' column.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        ds = load_dataset("csv", data_files={"train": path})
    else:
        ds = load_dataset("json", data_files={"train": path})

    train = ds["train"]
    if "text" not in train.column_names:
        def to_text(ex):
            merged = " ".join(str(v) for v in ex.values() if isinstance(v, str))
            return {"text": merged}
        train = train.map(to_text, remove_columns=train.column_names)

    split = train.train_test_split(test_size=0.05, seed=42)
    return DatasetDict(train=split["train"], test=split["test"])

def train_args() -> TrainingArguments:
    """
    Minimal CPU-only training arguments optimized for speed.

    Key choices:
    - use_cpu=True forces CPU.
    - Small per-device batch sizes; modest gradient_accumulation_steps to avoid overhead spikes on CPU.
    - max_steps limits runtime; evaluation disabled; single final checkpoint save.
    """
    return TrainingArguments(
        output_dir=OUT_DIR,
        use_cpu=True,                   # force CPU
        per_device_train_batch_size=2,  # small per-step memory
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # modest accumulation to reduce overhead
        learning_rate=2e-5,
        max_steps=600,                  # shorter run
        logging_steps=50,
        eval_strategy="no",             # disable mid-run eval for speed
        save_steps=250,                 # save only at end
        save_total_limit=1,             # only final checkpoint
        fp16=False, bf16=False,
        report_to=[]
    )

def get_params() -> Dict:
    """
    STUDENT: Return generation hyperparameters.

    Purpose:
    - Control diversity and length of generated text.

    Return keys:
    - temperature (float): Higher -> more randomness (e.g., 0.7-1.0).
    - top_p (float): Nucleus sampling threshold (e.g., 0.9-0.95).
    - max_new_tokens (int): Max tokens to generate beyond the prompt (e.g., 80-150).
    - repetition_penalty (float): Penalty for repeating tokens (e.g., 1.0-2.0).
    - do_sample (bool): True to sample; False for greedy decoding.
    - return_full_text (bool): If True, return the full text including the prompt.

    Pseudo code:
    - params = {
        "temperature": 0.8,
        "top_p": 0.95,
        "max_new_tokens": 100,
        "repetition_penalty": 1.5,
        "do_sample": True,
        "return_full_text": False
      }
    - return params
    """

    params = {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 80,
            "repetition_penalty": 1.2,
            "do_sample": True,
            "return_full_text": False
        }
        
    return params

def train_cpu(data_path: str):
    """
    CPU training pipeline:
    1) Load dataset and ensure a single 'text' column.
    2) Tokenize to a fixed max_length with truncation for causal LM (here: 128 for speed).
    3) Use DataCollatorForLanguageModeling(mlm=False) for next-token prediction.
    4) Train with Trainer for max_steps; save model and tokenizer to OUT_DIR.
    """
    # Load the dataset
    ds = load_text_dataset(data_path)
    # Load tokenizer from pre-trained model
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token  # GPT-2 family lacks a pad token

    # Define tokenization function
    def tok_func(ex):
        return tok(ex["text"], truncation=True, max_length=128)
    # Apply tokenization to dataset
    ds_tok = ds.map(tok_func, batched=True, remove_columns=ds["train"].column_names)

    # Set up data collator for language modeling
    col = DataCollatorForLanguageModeling(tok, mlm=False)
    # Load pre-trained model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    # Get training arguments
    args = train_args()

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_tok["train"],
        eval_dataset=None,  # eval disabled for speed
        data_collator=col
    )

    # Start training
    trainer.train()
    # Save the fine-tuned model
    trainer.save_model(OUT_DIR)
    # Save tokenizer
    tok.save_pretrained(OUT_DIR)

def generate_cpu(out_file="generated_distilgpt.txt"):
    """
    Generate text from prompts using the fine-tuned checkpoint if present,
    otherwise the base DistilGPT2 model. Results are written to generated_distilgpt.txt.

    Steps:
    1) Load tokenizer and model through the text-generation pipeline (CPU).
    2) Get generation params from get_params().
    3) For each prompt in PROMPTS, produce 3 samples and write them to file.

    STUDENT: implement the params call and the for-loops that write outputs.

    Pseudo code:
    - params = get_params()
    - open out_file for writing
    - for i, prompt in enumerate(PROMPTS):
        - for j in range(3):
            - result = gen(prompt, **params)[0]["generated_text"]
            - write a section with prompt index, sample index, prompt text, and generated text
    
    Expected output format in text file:
    
    === Prompt 0 (Sample 0) ===
    Prompt: Who is Michael Jordan and what sport is he known for?
    === Generation ===
    Michael Jordan is widely regarded as one of the greatest basketball players of all time...
    
    === Prompt 0 (Sample 1) ===
    Prompt: Who is Michael Jordan and what sport is he known for?
    === Generation ===
    Michael Jordan, often called "MJ," was a professional basketball player who played...
    
    (Continue for all 3 samples of each prompt)
    """
    # Choose model source: fine-tuned if available, otherwise base model
    src = OUT_DIR if os.path.exists(OUT_DIR) else MODEL_ID
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(src)
    # Create text generation pipeline
    gen = pipeline("text-generation", model=src, tokenizer=tok, device=-1,
                   pad_token_id=tok.eos_token_id, eos_token_id=tok.eos_token_id)  # CPU pipeline
    params = get_params()
    with open(out_file, "w", encoding="utf-8") as f:
        for i, prompt in enumerate(PROMPTS):
            for j in range(3):
                result = gen(prompt, **params)[0]["generated_text"]
                f.write(f"=== Prompt {i} (Sample {j}) ===\n")
                f.write(f"Prompt: {prompt}\n")
                f.write("=== Generation ===\n")
                f.write(result.strip() + "\n\n")

if __name__ == "__main__":
    """
    CLI usage:
      - Create dataset:  python make_data_csv.py  (produces data.csv)
      - Training (CPU):  python distilgpt2_sft_cpu.py --data data.csv --mode train
      - Generation:      python distilgpt2_sft_cpu.py --mode gen

    Data expectations:
      - data.csv is a single-column CSV with header 'text' (created by make_data_csv.py).
      - Keep data small (~3k lines) for fast CPU runs.
    """
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=False, help="Path to CSV dataset (data.csv).")
    ap.add_argument("--mode", type=str, choices=["train", "gen"], default="train", help="train or gen")
    args = ap.parse_args()

    if args.mode == "train":
        if not args.data:
            raise SystemExit("Provide --data pointing to data.csv (single 'text' column).")
        train_cpu(args.data)
    else:
        generate_cpu()
