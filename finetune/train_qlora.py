import os
import sys
import pandas as pd
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Unsloth MUST be imported first to apply its 
# 2x speed hot-patches to the Hugging Face ecosystem.
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

from datasets import Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# ... (The rest of your script starting with def train_model(): remains exactly the same)
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def train_model():
    print("Initializing Code Generation RAG Fine-Tuning Pipeline...")
    
    # 1. Hardware & Path Setup
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, "data", "training_data.jsonl")
    output_dir = os.path.join(project_root, "rag", "adapters")

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run prep_dataset.py first.")
        return

    # 2. Model Configuration
    max_seq_length = 2048 
    dtype = None 
    load_in_4bit = True 

    print("Loading model and tokenizer...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-Instruct-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # 3. LoRA Configuration
    print("Injecting LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0, 
        bias="none",
        use_gradient_checkpointing="unsloth", 
        random_state=3407,
    )

    # 4. Data Formatting (ChatML) & PyArrow Bypass
    print("Preparing dataset via Pandas (PyArrow Bypass)...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="chatml"
    )

    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    # Safely load the JSONL using Python's native engine to avoid C++ memory crashes
    df = pd.read_json(dataset_path, lines=True)
    dataset = Dataset.from_pandas(df)
    
    # Apply ChatML formatting
    dataset = dataset.map(formatting_prompts_func, batched=True)

    # 5. Training Arguments
    print("Configuring Trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60, 
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
        ),
    )

    # 6. Execution
    print("Starting fine-tuning...")
    trainer_stats = trainer.train()
    
    print(f"Saving final adapter weights to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Training sequence complete.")

if __name__ == "__main__":
    train_model()