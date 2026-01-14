# %%
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch

# %%
train_ds = load_dataset("kylelovesllms/alpaca-with-text-upper", split="train") 
test_ds = load_dataset("kylelovesllms/alpaca-cleaned-de-upper", split="train")

train_dataset_split = train_ds.train_test_split(train_size=0.8, seed=42)
train_ds = train_dataset_split["train"]

test_dataset_split = test_ds.train_test_split(test_size=0.2, seed=42)
test_ds = test_dataset_split["test"]

# %%
model_name = "Qwen/Qwen2.5-0.5B-Instruct"  # example
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,          # auto
    load_in_4bit=True,   # typical for LoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# %%
import wandb
wandb.init(
    project="unsloth-allcaps-alignment",
    name="qwen2.5-0.5b-lora-allcaps",
    config={
        "model": model_name,
        "max_seq_length": max_seq_length,
        "lora_r": 16,
        "lora_alpha": 16,
        "lr": 2e-4,
        "batch_size": 2,
        "grad_accum": 8,
        "dataset": "kylelovesllms/alpaca-with-text-upper",
    },
)


# %%
from trl import SFTTrainer
from transformers import TrainingArguments

def formatting_func(examples):
    to_apply_template = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": examples["instruction"] + ("\n\nInput: " + examples["input"] if examples["input"] and examples["input"].strip() != "" else "")},
        {"role": "assistant", "content": examples["output_upper"]},
    ]

    tokenized = tokenizer.apply_chat_template(
        to_apply_template,
        tokenize=False,
        add_generation_prompt=False
    )
    return [tokenized]

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    max_seq_length=max_seq_length,
    packing=True,  # packs multiple samples into one sequence -> faster if your samples are short
    formatting_func=formatting_func,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        max_steps=1000,              # or set num_train_epochs=...
        learning_rate=2e-4,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        output_dir="outputs",
        report_to="wandb",
        save_steps=200,
    ),
)

trainer.train()
wandb.finish()

# %%
from huggingface_hub import HfApi
model.save_pretrained("lora_adapters")
tokenizer.save_pretrained("lora_adapters")

merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model", safe_serialization=True)
tokenizer.save_pretrained("merged_model")


HfApi().upload_folder(
    folder_path="lora_adapters",
    repo_id="kylelovesllms/Qwen2.5-0.5B-Instruct-caps-en-lora",
    repo_type="model",
)

HfApi().upload_folder(
    folder_path="merged_model",
    repo_id="kylelovesllms/Qwen2.5-0.5B-Instruct-caps-en-lora-merged",
    repo_type="model",
)
