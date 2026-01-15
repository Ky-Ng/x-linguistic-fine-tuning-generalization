# %%
from datasets import load_dataset
from unsloth import FastLanguageModel
import torch

# %%
# train_ds = load_dataset("kylelovesllms/alpaca-with-text-upper", split="train") 


# train_dataset_split = train_ds.train_test_split(train_size=0.8, seed=42)
# train_ds = train_dataset_split["train"]

# test_dataset_split = test_ds.train_test_split(test_size=0.2, seed=42)
# test_ds = test_dataset_split["test"]

dataset = load_dataset("kylelovesllms/alpaca-with-text-upper", split="train") 

# 2. Create the split (e.g., 90% Train, 10% Test)
# We shuffle with a seed to ensure the split is reproducible
dataset_split = dataset.train_test_split(test_size=0.1, seed=42)

train_ds = dataset_split["train"]
test_ds = dataset_split["test"]

# 3. Verify the counts
print(f"Training samples: {len(train_ds)}")
print(f"Testing samples:  {len(test_ds)}")

# Check a sample to ensure it looks correct
print("\nSample Input:", train_ds[0]['instruction'])
print("Sample Output:", train_ds[0]['output_upper'])

# %%
from unsloth.chat_templates import get_chat_template
model_name = "Qwen/Qwen2.5-0.5B"  # example
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,          # auto
    load_in_4bit=True,   # typical for LoRA
)

# tokenizer = get_chat_template(
#     tokenizer,
#     chat_template = "chatml",
# )

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
import torch
# from transformers import DataCollatorForLanguageModeling

# class QwenMaskingCollator(DataCollatorForLanguageModeling):
#     def __init__(self, tokenizer, response_template_ids, mlm=False):
#         super().__init__(tokenizer=tokenizer, mlm=mlm)
#         self.response_template_ids = response_template_ids

#     def __call__(self, examples):
#         # 1. Let the base class handle padding and tensor conversion
#         batch = super().__call__(examples)
        
#         # 2. Create labels if they don't exist (copy of input_ids)
#         if "labels" not in batch:
#             batch["labels"] = batch["input_ids"].clone()
            
#         # 3. Iterate over the batch to apply masking
#         for i in range(len(batch["input_ids"])):
#             # Find the start index of the assistant's response
#             response_start = self.find_template_index(
#                 batch["input_ids"][i], 
#                 self.response_template_ids
#             )
            
#             if response_start != -1:
#                 # Mask everything BEFORE the response starts (User + System prompt)
#                 # We use -100 because PyTorch's CrossEntropyLoss ignores this value
#                 batch["labels"][i, :response_start] = -100
#             else:
#                 # If template not found (rare error), mask everything to be safe
#                 # so we don't train on garbage
#                 batch["labels"][i, :] = -100
                
#         return batch

#     def find_template_index(self, input_ids, template_ids):
#         """
#         Finds the end index of the template_ids inside input_ids.
#         Returns the index where the ACTUAL response begins.
#         """
#         # Convert tensors to lists for easier matching
#         input_list = input_ids.tolist()
#         n = len(template_ids)
        
#         # Simple sliding window search
#         for i in range(len(input_list) - n + 1):
#             if input_list[i : i+n] == template_ids:
#                 return i + n  # Return index *after* the template
#         return -1

# response_template_ids = [151644, 77091, 198]

# # 2. Initialize the Custom Collator
# collator = QwenMaskingCollator(
#     tokenizer=tokenizer,
#     response_template_ids=response_template_ids,
#     mlm=False # Important: We are doing Causal LM, not Masked LM
# )


# %%
from trl import SFTTrainer
from transformers import TrainingArguments

def formatting_func(examples):
    texts = []

    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output_upper"]):

        if input_text and input_text.strip():
            user_content = f"{instruction}\n\nInput: {input_text}"
        else:
            user_content = instruction
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        print(text)
        texts.append(text)
    return texts

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    max_seq_length=max_seq_length,
    packing=False,  # packs multiple samples into one sequence -> faster if your samples are short
    # data_collator=collator,
    formatting_func=formatting_func,
    response_template="<|im_start|>assistant\n",  # Mask tokens before assistant response
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