# code from: https://tutorials.inferless.com/
import torch

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import prepare_model_for_kbit_training, LoraConfig, PeftModel
from trl import DPOTrainer

model_name = "microsoft/phi-2"

#Load the Tokenizer
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

def format_data(example):

    system = tokenizer.apply_chat_template([{"role": "system", "content": example['system']}], tokenize=False) if example['system'] else ""
    prompt = tokenizer.apply_chat_template([{"role": "user", "content": example['input']}], tokenize=False, add_generation_prompt=True)

    return {
        "prompt": system + prompt,
        "chosen": example['chosen'] + "\n",
        "rejected": example['rejected'] + "\n",
    }

#Spliting the data in 95%(train) and 5%(eval)
train_ds = load_dataset('argilla/distilabel-intel-orca-dpo-pairs', split='train[:95%]')
eval_ds = load_dataset('argilla/distilabel-intel-orca-dpo-pairs', split='train[95%:]')

# Save columns
train_original_columns = train_ds.column_names
eval_original_columns = eval_ds.column_names

# Format dataset
train_ds = train_ds.map(
    format_data,
    remove_columns=train_original_columns
)
eval_ds = eval_ds.map(
    format_data,
    remove_columns=eval_original_columns
)

#Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id =  tokenizer.unk_token_id
tokenizer.padding_side = 'left'

#Load the model
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=getattr(torch, "float16"), load_in_4bit=True, device_map={"": 0}, trust_remote_code=True)
model = prepare_model_for_kbit_training(model)

#Configure the pad token in the model
model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False

#Load the Reference Model
model_ref = AutoModelForCausalLM.from_pretrained(model_name,load_in_4bit=True, torch_dtype=getattr(torch, "float16"), trust_remote_code=True, device_map={"": 0})

# LoRA configuration
peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules= ["q_proj","k_proj","v_proj", "dense"]
)

# Training arguments
training_arguments = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="steps",
        do_eval=True,
        optim="paged_adamw_32bit",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=2,
        log_level="debug",
        save_steps=10,
        logging_steps=1,
        learning_rate=5e-5,
        eval_steps=20,
        #num_train_epochs=1,
        max_steps=500,
        warmup_steps=100,
        bf16=True,
        lr_scheduler_type="cosine"
        # report_to="wandb",
)

# Create DPO trainer
trainer = DPOTrainer(
    model,
    model_ref,
    args=training_arguments,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    peft_config=peft_config,
    beta=0.1,
    max_prompt_length=1024,
    max_length=1536,
)

# START DPO fine-tuning
trainer.train()

# Save the adapter
trainer.model.save_pretrained("final_checkpoint")
tokenizer.save_pretrained("final_checkpoint")

#Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Load the base model
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True,device_map={"": 0})

#Merge base model with the adapter
model = PeftModel.from_pretrained(base_model, "final_checkpoint")
model = model.merge_and_unload()

#Save the model and the tokenizer
model.save_pretrained("./dpo-phi2")
tokenizer.save_pretrained("./dpo-phi2")