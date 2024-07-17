import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Disable MPS fallback

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import torch

# Set device to CPU explicitly
device = torch.device("cpu")
# Alternatively, you can set to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the dataset with trust_remote_code argument
dataset = load_dataset("xsum", trust_remote_code=True)

# Initialize the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn").to(device)

# Preprocessing function
def preprocess_function(examples):
    inputs = examples["document"]
    targets = examples["summary"]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply preprocessing
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=["document", "summary"])

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Train the model
trainer.train()
