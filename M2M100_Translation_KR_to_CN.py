
import torch
from datasets import load_dataset, load_metric
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer

# Load BLEU metric
bleu_metric = load_metric("bleu")

# Load and preprocess dataset
data_files = {
    "train": "path_to_train.csv",
    "validation": "path_to_validation.csv"
}
datasets = load_dataset("csv", data_files=data_files)

tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="ko", tgt_lang="zh")


def preprocess_function(examples):
    # Example of specifying source and target languages directly
    inputs = ["translate Korean to Chinese: " + ex for ex in examples["korean"]]
    targets = [ex for ex in examples["chinese"]]

    # Tokenize inputs and targets together using text_target for the targets
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding="max_length")

    # Set the labels for training
    # import ipdb
    # ipdb.set_trace()

    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs


tokenized_datasets = datasets.map(preprocess_function, batched=True)

# Initialize the Model
model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
model.config.decoder_start_token_id = tokenizer.get_lang_id("zh")


# Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

# Set Up Trainer


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = [[label] for label in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_flags=True)

    # Clean some special tokens not handled by the tokenizer
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]

    bleu_result = bleumetric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": bleu_result["score"]}


trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train the Model
trainer.train()
