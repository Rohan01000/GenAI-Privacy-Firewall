import os
import json
import torch
import numpy as np
from typing import List, Tuple, Dict
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

# Import the identical data generator used for the Scratch Model
from ml_engine.scratch_model.dataset import generate_dataset

# ==========================================
# 1. DATASET PREPARATION & ALIGNMENT
# ==========================================
def align_labels_with_tokens(
    tokens: List[str], 
    tags: List[str], 
    tokenizer: AutoTokenizer, 
    label2idx: Dict[str, int]
) -> Tuple[List[int], List[int], List[int]]:
    
    tokenized_input = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        padding=False,
        max_length=512
    )
    
    word_ids = tokenized_input.word_ids()
    aligned_labels = []
    
    previous_word_idx = None
    for word_idx in word_ids:
        # Special tokens (like [CLS] and [SEP]) get word_idx None
        if word_idx is None:
            aligned_labels.append(-100)
        # Only the FIRST subword of a given word gets the actual label
        elif word_idx != previous_word_idx:
            aligned_labels.append(label2idx[tags[word_idx]])
        # Subsequent subwords get -100 to ignore them in the loss function
        else:
            aligned_labels.append(-100)
            
        previous_word_idx = word_idx

    return tokenized_input["input_ids"], tokenized_input["attention_mask"], aligned_labels

def prepare_huggingface_dataset(
    raw_data: List[Tuple[List[str], List[str]]], 
    tokenizer: AutoTokenizer, 
    label2idx: Dict[str, int]
) -> DatasetDict:
    
    input_ids_list = []
    attention_mask_list = []
    labels_list = []
    
    for tokens, tags in raw_data:
        input_ids, attention_mask, aligned_labels = align_labels_with_tokens(
            tokens, tags, tokenizer, label2idx
        )
        input_ids_list.append(input_ids)
        attention_mask_list.append(attention_mask)
        labels_list.append(aligned_labels)
        
    dataset_dict = {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    # Split: Train(70%), Val(15%), Test(15%)
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_ds = dataset.select(range(n_train))
    val_ds = dataset.select(range(n_train, n_train + n_val))
    test_ds = dataset.select(range(n_train + n_val, n_total))
    
    return DatasetDict({
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds
    })

# ==========================================
# 2. METRICS COMPUTATION
# ==========================================
def get_compute_metrics(idx2label: Dict[int, str]):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        # Remove ignored index (-100)
        true_predictions = [
            [idx2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        true_labels = [
            [idx2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        return {
            "precision": precision_score(true_labels, true_predictions),
            "recall": recall_score(true_labels, true_predictions),
            "f1": f1_score(true_labels, true_predictions),
        }
    return compute_metrics

# ==========================================
# 3. MAIN TRAINING SCRIPT
# ==========================================
if __name__ == "__main__":
    print("Initializing BERT Fine-Tuning Pipeline...")
    
    # 1. Generate Data
    print("Generating synthetic dataset (identical to scratch model)...")
    raw_data = generate_dataset()
    
    # 2. Build Label Vocabularies
    unique_tags = set()
    for _, tags in raw_data:
        unique_tags.update(tags)
        
    # Sort to ensure consistent mapping
    sorted_tags = sorted(list(unique_tags))
    
    label2idx = {tag: i for i, tag in enumerate(sorted_tags)}
    idx2label = {i: tag for i, tag in enumerate(sorted_tags)}
    
    # Save label mappings for inference
    os.makedirs("models/bert_ner", exist_ok=True)
    with open("models/bert_ner/label_map.json", "w") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=4)
        
    # 3. Initialize Tokenizer and Prepare Dataset
    print("Loading BERT tokenizer and aligning subwords...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    hf_dataset = prepare_huggingface_dataset(raw_data, tokenizer, label2idx)
    
    print(f"Dataset splits -> Train: {len(hf_dataset['train'])}, Val: {len(hf_dataset['validation'])}, Test: {len(hf_dataset['test'])}")
    
    # 4. Initialize Model
    print("Loading pre-trained BERT model for Token Classification...")
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=len(label2idx),
        id2label=idx2label,
        label2id=label2idx
    )
    
    # 5. Training Arguments
    device_has_cuda = torch.cuda.is_available()
    
    training_args = TrainingArguments(
        output_dir="models/bert_ner",
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_steps=200,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
        logging_dir=None,
        logging_steps=50,
        fp16=True if device_has_cuda else False,
        report_to="none" # Disable external logging like WandB for this standalone script
    )
    
    # 6. Data Collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset["train"],
        eval_dataset=hf_dataset["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=get_compute_metrics(idx2label)
    )
    
    print("\nStarting Training Loop...\n")
    trainer.train()
    
    print("\nEvaluating on Test Set...")
    test_results = trainer.evaluate(hf_dataset["test"])
    print("Test Set Metrics:")
    for key, value in test_results.items():
        if key.startswith("eval_"):
            print(f"  {key[5:]:>15}: {value:.4f}")
            
    # 8. Save Final Artifacts
    print("\nSaving final model and tokenizer...")
    trainer.save_model("models/bert_ner")
    tokenizer.save_pretrained("models/bert_ner")
    
    print("\nBERT fine-tuning complete. Model saved to models/bert_ner")