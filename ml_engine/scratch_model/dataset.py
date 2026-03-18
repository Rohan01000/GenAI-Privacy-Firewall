import os
import json
import random
import re
import string
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. CONSTANTS & CONFIGURATION
# ==========================================
ENTITIES = [
    "PERSON", "EMAIL", "PHONE", "SSN", "API_KEY", 
    "PASSWORD", "CREDIT_CARD", "IP_ADDRESS", 
    "ORG_INTERNAL", "PROPRIETARY_CODE"
]

NUM_CLEAN = 400
NUM_SINGLE = 1000
NUM_MULTI = 600
TOTAL_SENTENCES = NUM_CLEAN + NUM_SINGLE + NUM_MULTI

# ==========================================
# 2. DATA GENERATORS
# ==========================================
def gen_person():
    firsts = ["John", "Alice", "Michael", "Sarah", "David", "Emma", "James", "Olivia", "Robert", "Sophia"]
    lasts = ["Smith", "Johnson", "Brown", "Williams", "Jones", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez"]
    return f"{random.choice(firsts)} {random.choice(lasts)}"

def gen_email():
    users = ["admin", "test.user", "j.doe", "support", "billing", "ceo", "dev.ops"]
    domains = ["example.com", "acme.org", "internal.net", "test.io", "company.co"]
    return f"{random.choice(users)}@{random.choice(domains)}"

def gen_phone():
    return f"{random.randint(200,999)}-{random.randint(200,999)}-{random.randint(1000,9999)}"

def gen_ssn():
    return f"{random.randint(100,999)}-{random.randint(10,99)}-{random.randint(1000,9999)}"

def gen_api_key():
    prefix = random.choice(["AKIA", "sk_live_", "pk_test_", "bearer_"])
    body = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(20, 40)))
    return prefix + body

def gen_password():
    chars = string.ascii_letters + string.digits + "!@#$%^&*"
    return ''.join(random.choices(chars, k=random.randint(8, 16)))

def gen_credit_card():
    return f"{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}"

def gen_ip_address():
    return f"{random.randint(1,255)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,255)}"

def gen_org_internal():
    names = ["Project Apollo", "Operation RedDawn", "AcmeCorp Internal", "Zeus Database", "Falcon X Architecture"]
    return random.choice(names)

def gen_proprietary_code():
    codes = [
        "def calculate_revenue(tax_rate): return base * tax_rate",
        "SELECT * FROM secure_users WHERE is_admin = 1;",
        "class AuthManager(Singleton):",
        "const DB_URI = 'mongodb://localhost:27017';",
        "git commit -m 'Fixed critical auth bypass'"
    ]
    return random.choice(codes)

GENERATORS = {
    "PERSON": gen_person,
    "EMAIL": gen_email,
    "PHONE": gen_phone,
    "SSN": gen_ssn,
    "API_KEY": gen_api_key,
    "PASSWORD": gen_password,
    "CREDIT_CARD": gen_credit_card,
    "IP_ADDRESS": gen_ip_address,
    "ORG_INTERNAL": gen_org_internal,
    "PROPRIETARY_CODE": gen_proprietary_code
}

# ==========================================
# 3. TEMPLATE ENGINE
# ==========================================
CLEAN_TEMPLATES = [
    "Can you summarize this public article?",
    "Write a poem about the ocean.",
    "How do I reverse a binary tree in Python?",
    "What is the capital of France?",
    "Give me a recipe for chocolate chip cookies.",
    "Explain quantum computing to a 5-year-old.",
    "Translate 'hello world' to Spanish.",
    "What's the weather like today?",
    "Create a workout plan for beginners.",
    "List the planets in the solar system."
]

SINGLE_TEMPLATES = [
    "Can you summarize the contract for [E1]?",
    "Debug this function, our DB password is [E1].",
    "My AWS key is [E1], why won't it connect?",
    "Analyze sales data for SSN [E1].",
    "Please send the report to [E1] by Friday.",
    "User logged in from [E1] at midnight.",
    "The payment failed for card [E1].",
    "Can you review the architecture for [E1]?",
    "I need help understanding this code: [E1]",
    "Call me back at [E1] when you are free."
]

MULTI_TEMPLATES = [
    "Contact [E1] at [E2] immediately.",
    "User [E1] logged in from [E2] using password [E3].",
    "The account with SSN [E1] and Card [E2] belongs to [E3].",
    "Please update [E1] regarding [E2] and cc [E3].",
    "Server at [E1] rejected the key [E2] for user [E3].",
    "Review [E1] and check if it exposes [E2]."
]

def generate_sentence_structure(template_type):
    if template_type == "clean":
        return [(random.choice(CLEAN_TEMPLATES), "O")]
    
    elif template_type == "single":
        template = random.choice(SINGLE_TEMPLATES)
        ent_type = random.choice(ENTITIES)
        parts = template.split("[E1]")
        ent_val = GENERATORS[ent_type]()
        return [(parts[0], "O"), (ent_val, ent_type), (parts[1], "O")]
        
    elif template_type == "multi":
        template = random.choice(MULTI_TEMPLATES)
        num_ents = template.count("[E")
        
        structure = []
        current_text = template
        for i in range(1, num_ents + 1):
            ent_type = random.choice(ENTITIES)
            ent_val = GENERATORS[ent_type]()
            
            parts = current_text.split(f"[E{i}]", 1)
            if parts[0]:
                structure.append((parts[0], "O"))
            structure.append((ent_val, ent_type))
            current_text = parts[1] if len(parts) > 1 else ""
            
        if current_text:
            structure.append((current_text, "O"))
        return structure

def tokenize_and_tag(structure):
    tokens = []
    tags = []
    
    for text, ent_type in structure:
        if not text.strip():
            continue
            
        # Tokenize by words and punctuation
        raw_tokens = re.findall(r"\w+|[^\w\s]", text)
        
        for i, token in enumerate(raw_tokens):
            tokens.append(token)
            if ent_type == "O":
                tags.append("O")
            else:
                if i == 0:
                    tags.append(f"B-{ent_type}")
                else:
                    tags.append(f"I-{ent_type}")
                    
    return tokens, tags

def generate_dataset():
    data = []
    
    # Generate distribution
    for _ in range(NUM_CLEAN):
        struct = generate_sentence_structure("clean")
        data.append(tokenize_and_tag(struct))
        
    for _ in range(NUM_SINGLE):
        struct = generate_sentence_structure("single")
        data.append(tokenize_and_tag(struct))
        
    for _ in range(NUM_MULTI):
        struct = generate_sentence_structure("multi")
        data.append(tokenize_and_tag(struct))
        
    random.shuffle(data)
    return data

# ==========================================
# 4. VOCABULARY BUILDER
# ==========================================
def build_vocabularies(dataset):
    word2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    char2idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
    label2idx = {"<PAD>": 0, "O": 1}
    
    for tokens, tags in dataset:
        for token, tag in zip(tokens, tags):
            if token not in word2idx:
                word2idx[token] = len(word2idx)
            for char in token:
                if char not in char2idx:
                    char2idx[char] = len(char2idx)
            if tag not in label2idx:
                label2idx[tag] = len(label2idx)
                
    idx2label = {v: k for k, v in label2idx.items()}
    
    # Save vocabularies
    os.makedirs("models", exist_ok=True)
    with open("models/vocab.json", "w", encoding="utf-8") as f:
        json.dump({"word2idx": word2idx, "char2idx": char2idx}, f, indent=2)
    with open("models/label_map.json", "w", encoding="utf-8") as f:
        json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)
        
    return word2idx, char2idx, label2idx

# ==========================================
# 5. PYTORCH DATASET & DATALOADER
# ==========================================
class NERDataset(Dataset):
    def __init__(self, data, word2idx, char2idx, label2idx):
        self.data = data
        self.word2idx = word2idx
        self.char2idx = char2idx
        self.label2idx = label2idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokens, tags = self.data[idx]
        
        word_indices = [self.word2idx.get(t, self.word2idx["<UNK>"]) for t in tokens]
        char_indices = [[self.char2idx.get(c, self.char2idx["<UNK>"]) for c in t] for t in tokens]
        label_indices = [self.label2idx[tag] for tag in tags]
        
        return word_indices, char_indices, label_indices, tokens

def custom_collate_fn(batch):
    max_words = max(len(item[0]) for item in batch)
    max_chars = max(max((len(chars) for chars in item[1]), default=0) for item in batch)
    
    word_batch = []
    char_batch = []
    label_batch = []
    orig_batch = []
    
    for w_idx, c_idx, l_idx, orig in batch:
        pad_len = max_words - len(w_idx)
        
        # Pad words and labels
        word_batch.append(w_idx + [0] * pad_len)
        label_batch.append(l_idx + [0] * pad_len)
        
        # Pad chars
        c_padded = []
        for chars in c_idx:
            c_padded.append(chars + [0] * (max_chars - len(chars)))
        for _ in range(pad_len):
            c_padded.append([0] * max_chars)
            
        char_batch.append(c_padded)
        orig_batch.append(orig)
        
    return (
        torch.tensor(word_batch, dtype=torch.long),
        torch.tensor(char_batch, dtype=torch.long),
        torch.tensor(label_batch, dtype=torch.long),
        orig_batch
    )

def create_dataloaders(batch_size=32, word2idx=None, char2idx=None, label2idx=None):
    dataset = generate_dataset()
    
    # Use saved vocabularies if provided, otherwise build fresh ones
    if word2idx is None or char2idx is None or label2idx is None:
        word2idx, char2idx, label2idx = build_vocabularies(dataset)
    
    # Calculate Splits
    n_total = len(dataset)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    train_data = dataset[:n_train]
    val_data = dataset[n_train:n_train + n_val]
    test_data = dataset[n_train + n_val:]
    train_dataset = NERDataset(train_data, word2idx, char2idx, label2idx)
    val_dataset = NERDataset(val_data, word2idx, char2idx, label2idx)
    test_dataset = NERDataset(test_data, word2idx, char2idx, label2idx)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    return train_loader, val_loader, test_loader, dataset, word2idx, label2idx

# ==========================================
# 6. STATISTICS REPORTING
# ==========================================
def print_statistics(dataset, word2idx, train_loader, val_loader, test_loader):
    total_sentences = len(dataset)
    total_tokens = sum(len(tokens) for tokens, _ in dataset)
    vocab_size = len(word2idx)
    
    entity_counts = defaultdict(int)
    total_entities = 0
    max_len = 0
    
    for tokens, tags in dataset:
        max_len = max(max_len, len(tokens))
        for tag in tags:
            if tag.startswith("B-"):
                ent_type = tag[2:]
                entity_counts[ent_type] += 1
                total_entities += 1
                
    avg_len = total_tokens / total_sentences if total_sentences else 0
    
    print("\n" + "="*50)
    print(" 📊 DATASET STATISTICS REPORT ")
    print("="*50)
    print(f"Total Sentences    : {total_sentences}")
    print(f"Total Tokens       : {total_tokens}")
    print(f"Vocabulary Size    : {vocab_size}")
    print(f"Average Sent Length: {avg_len:.2f} tokens")
    print(f"Max Sent Length    : {max_len} tokens")
    print("-" * 50)
    print("Splits:")
    print(f"  Train: {len(train_loader.dataset)} sentences (70%)")
    print(f"  Val  : {len(val_loader.dataset)} sentences (15%)")
    print(f"  Test : {len(test_loader.dataset)} sentences (15%)")
    print("-" * 50)
    print("Entity Distribution:")
    for ent, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_entities * 100) if total_entities else 0
        print(f"  {ent:<18}: {count:<5} ({pct:.1f}%)")
    print("="*50 + "\n")

if __name__ == "__main__":
    print("Generating synthetic dataset and building DataLoaders...")
    train_loader, val_loader, test_loader, full_dataset, word_vocab, label_vocab = create_dataloaders(batch_size=32)
    print_statistics(full_dataset, word_vocab, train_loader, val_loader, test_loader)
    print("Success! Vocabularies saved to models/ directory.")