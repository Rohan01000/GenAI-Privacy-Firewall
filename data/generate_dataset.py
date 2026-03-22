"""
generate_dataset.py  —  Synthetic PII corpus generator
========================================================
Outputs:
  data/pii_raw.csv          raw PII values (reference / inspection)
  data/pii_ner_dataset.csv  CoNLL-style NER tokens with BIO labels
"""

import csv
import os
import random
import re
import uuid
from collections import Counter
from faker import Faker

fake = Faker()
random.seed(42)
Faker.seed(42)

# ── Sentence templates ────────────────────────────────────────────────────────
TEMPLATES = [
    "Hi my name is {NAME} and my email is {EMAIL}",
    "Please contact {NAME} at {EMAIL} for further details",
    "My phone number is {PHONE} call me anytime",
    "You can reach me at {PHONE} or email {EMAIL}",
    "My credit card number is {CREDIT_CARD} expires next year",
    "I paid using card {CREDIT_CARD} but it got declined",
    "My IP address is {IP_ADDRESS} and I cannot connect",
    "The server at {IP_ADDRESS} is down please check",
    "My API key is {API_KEY} why is it returning 401",
    "Use the token {API_KEY} to authenticate the request",
    "The database password is {PASSWORD} please update it",
    "Login with password {PASSWORD} for admin access",
    "My SSN is {SSN} and I need help with my account",
    "Social security number {SSN} on the form is wrong",
    "Hi I am {NAME} my SSN is {SSN} and email is {EMAIL}",
    "Contact {NAME} at {EMAIL} her phone is {PHONE}",
    "Server {IP_ADDRESS} uses password {PASSWORD} for ssh",
    "API key {API_KEY} and card {CREDIT_CARD} are compromised",
    "User {NAME} from {IP_ADDRESS} needs access",
    "My email {EMAIL} and phone {PHONE} are linked to card {CREDIT_CARD}",
    "Please reset the password {PASSWORD} for user {NAME}",
    "Credentials: email={EMAIL} password={PASSWORD}",
    "The key {API_KEY} was issued to {NAME}",
    "Call {NAME} at {PHONE} his card is {CREDIT_CARD}",
    "IP {IP_ADDRESS} attempted login for {EMAIL}",
    "Debug: db.connect host=localhost password={PASSWORD}",
    "AWS key {API_KEY} is leaking in the logs",
    "My number is {PHONE} and I forgot my card {CREDIT_CARD}",
    "Send the invoice to {EMAIL} from {NAME}",
    "User {NAME} has SSN {SSN} and lives at unknown address",
    "The token is {API_KEY} use it before it expires",
    "I got an email from {EMAIL} claiming to be {NAME}",
    "Server password is {PASSWORD} for IP {IP_ADDRESS}",
    "Card number {CREDIT_CARD} was used by {NAME}",
    "My email is {EMAIL} and my API key is {API_KEY}",
    "Hi there I am {NAME}",
    "Reset my password {PASSWORD} account {EMAIL}",
    "Contact support using phone {PHONE} or email {EMAIL}",
    "My card {CREDIT_CARD} was charged incorrectly",
    "Key {API_KEY} expired please issue a new one",
    "The host {IP_ADDRESS} is unreachable",
    "Employee {NAME} SSN {SSN} needs benefits update",
    "Forgot password {PASSWORD} please help me",
    "Billing email is {EMAIL} and phone is {PHONE}",
    "Deploy to {IP_ADDRESS} using token {API_KEY}",
    "Name {NAME} email {EMAIL} phone {PHONE} card {CREDIT_CARD}",
    # Clean (no PII) sentences for balance
    "What is the weather today",
    "Can you help me debug this Python code",
    "Tell me about machine learning models",
    "The quick brown fox jumps over the lazy dog",
    "How do neural networks learn representations",
    "Explain the concept of backpropagation",
    "What is the capital of France",
    "Help me write a function to sort a list",
]

# ── PII generators ─────────────────────────────────────────────────────────────
def gen_email():
    return fake.email()

def gen_phone():
    return random.choice([
        fake.numerify("###-###-####"),
        fake.numerify("(###) ###-####"),
        fake.numerify("+1-###-###-####"),
        fake.numerify("+91-##########"),
    ])

def gen_credit_card():
    return random.choice([
        fake.numerify("4###-####-####-####"),
        fake.numerify("5###-####-####-####"),
        fake.numerify("3###-######-#####"),
    ])

def gen_password():
    return (
        fake.lexify("????").capitalize()
        + str(random.randint(10, 99))
        + random.choice("!@#$%^&*")
        + fake.lexify("??").upper()
    )

def gen_ip():
    return fake.ipv4()

def gen_api_key():
    prefix = random.choice(["sk-", "AKIA", "ghp_", "xoxb-", "pk_live_", "rk_live_"])
    return prefix + uuid.uuid4().hex[:20].upper()

def gen_ssn():
    return fake.numerify("###-##-####")

def gen_name():
    return fake.name()

GENERATORS = {
    "EMAIL":       gen_email,
    "PHONE":       gen_phone,
    "CREDIT_CARD": gen_credit_card,
    "PASSWORD":    gen_password,
    "IP_ADDRESS":  gen_ip,
    "API_KEY":     gen_api_key,
    "SSN":         gen_ssn,
    "NAME":        gen_name,
}

# ── Fill template → (sentence, entity_spans) ──────────────────────────────────
def fill_template(template: str):
    sentence = template
    entities = []
    offset = 0
    for m in re.finditer(r"\{(\w+)\}", template):
        label = m.group(1)
        if label not in GENERATORS:
            continue
        value = GENERATORS[label]()
        adj_start = m.start() + offset
        adj_end   = adj_start + len(value)
        entities.append((adj_start, adj_end, label))
        placeholder_len = m.end() - m.start()
        sentence = sentence[:adj_start] + value + sentence[adj_start + placeholder_len:]
        offset  += len(value) - placeholder_len
    return sentence, entities

# ── Tokenise and assign BIO labels ────────────────────────────────────────────
def tokenise_and_label(sentence: str, entities):
    char_labels = ["O"] * len(sentence)
    for start, end, etype in entities:
        for i in range(start, min(end, len(sentence))):
            char_labels[i] = f"B-{etype}" if i == start else f"I-{etype}"

    tokens = []
    i = 0
    while i < len(sentence):
        if sentence[i].isspace():
            i += 1
            continue
        j = i
        while j < len(sentence) and not sentence[j].isspace():
            j += 1
        span_lbls = char_labels[i:j]
        non_o = [l for l in span_lbls if l != "O"]
        if non_o:
            b = [l for l in non_o if l.startswith("B-")]
            label = b[0] if b else non_o[0]
        else:
            label = "O"
        tokens.append((sentence[i:j], label))
        i = j
    return tokens

# ── Main ───────────────────────────────────────────────────────────────────────
def generate(
    n_sentences: int = 5000,
    ner_path: str = "data/pii_ner_dataset.csv",
    raw_path: str = "data/pii_raw.csv",
):
    ner_rows, raw_rows = [], []

    for sid in range(n_sentences):
        tmpl = random.choice(TEMPLATES)
        sentence, entities = fill_template(tmpl)
        for word, label in tokenise_and_label(sentence, entities):
            ner_rows.append({"sentence_id": sid, "word": word, "label": label})
        for start, end, etype in entities:
            raw_rows.append({
                "sentence_id": sid,
                "entity_type": etype,
                "value":       sentence[start:end],
            })

    os.makedirs(os.path.dirname(ner_path) or ".", exist_ok=True)

    with open(ner_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sentence_id", "word", "label"])
        w.writeheader()
        w.writerows(ner_rows)

    with open(raw_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sentence_id", "entity_type", "value"])
        w.writeheader()
        w.writerows(raw_rows)

    label_dist = Counter(r["label"] for r in ner_rows)
    print(f"\n✅  Generated {n_sentences} sentences  →  {len(ner_rows)} token rows")
    print(f"   NER dataset : {ner_path}")
    print(f"   Raw PII ref : {raw_path}\n")
    print("📊  Label distribution:")
    for lbl, cnt in sorted(label_dist.items()):
        print(f"   {lbl:<28} {cnt:>7}")


if __name__ == "__main__":
    generate()
