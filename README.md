# GenAI Privacy Firewall

A middleware proxy that sits between your chat UI and Ollama. Every prompt passes through a **BiLSTM-CRF + regex** PII detection pipeline — sensitive data is redacted before it ever reaches the LLM. The model trains and runs entirely on your local machine.

```
You type     →  "My email is rohan@gmail.com and my API key is sk-abc123"
                              ↓  firewall intercepts
Ollama gets  →  "My email is [EMAIL_1] and my API key is [API_KEY_1]"
```

---

## Why This Matters

Modern apps send user data to LLMs. Without safeguards, sensitive information like emails, SSNs, or API keys can leak to external servers.

**GenAI Privacy Firewall prevents this** — it sits as a proxy, scrubs PII from every prompt, and only forwards the sanitized version to the LLM.

---

## Core Features

- **Hybrid PII Detection** — regex patterns (high precision for structured data) + BiLSTM-CRF neural model (catches contextual PII)
- **Locally Trained Model** — the BiLSTM-CRF trains on your machine using synthetic data, no cloud APIs
- **Real-time Redaction** — PII is replaced with placeholders like `[EMAIL_1]`, `[SSN_1]` before the LLM ever sees it
- **Audit Log** — every request stored with original vs redacted side-by-side
- **Live Dashboard** — stats, charts, entity breakdown, request history
- **Chat UI** — built-in chat interface that shows redaction details after each response
- **Rate Limiting** — per-IP request throttling
- **Simple API** — `POST /chat` with JSON messages

---

## Example

**Input:**
```
Hi, my email is rohan@gmail.com. My API key is sk-123456abc.
```

**What Ollama receives:**
```
Hi, my email is [EMAIL_1]. My API key is [API_KEY_1].
```

**Detected entities:** EMAIL, API_KEY — visible in the audit log and chat UI.

---

## Detection Architecture

```
Incoming prompt
      │
      ├──► Regex detector (always runs)
      │    EMAIL, IP_ADDRESS, CREDIT_CARD, SSN, PHONE, API_KEY, PASSWORD
      │
      ├──► BiLSTM-CRF model (if trained model exists)
      │    Char-CNN + Word Embeddings → BiLSTM → CRF decoding
      │
      └──► Merge + Filter
           • Regex wins on overlapping spans
           • Placeholders applied: value → [TYPE_N]
           • Sanitised prompt forwarded to Ollama
```

If the trained model doesn't exist, the server runs in **regex-only mode** automatically.

---

## How the Model Works

The BiLSTM-CRF is a locally trained neural network for Named Entity Recognition:

1. **Char-CNN** — learns character-level features (catches patterns like `sk-abc123` even if never seen in training)
2. **Word Embeddings** — learned from the training vocabulary
3. **Bidirectional LSTM** — reads context left-to-right and right-to-left
4. **CRF Layer** — ensures label sequence consistency (no `I-EMAIL` without `B-EMAIL` before it)

Trained weights are stored in `best_model.pt` (~13MB, ~3.4M parameters). The model loads once on first request and stays in memory.

---

## Project Structure

```
GenAI-Privacy-Firewall/
│
├── data/
│   └── generate_dataset.py      ← Faker: generates 5000 labelled PII sentences
│
├── model/
│   ├── bilstm_crf.py            ← BiLSTM + CRF + Char-CNN (PyTorch)
│   ├── data_loader.py           ← Dataset, vocab builders, encoding helpers
│   ├── train.py                 ← Training loop → saves to model/saved/
│   └── saved/                   ← Trained model artifacts
│       ├── best_model.pt
│       ├── word_vocab.json
│       ├── char_vocab.json
│       ├── label_vocab.json
│       └── model_config.json
│
├── server/
│   ├── config.py                ← Settings loaded from .env
│   ├── redactor.py              ← Hybrid PII engine: regex + BiLSTM-CRF
│   └── main.py                  ← FastAPI server: proxy, stats, audit, serves frontend
│
├── frontend/
│   ├── firewall_chat.html       ← Chat UI
│   ├── index.html               ← Dashboard (stats, charts, request table)
│   └── audit.html               ← Audit log (original vs redacted side-by-side)
│
├── .env.example                 ← Copy → .env, edit as needed
├── requirements.txt
├── setup.py                     ← Setup & launcher script
└── README.md
```

---

## Tech Stack

- **Backend:** FastAPI, Uvicorn, httpx
- **ML/NLP:** PyTorch (BiLSTM-CRF, custom Char-CNN, CRF from scratch)
- **Data Generation:** Faker
- **Frontend:** Vanilla HTML/CSS/JS, Chart.js
- **LLM Backend:** Ollama (local)

---

## Getting Started

### Prerequisites

| Tool   | Minimum | Notes |
|--------|---------|-------|
| Python | 3.10    | 3.11+ recommended |
| pip    | 23+     | `pip install --upgrade pip` |
| Ollama | any     | https://ollama.com |

### 1. Configure environment

```bash
cp .env.example .env
```

Edit `.env` — set `DEFAULT_MODEL` to whatever model you have pulled in Ollama (e.g. `llama3`).

### 2. Full setup

```bash
python setup.py setup
```

This runs three stages:

| Stage | What happens |
|-------|-------------|
| Install deps | `pip install -r requirements.txt` |
| Generate data | Creates 5000 synthetic PII sentences using Faker |
| Train model | Trains BiLSTM-CRF locally (~5-10 min CPU, <2 min GPU) |

### 3. Start Ollama + the firewall

```bash
# Terminal 1
ollama serve

# Terminal 2
python setup.py server
```

### 4. Open the UI

| Page | URL |
|------|-----|
| Chat | http://localhost:8000/chat.html |
| Dashboard | http://localhost:8000/dashboard.html |
| Audit Log | http://localhost:8000/audit.html |

---

## Setup Commands

| Command | What it does |
|---------|-------------|
| `python setup.py setup` | Full setup: install deps + generate data + train model |
| `python setup.py data` | Generate PII dataset only |
| `python setup.py train` | Train BiLSTM-CRF model only |
| `python setup.py server` | Start the FastAPI server (default) |

---

## Frontend Pages

### Chat (`/chat.html`)
Type a message (or use a quick prompt). The firewall scans for PII, redacts it, forwards to Ollama. The response shows with a redaction details box — what was found and what it was replaced with. Live stats in the sidebar.

### Dashboard (`/dashboard.html`)
4 stat cards (total requests, entities redacted, blocked, avg latency). Donut chart for entity type breakdown. Line chart for requests per hour. Live request log table.

### Audit Log (`/audit.html`)
Every intercepted request as a side-by-side card — original (PII in red) vs redacted (placeholders in green). Entity mapping strip at the bottom. Filter by: All / Redacted / Clean / entity type. Auto-refreshes every 5 seconds.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/chat` | Redacts PII from prompt, forwards to Ollama, returns response |
| `GET` | `/config` | Server config (default model, detection type) |
| `GET` | `/admin/stats` | Aggregate stats (requests, entities, latency, breakdown) |
| `GET` | `/admin/recent-requests` | Last 50 requests |
| `GET` | `/audit/requests?limit=100` | Full audit log with original + redacted prompts |
| `GET` | `/health` | Liveness check |

All endpoints are open — no authentication required.

---

## Configuration (`.env`)

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where Ollama is running |
| `DEFAULT_MODEL` | `llama3` | Model name sent to Ollama |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `RATE_LIMIT_PER_MIN` | `60` | Requests per minute per IP |
| `MODEL_DIR` | `model/saved` | Path to trained model artifacts |
| `CONFIDENCE_THRESHOLD` | `0.80` | Minimum confidence to redact (0.0–1.0) |
| `MODEL_TYPE` | `scratch` | `scratch` = BiLSTM-CRF + regex, `regex` = regex only |

---

## PII Types Detected

| Entity | Detection | Examples |
|--------|-----------|----------|
| EMAIL | Regex + Model | `user@example.com` |
| PHONE | Regex + Model | `+1-800-555-1234`, `(212) 555-9876` |
| CREDIT_CARD | Regex + Model | `4532-1234-5678-9012` |
| SSN | Regex + Model | `123-45-6789` |
| IP_ADDRESS | Regex + Model | `192.168.1.50` |
| API_KEY | Regex + Model | `sk-abc123…`, `AKIAIOSFODNN7…`, `ghp_…` |
| PASSWORD | Regex + Model | `password='Xk9#mNp2!@3'` |

---

## Adding a New PII Type

Example: adding `PASSPORT` detection.

1. Add a generator in `data/generate_dataset.py`: `def gen_passport(): …`
2. Add sentence templates using `{PASSPORT}` placeholder
3. Regenerate data: `python setup.py data`
4. Retrain model: `python setup.py train`
5. Add a regex pattern in `server/redactor.py` under `REGEX_PATTERNS`

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: torchcrf` | `pip install pytorch-crf` |
| `ModuleNotFoundError: faker` | `pip install faker` |
| `ModuleNotFoundError: dotenv` | `pip install python-dotenv` |
| Ollama returns 502 | Run `ollama serve` in a separate terminal |
| Model not loading | Run `python setup.py train` |
| Training OOM on CPU | Reduce `BATCH_SIZE` in `model/train.py` (default 32) |
| Pages show "Connection error" | Make sure server is running on port 8000 |

---

## Contributors

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/Rohan01000">
        <img src="https://github.com/Rohan01000.png" width="100px;" alt="Rohan"/>
        <br />
        <sub><b>Rohan Kumar</b></sub>
      </a>
      <br />
      <sub>@Rohan01000</sub>
    </td>

<td align="center">
  <a href="https://github.com/syren-12">
    <img src="https://github.com/syren-12.png" width="100px;" alt="Teammate2"/>
    <br />
    <sub><b> Shreya Chatterjee</b></sub>
  </a>
  <br />
  <sub>@syren-12</sub>
</td>

<td align="center">
    <a href="https://github.com/Pager-dot">
    <img src="https://github.com/Pager-dot.png" width="100px;" alt="Teammate3"/>
    <br />
    <sub><b> Paritosh</b></sub>
    </a>
    <br />
    <sub>@Pager-dot</sub>
</td>


<td align="center">
  <a href="https://github.com/SaswatRout05">
    <img src="https://github.com/SaswatRout05.png" width="100px;" alt="Teammate4"/>
    <br />
    <sub><b> Saswat Rout</b></sub>
  </a>
  <br />
  <sub>@SaswatRout05</sub>
</td>

<td align="center">
  <a href="https://github.com/saisohan-eng">
    <img src="https://github.com/saisohan-eng.png" width="100px;" alt="Teammate5"/>
    <br />
    <sub><b> Sai Sohan Rout</b></sub>
  </a>
  <br />
  <sub>@saisohan-eng</sub>
</td>

</table>

---

## Author

**Rohan Kumar**

---

## If you found this useful

Give it a star and feel free to contribute!
