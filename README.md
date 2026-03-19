# 🛡️ GenAI Privacy Firewall

An enterprise-grade, ML-powered proxy firewall designed to secure Large Language Model (LLM) interactions.

This application acts as a secure middleware layer that intercepts outgoing LLM prompts, automatically detects Personally Identifiable Information (PII) or sensitive data, redacts it, and forwards the sanitized prompt to your target LLM.

---

## 🎯 Why This Matters

Modern applications frequently send user data to LLM APIs (e.g., OpenAI, Claude).
Without safeguards, sensitive information like emails, phone numbers, or API keys can leak.

**GenAI Privacy Firewall prevents this by acting as a protective layer between your application and LLMs.**

---

## ✨ Core Features

* **Hybrid PII Detection Engine**
  Combines rule-based detection (Microsoft Presidio) with ML-based models (BERT / LSTM) for high accuracy.

* **Smart Result Merging**
  Prioritizes deterministic matches and supplements them with ML predictions to reduce false positives.

* **Multiple ML Backends**
  Switch between:

  * Custom NER (LSTM)
  * Pretrained BERT model

* **Async & Batch Processing**
  Built with FastAPI and asyncio for concurrent processing.

* **Enterprise Middleware**

  * Audit Logging
  * Request Timing
  * Request ID tracking

* **Rate Limiting**
  Prevents abuse using `slowapi`.

* **Admin Dashboard**
  Includes `/admin` routes and `/dashboard` for monitoring.

---

## 🧪 Example

**Input:**

```
Hi, my name is Rohan and my email is rohan@gmail.com. My API key is sk-123456.
```

**Output:**

```
Hi, my name is [PERSON_1] and my email is [EMAIL_1]. My API key is [API_KEY_1].
```

---

## 🏗️ Architecture

The core component is the `CombinedDetector`:

1. **Concurrent Execution**
   Runs rule-based (Presidio) and ML models in parallel.

2. **Rule-Based Priority**
   Structured data (IPs, keys, SSNs) treated as ground truth.

3. **ML Supplementation**
   Contextual entities added only when non-overlapping.

4. **Deduplication & Sorting**
   Final entities cleaned before redaction.

---

## 📂 Project Structure

```
genai-privacy-firewall/
├── main.py
├── requirements.txt
├── .env.example
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py
├── proxy/
│   ├── __init__.py
│   ├── proxy_handler.py
│   ├── middleware.py
│   └── admin_routes.py
├── ml_engine/
│   ├── __init__.py
│   ├── scratch_model/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── embeddings.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── inference.py
│   ├── bert_model/
│   │   ├── __init__.py
│   │   ├── train_bert.py
│   │   └── bert_inference.py
│   ├── rule_based_detector.py
│   ├── combined_detector.py
│   ├── redactor.py
│   └── model_comparison.py
├── dashboard/
│   └── index.html
├── models/
│   └── .gitkeep
├── demo/
│   └── run_demo.py
└── tests/
    ├── __init__.py
    ├── test_redactor.py
    ├── test_detector.py
    └── test_integration.py
```

---

## 🚀 Getting Started

### Prerequisites

* Python 3.10+
* PyTorch
* Transformers
* SpaCy

---

### 1️⃣ Installation

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

---

### 2️⃣ Configuration

```bash
cp .env.example .env
```

#### Key Configurations

* `MODEL_TYPE` → `"scratch"` or `"bert"`
* `PROXY_PORT` → default `8000`
* `LOG_LEVEL` → `INFO / DEBUG`

---

### 3️⃣ Run the Firewall

#### Option A: Development Server

```bash
uvicorn main:app --reload --port 8000 --host 0.0.0.0
```

#### Option B: Demo Script

```bash
python  -m demo.run_demo
```

---

## 📡 API Reference

* `GET /health` → System health + model status
* `POST /v1/...` → Main proxy endpoints
* `/admin/...` → Admin routes
* `GET /dashboard` → Monitoring dashboard

---

## 🤖 Model Details

* Supports:

  * Custom NER (LSTM-based)
  * BERT-based transformer model

* Designed to detect:

  * Names (**PERSON**)
  * Emails (**EMAIL**)
  * Phone Numbers (**PHONE**)
  * SSNs (**SSN**)
  * Credit Cards (**CREDIT_CARD**)
  * API Keys (**API_KEY**)
  * Passwords (**PASSWORD**)
  * IP Addresses (**IP_ADDRESS**)
  * Internal Org Names (**ORG_INTERNAL**)
  * Proprietary Code (**PROPRIETARY_CODE**)

* Trained on a custom labeled dataset with 10 entity classes

* Hybrid approach improves detection accuracy and reduces false positives

---

## 🛠️ Tech Stack

* **Backend:** FastAPI, Uvicorn
* **ML/NLP:** PyTorch, HuggingFace Transformers, SpaCy
* **PII Detection:** Microsoft Presidio
* **Utilities:** slowapi, pydantic-settings

---

## 🧪 Testing

```bash
pytest tests/
```

Includes:

* Detector tests
* Redactor tests
* Integration tests

---

## 🤝 Sharing Models

Model files are **not included in the repository**.

To use the project:

1. Download models from shared storage (Drive / HuggingFace)
2. Place them inside the `models/` directory

---

## 🚀 Future Improvements

* Model performance benchmarking
* Docker containerization
* Cloud deployment (AWS/GCP)

---

## 👥 Contributors

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

## 📌 Author

**Rohan Kumar** 🚀

---

## ⭐ If you found this useful

Give it a star ⭐ and feel free to contribute!
