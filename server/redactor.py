"""
redactor.py  —  PII detection & redaction engine
=================================================
Strategy (hybrid):
  1. Regex patterns  — high-precision, structured PII (email, IP, credit card…)
  2. BiLSTM-CRF      — context-aware NER for NAME, PASSWORD, API_KEY in free text
  3. Results are merged; regex wins on overlapping spans.

The redact() function returns:
  RedactionResult(
      original_text   : str
      redacted_text   : str
      entities        : list[Entity]   ← what was found + replaced
  )
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch

# ── Data classes ──────────────────────────────────────────────────────────────
@dataclass
class Entity:
    start:       int
    end:         int
    entity_type: str
    original:    str
    placeholder: str
    source:      str   # "regex" or "model"
    confidence:  float = 1.0


@dataclass
class RedactionResult:
    original_text:  str
    redacted_text:  str
    entities:       List[Entity] = field(default_factory=list)
    latency_ms:     float = 0.0
    model_type:     str   = "regex"


# ── Regex patterns ────────────────────────────────────────────────────────────
REGEX_PATTERNS: List[Tuple[str, str]] = [
    ("EMAIL",
     r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    ("IP_ADDRESS",
     r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
    ("CREDIT_CARD",
     r"\b(?:\d{4}[- ]){3}\d{4}\b|\b3[47]\d{2}[- ]\d{6}[- ]\d{5}\b"),
    ("SSN",
     r"\b\d{3}-\d{2}-\d{4}\b"),
    ("PHONE",
     r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
     r"|\+91[-\s]?\d{10}\b"),
    ("API_KEY",
     r"\b(?:sk-|AKIA|ghp_|xoxb-|pk_live_|rk_live_)[A-Za-z0-9]{10,}\b"),
    ("PASSWORD",
     r"(?i)(?:password|passwd|pwd)\s*[=:]\s*\S+"),
]

_compiled = [(name, re.compile(pat)) for name, pat in REGEX_PATTERNS]


def _regex_detect(text: str) -> List[Entity]:
    entities: List[Entity] = []
    counts: Dict[str, int] = {}
    for etype, pattern in _compiled:
        for m in pattern.finditer(text):
            counts[etype] = counts.get(etype, 0) + 1
            ph = f"[{etype}_{counts[etype]}]"
            # For PASSWORD=value, only redact the value part
            original = m.group()
            if etype == "PASSWORD":
                # keep "password=" prefix, redact only the value
                kv_match = re.match(
                    r"(?i)(password|passwd|pwd\s*[=:]\s*)(\S+)", original
                )
                if kv_match:
                    val_start = m.start() + kv_match.start(2)
                    val_end   = m.start() + kv_match.end(2)
                    entities.append(Entity(
                        start=val_start, end=val_end,
                        entity_type=etype,
                        original=kv_match.group(2),
                        placeholder=ph,
                        source="regex",
                    ))
                    continue
            entities.append(Entity(
                start=m.start(), end=m.end(),
                entity_type=etype,
                original=original,
                placeholder=ph,
                source="regex",
            ))
    return entities


# ── BiLSTM-CRF inference ──────────────────────────────────────────────────────
class ModelRedactor:
    """
    Lazy-loaded BiLSTM-CRF inference engine.
    Falls back to regex-only if model is not yet trained.
    """
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self._loaded   = False
        self.model     = None
        self.word_vocab: Optional[Dict] = None
        self.char_vocab: Optional[Dict] = None
        self.id2label:   Optional[Dict] = None
        self._try_load()

    def _try_load(self):
        required = [
            "best_model.pt", "word_vocab.json",
            "char_vocab.json", "label_vocab.json", "model_config.json",
        ]
        if not all(os.path.exists(os.path.join(self.model_dir, f)) for f in required):
            print("⚠️   BiLSTM-CRF weights not found — using regex-only mode.")
            return

        # Dynamically import to avoid top-level dependency when model not trained
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from model.bilstm_crf import BiLSTMCRF
        from model.data_loader import (
            PAD_IDX, UNK_IDX, MAX_WORD_LEN,
            encode_word, encode_chars,
        )

        with open(os.path.join(self.model_dir, "model_config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(self.model_dir, "word_vocab.json"))  as f:
            self.word_vocab = json.load(f)
        with open(os.path.join(self.model_dir, "char_vocab.json"))  as f:
            self.char_vocab = json.load(f)
        with open(os.path.join(self.model_dir, "label_vocab.json")) as f:
            label_vocab = json.load(f)

        self.id2label = {v: k for k, v in label_vocab.items()}
        self._encode_word  = encode_word
        self._encode_chars = encode_chars
        self._PAD_IDX      = PAD_IDX
        self._MAX_WORD_LEN = MAX_WORD_LEN

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = BiLSTMCRF(
            vocab_size      = cfg["vocab_size"],
            char_vocab_size = cfg["char_vocab_size"],
            num_tags        = cfg["num_tags"],
            word_embed_dim  = cfg["word_embed_dim"],
            char_embed_dim  = cfg["char_embed_dim"],
            char_cnn_out    = cfg["char_cnn_out"],
            hidden_size     = cfg["hidden_size"],
            num_lstm_layers = cfg["num_lstm_layers"],
            dropout         = cfg.get("dropout", 0.3),
        ).to(self.device)
        state = torch.load(
            os.path.join(self.model_dir, "best_model.pt"),
            map_location=self.device,
        )
        self.model.load_state_dict(state)
        self.model.eval()
        self._loaded = True
        print("✅  BiLSTM-CRF model loaded.")

    @property
    def available(self) -> bool:
        return self._loaded

    def predict(self, text: str) -> List[Tuple[str, str]]:
        """Returns [(word, label), …] for the input text."""
        if not self._loaded or self.model is None:
            return []

        # Simple whitespace tokenisation (same as training)
        words = text.split()
        if not words:
            return []

        word_ids  = [self._encode_word(w, self.word_vocab) for w in words]
        char_ids  = [self._encode_chars(w, self.char_vocab) for w in words]
        max_wl    = self._MAX_WORD_LEN

        words_t = torch.tensor([word_ids],  dtype=torch.long).to(self.device)
        chars_t = torch.tensor([char_ids],  dtype=torch.long).to(self.device)
        mask_t  = torch.ones(1, len(words), dtype=torch.float).to(self.device)

        preds = self.model.predict(words_t, chars_t, mask_t)
        labels = [self.id2label.get(idx, "O") for idx in preds[0]]
        return list(zip(words, labels))


def _model_detect(text: str, model_redactor: "ModelRedactor") -> List[Entity]:
    """Convert model token predictions to Entity spans."""
    if not model_redactor.available:
        return []

    token_labels = model_redactor.predict(text)
    entities: List[Entity] = []
    counts: Dict[str, int] = {}

    # Rebuild character offsets
    pos = 0
    current_entity: Optional[Dict] = None

    for word, label in token_labels:
        # Find word start in text
        idx = text.find(word, pos)
        if idx == -1:
            idx = pos
        word_end = idx + len(word)

        if label.startswith("B-"):
            if current_entity:
                etype = current_entity["type"]
                counts[etype] = counts.get(etype, 0) + 1
                ph = f"[{etype}_{counts[etype]}]"
                entities.append(Entity(
                    start=current_entity["start"],
                    end=current_entity["end"],
                    entity_type=etype,
                    original=text[current_entity["start"]:current_entity["end"]],
                    placeholder=ph,
                    source="model",
                ))
            current_entity = {"start": idx, "end": word_end, "type": label[2:]}

        elif label.startswith("I-") and current_entity:
            current_entity["end"] = word_end

        else:
            if current_entity:
                etype = current_entity["type"]
                counts[etype] = counts.get(etype, 0) + 1
                ph = f"[{etype}_{counts[etype]}]"
                entities.append(Entity(
                    start=current_entity["start"],
                    end=current_entity["end"],
                    entity_type=etype,
                    original=text[current_entity["start"]:current_entity["end"]],
                    placeholder=ph,
                    source="model",
                ))
            current_entity = None

        pos = word_end

    if current_entity:
        etype = current_entity["type"]
        counts[etype] = counts.get(etype, 0) + 1
        ph = f"[{etype}_{counts[etype]}]"
        entities.append(Entity(
            start=current_entity["start"],
            end=current_entity["end"],
            entity_type=etype,
            original=text[current_entity["start"]:current_entity["end"]],
            placeholder=ph,
            source="model",
        ))

    return entities


# ── Merge + apply redactions ──────────────────────────────────────────────────
# Entity types to ignore (not considered PII for this firewall)
_IGNORED_ENTITY_TYPES = {"NAME"}

# Patterns to validate model-detected entities (reject false positives)
_PHONE_PATTERN = re.compile(
    r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}"
    r"|\+91[-\s]?\d{10}"
    r"|\d{10,}"
)
_EMAIL_PATTERN = re.compile(
    r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"
)


def _filter_entities(entities: List[Entity]) -> List[Entity]:
    """Remove ignored types and model false-positives."""
    filtered = []
    for e in entities:
        if e.entity_type in _IGNORED_ENTITY_TYPES:
            continue
        # Model tagged as PHONE but doesn't look like a phone number
        if e.entity_type == "PHONE" and e.source == "model":
            if not _PHONE_PATTERN.fullmatch(e.original.strip()):
                continue
        # Model tagged as EMAIL but doesn't contain @
        if e.entity_type == "EMAIL" and e.source == "model":
            if not _EMAIL_PATTERN.fullmatch(e.original.strip()):
                continue
        filtered.append(e)
    return filtered


def _merge_entities(regex_ents: List[Entity], model_ents: List[Entity]) -> List[Entity]:
    """Merge lists, regex wins on overlap, then filter out non-PII."""
    regex_spans = [(e.start, e.end) for e in regex_ents]

    def overlaps(e: Entity) -> bool:
        for rs, re_ in regex_spans:
            if not (e.end <= rs or e.start >= re_):
                return True
        return False

    merged = list(regex_ents)
    for e in model_ents:
        if not overlaps(e):
            merged.append(e)

    merged = _filter_entities(merged)
    return sorted(merged, key=lambda x: x.start)


def _apply_redaction(text: str, entities: List[Entity]) -> str:
    result = []
    pos = 0
    for e in sorted(entities, key=lambda x: x.start):
        result.append(text[pos:e.start])
        result.append(e.placeholder)
        pos = e.end
    result.append(text[pos:])
    return "".join(result)


# ── Public API ────────────────────────────────────────────────────────────────
_model_instance: Optional[ModelRedactor] = None

def get_model(model_dir: str) -> ModelRedactor:
    global _model_instance
    if _model_instance is None:
        _model_instance = ModelRedactor(model_dir)
    return _model_instance


def redact(
    text:     str,
    model_dir: str = "model/saved",
    use_model: bool = True,
) -> RedactionResult:
    t0 = time.time()

    regex_entities = _regex_detect(text)

    model_entities: List[Entity] = []
    if use_model:
        mr = get_model(model_dir)
        model_entities = _model_detect(text, mr)
        model_type = "scratch" if mr.available else "regex"
    else:
        model_type = "regex"

    all_entities = _merge_entities(regex_entities, model_entities)
    redacted     = _apply_redaction(text, all_entities)
    latency_ms   = (time.time() - t0) * 1000

    return RedactionResult(
        original_text = text,
        redacted_text = redacted,
        entities      = all_entities,
        latency_ms    = latency_ms,
        model_type    = model_type,
    )
