import argparse, json, pickle, warnings, random
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset
from tqdm.auto import tqdm
tqdm.pandas()

# ──────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEVERITY_ORDER = {0: "low", 1: "medium", 2: "high", 3: "critical"}
W_KW, W_EMB, W_RT = 0.45, 0.35, 0.20
MODEL_NAME = "microsoft/deberta-v3-small"
EMB_MODEL  = "all-MiniLM-L6-v2"
MAX_LEN    = 256
OUTPUT_DIR = "sia_model"

# ──────────────────────────────────────────────────────────
# Keyword patterns
# ──────────────────────────────────────────────────────────
import re

CRITICAL_KW = [
    r"\bdata.?loss\b", r"\bsystem.?down\b", r"\bcritical\b", r"\bsecurity.?breach\b",
    r"\bransomware\b", r"\bcompromised\b", r"\bproduction.?outage\b", r"\bescalat\w+\b",
    r"\bregulatory\b", r"\bcompliance\b", r"\bsla.?breach\b", r"\bimpossible.?to.?work\b",
    r"\bentirely.?broken\b", r"\blegal\b", r"\bfinancial.?loss\b", r"\bceos?\b",
    r"\bcto\b", r"\bexecutive\b", r"\burgen\w+\b", r"\bassistance.?immediately\b"
]
HIGH_KW = [
    r"\bnot.?working\b", r"\bbroken\b", r"\berror\b", r"\bcrash\b", r"\bfail\w*\b",
    r"\bblocking\b", r"\bcannot.?access\b", r"\bunable.?to\b", r"\bdown\b",
    r"\bproductivity.?impact\b", r"\bmultiple.?users\b", r"\bwhole.?team\b",
    r"\bteam.?affected\b", r"\bstuck\b"
]
MED_KW = [
    r"\bslow\b", r"\bdegraded\b", r"\bintermittent\b", r"\boccasionally\b",
    r"\bperformance\b", r"\bdelay\b", r"\blag\b", r"\bsometimes\b"
]
LOW_KW = [
    r"\bquestion\b", r"\bhow.?to\b", r"\bwhere.?can\b", r"\bcan.?i\b",
    r"\bfeedback\b", r"\bsuggestion\b", r"\bfeature.?request\b", r"\bwhen.?will\b"
]
NEGATION = [r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bwithout\b", r"\bdon.?t\b", r"\bcan.?t\b"]

ANCHORS = {
    0: "A simple question about how to use a feature or general feedback.",
    1: "Occasional slowness or minor inconvenience that does not block work.",
    2: "A significant error or failure affecting individual productivity or access.",
    3: "Critical system outage, data loss, or security breach affecting multiple users immediately."
}


# ──────────────────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────────────────
def keyword_score(text: str) -> int:
    text = str(text).lower()
    crit = sum(1 for p in CRITICAL_KW if re.search(p, text))
    high = sum(1 for p in HIGH_KW if re.search(p, text))
    med  = sum(1 for p in MED_KW  if re.search(p, text))
    low  = sum(1 for p in LOW_KW  if re.search(p, text))
    neg  = sum(1 for p in NEGATION if re.search(p, text))
    if neg > 0: high += 1
    raw = crit * 3 + high * 2 + med * 1 - low * 0.5
    if raw >= 6:   return 3
    elif raw >= 3: return 2
    elif raw >= 1: return 1
    else:          return 0


def build_signals(df: pd.DataFrame) -> pd.DataFrame:
    print("Computing keyword scores...")
    df["combined_text"] = df.get("ticket_subject", "").fillna("") + " " + df.get("ticket_description", "").fillna("")
    df["kw_score"] = df["combined_text"].progress_apply(keyword_score)

    print("Computing embeddings...")
    embedder = SentenceTransformer(EMB_MODEL)
    embs = embedder.encode(df["combined_text"].tolist(), batch_size=256,
                           show_progress_bar=True, normalize_embeddings=True)
    anchor_embs = embedder.encode(list(ANCHORS.values()), normalize_embeddings=True)
    km = KMeans(n_clusters=4, random_state=SEED, n_init=10)
    clusters = km.fit_predict(embs)
    centroid_sims = cosine_similarity(km.cluster_centers_, anchor_embs)
    c2s = {c: int(np.argmax(centroid_sims[c])) for c in range(4)}
    df["emb_score"] = [c2s[c] for c in clusters]

    # Save km model
    with open(f"{OUTPUT_DIR}/km_model.pkl", "wb") as f: pickle.dump(km, f)
    with open(f"{OUTPUT_DIR}/cluster_to_sev.json", "w") as f: json.dump(c2s, f)

    le_chan = LabelEncoder()
    le_type = LabelEncoder()
    df["chan_enc"] = le_chan.fit_transform(df.get("ticket_channel", pd.Series(["email"]*len(df))).fillna("unknown"))
    df["type_enc"] = le_type.fit_transform(df.get("ticket_type",    pd.Series(["general"]*len(df))).fillna("unknown"))
    df["prio_enc"] = df["ticket_priority"].map(PRIORITY_ORDER)

    if "resolution_hours" in df.columns and df["resolution_hours"].notna().sum() > 100:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import train_test_split as tts
        mask = df["resolution_hours"].notna()
        X_r = df.loc[mask, ["prio_enc", "chan_enc", "type_enc", "kw_score"]]
        y_r = df.loc[mask, "resolution_hours"]
        reg = GradientBoostingRegressor(n_estimators=100, random_state=SEED)
        reg.fit(*tts(X_r, y_r, test_size=0.2, random_state=SEED)[:2])
        preds = reg.predict(df[["prio_enc", "chan_enc", "type_enc", "kw_score"]])
        q25, q50, q75 = np.percentile(preds, [25, 50, 75])
        df["rt_score"] = [3 if v>=q75 else (2 if v>=q50 else (1 if v>=q25 else 0)) for v in preds]
        with open(f"{OUTPUT_DIR}/regressor.pkl", "wb") as f: pickle.dump(reg, f)
    else:
        df["rt_score"] = df["prio_enc"]

    df["fused_severity"] = (
        W_KW * df["kw_score"] + W_EMB * df["emb_score"] + W_RT * df["rt_score"]
    ).round().astype(int).clip(0, 3)

    df["assigned_severity_num"] = df["ticket_priority"].map(PRIORITY_ORDER)
    df["severity_delta"] = df["fused_severity"] - df["assigned_severity_num"]
    df["pseudo_label"]   = (df["severity_delta"] != 0).astype(int)
    df["mismatch_type"]  = df["severity_delta"].apply(
        lambda d: "Hidden Crisis" if d > 0 else ("False Alarm" if d < 0 else "Consistent")
    )

    with open(f"{OUTPUT_DIR}/le_chan.pkl", "wb") as f: pickle.dump(le_chan, f)
    with open(f"{OUTPUT_DIR}/le_type.pkl", "wb") as f: pickle.dump(le_type, f)

    return df


class TicketDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(texts, truncation=True, padding=True,
                             max_length=max_len, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, i):
        return {k: v[i] for k, v in self.enc.items()} | {"labels": self.labels[i]}


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cw = class_weights
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss_fn = nn.CrossEntropyLoss(weight=self.cw.to(outputs.logits.device))
        loss = loss_fn(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "recall_consistent": recall_score(labels, preds, pos_label=0),
        "recall_mismatch":   recall_score(labels, preds, pos_label=1),
    }


def build_model_input(row):
    return (
        f"[PRIORITY:{row.get('ticket_priority','')}] "
        f"[CHANNEL:{row.get('ticket_channel','')}] "
        f"[RT:{row.get('resolution_hours','unknown')}h] "
        f"{row.get('combined_text','')}"
    )


def train(data_path: str):
    Path(OUTPUT_DIR).mkdir(exist_ok=True)

    # Load
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {"time_to_resolution": "resolution_hours"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    text_cols = [c for c in ["ticket_subject", "ticket_description", "ticket_type",
                              "ticket_channel", "product_purchased"] if c in df.columns]
    df[text_cols] = df[text_cols].fillna("")
    df["ticket_priority"] = df["ticket_priority"].str.strip().str.lower()
    df = df[df["ticket_priority"].isin(PRIORITY_ORDER.keys())].reset_index(drop=True)

    if "resolution_hours" in df.columns:
        df["resolution_hours"] = pd.to_numeric(
            df["resolution_hours"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
        )

    # Skip signal generation if already done
    if "pseudo_label" not in df.columns:
        df = build_signals(df)
        df.to_csv("data/pseudo_labeled.csv", index=False)
    else:
        print("Using existing pseudo labels")

    # Prepare inputs
    df["model_input"] = df.apply(build_model_input, axis=1)
    X = df["model_input"].tolist()
    y = df["pseudo_label"].tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=SEED, stratify=y)
    X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.15, random_state=SEED, stratify=y_train)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds  = TicketDataset(X_train, y_train, tokenizer, MAX_LEN)
    val_ds    = TicketDataset(X_val,   y_val,   tokenizer, MAX_LEN)
    test_ds   = TicketDataset(X_test,  y_test,  tokenizer, MAX_LEN)

    counts = Counter(y_train)
    total  = len(y_train)
    cw = torch.tensor([total/(2*counts[0]), total/(2*counts[1])], dtype=torch.float)

    base_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    lora_cfg   = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["query_proj", "value_proj"], bias="none"
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="best",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=(DEVICE=="cuda"),
        seed=SEED,
        report_to="none",
    )

    trainer = WeightedTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, class_weights=cw,
    )
    trainer.train()

    # Evaluate
    results = trainer.evaluate(test_ds)
    print("\n=== TEST RESULTS ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    preds_out = trainer.predict(test_ds)
    pred_labels = np.argmax(preds_out.predictions, axis=-1)
    print("\n=== Classification Report ===")
    print(classification_report(y_test, pred_labels, target_names=["Consistent","Mismatch"]))

    # Save
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    with open(f"{OUTPUT_DIR}/config_extra.json", "w") as f:
        json.dump({"emb_model": EMB_MODEL, "max_len": MAX_LEN, "model_name": MODEL_NAME}, f)
    print(f"\nModel saved to {OUTPUT_DIR}/")

    # Save metrics
    with open(f"{OUTPUT_DIR}/metrics.json", "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/pseudo_labeled.csv",
                        help="Path to CSV (raw or pseudo-labeled)")
    args = parser.parse_args()
    train(args.data)
