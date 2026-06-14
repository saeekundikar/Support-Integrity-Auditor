"""
predict.py — SIA Inference Script
Usage:
    python predict.py --input tickets.csv --output results/
    python predict.py --text "System completely down, all users affected" --priority low --channel email
"""
import argparse, json, pickle, re, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────
MODEL_DIR = "sia_model"
W_KW, W_EMB, W_RT = 0.45, 0.35, 0.20
PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEVERITY_ORDER = {0: "low", 1: "medium", 2: "high", 3: "critical"}

CRITICAL_KW = [
    r"\bdata.?loss\b", r"\bsystem.?down\b", r"\bcritical\b", r"\bsecurity.?breach\b",
    r"\bransomware\b", r"\bcompromised\b", r"\bproduction.?outage\b", r"\bescalat\w+\b",
    r"\bregulatory\b", r"\bcompliance\b", r"\bsla.?breach\b", r"\bentirely.?broken\b",
    r"\blegal\b", r"\bfinancial.?loss\b", r"\bceos?\b", r"\bcto\b", r"\bexecutive\b",
    r"\burgen\w+\b"
]
HIGH_KW = [
    r"\bnot.?working\b", r"\bbroken\b", r"\berror\b", r"\bcrash\b", r"\bfail\w*\b",
    r"\bblocking\b", r"\bcannot.?access\b", r"\bunable.?to\b", r"\bdown\b",
    r"\bmultiple.?users\b", r"\bwhole.?team\b", r"\bstuck\b"
]
MED_KW = [
    r"\bslow\b", r"\bdegraded\b", r"\bintermittent\b", r"\boccasionally\b",
    r"\bperformance\b", r"\bdelay\b", r"\blag\b"
]
LOW_KW = [
    r"\bquestion\b", r"\bhow.?to\b", r"\bfeedback\b", r"\bsuggestion\b",
    r"\bfeature.?request\b"
]
NEGATION = [r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bdon.?t\b", r"\bcan.?t\b"]

ANCHORS = {
    0: "A simple question about how to use a feature or general feedback.",
    1: "Occasional slowness or minor inconvenience that does not block work.",
    2: "A significant error or failure affecting individual productivity or access.",
    3: "Critical system outage, data loss, or security breach affecting multiple users immediately."
}


# ──────────────────────────────────────────────────────────
# Load models (cached globally)
# ──────────────────────────────────────────────────────────
_tokenizer = None
_classifier = None
_embedder   = None
_km_model   = None
_c2s        = None
_extra_cfg  = None


def _load_models():
    global _tokenizer, _classifier, _embedder, _km_model, _c2s, _extra_cfg
    if _classifier is not None:
        return

    print("Loading SIA models...")
    with open(f"{MODEL_DIR}/config_extra.json") as f:
        _extra_cfg = json.load(f)

    _tokenizer  = AutoTokenizer.from_pretrained(MODEL_DIR)
    base_model  = AutoModelForSequenceClassification.from_pretrained(
        _extra_cfg["model_name"], num_labels=2
    )
    _classifier = PeftModel.from_pretrained(base_model, MODEL_DIR)
    _classifier.eval()

    _embedder = SentenceTransformer(_extra_cfg["emb_model"])

    with open(f"{MODEL_DIR}/km_model.pkl", "rb") as f:
        _km_model = pickle.load(f)
    with open(f"{MODEL_DIR}/cluster_to_sev.json") as f:
        _c2s = {int(k): v for k, v in json.load(f).items()}

    print("Models loaded.")


# ──────────────────────────────────────────────────────────
# Scoring functions
# ──────────────────────────────────────────────────────────
def keyword_score(text: str) -> int:
    text = str(text).lower()
    crit = sum(1 for p in CRITICAL_KW if re.search(p, text))
    high = sum(1 for p in HIGH_KW  if re.search(p, text))
    med  = sum(1 for p in MED_KW   if re.search(p, text))
    low  = sum(1 for p in LOW_KW   if re.search(p, text))
    neg  = sum(1 for p in NEGATION if re.search(p, text))
    if neg > 0: high += 1
    raw = crit * 3 + high * 2 + med * 1 - low * 0.5
    if raw >= 6: return 3
    elif raw >= 3: return 2
    elif raw >= 1: return 1
    else: return 0


def embedding_score(text: str) -> int:
    emb = _embedder.encode([text], normalize_embeddings=True)
    cluster = _km_model.predict(emb)[0]
    return _c2s.get(cluster, 1)


def fuse_scores(kw, emb, rt=1):
    return int(round(W_KW * kw + W_EMB * emb + W_RT * rt))


def classify_ticket(text: str, priority: str, channel: str = "email",
                    resolution_hours: float = None) -> dict:
    """Core inference for a single ticket."""
    _load_models()

    model_input = (
        f"[PRIORITY:{priority}] [CHANNEL:{channel}] "
        f"[RT:{resolution_hours if resolution_hours else 'unknown'}h] {text}"
    )
    enc = _tokenizer(model_input, return_tensors="pt",
                     truncation=True, max_length=_extra_cfg["max_len"])

    with torch.no_grad():
        logits = _classifier(**enc).logits
    probs = F.softmax(logits, dim=-1)[0].numpy()
    pred_label = int(np.argmax(probs))
    confidence = float(probs[pred_label])

    # Signal scores
    kw_s  = keyword_score(text)
    emb_s = embedding_score(text)
    rt_s  = 1  # default if no RT

    fused = fuse_scores(kw_s, emb_s, rt_s)
    fused = max(0, min(3, fused))

    assigned_num = PRIORITY_ORDER.get(priority.lower(), 1)
    delta        = fused - assigned_num
    mtype        = "Hidden Crisis" if delta > 0 else ("False Alarm" if delta < 0 else "Consistent")

    # Triggered keywords
    triggered = []
    for p in CRITICAL_KW + HIGH_KW + MED_KW:
        m = re.search(p, text.lower())
        if m:
            triggered.append(m.group())
    triggered_str = ", ".join(set(triggered[:5])) if triggered else "none"

    inferred_label = SEVERITY_ORDER[fused]
    sev_labels = ["Low", "Medium", "High", "Critical"]

    result = {
        "classifier_prediction": "Mismatch" if pred_label == 1 else "Consistent",
        "classifier_confidence": round(confidence, 3),
        "dossier": None
    }

    if pred_label == 1 or delta != 0:
        result["dossier"] = {
            "ticket_id": "N/A",
            "assigned_priority": priority.capitalize(),
            "inferred_severity": inferred_label.capitalize(),
            "mismatch_type": mtype,
            "severity_delta": f"+{delta}" if delta > 0 else str(delta),
            "feature_evidence": [
                {"signal": "keyword",          "value": triggered_str,
                 "weight": f"{W_KW} — score {kw_s}/3"},
                {"signal": "embedding_cluster", "value": f"Cluster severity: {emb_s}/3",
                 "weight": f"{W_EMB} — semantic similarity"},
                {"signal": "resolution_time",   "value": f"{resolution_hours}h" if resolution_hours else "N/A",
                 "interpretation": f"RT severity proxy: {rt_s}/3"},
                {"signal": "channel",           "value": channel,
                 "weight": "Structured metadata feature"}
            ],
            "constraint_analysis": (
                f"Ticket was assigned '{priority}' but signals infer '{inferred_label}'. "
                f"Keyword patterns detected: '{triggered_str}' (score {kw_s}/3). "
                f"Semantic embedding placed this in severity band {emb_s}/3. "
                f"The {abs(delta)}-level delta flags this as '{mtype}'."
            ),
            "confidence": f"{confidence:.3f}"
        }

    return result


def predict_csv(input_path: str, output_dir: str):
    """Batch prediction on a CSV file."""
    _load_models()
    df = pd.read_csv(input_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    col_map = {"time_to_resolution": "resolution_hours"}
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    for col in ["ticket_subject", "ticket_description", "ticket_channel",
                "ticket_priority", "ticket_type"]:
        if col not in df.columns:
            df[col] = ""

    df["combined_text"] = df["ticket_subject"].fillna("") + " " + df["ticket_description"].fillna("")
    df["ticket_priority"] = df["ticket_priority"].str.strip().str.lower().fillna("medium")
    df["ticket_channel"]  = df["ticket_channel"].fillna("email")

    if "resolution_hours" in df.columns:
        df["resolution_hours"] = pd.to_numeric(
            df["resolution_hours"].astype(str).str.extract(r"(\d+\.?\d*)")[0], errors="coerce"
        )

    results = []
    dossiers = []

    from tqdm.auto import tqdm
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        res = classify_ticket(
            text=str(row["combined_text"]),
            priority=str(row["ticket_priority"]),
            channel=str(row.get("ticket_channel", "email")),
            resolution_hours=row.get("resolution_hours")
        )
        results.append({
            "ticket_id": str(row.get("ticket_id", idx)),
            "prediction": res["classifier_prediction"],
            "confidence": res["classifier_confidence"],
            "mismatch_type": res["dossier"]["mismatch_type"] if res["dossier"] else "Consistent",
            "severity_delta": res["dossier"]["severity_delta"] if res["dossier"] else "0"
        })
        if res["dossier"]:
            d = res["dossier"].copy()
            d["ticket_id"] = str(row.get("ticket_id", idx))
            dossiers.append(d)

    Path(output_dir).mkdir(exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    with open(f"{output_dir}/dossiers.json", "w") as f:
        json.dump(dossiers, f, indent=2)

    print(f"\nPredictions: {output_dir}/predictions.csv")
    print(f"Dossiers:    {output_dir}/dossiers.json")
    print(f"\nMismatch breakdown:")
    print(results_df["prediction"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SIA Inference")
    subparsers = parser.add_subparsers(dest="mode")

    # CSV mode
    csv_p = subparsers.add_parser("csv", help="Batch predict from CSV")
    csv_p.add_argument("--input",  required=True)
    csv_p.add_argument("--output", default="results")

    # Single ticket mode
    single_p = subparsers.add_parser("single", help="Predict single ticket")
    single_p.add_argument("--text",     required=True)
    single_p.add_argument("--priority", required=True, choices=["low","medium","high","critical"])
    single_p.add_argument("--channel",  default="email")
    single_p.add_argument("--rt",       type=float, default=None, help="Resolution hours")

    args = parser.parse_args()

    if args.mode == "csv":
        predict_csv(args.input, args.output)
    elif args.mode == "single":
        result = classify_ticket(args.text, args.priority, args.channel, args.rt)
        print(f"\nPrediction: {result['classifier_prediction']} (confidence: {result['classifier_confidence']})")
        if result["dossier"]:
            print("\nEvidence Dossier:")
            print(json.dumps(result["dossier"], indent=2))
    else:
        parser.print_help()
