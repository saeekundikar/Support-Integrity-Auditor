import streamlit as st
import pandas as pd
import numpy as np
import json, re, io, time
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


st.set_page_config(
    page_title="Support Integrity Auditor",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem; font-weight: 800; color: #1a1a2e;
        border-left: 6px solid #e94560; padding-left: 16px; margin-bottom: 4px;
    }
    .sub-header { color: #555; font-size: 1rem; margin-bottom: 24px; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #e94560; border-radius: 12px;
        padding: 20px; text-align: center; color: white;
    }
    .metric-card h2 { font-size: 2rem; margin: 0; color: #e94560; }
    .metric-card p  { margin: 0; opacity: 0.8; font-size: 0.9rem; }
    .dossier-box {
        background: #f8f9fa; border: 1px solid #dee2e6;
        border-radius: 10px; padding: 20px; margin-top: 16px;
    }
    .hidden-crisis { background:#fff0f0; border-left: 4px solid #e74c3c; padding: 12px; border-radius:6px; }
    .false-alarm   { background:#fff8f0; border-left: 4px solid #f39c12; padding: 12px; border-radius:6px; }
    .consistent    { background:#f0fff4; border-left: 4px solid #27ae60; padding: 12px; border-radius:6px; }
    .evidence-item { background:#fff; border: 1px solid #e0e0e0; border-radius:8px;
                     padding:12px; margin:6px 0; font-size:0.9rem; }
    .stTabs [data-baseweb="tab"] { font-size: 1rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
SEVERITY_ORDER = {0: "low", 1: "medium", 2: "high", 3: "critical"}
W_KW, W_EMB, W_RT = 0.45, 0.35, 0.20

CRITICAL_KW = [
    r"\bdata.?loss\b", r"\bsystem.?down\b", r"\bcritical\b", r"\bsecurity.?breach\b",
    r"\bransomware\b", r"\bcompromised\b", r"\bproduction.?outage\b", r"\bescalat\w+\b",
    r"\bregulatory\b", r"\bcompliance\b", r"\bsla.?breach\b", r"\bentirely.?broken\b",
    r"\blegal\b", r"\bfinancial.?loss\b", r"\bceos?\b", r"\bcto\b", r"\bexecutive\b",
    r"\burgen\w+\b", r"\bno.?one.?can\b", r"\bcompletely.?broken\b"
]
HIGH_KW = [
    r"\bnot.?working\b", r"\bbroken\b", r"\berror\b", r"\bcrash\b", r"\bfail\w*\b",
    r"\bblocking\b", r"\bcannot.?access\b", r"\bunable.?to\b", r"\bdown\b",
    r"\bmultiple.?users\b", r"\bwhole.?team\b", r"\bstuck\b", r"\baffected\b"
]
MED_KW = [
    r"\bslow\b", r"\bdegraded\b", r"\bintermittent\b", r"\boccasionally\b",
    r"\bperformance\b", r"\bdelay\b", r"\blag\b"
]
LOW_KW = [
    r"\bquestion\b", r"\bhow.?to\b", r"\bfeedback\b", r"\bsuggestion\b", r"\bfeature.?request\b"
]
NEGATION = [r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bdon.?t\b", r"\bcan.?t\b"]


@st.cache_resource
def load_ml_models():
    """Load actual trained models — only called if sia_model/ exists."""
    model_path = Path("sia_model")
    if not model_path.exists():
        return None, None, None, None, None

    try:
        import torch
        import torch.nn.functional as F
        import pickle
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from peft import PeftModel
        from sentence_transformers import SentenceTransformer

        with open("sia_model/config_extra.json") as f:
            cfg = json.load(f)

        tokenizer   = AutoTokenizer.from_pretrained("sia_model")
        base_model  = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)
        classifier  = PeftModel.from_pretrained(base_model, "sia_model")
        classifier.eval()
        embedder    = SentenceTransformer(cfg["emb_model"])

        with open("sia_model/km_model.pkl", "rb") as f:
            km_model = pickle.load(f)
        with open("sia_model/cluster_to_sev.json") as f:
            c2s = {int(k): v for k, v in json.load(f).items()}

        return tokenizer, classifier, embedder, km_model, c2s
    except Exception as e:
        st.warning(f"Could not load trained models: {e}. Running in demo mode with rule-based signals only.")
        return None, None, None, None, None


def keyword_score(text: str) -> int:
    text = str(text).lower()
    crit = sum(1 for p in CRITICAL_KW if re.search(p, text))
    high = sum(1 for p in HIGH_KW     if re.search(p, text))
    med  = sum(1 for p in MED_KW      if re.search(p, text))
    low  = sum(1 for p in LOW_KW      if re.search(p, text))
    neg  = sum(1 for p in NEGATION    if re.search(p, text))
    if neg > 0: high += 1
    raw = crit * 3 + high * 2 + med * 1 - low * 0.5
    if raw >= 6: return 3
    elif raw >= 3: return 2
    elif raw >= 1: return 1
    else: return 0


def analyse_ticket(subject, description, priority, channel, rt_hours, ticket_id="N/A"):
    """Analyse a ticket and return prediction + dossier."""
    text = f"{subject} {description}"
    kw_s = keyword_score(text)
    emb_s = 1  # placeholder without embedder
    rt_s  = 1  # placeholder without regressor

    tokenizer, classifier, embedder, km_model, c2s = load_ml_models()

    if embedder is not None:
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            import torch
            import torch.nn.functional as F

            emb = embedder.encode([text], normalize_embeddings=True)
            cluster = km_model.predict(emb)[0]
            emb_s = c2s.get(cluster, 1)

            model_input = f"[PRIORITY:{priority}] [CHANNEL:{channel}] [RT:{rt_hours}h] {text}"
            enc = tokenizer(model_input, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = classifier(**enc).logits
            probs = F.softmax(logits, dim=-1)[0].numpy()
            pred_label  = int(np.argmax(probs))
            confidence  = float(probs[pred_label])
        except Exception:
            pred_label, confidence = None, None
    else:
        pred_label, confidence = None, None

    fused = max(0, min(3, round(W_KW * kw_s + W_EMB * emb_s + W_RT * rt_s)))
    assigned_num = PRIORITY_ORDER.get(priority.lower(), 1)
    delta = fused - assigned_num
    mtype = "Hidden Crisis" if delta > 0 else ("False Alarm" if delta < 0 else "Consistent")

    if pred_label is None:
        is_mismatch = delta != 0
        pred_label  = 1 if is_mismatch else 0
        confidence  = min(0.60 + abs(delta) * 0.12, 0.95)

    inferred_label = SEVERITY_ORDER[fused].capitalize()

    triggered = []
    for p in CRITICAL_KW + HIGH_KW + MED_KW:
        m = re.search(p, text.lower())
        if m:
            triggered.append(m.group())
    triggered_str = ", ".join(set(triggered[:6])) if triggered else "none"

    sev_map = ["Low", "Medium", "High", "Critical"]
    constraint = (
        f"Ticket assigned priority '{priority.capitalize()}' but fused signals infer '{inferred_label}'. "
        f"Keyword analysis detected: '{triggered_str}' (score {kw_s}/3). "
        f"Semantic embedding placed this in severity band {emb_s}/3. "
        f"The {abs(delta)}-level delta classifies this as a '{mtype}' case."
    )

    return {
        "prediction": "Mismatch" if pred_label == 1 else "Consistent",
        "confidence": confidence,
        "dossier": {
            "ticket_id": ticket_id,
            "assigned_priority": priority.capitalize(),
            "inferred_severity": inferred_label,
            "mismatch_type": mtype,
            "severity_delta": f"+{delta}" if delta > 0 else str(delta),
            "feature_evidence": [
                {"signal": "keyword",           "value": triggered_str,
                 "weight": f"{W_KW} — keyword score {kw_s}/3"},
                {"signal": "embedding_cluster", "value": f"Cluster severity: {emb_s}/3",
                 "weight": f"{W_EMB} — semantic similarity"},
                {"signal": "resolution_time",   "value": f"{rt_hours}h",
                 "interpretation": f"RT severity proxy: {rt_s}/3"},
                {"signal": "channel",           "value": channel,
                 "weight": "Structured metadata feature"}
            ],
            "constraint_analysis": constraint,
            "confidence": f"{confidence:.3f}"
        }
    }


def render_dossier(dossier):
    """Render a dossier as formatted Streamlit components."""
    mtype  = dossier["mismatch_type"]
    css_cls = "hidden-crisis" if mtype=="Hidden Crisis" else \
              ("false-alarm" if mtype=="False Alarm" else "consistent")
    icon    = "🔴" if mtype=="Hidden Crisis" else ("🟡" if mtype=="False Alarm" else "🟢")

    st.markdown(f"""
    <div class="{css_cls}">
        <b>{icon} {mtype}</b> &nbsp;|&nbsp;
        Assigned: <b>{dossier['assigned_priority']}</b> →
        Inferred: <b>{dossier['inferred_severity']}</b> &nbsp;|&nbsp;
        Δ = <b>{dossier['severity_delta']}</b> &nbsp;|&nbsp;
        Confidence: <b>{dossier['confidence']}</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Feature Evidence**")
    for ev in dossier["feature_evidence"]:
        val = ev.get("value","")
        extra = ev.get("weight", ev.get("interpretation",""))
        st.markdown(f"""
        <div class="evidence-item">
            <b>Signal:</b> {ev['signal'].replace('_',' ').title()} &nbsp;|&nbsp;
            <b>Value:</b> {val} &nbsp;|&nbsp;
            <small>{extra}</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("**Constraint Analysis**")
    st.info(dossier["constraint_analysis"])


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield-with-check.png", width=60)
    st.title("SIA Controls")
    st.markdown("---")
    mode = st.radio("Mode", ["Single Ticket", "Batch CSV Upload", "Dashboard"], index=0)
    st.markdown("---")
    st.markdown("**Fusion Weights**")
    w_kw  = st.slider("Keyword",   0.0, 1.0, W_KW,  0.05)
    w_emb = st.slider("Embedding", 0.0, 1.0, W_EMB, 0.05)
    w_rt  = st.slider("Resolution Time", 0.0, 1.0, W_RT, 0.05)
    st.caption(f"Sum: {w_kw+w_emb+w_rt:.2f}")

st.markdown('<div class="main-header">🔍 Support Integrity Auditor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered priority mismatch detection for enterprise CRM tickets</div>',
            unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(["🎫 Single Ticket", "📂 Batch Upload", "📊 Dashboard", "⚔️ Adversarial Test"])

with tab1:
    st.subheader("Analyse a Single Support Ticket")
    c1, c2 = st.columns([2, 1])

    with c1:
        subject  = st.text_input("Ticket Subject", placeholder="e.g. Users cannot log in to platform")
        desc     = st.text_area("Ticket Description", height=150,
                                placeholder="Provide the full ticket description here...")

    with c2:
        priority = st.selectbox("Assigned Priority", ["low","medium","high","critical"], index=1)
        channel  = st.selectbox("Channel", ["email","chat","phone","social media"])
        rt       = st.number_input("Resolution Time (hours)", min_value=0.0, value=24.0, step=0.5)
        t_id     = st.text_input("Ticket ID (optional)", value="T-001")

    if st.button("🔍 Analyse Ticket", type="primary", use_container_width=True):
        if not subject and not desc:
            st.warning("Please enter at least a subject or description.")
        else:
            with st.spinner("Analysing..."):
                result = analyse_ticket(subject, desc, priority, channel, rt, t_id)

            col1, col2, col3, col4 = st.columns(4)
            pred = result["prediction"]
            col1.metric("Verdict", pred, delta="⚠️ Review" if pred=="Mismatch" else "✅ OK")
            col2.metric("Confidence", f"{result['confidence']:.1%}")
            col3.metric("Assigned", priority.capitalize())
            col4.metric("Inferred", result["dossier"]["inferred_severity"])

            st.divider()
            if pred == "Mismatch":
                st.error(f"**Priority Mismatch Detected!** — {result['dossier']['mismatch_type']}")
            else:
                st.success("**Priority Consistent** — No mismatch detected.")

            if result["dossier"]:
                render_dossier(result["dossier"])

                with st.expander("📋 Raw JSON Dossier"):
                    st.json(result["dossier"])

                # Download
                st.download_button(
                    "⬇️ Download Dossier (JSON)",
                    data=json.dumps(result["dossier"], indent=2),
                    file_name=f"dossier_{t_id}.json",
                    mime="application/json"
                )


with tab2:
    st.subheader("Batch Ticket Analysis")
    st.markdown("Upload a CSV with columns: `ticket_subject`, `ticket_description`, `ticket_priority`, `ticket_channel`")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        df_raw.columns = [c.strip().lower().replace(" ","_") for c in df_raw.columns]
        st.write(f"Loaded **{len(df_raw)}** tickets")
        st.dataframe(df_raw.head(5))

        if st.button(" Run Batch Analysis", type="primary"):
            progress = st.progress(0)
            results_list = []

            for i, row in df_raw.iterrows():
                subj = str(row.get("ticket_subject", ""))
                desc = str(row.get("ticket_description", ""))
                prio = str(row.get("ticket_priority", "medium")).lower()
                chan = str(row.get("ticket_channel", "email"))
                rt   = float(row.get("resolution_hours", row.get("time_to_resolution", 24)) or 24)
                tid  = str(row.get("ticket_id", i))

                res = analyse_ticket(subj, desc, prio, chan, rt, tid)
                results_list.append({
                    "ticket_id":    tid,
                    "prediction":   res["prediction"],
                    "confidence":   round(res["confidence"], 3),
                    "mismatch_type": res["dossier"]["mismatch_type"],
                    "severity_delta": res["dossier"]["severity_delta"],
                    "assigned_priority": prio.capitalize(),
                    "inferred_severity": res["dossier"]["inferred_severity"]
                })
                progress.progress((i+1)/len(df_raw))

            res_df = pd.DataFrame(results_list)
            st.success("Analysis complete!")
            st.dataframe(res_df)

            # Store for dashboard
            st.session_state["batch_results"] = res_df

            col1, col2, col3 = st.columns(3)
            total    = len(res_df)
            mismatch = (res_df["prediction"]=="Mismatch").sum()
            hc       = (res_df["mismatch_type"]=="Hidden Crisis").sum()
            fa       = (res_df["mismatch_type"]=="False Alarm").sum()

            col1.metric("Total Tickets", total)
            col2.metric("Mismatches", mismatch, delta=f"{mismatch/total:.1%}")
            col3.metric("Hidden Crises", hc)

            csv_out = res_df.to_csv(index=False)
            st.download_button("⬇️ Download Results CSV", csv_out, "sia_results.csv", "text/csv")

with tab3:
    st.subheader("Priority Mismatch Dashboard")

    # Use batch results if available, else generate demo data
    if "batch_results" in st.session_state:
        res_df = st.session_state["batch_results"]
    else:
        st.info("Run a batch analysis first, or viewing demo data below.")
        # Demo synthetic data
        np.random.seed(42)
        n = 300
        priorities  = np.random.choice(["low","medium","high","critical"], n, p=[0.3,0.35,0.25,0.1])
        mtypes      = np.random.choice(["Consistent","Hidden Crisis","False Alarm"], n, p=[0.6,0.25,0.15])
        deltas      = [np.random.randint(1,3) if m=="Hidden Crisis" else
                       (-np.random.randint(1,3) if m=="False Alarm" else 0) for m in mtypes]
        channels    = np.random.choice(["email","chat","phone","social media"], n)
        ticket_types = np.random.choice(["Billing","Technical","Account","Feature Request"], n)
        res_df = pd.DataFrame({
            "prediction":       ["Mismatch" if m!="Consistent" else "Consistent" for m in mtypes],
            "mismatch_type":    mtypes,
            "severity_delta":   deltas,
            "assigned_priority": priorities,
            "ticket_channel":   channels,
            "ticket_type":      ticket_types
        })

    m_df = res_df[res_df["prediction"]=="Mismatch"] if "prediction" in res_df.columns else res_df

    # Top metrics
    c1, c2, c3, c4 = st.columns(4)
    total = len(res_df)
    mismatch_n = len(m_df)
    hc_n = (res_df.get("mismatch_type", pd.Series()) == "Hidden Crisis").sum()
    fa_n = (res_df.get("mismatch_type", pd.Series()) == "False Alarm").sum()
    for col, label, val, color in zip(
        [c1,c2,c3,c4],
        ["Total Tickets","Mismatches","Hidden Crises","False Alarms"],
        [total, mismatch_n, hc_n, fa_n],
        ["#3498db","#e74c3c","#c0392b","#f39c12"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <h2 style="color:{color}">{val}</h2>
            <p>{label}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        mtype_counts = res_df["mismatch_type"].value_counts() if "mismatch_type" in res_df.columns else pd.Series()
        fig = px.pie(
            values=mtype_counts.values, names=mtype_counts.index,
            title="Mismatch Type Distribution",
            color_discrete_map={"Consistent":"#27ae60","Hidden Crisis":"#e74c3c","False Alarm":"#f39c12"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        if "assigned_priority" in res_df.columns and "mismatch_type" in res_df.columns:
            heat_data = pd.crosstab(res_df["assigned_priority"], res_df["mismatch_type"])
            fig2 = px.imshow(heat_data, title="Priority vs Mismatch Type Heatmap",
                             color_continuous_scale="Reds", text_auto=True)
            st.plotly_chart(fig2, use_container_width=True)

    # Severity delta heatmap by channel and ticket type
    if "ticket_channel" in res_df.columns and "ticket_type" in res_df.columns:
        pivot = res_df.pivot_table(
            values="severity_delta", index="ticket_type",
            columns="ticket_channel", aggfunc="mean"
        ).fillna(0)
        fig3 = px.imshow(
            pivot, title="Mean Severity Delta by Ticket Type & Channel",
            color_continuous_scale="RdBu", color_continuous_midpoint=0,
            text_auto=".1f"
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Signal importance bar
    st.subheader("Signal Contribution (Current Weights)")
    fig4 = go.Figure(go.Bar(
        x=[f"Keyword ({w_kw:.0%})", f"Embedding ({w_emb:.0%})", f"Resolution Time ({w_rt:.0%})"],
        y=[w_kw, w_emb, w_rt],
        marker_color=["#e74c3c","#3498db","#2ecc71"]
    ))
    fig4.update_layout(title="Fusion Signal Weights", yaxis_title="Weight", showlegend=False)
    st.plotly_chart(fig4, use_container_width=True)



with tab4:
    st.subheader("⚔️ Adversarial Ticket Test")
    st.markdown("Test SIA against 10 adversarially crafted tickets designed to fool keyword-based systems.")

    ADVERSARIAL = [
        {"subject":"Minor cosmetic issue","description":"Just noticed the dashboard looks a bit off. By the way, none of our 500 users can log in to process payroll today.","priority":"low","true":"critical","channel":"email"},
        {"subject":"System is completely unusable — end of world","description":"I might be exaggerating slightly. The tooltip on the help page has a typo.","priority":"critical","true":"low","channel":"phone"},
        {"subject":"Feedback on recent update","description":"Wanted to share some thoughts. Customer data appears to be visible across different accounts since last Tuesday's deployment.","priority":"medium","true":"critical","channel":"email"},
        {"subject":"URGENT CRITICAL EMERGENCY HELP","description":"Hi, when you get a chance, could you update the color of the submit button to blue? Thanks!","priority":"critical","true":"low","channel":"chat"},
        {"subject":"API not responding","description":"Our payment processing API has been returning 500 errors for 3 hours. Revenue operations for 200 merchants are impacted.","priority":"low","true":"critical","channel":"email"},
        {"subject":"Production database issue","description":"It seems the prod DB might need a restart at some convenient time. It's throwing warnings.","priority":"medium","true":"high","channel":"email"},
        {"subject":"The sky is falling","description":"My font size in reports seems slightly larger than before the update.","priority":"critical","true":"low","channel":"phone"},
        {"subject":"Gentle inquiry","description":"I hope this isn't an issue but all order confirmations sent since Monday have been going to wrong customer emails. We have 10,000 orders affected.","priority":"low","true":"critical","channel":"chat"},
        {"subject":"Feature not working as expected","description":"The export button exports everything correctly, it just takes an extra half second. The button is in the wrong place too.","priority":"high","true":"low","channel":"email"},
        {"subject":"Quick check on billing","description":"Just wanted to verify — we've been double-charged for 6 months across all 300 enterprise accounts. Is that expected?","priority":"medium","true":"critical","channel":"email"},
    ]

    if st.button("🚀 Run Adversarial Tests", type="primary"):
        correct = 0
        rows = []
        for i, t in enumerate(ADVERSARIAL):
            res = analyse_ticket(t["subject"], t["description"], t["priority"], t["channel"], 24.0)
            inferred = res["dossier"]["inferred_severity"].lower()
            hit = inferred == t["true"]
            if hit: correct += 1
            rows.append({
                "Test #": i+1, "Subject (truncated)": t["subject"][:40]+"...",
                "Assigned": t["priority"].capitalize(),
                "True Severity": t["true"].capitalize(),
                "SIA Inferred": inferred.capitalize(),
                "✓": "✅" if hit else "❌"
            })

        score_pct = correct / 10
        if score_pct >= 0.7:
            st.success(f"🏆 Adversarial Score: **{correct}/10** ({score_pct:.0%}) — Bonus threshold met!")
        else:
            st.warning(f"Adversarial Score: **{correct}/10** ({score_pct:.0%})")

        adv_df = pd.DataFrame(rows)
        st.dataframe(adv_df, use_container_width=True, hide_index=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=correct * 10,
            title={"text": "Adversarial Accuracy (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#27ae60" if correct >= 7 else "#e74c3c"},
                   "steps": [{"range":[0,70],"color":"#f9ecec"},
                              {"range":[70,100],"color":"#e8f8f5"}],
                   "threshold": {"value":70,"line":{"color":"red","width":3}}}
        ))
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown(
    "<center><small>Support Integrity Auditor (SIA) · "
    "AI/ML · NLP · CRM Systems</small></center>",
    unsafe_allow_html=True
)
