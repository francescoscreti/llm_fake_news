"""
app streamlit unificata — llm fake news
lancia con: streamlit run app.py
ollama deve girare in locale per la demo live
"""

import os
import sys
import json
import requests
import torch
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# importa i prompt dalla pipeline ollama senza __init__.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ollama_pipeline"))
from ollama_pipeline.prompts import zero_shot_prompt, few_shot_prompt

# ── configurazione pagina ──────────────────────────────────────────────────────

st.set_page_config(
    page_title="LLM Fake News",
    page_icon="📰",
    layout="wide",  # TODO : valutare se lasciarlo o meno
    initial_sidebar_state="collapsed",
)

# ── tema chiaro e stile ────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* forza tema chiaro */
[data-testid="stAppViewContainer"] { background: #f8f9fa; }
[data-testid="stHeader"] { background: #f8f9fa; }

/* tab bar come navigazione principale */
[data-testid="stTabs"] button {
    font-size: 0.95rem;
    font-weight: 600;
    padding: 0.6rem 1.2rem;
    color: #555;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #1a1a2e;
    border-bottom: 3px solid #e63946;
}

/* card metriche */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
}

/* badge risultato */
.badge-fake {
    background: #ffe0e0;
    color: #c0392b;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
}
.badge-real {
    background: #e0f5e0;
    color: #27ae60;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
}
.badge-unknown {
    background: #f0f0f0;
    color: #888;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
}

/* sezione highlight */
.info-box {
    background: #eef2ff;
    border-left: 4px solid #4361ee;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}
.warn-box {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 6px;
    padding: 1rem 1.2rem;
    margin: 0.8rem 0;
}

/* Pulsante Classifica: testo bianco e sfondo rosso, selettore robusto */
.stButton > button:first-child {
    color: #fff !important;
    background-color: #e63946 !important;
    border: none !important;
    font-weight: 700;
}
.stButton > button:first-child:hover, .stButton > button:first-child:focus {
    color: #fff !important;
    background-color: #b71c1c !important;
    border: none !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── dati configurazioni fine-tuning ───────────────────────────────────────────

CONFIGS = [
    {
        "name": "baseline",
        "lr": 2e-5,
        "batch": 16,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9740,
    },
    {
        "name": "high_lr",
        "lr": 5e-5,
        "batch": 16,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9740,
    },
    {
        "name": "more_epochs",
        "lr": 2e-5,
        "batch": 16,
        "epochs": 5,
        "wd": 0.01,
        "accuracy": 0.9720,
    },
    {
        "name": "small_batch",
        "lr": 2e-5,
        "batch": 8,
        "epochs": 3,
        "wd": 0.01,
        "accuracy": 0.9800,
    },
]

COLORS = {
    "primary": "#1a1a2e",
    "accent": "#e63946",
    "best": "#b3ffb3",
    "default": "#d0d0d0",
    "zero_shot": "#4361ee",
    "few_shot": "#f4a261",
    "distilbert": "#2a9d8f",
}

# ── helper: trova checkpoint ───────────────────────────────────────────────────


def find_checkpoint_dir(config_name: str) -> str | None:
    base = os.path.join("fine_tuning", "results", config_name)
    if not os.path.exists(base):
        return None
    checkpoints = [d for d in os.listdir(base) if d.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return os.path.join(base, checkpoints[-1])


# ── caching ────────────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Caricamento modello DistilBERT...")
def load_distilbert():
    ckpt = find_checkpoint_dir("small_batch")
    if ckpt is None:
        raise FileNotFoundError("checkpoint small_batch non trovato")
    model = AutoModelForSequenceClassification.from_pretrained(ckpt)
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model.eval()
    return model, tokenizer


@st.cache_data(show_spinner=False)
def load_trainer_state(config_name: str) -> dict | None:
    ckpt = find_checkpoint_dir(config_name)
    if ckpt is None:
        return None
    path = os.path.join(ckpt, "trainer_state.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_ollama_runs() -> list[list]:
    results_dir = os.path.join("ollama_pipeline", "results")
    if not os.path.exists(results_dir):
        return []
    files = sorted(
        f
        for f in os.listdir(results_dir)
        if f.startswith("outputs_run") and f.endswith(".json")
    )
    runs = []
    for f in files:
        with open(os.path.join(results_dir, f)) as j:
            runs.append(json.load(j))
    return runs


def parse_label(text: str) -> str:
    upper = text.strip().upper()
    if "FAKE" in upper:
        return "FAKE"
    if "REAL" in upper:
        return "REAL"
    return "UNKNOWN"


def distilbert_predict(text: str) -> str:
    model, tokenizer = load_distilbert()
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    with torch.no_grad():
        outputs = model(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
    return "FAKE" if pred == 0 else "REAL"


def ollama_predict(prompt: str) -> str:
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "gemma2:2b", "prompt": prompt, "stream": False},
        timeout=30,
    )
    resp.raise_for_status()
    return parse_label(resp.json()["response"])


def label_badge(label: str) -> str:
    cls = {"FAKE": "badge-fake", "REAL": "badge-real"}.get(label, "badge-unknown")
    return f'<span class="{cls}">{label}</span>'


# ── navigazione con tabs ───────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "📋 Introduzione",
        "🧠 Fine-tuning",
        "🤖 Pipeline Ollama",
        "⚖️ Confronto",
        "🎯 Demo live",
    ]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INTRODUZIONE
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("📰 LLM Fake News Detection")
    st.markdown(
        "**Classificazione automatica di notizie FAKE / REAL** tramite due approcci: fine-tuning di un modello encoder-only e prompting di un LLM locale."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Il progetto")
        st.markdown(
            """
        <div class="info-box">
        <b>Task:</b> Classificazione binaria — ogni articolo viene classificato come <b>FAKE</b> o <b>REAL</b>.<br><br>
        <b>Dataset:</b> <a href="https://huggingface.co/datasets/GonzaloA/fake_news">GonzaloA/fake_news</a> su Hugging Face —
        dataset bilanciato di articoli in inglese con etichette 0 = FAKE, 1 = REAL.<br><br>
        <b>Perché questo task?</b> La classificazione di fake news è un problema reale e rilevante,
        con un dataset pubblico e ben strutturato. Permette un confronto diretto tra le due
        strategie del progetto sullo stesso problema.
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.subheader("I due approcci")
        st.markdown("""
        | | Parte 1 | Parte 2 |
        |---|---|---|
        | Modello | DistilBERT | gemma2:2b |
        | Strategia | Fine-tuning | Prompting |
        | Training | Sì (su 1000 esempi) | No |
        | Accuracy | **0.980** | 0.95 (zero-shot) |
        """)

    with col2:
        st.subheader("I modelli")
        st.markdown("""
        **DistilBERT** (Parte 1)
        - Versione distillata di BERT: 40% meno parametri, 97% delle performance
        - Architettura encoder-only, ideale per classificazione di testo
        - Pre-addestrato su corpus generale, specializzato tramite fine-tuning
        - 66.9M parametri, tutti allenabili

        **gemma2:2b** (Parte 2)
        - Modello Google da 2 miliardi di parametri
        - Eseguito localmente tramite Ollama — nessuna API esterna
        - Usato senza training specifico, solo tramite prompt engineering
        """)

    st.divider()
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("⚠️ Difficoltà incontrate")
        st.markdown("""
        **Teoriche**
        - **Generalizzazione vs Specializzazione**: Analizzare perché un modello compatto (66M parametri) fine-tuned superi un modello da 2B parametri in un task verticale.
        - **Bias indotto dal prompting**: Comprendere il motivo del calo di performance nel few-shot dovuto all'ancoraggio stilistico del modello agli esempi forniti.

        **Pratiche**
        - **Normalizzazione dell'output**: La tendenza del modello a fornire spiegazioni discorsive ha richiesto l'implementazione di un parser robusto per ricondurre l'output alle label FAKE/REAL.
        - **Sincronizzazione dei checkpoint**: Assicurare la coerenza tra i log di training e i file di stato del modello per una corretta visualizzazione delle curve di loss.
        """)

    with col4:
        st.subheader("🤖 Uso degli LLM come supporto")
        st.markdown("""
        - _Claude_ è stato utilizzato per la progettazione dell'architettura della pipeline e per il ragionamento strutturato sulla risoluzione dei problemi di parsing; 
        - _Copilot_ per la generazione rapida di codice boilerplate per l'interfaccia Streamlit.

        **Cosa ha funzionato bene:** Claude per ragionamento strutturato su problemi complessi e spiegazioni dettagliate; Copilot per generazione rapida di boilerplate.

        **Cosa non ha funzionato:** Alcune spiegazioni sugli aspetti di configurazione dei path erano inizialmente generiche e hanno richiesto un affinamento manuale basato sulla struttura specifica della repository.
        """)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — FINE-TUNING
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("🧠 Fine-tuning DistilBERT")
    st.markdown(
        "Fine-tuning di `distilbert-base-uncased` su classificazione binaria fake news con 4 configurazioni di iperparametri."
    )

    st.divider()

    # ── SEZIONE: STRUTTURA DEL CODICE ──
    st.subheader("Struttura del codice")

    st.markdown("""
    Abbiamo riadattato il codice della fine-tuning su SST-2 (sentiment analysis) sostituendo il dataset 
    e aggiungendo il supporto per 4 configurazioni di iperparametri da confrontare in sequenza.
    """)

    col_struct1, col_struct2, col_struct3 = st.columns(3)

    with col_struct1:
        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">config.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Pannello di controllo. Tutte le costanti del progetto: modello, dataset, 4 configurazioni iperparametri. 
        Nessun numero hardcoded altrove.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">data/dataset.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Scarica il dataset da Hugging Face, tokenizza gli articoli, riduce a 1.000 esempi.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_struct2:
        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">model/model.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Carica DistilBERT pre-addestrato e aggiunge un layer di classificazione con 2 output (FAKE/REAL).
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">training/metrics.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Calcola l'accuracy. Il Trainer la chiama automaticamente alla fine di ogni epoca.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col_struct3:
        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">training/trainer.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Assembla Trainer HuggingFace con dataset, modello e metrica. Gestisce salvataggio checkpoint.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
        <div class="card" style="background:#fff; border:1px solid #e8e4df; border-radius:12px; padding:16px; margin:8px 0">
        <b style="font-size:0.85rem; color:#7a7470; display:block; margin-bottom:8px">train.py</b>
        <div style="font-size:0.85rem; color:#7a7470; line-height:1.5">
        Entry point. Carica i dati una volta, poi scorre le 4 configurazioni. Per ognuna carica un modello fresco e addestra da zero.
        </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── SEZIONE: IPERPARAMETRI ──
    st.subheader("I 4 iperparametri: cosa significano")

    st.markdown("""
    | Parametro | Cosa significa| Valore testato |
    |---|---|---|
    | **lr** (learning rate) | Il "passo" con cui il modello corregge i propri errori. **2e-5** = passi piccoli e stabili. **5e-5** = passi grandi, impara più in fretta ma rischia di saltare il punto ottimale. | `2e-5` vs `5e-5` |
    | **batch** | Quanti articoli vede il modello prima di aggiornare i pesi. **Batch 16**: il modello vede 16 articoli, calcola l'errore medio su tutti, aggiorna. **Batch 8**: aggiorna il doppio delle volte, su meno esempi. | `16` vs `8` |
    | **epochs** | Quante volte il modello vede l'intero dataset. Con 1.000 articoli e 3 epoche, il modello li vede 3.000 volte in totale. Più epoche → rischio di imparare a memoria. | `3` vs `5` |
    | **wd** (weight decay) | Regolarizzazione. Ad ogni step i pesi vengono ridotti dell'1% per evitare che crescano troppo e causino overfitting. | `0.01` (fisso) |
    """)

    st.divider()

    # metriche riassuntive
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Modello", "DistilBERT")
    c2.metric("Parametri", "66.9M")
    c3.metric("Best accuracy", "0.9800")
    c4.metric("Best config", "small_batch")

    st.divider()
    st.subheader("Configurazioni testate")

    df_cfg = pd.DataFrame(CONFIGS)
    best_idx = df_cfg["accuracy"].idxmax()

    def highlight_best(row):
        return [
            "background-color: #b3ffb3" if row.name == best_idx else "" for _ in row
        ]

    st.dataframe(
        df_cfg[["name", "lr", "batch", "epochs", "wd", "accuracy"]]
        .style.apply(highlight_best, axis=1)
        .format({"lr": "{:.0e}", "accuracy": "{:.4f}"}),
        use_container_width=True,
    )

    # bar chart accuracy
    fig_acc = go.Figure(
        [
            go.Bar(
                x=[c["name"] for c in CONFIGS],
                y=[c["accuracy"] for c in CONFIGS],
                marker_color=[
                    COLORS["best"] if i == best_idx else COLORS["default"]
                    for i in range(len(CONFIGS))
                ],
                text=[f"{c['accuracy']:.4f}" for c in CONFIGS],
                textposition="outside",
            )
        ]
    )
    fig_acc.update_layout(
        title="Accuracy sul validation set per configurazione",
        yaxis=dict(title="Accuracy", range=[0.96, 0.985]),
        xaxis_title="Configurazione",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380,
    )
    st.plotly_chart(fig_acc, use_container_width=True)

    st.divider()
    st.subheader("Andamento loss durante il training")

    for cfg in CONFIGS:
        state = load_trainer_state(cfg["name"])
        if state is None:
            st.warning(f"trainer_state.json non trovato per {cfg['name']}")
            continue

        log = state.get("log_history", [])

        # separa correttamente training loss e validation loss
        train_logs = [
            (x["step"], x["loss"]) for x in log if "loss" in x and "eval_loss" not in x
        ]
        eval_logs = [(x["step"], x["eval_loss"]) for x in log if "eval_loss" in x]

        if not train_logs:
            st.warning(f"Nessun dato di training loss per {cfg['name']}")
            continue

        fig_loss = go.Figure()
        fig_loss.add_trace(
            go.Scatter(
                x=[s for s, _ in train_logs],
                y=[l for _, l in train_logs],
                mode="lines+markers",
                name="Training loss",
                line=dict(color=COLORS["zero_shot"]),
            )
        )
        if eval_logs:
            fig_loss.add_trace(
                go.Scatter(
                    x=[s for s, _ in eval_logs],
                    y=[l for _, l in eval_logs],
                    mode="lines+markers",
                    name="Validation loss",
                    line=dict(color=COLORS["accent"], dash="dash"),
                )
            )
        fig_loss.update_layout(
            title=f"{cfg['name']} — lr={cfg['lr']:.0e}, batch={cfg['batch']}, epochs={cfg['epochs']}",
            xaxis_title="Step",
            yaxis_title="Loss",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=320,
        )
        st.plotly_chart(fig_loss, use_container_width=True)

    st.divider()
    st.subheader("Analisi impatto iperparametri")

    ca, cb, cc = st.columns(3)
    with ca:
        st.markdown("**Learning rate**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Config": ["baseline (2e-5)", "high_lr (5e-5)"],
                    "Accuracy": [0.9740, 0.9740],
                    "Δ": ["-", "="],
                }
            ),
            use_container_width=True,
        )
        st.markdown(
            '<div class="warn-box">LR elevato non migliora — passi di aggiornamento troppo grandi destabilizzano il training a fine epoche.</div>',
            unsafe_allow_html=True,
        )
    with cb:
        st.markdown("**Batch size**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Config": ["baseline (16)", "small_batch (8)"],
                    "Accuracy": [0.9740, 0.9800],
                    "Δ": ["-", "+0.006"],
                }
            ),
            use_container_width=True,
        )
        st.markdown(
            '<div class="info-box">Batch più piccolo → aggiornamenti dei pesi più frequenti → migliore generalizzazione.</div>',
            unsafe_allow_html=True,
        )
    with cc:
        st.markdown("**Epoche**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Config": ["baseline (3)", "more_epochs (5)"],
                    "Accuracy": [0.9740, 0.9720],
                    "Δ": ["-", "-0.002"],
                }
            ),
            use_container_width=True,
        )
        st.markdown(
            '<div class="warn-box">Più epoche non aiutano — il modello converge già a epoca 2. Epoche extra rischiano overfitting.</div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PIPELINE OLLAMA
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.title("🤖 Pipeline Ollama")
    st.markdown(
        "Classificazione fake news tramite **gemma2:2b** in locale con due strategie di prompting: zero-shot e few-shot."
    )

    st.divider()

    # schema pipeline
    st.subheader("Schema della pipeline")
    st.markdown(
        """
    <div class="info-box">
    <b>Input</b> → testo articolo da <code>data/samples.json</code><br><br>
    <b>Zero-shot</b>: istruzione + testo → <code>POST localhost:11434/api/generate</code> → risposta grezza → <code>parse_label()</code> → FAKE / REAL<br><br>
    <b>Few-shot</b>: istruzione + 4 esempi etichettati + testo → stessa pipeline → FAKE / REAL<br><br>
    <b>Output</b> → salvato in <code>results/outputs_run{N}.json</code> con predizioni, label raw, correttezza per ogni sample
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.markdown("""
    **Perché `parse_label()`?**
    Ollama non risponde sempre con una sola parola anche se richiesto.
    Risposte tipo `"The label is FAKE."` o `"fake"` vengono normalizzate cercando
    le sottostringhe `FAKE` o `REAL` nel testo in maiuscolo.
    """)

    st.divider()

    all_runs = load_ollama_runs()
    if not all_runs:
        st.error("Nessun file outputs_run*.json trovato in ollama_pipeline/results/")
    else:
        # selettore run
        run_labels = [f"Run {i+1}" for i in range(len(all_runs))]
        selected_run_idx = 0
        if len(all_runs) > 1:
            selected_run_label = st.selectbox("Seleziona run da analizzare", run_labels)
            selected_run_idx = run_labels.index(selected_run_label)

        run = all_runs[selected_run_idx]
        df = pd.DataFrame(run)
        df["zs_label"] = df["zero_shot"].apply(lambda x: x["label"])
        df["fs_label"] = df["few_shot"].apply(lambda x: x["label"])
        df["zs_correct"] = df["zero_shot"].apply(lambda x: x["correct"])
        df["fs_correct"] = df["few_shot"].apply(lambda x: x["correct"])

        # metriche
        st.subheader("Metriche")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        zs_acc = df["zs_correct"].mean()
        fs_acc = df["fs_correct"].mean()
        zs_fake = df[df["ground_truth"] == "FAKE"]["zs_correct"].mean()
        zs_real = df[df["ground_truth"] == "REAL"]["zs_correct"].mean()
        fs_fake = df[df["ground_truth"] == "FAKE"]["fs_correct"].mean()
        fs_real = df[df["ground_truth"] == "REAL"]["fs_correct"].mean()
        m1.metric("Zero-shot accuracy", f"{zs_acc:.2f}")
        m2.metric("Zero-shot FAKE", f"{zs_fake:.2f}")
        m3.metric("Zero-shot REAL", f"{zs_real:.2f}")
        m4.metric("Few-shot accuracy", f"{fs_acc:.2f}")
        m5.metric("Few-shot FAKE", f"{fs_fake:.2f}")
        m6.metric("Few-shot REAL", f"{fs_real:.2f}")

        # bar chart metriche
        fig_m = go.Figure(
            [
                go.Bar(
                    name="Zero-shot",
                    x=["Globale", "FAKE", "REAL"],
                    y=[zs_acc, zs_fake, zs_real],
                    marker_color=COLORS["zero_shot"],
                ),
                go.Bar(
                    name="Few-shot",
                    x=["Globale", "FAKE", "REAL"],
                    y=[fs_acc, fs_fake, fs_real],
                    marker_color=COLORS["few_shot"],
                ),
            ]
        )
        fig_m.update_layout(
            barmode="group",
            title="Accuracy per strategia e per classe",
            yaxis=dict(title="Accuracy", range=[0.7, 1.05]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=340,
        )
        st.plotly_chart(fig_m, use_container_width=True)

        st.divider()

        # tabella sample per sample
        st.subheader("Risultati per sample")
        display_df = df[
            [
                "id",
                "text_preview",
                "ground_truth",
                "zs_label",
                "fs_label",
                "zs_correct",
                "fs_correct",
            ]
        ].copy()
        display_df.columns = [
            "ID",
            "Testo",
            "Ground truth",
            "Zero-shot",
            "Few-shot",
            "ZS ✓",
            "FS ✓",
        ]

        filter_col1, filter_col2 = st.columns(2)
        filter_gt = filter_col1.selectbox(
            "Filtra per ground truth", ["Tutti", "FAKE", "REAL"]
        )
        filter_err = filter_col2.selectbox(
            "Filtra per errori",
            [
                "Tutti",
                "Solo errori zero-shot",
                "Solo errori few-shot",
                "Entrambi sbagliano",
            ],
        )

        filtered = display_df.copy()
        if filter_gt != "Tutti":
            filtered = filtered[filtered["Ground truth"] == filter_gt]
        if filter_err == "Solo errori zero-shot":
            filtered = filtered[~filtered["ZS ✓"]]
        elif filter_err == "Solo errori few-shot":
            filtered = filtered[~filtered["FS ✓"]]
        elif filter_err == "Entrambi sbagliano":
            filtered = filtered[~filtered["ZS ✓"] & ~filtered["FS ✓"]]

        st.dataframe(filtered, use_container_width=True)

        st.divider()

        # casi di disaccordo
        st.subheader("Casi di disaccordo tra le due strategie")
        disag = df[df["zs_label"] != df["fs_label"]]
        if disag.empty:
            st.success("Le due strategie concordano su tutti i sample.")
        else:
            st.markdown(
                f"**{len(disag)} caso/i** in cui zero-shot e few-shot danno risposte diverse:"
            )
            for _, row in disag.iterrows():
                winner = (
                    "zero-shot"
                    if row["zs_correct"]
                    else ("few-shot" if row["fs_correct"] else "nessuno")
                )
                st.markdown(
                    f"""
                <div class="warn-box">
                <b>ID {int(row['id'])} — GT: {row['ground_truth']}</b><br>
                Zero-shot: <b>{row['zs_label']}</b> &nbsp;|&nbsp; Few-shot: <b>{row['fs_label']}</b> &nbsp;|&nbsp; Corretto: <b>{winner}</b><br><br>
                <i>{row['text_preview'][:200]}...</i>
                </div>
                """,
                    unsafe_allow_html=True,
                )

        # stabilità multi-run
        if len(all_runs) > 1:
            st.divider()
            st.subheader(f"Stabilità su {len(all_runs)} run")

            zs_accs = [
                pd.DataFrame(r)["zero_shot"].apply(lambda x: x["correct"]).mean()
                for r in all_runs
            ]
            fs_accs = [
                pd.DataFrame(r)["few_shot"].apply(lambda x: x["correct"]).mean()
                for r in all_runs
            ]

            sr1, sr2 = st.columns(2)
            sr1.metric(
                "Zero-shot media ± std",
                f"{pd.Series(zs_accs).mean():.3f} ± {pd.Series(zs_accs).std():.3f}",
            )
            sr2.metric(
                "Few-shot media ± std",
                f"{pd.Series(fs_accs).mean():.3f} ± {pd.Series(fs_accs).std():.3f}",
            )

            fig_runs = go.Figure()
            fig_runs.add_trace(
                go.Scatter(
                    x=list(range(1, len(all_runs) + 1)),
                    y=zs_accs,
                    mode="lines+markers",
                    name="Zero-shot",
                    line=dict(color=COLORS["zero_shot"]),
                )
            )
            fig_runs.add_trace(
                go.Scatter(
                    x=list(range(1, len(all_runs) + 1)),
                    y=fs_accs,
                    mode="lines+markers",
                    name="Few-shot",
                    line=dict(color=COLORS["few_shot"]),
                )
            )
            fig_runs.update_layout(
                title="Accuracy per run",
                xaxis_title="Run",
                yaxis_title="Accuracy",
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=300,
            )
            st.plotly_chart(fig_runs, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — CONFRONTO
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.title("⚖️ Confronto tra approcci")
    st.markdown("Fine-tuning vs prompting: quando usare quale approccio?")

    st.divider()

    # tabella comparativa
    comp_data = {
        "Approccio": ["DistilBERT (best)", "Zero-shot gemma2", "Few-shot gemma2"],
        "Accuracy": [0.9800, 0.9500, 0.9000],
        "Acc. FAKE": ["-", "1.00", "0.90"],
        "Acc. REAL": ["-", "0.90", "0.90"],
        "Training": ["Sì", "No", "No"],
        "Risorse": ["Alta", "Bassa", "Bassa"],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    # bar chart confronto
    fig_comp = go.Figure(
        [
            go.Bar(
                x=comp_data["Approccio"],
                y=comp_data["Accuracy"],
                marker_color=[
                    COLORS["distilbert"],
                    COLORS["zero_shot"],
                    COLORS["few_shot"],
                ],
                text=[f"{a:.2f}" for a in comp_data["Accuracy"]],
                textposition="outside",
            )
        ]
    )
    fig_comp.update_layout(
        title="Accuracy a confronto",
        yaxis=dict(title="Accuracy", range=[0.85, 1.0]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.divider()

    st.subheader("Esempi concreti")

    ex1, ex2 = st.columns(2)
    with ex1:
        st.markdown(
            """
        <div class="warn-box">
        <b>Caso interessante — zero-shot corretto, few-shot sbaglia</b><br><br>
        <i>"History is once again being made thanks to President Obama. On Wednesday,
        Obama nominated Abid Riaz Qureshi to serve as..."</i><br><br>
        Ground truth: <b>FAKE</b><br>
        Zero-shot: ✅ FAKE<br>
        Few-shot: ❌ REAL<br><br>
        Il few-shot ha classificato questo articolo come REAL perché il tono istituzionale
        ricordava gli esempi reali forniti nel prompt. Gli esempi hanno introdotto un bias
        stilistico invece di aiutare.
        </div>
        """,
            unsafe_allow_html=True,
        )
    with ex2:
        st.markdown(
            """
        <div class="warn-box">
        <b>Errore condiviso — entrambi sbagliano</b><br><br>
        <i>"hillary clinton campaign still whining about the fbi november —
        the hillary clinton whineaton continues after having m..."</i><br><br>
        Ground truth: <b>REAL</b><br>
        Zero-shot: ❌ FAKE<br>
        Few-shot: ❌ FAKE<br><br>
        Linguaggio aggressivo e partigiano che stilisticamente somiglia a contenuto fake,
        ma l'articolo è reale. Questo tipo di errore richiede training specifico per essere corretto.
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Quando usare quale approccio?")

    ca, cb = st.columns(2)
    with ca:
        st.markdown(
            """
        <div class="info-box">
        <b>Fine-tuning</b> quando:<br>
        • hai dati etichettati sufficienti (anche poche centinaia)<br>
        • vuoi la massima accuracy possibile<br>
        • il task è sempre lo stesso (produzione)<br>
        • hai risorse per eseguire il training<br>
        • la consistenza delle risposte è critica
        </div>
        """,
            unsafe_allow_html=True,
        )
    with cb:
        st.markdown(
            """
        <div class="info-box">
        <b>Prompting</b> quando:<br>
        • non hai dati etichettati<br>
        • hai bisogno di risultati subito<br>
        • il task cambia frequentemente<br>
        • vuoi prototipare rapidamente<br>
        • 0.95 di accuracy è sufficiente
        </div>
        """,
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — DEMO LIVE
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.title("🎯 Demo live")
    st.markdown(
        "Inserisci un articolo e confronta in tempo reale le predizioni dei tre classificatori."
    )

    st.markdown(
        """
    <div class="warn-box">
    ⚠️ Ollama deve essere in esecuzione in locale (<code>ollama serve</code>) e il modello <code>gemma2:2b</code> deve essere installato.
    DistilBERT viene caricato dal checkpoint <code>fine_tuning/results/small_batch/</code>.
    </div>
    """,
        unsafe_allow_html=True,
    )

    # esempi rapidi
    st.markdown("**Esempi rapidi:**")
    ex_col1, ex_col2 = st.columns(2)
    example_fake = "Scientists confirm that drinking bleach cures cancer. The government has been hiding this information for decades to protect pharmaceutical companies."
    example_real = "The Federal Reserve raised interest rates by 0.25 percentage points on Wednesday, citing persistent inflation concerns and a resilient labor market."

    if ex_col1.button("📋 Inserisci esempio FAKE"):
        st.session_state["demo_text"] = example_fake
    if ex_col2.button("📋 Inserisci esempio REAL"):
        st.session_state["demo_text"] = example_real

    user_text = st.text_area(
        "Testo dell'articolo:",
        value=st.session_state.get("demo_text", ""),
        height=150,
        placeholder="Incolla qui il testo di un articolo di notizie...",
    )

    if st.button("🔍 Classifica", type="primary") and user_text.strip():
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### DistilBERT")
            st.caption("fine-tuned su GonzaloA/fake_news")
            with st.spinner("classificazione in corso..."):
                try:
                    label = distilbert_predict(user_text)
                    st.markdown(label_badge(label), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Errore: {e}")

        with col2:
            st.markdown("### Zero-shot")
            st.caption("gemma2:2b — nessun esempio")
            with st.spinner("interrogazione Ollama..."):
                try:
                    label = ollama_predict(zero_shot_prompt(user_text))
                    st.markdown(label_badge(label), unsafe_allow_html=True)
                except Exception as e:
                    st.error(
                        "Ollama non disponibile — assicurati che `ollama serve` sia in esecuzione"
                    )

        with col3:
            st.markdown("### Few-shot")
            st.caption("gemma2:2b — 4 esempi nel prompt")
            with st.spinner("interrogazione Ollama..."):
                try:
                    label = ollama_predict(few_shot_prompt(user_text))
                    st.markdown(label_badge(label), unsafe_allow_html=True)
                except Exception as e:
                    st.error(
                        "Ollama non disponibile — assicurati che `ollama serve` sia in esecuzione"
                    )
