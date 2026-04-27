# ════════════════════════════════════════════════════════════════
#  DTI Predictor + ADMET Analysis — Streamlit App
#  BT305 Course Project
# ════════════════════════════════════════════════════════════════

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
from PIL import Image

# ── Page config (MUST be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="DTI Predictor | Drug-Target Interaction & ADMET",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Fonts & base ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ───────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Gradient page header ───────────────────────────── */
.page-header {
    background: linear-gradient(135deg, #0d0e1a 0%, #131729 60%, #0a1628 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid rgba(108,99,255,.25);
    position: relative;
    overflow: hidden;
}
.page-header::before {
    content: '';
    position: absolute;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(108,99,255,.22), transparent 70%);
    top: -120px; right: -80px;
    border-radius: 50%;
}
.page-header::after {
    content: '';
    position: absolute;
    width: 250px; height: 250px;
    background: radial-gradient(circle, rgba(0,212,170,.15), transparent 70%);
    bottom: -80px; left: 10%;
    border-radius: 50%;
}
.page-title {
    font-size: 2.5rem;
    font-weight: 800;
    letter-spacing: -.03em;
    line-height: 1.2;
    margin: 0 0 .6rem;
    background: linear-gradient(135deg, #e8eaff, #fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    position: relative; z-index: 1;
}
.page-subtitle {
    font-size: 1rem;
    color: #8b95c4;
    max-width: 680px;
    line-height: 1.7;
    position: relative; z-index: 1;
    margin: 0;
}
.badge-row {
    display: flex; flex-wrap: wrap; gap: .5rem;
    margin-top: 1.2rem;
    position: relative; z-index: 1;
}
.badge {
    background: rgba(108,99,255,.12);
    border: 1px solid rgba(108,99,255,.3);
    color: #9b95ff;
    border-radius: 100px;
    padding: .25rem .85rem;
    font-size: .75rem;
    font-weight: 600;
    letter-spacing: .05em;
}

/* ── Glass result cards ─────────────────────────────── */
.result-card {
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.1);
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-title-row {
    font-size: .75rem;
    font-weight: 700;
    letter-spacing: .09em;
    text-transform: uppercase;
    color: #5a6286;
    margin-bottom: .7rem;
    display: flex;
    align-items: center;
    gap: .4rem;
}

/* ── Metric boxes ───────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: .8rem;
    margin-bottom: 1rem;
}
.metric-box {
    background: rgba(255,255,255,.03);
    border: 1px solid rgba(255,255,255,.07);
    border-radius: 12px;
    padding: .9rem 1rem;
    text-align: center;
}
.metric-label {
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #5a6286;
    margin-bottom: .3rem;
}
.metric-value {
    font-size: 1.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #6c63ff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* ── Flag banners ───────────────────────────────────── */
.flag-ok   { background: rgba(0,212,170,.08);  border: 1px solid rgba(0,212,170,.2);  color: #00d4aa; border-radius: 10px; padding: .6rem 1rem; margin: .4rem 0; font-size: .88rem; font-weight: 500; }
.flag-warn { background: rgba(255,179,71,.08); border: 1px solid rgba(255,179,71,.2); color: #ffb347; border-radius: 10px; padding: .6rem 1rem; margin: .4rem 0; font-size: .88rem; font-weight: 500; }
.flag-bad  { background: rgba(255,77,109,.08); border: 1px solid rgba(255,77,109,.2); color: #ff4d6d; border-radius: 10px; padding: .6rem 1rem; margin: .4rem 0; font-size: .88rem; font-weight: 500; }

/* ── Probability big display ────────────────────────── */
.prob-display {
    text-align: center;
    padding: 1.5rem;
}
.prob-number {
    font-size: 3.5rem;
    font-weight: 800;
    font-family: 'JetBrains Mono', monospace;
    background: linear-gradient(135deg, #6c63ff, #00d4aa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: .5rem;
}
.prob-label { font-size: .8rem; color: #5a6286; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; }
.interp-badge {
    display: inline-block;
    padding: .5rem 1.4rem;
    border-radius: 100px;
    font-size: .9rem;
    font-weight: 700;
    margin-top: .8rem;
}
.interp-high     { background: rgba(0,212,170,.12);  color: #00d4aa; border: 1px solid rgba(0,212,170,.3);  }
.interp-moderate { background: rgba(255,179,71,.12); color: #ffb347; border: 1px solid rgba(255,179,71,.3); }
.interp-low      { background: rgba(255,77,109,.12); color: #ff4d6d; border: 1px solid rgba(255,77,109,.3);  }

/* ── Similar drug cards ─────────────────────────────── */
.sim-card {
    background: rgba(255,255,255,.035);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 14px;
    padding: .9rem;
    text-align: center;
    transition: all .2s;
}
.sim-card:hover { border-color: rgba(108,99,255,.4); }
.sim-card-rank {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 22px; height: 22px;
    border-radius: 50%;
    background: linear-gradient(135deg,#6c63ff,#00d4aa);
    color:#fff;
    font-size:.68rem;
    font-weight:800;
    margin-bottom:.5rem;
}
.sim-card-name { font-size: .85rem; font-weight: 700; margin-bottom: .3rem; }
.sim-card-smiles {
    font-family: 'JetBrains Mono', monospace;
    font-size: .65rem;
    color: #5a6286;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin: .3rem 0;
}
.sim-score-bar-wrap { margin: .5rem 0 .2rem; }

/* ── Section divider ────────────────────────────────── */
.sec-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(108,99,255,.3), transparent);
    margin: 1.8rem 0;
}

/* ── Streamlit widget overrides ─────────────────────── */
div[data-testid="stTextInput"] input,
div[data-testid="stTextArea"] textarea {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .85rem !important;
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    border-radius: 10px !important;
    color: #e8eaff !important;
}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {
    border-color: rgba(108,99,255,.6) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,.12) !important;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #6c63ff, #00d4aa) !important;
    border: none !important;
    border-radius: 100px !important;
    font-weight: 700 !important;
    padding: .6rem 2rem !important;
    box-shadow: 0 4px 20px rgba(108,99,255,.4) !important;
}
button[kind="primary"]:hover { box-shadow: 0 6px 30px rgba(108,99,255,.6) !important; }
div[data-testid="stTabs"] button {
    font-weight: 600 !important;
    letter-spacing: .02em !important;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  CHEMISTRY & MODEL DEFINITIONS (mirrors the original notebook)
# ════════════════════════════════════════════════════════════════

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw
from rdkit.DataStructs import TanimotoSimilarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score

# ── Amino-acid encoding ──────────────────────────────────────────
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
aa_dict     = {aa: i for i, aa in enumerate(amino_acids)}
MAX_LEN     = 512

def smiles_to_fp(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    return torch.tensor(list(fp), dtype=torch.float)

def encode_protein(seq: str):
    arr = [aa_dict.get(a, 0) for a in seq[:MAX_LEN]]
    if len(arr) < MAX_LEN:
        arr += [0] * (MAX_LEN - len(arr))
    return torch.tensor(arr, dtype=torch.long)


# ── Model architecture ───────────────────────────────────────────
class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.fc(x), dim=0)
        output  = (weights * x).sum(dim=0)
        return output, weights


class Final_DTI_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.drug_fc    = nn.Linear(1024, 256)
        self.embed      = nn.Embedding(20, 128)
        encoder_layer   = nn.TransformerEncoderLayer(d_model=128, nhead=4)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn       = Attention(128)
        self.fc1        = nn.Linear(256 + 128, 128)
        self.fc2        = nn.Linear(128, 1)
        self.relu       = nn.ReLU()
        self.sigmoid    = nn.Sigmoid()

    def forward(self, drug, protein):
        d               = self.relu(self.drug_fc(drug))
        p               = self.embed(protein).unsqueeze(1)
        p               = self.transformer(p)
        p, attn_weights = self.attn(p)
        combined        = torch.cat((d, p.squeeze()), dim=0)
        out             = self.relu(self.fc1(combined))
        out             = self.sigmoid(self.fc2(out))
        return out, attn_weights


# ════════════════════════════════════════════════════════════════
#  CACHED RESOURCE: Load dataset + train model once per session
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_and_train():
    """Downloads BindingDB, builds dataset, trains model. Cached."""
    # ── Dataset ────────────────────────────────────────────────
    from tdc.multi_pred import DTI as TDC_DTI
    data = TDC_DTI(name="BindingDB_Kd")
    data.harmonize_affinities(mode="max_affinity")
    df   = data.get_data()
    df["interaction"] = df["Y"].apply(lambda x: 1 if x < 1000 else 0)
    df   = df[["Drug", "Target", "interaction"]].sample(2000, random_state=42)

    # ── Featurise ──────────────────────────────────────────────
    dataset = []
    for _, row in df.iterrows():
        d = smiles_to_fp(row["Drug"])
        p = encode_protein(row["Target"])
        if d is not None:
            dataset.append((d, p, row["interaction"]))

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # ── Train ──────────────────────────────────────────────────
    model     = Final_DTI_Model()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    losses    = []

    for epoch in range(5):
        total_loss = 0
        for d, p, label in train_data:
            optimizer.zero_grad()
            pred, _ = model(d, p)
            loss    = criterion(pred, torch.tensor([label], dtype=torch.float))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        losses.append(total_loss)

    model.eval()

    # ── Evaluate ───────────────────────────────────────────────
    y_true, y_pred, y_prob = [], [], []
    for d, p, label in dataset[500:800]:
        pred, _ = model(d, p)
        prob    = pred.item()
        y_true.append(label)
        y_prob.append(prob)
        y_pred.append(1 if prob > 0.5 else 0)

    metrics = {
        "Accuracy": round(accuracy_score(y_true, y_pred), 4),
        "F1 Score": round(f1_score(y_true, y_pred), 4),
        "ROC-AUC":  round(roc_auc_score(y_true, y_prob), 4),
        "PR-AUC":   round(average_precision_score(y_true, y_prob), 4),
    }

    return model, df, losses, metrics, (y_true, y_prob)


# ════════════════════════════════════════════════════════════════
#  ANALYSIS HELPERS
# ════════════════════════════════════════════════════════════════

def compute_admet(smiles: str) -> dict | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mw          = round(Descriptors.MolWt(mol), 2)
    logp        = round(Descriptors.MolLogP(mol), 2)
    h_donors    = Descriptors.NumHDonors(mol)
    h_acceptors = Descriptors.NumHAcceptors(mol)
    lipinski_ok = mw < 500 and logp < 5 and h_donors <= 5 and h_acceptors <= 10
    bbb         = logp > 2
    toxicity    = mw > 600
    return {
        "mw": mw, "logp": logp,
        "h_donors": h_donors, "h_acceptors": h_acceptors,
        "lipinski_ok": lipinski_ok, "bbb": bbb, "toxicity": toxicity,
    }


def find_similar_drugs(query_smiles: str, df: pd.DataFrame, top_n: int = 5):
    query_mol = Chem.MolFromSmiles(query_smiles)
    if query_mol is None:
        return []
    query_fp   = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=1024)
    similarities = []
    for sm in df["Drug"].head(200):
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            continue
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        sim = TanimotoSimilarity(query_fp, fp)
        similarities.append((sm, round(sim, 3), mol))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]


def mol_to_pil(mol, size=(280, 200)):
    return Draw.MolToImage(mol, size=size)


def mol_grid_pil(mols, legends=None, per_row=5, sub_size=(220, 180)):
    return Draw.MolsToGridImage(
        mols,
        molsPerRow=per_row,
        subImgSize=sub_size,
        legends=legends or [""] * len(mols),
        returnPNG=False,
    )


# ════════════════════════════════════════════════════════════════
#  PLOT HELPERS
# ════════════════════════════════════════════════════════════════

_DARK = "#0d0e1a"
_GRID = "#1e2035"
_TEXT = "#8b95c4"

def _fig_style(fig, ax):
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)
    ax.yaxis.label.set_color(_TEXT)
    ax.title.set_color("#e8eaff")
    ax.spines[:].set_color(_GRID)
    ax.grid(color=_GRID, linewidth=.6)
    return fig, ax

def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130,
                facecolor=_DARK, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

def plot_loss(losses):
    fig, ax = plt.subplots(figsize=(5, 2.8))
    _fig_style(fig, ax)
    ax.plot(range(1, len(losses) + 1), losses,
            color="#6c63ff", linewidth=2, marker="o", markersize=5,
            markerfacecolor="#00d4aa", markeredgecolor="#6c63ff")
    ax.fill_between(range(1, len(losses) + 1), losses,
                    alpha=.15, color="#6c63ff")
    ax.set_xlabel("Epoch"), ax.set_ylabel("Total Loss")
    ax.set_title("Training Loss Curve", fontweight="bold")
    ax.set_xticks(range(1, len(losses) + 1))
    return fig_to_bytes(fig)

def plot_roc(y_true, y_prob):
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc     = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _fig_style(fig, ax)
    ax.plot(fpr, tpr, color="#6c63ff", lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.fill_between(fpr, tpr, alpha=.12, color="#6c63ff")
    ax.plot([0, 1], [0, 1], "--", color=_TEXT, lw=1)
    ax.set_xlabel("False Positive Rate"), ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", fontweight="bold")
    ax.legend(facecolor=_DARK, edgecolor=_GRID, labelcolor="#e8eaff")
    return fig_to_bytes(fig)

def plot_pr(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    _fig_style(fig, ax)
    ax.plot(recall, precision, color="#00d4aa", lw=2)
    ax.fill_between(recall, precision, alpha=.12, color="#00d4aa")
    ax.set_xlabel("Recall"), ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", fontweight="bold")
    return fig_to_bytes(fig)

def score_gauge_fig(prob: float):
    """Circular gauge for interaction probability."""
    fig, ax = plt.subplots(figsize=(2.8, 2.8),
                           subplot_kw=dict(projection="polar"))
    fig.patch.set_facecolor(_DARK)
    ax.set_facecolor(_DARK)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticks([])

    # Background ring
    theta_bg = np.linspace(0, 2 * np.pi, 300)
    ax.plot(theta_bg, [0.75] * 300, color=_GRID, lw=14, solid_capstyle="round")

    # Filled arc
    theta_fill = np.linspace(0, 2 * np.pi * prob, 300)
    if len(theta_fill) > 1:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("g", ["#6c63ff", "#00d4aa"])
        for j in range(len(theta_fill) - 1):
            c = cmap(j / len(theta_fill))
            ax.plot(theta_fill[j:j+2], [0.75, 0.75], color=c, lw=14,
                    solid_capstyle="round")

    ax.text(0, 0, f"{prob:.3f}", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#e8eaff",
            fontfamily="monospace")
    ax.spines[:].set_visible(False)
    return fig_to_bytes(fig)


# ════════════════════════════════════════════════════════════════
#  UI — SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("### ⚛️ DTIPredict")
    st.caption("BT305 Course Project")
    st.markdown("---")

    st.markdown("**⚡ Quick Examples**")
    ex = st.radio(
        "Load example",
        ["Custom input", "💊 Aspirin", "☕ Caffeine", "🧪 Ethanol"],
        label_visibility="collapsed",
    )

    EXAMPLES = {
        "💊 Aspirin": {
            "smiles":  "CC(=O)OC1=CC=CC=C1C(=O)O",
            "protein": "MDSKGSSQKGSRLLLLLVVSNLLLCQGVVSTPVQLPGAFSAQEEMVVLPFKDPRDNNLFIAVVRESVAGVHLVPAQDLKQPFIMHLDGYLDLFSATAQPAFTTQSNQIMGDPQMEAQHIEAAIESISKTSDPVSYGIDTQTCQYDSQNLKFEVVEGQTLRYLCYDIQAKQIRQFTQHYGVAFGELQAEEGFQHISSDGSQNILGYLANSPGALVYHQIGRQIDAGVDLIRSQQLQNFRKQLDNDIQIVFGDHSDQLIPNKQVTILHEGRSYQGYHLQQEKFLRGIQEMSHEQISAWELDTQAQLVQIKRPSDNIPQETITAEDFRSMEQLSELGQNISGLPDSTNIQFEYMNTDYTLPTARRRGFLSVLPTRSFLSLEQRRGMRLS",
        },
        "☕ Caffeine": {
            "smiles":  "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
            "protein": "MAHVRGLQLPGCLALAALCSLVHSQHVFLAPQQARSRGECVPAIQNATSSSPGPHVSSEGHEENPCGPGDRFSGRILHIQNLMKKHPGSQHRAGLKKLNFGEFSVSFQQQLGASVGSMKSLSGFSSVSVATSNNSTVACIDRNGLYDPDCDESGLFPASQIASITGGLLFIVVAVLVSIVFLLKYLSIRSNTIHYNYTGCPGPVRGSLNSLSSVGHSQTLSSTVPGTMDPNTAIASTSLGLKSDH",
        },
        "🧪 Ethanol": {
            "smiles":  "CCO",
            "protein": "MAHVRGLQLPGCLALAALCSLVH",
        },
    }

    st.markdown("---")
    st.markdown("**🔬 Model Info**")
    st.markdown("""
- Dataset: **BindingDB Kd** (2000 samples)
- Drug: **Morgan FP** (1024-bit, r=2)
- Protein: **Transformer Encoder** (2L, 4H, d=128)
- Training: **5 epochs**, Adam, lr=0.001
- Threshold: **Kd < 1000 nM**
    """)

    st.markdown("---")
    st.caption("Results are heuristic & for educational use only.")


# ════════════════════════════════════════════════════════════════
#  UI — HEADER
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div class="page-header">
  <div class="page-title">Drug–Target Interaction Predictor<br>&amp; ADMET Analyser</div>
  <p class="page-subtitle">
    Enter a drug's SMILES string and a protein sequence to predict binding probability,
    explore ADMET properties, and discover structurally similar molecules — powered by
    a Transformer + Attention deep-learning model trained on BindingDB.
  </p>
  <div class="badge-row">
    <span class="badge">Morgan Fingerprints</span>
    <span class="badge">Transformer Encoder</span>
    <span class="badge">Attention Mechanism</span>
    <span class="badge">Lipinski Rules</span>
    <span class="badge">Tanimoto Similarity</span>
    <span class="badge">RDKit</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  UI — MODEL LOADING (with spinner)
# ════════════════════════════════════════════════════════════════

with st.spinner("🔬 Loading BindingDB dataset & training model… (first run only, ~2 min)"):
    try:
        model, df_data, losses, metrics, eval_data = load_and_train()
        model_ready = True
    except Exception as e:
        st.error(f"⚠️ Model loading failed: {e}")
        model_ready = False


# ════════════════════════════════════════════════════════════════
#  UI — STATS BAR
# ════════════════════════════════════════════════════════════════

if model_ready:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  metrics["Accuracy"])
    c2.metric("F1 Score",  metrics["F1 Score"])
    c3.metric("ROC-AUC",   metrics["ROC-AUC"])
    c4.metric("PR-AUC",    metrics["PR-AUC"])

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  UI — INPUT FORM
# ════════════════════════════════════════════════════════════════

# Pre-fill from example picker
default_smiles  = ""
default_protein = ""
if ex != "Custom input" and ex in EXAMPLES:
    default_smiles  = EXAMPLES[ex]["smiles"]
    default_protein = EXAMPLES[ex]["protein"]

col_in1, col_in2 = st.columns([1, 1], gap="large")

with col_in1:
    st.markdown("#### 💊 Drug SMILES String")
    smiles_input = st.text_input(
        "SMILES",
        value=default_smiles,
        placeholder="e.g. CC(=O)OC1=CC=CC=C1C(=O)O",
        label_visibility="collapsed",
    )

with col_in2:
    st.markdown("#### 🧬 Protein Amino-Acid Sequence")
    protein_input = st.text_area(
        "Protein",
        value=default_protein,
        placeholder="e.g. MAHVRGLQLPGCLALAALCSLVH...",
        height=100,
        label_visibility="collapsed",
    )

if protein_input:
    st.caption(f"Sequence length: **{len(protein_input.strip())}** residues "
               f"({'✅ within limit' if len(protein_input.strip()) <= 512 else '⚠️ truncated to 512'})")

run_btn = st.button("⚡ Run Full Analysis", type="primary", use_container_width=False,
                    disabled=not model_ready)


# ════════════════════════════════════════════════════════════════
#  ANALYSIS
# ════════════════════════════════════════════════════════════════

if run_btn:
    smiles  = smiles_input.strip()
    protein = protein_input.strip().replace(" ", "").replace("\n", "")

    # ── Input validation ────────────────────────────────────────
    if not smiles:
        st.error("⚠️ Please enter a valid SMILES string.")
        st.stop()

    mol_query = Chem.MolFromSmiles(smiles)
    if mol_query is None:
        st.error("⚠️ Invalid SMILES — couldn't parse the molecule. Please check your input.")
        st.stop()

    if not protein:
        st.error("⚠️ Please enter a protein sequence.")
        st.stop()

    # ── Compute ─────────────────────────────────────────────────
    with st.spinner("Running prediction…"):
        d_fp = smiles_to_fp(smiles)
        p_enc = encode_protein(protein)
        with torch.no_grad():
            pred, _ = model(d_fp, p_enc)
        prob = pred.item()

        admet   = compute_admet(smiles)
        similar = find_similar_drugs(smiles, df_data, top_n=5)

    st.markdown('<div class="sec-divider"></div>', unsafe_allow_html=True)
    st.markdown("## 📊 Analysis Report")

    # ── Tabs ─────────────────────────────────────────────────────
    tab_pred, tab_admet, tab_struct, tab_similar, tab_model = st.tabs([
        "🔵 Prediction", "🧪 ADMET Profile",
        "⚛️ Molecule Structure", "🔍 Similar Drugs", "📈 Model Metrics",
    ])

    # ═══════════════════════════════════════════════════
    #  TAB 1 — Prediction
    # ═══════════════════════════════════════════════════
    with tab_pred:
        col_gauge, col_interp = st.columns([1, 2], gap="large")

        with col_gauge:
            gauge_bytes = score_gauge_fig(prob)
            st.image(gauge_bytes, use_container_width=True)

        with col_interp:
            st.markdown("### Interaction Probability")
            st.markdown(
                f'<div class="prob-display">'
                f'<div class="prob-number">{prob:.4f}</div>'
                f'<div class="prob-label">binding probability</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if prob > 0.8:
                badge_cls, badge_txt = "interp-high", "⚡ Very High Binding"
                interp_txt = "Very high probability that the drug strongly binds to the protein."
            elif prob > 0.6:
                badge_cls, badge_txt = "interp-high", "✅ High Probability"
                interp_txt = "High probability of binding between the drug and the protein."
            elif prob > 0.4:
                badge_cls, badge_txt = "interp-moderate", "⚠️ Moderate Interaction"
                interp_txt = "Moderate interaction possible — further wet-lab validation recommended."
            else:
                badge_cls, badge_txt = "interp-low", "❌ Low Probability"
                interp_txt = "Low probability — drug is unlikely to bind effectively to this target."

            st.markdown(
                f'<div style="text-align:center;">'
                f'<span class="interp-badge {badge_cls}">{badge_txt}</span></div>'
                f'<p style="color:#8b95c4;margin-top:.8rem;font-size:.95rem;">{interp_txt}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.markdown("**Input Summary**")
        ic1, ic2 = st.columns(2)
        ic1.code(smiles, language=None)
        ic2.code(protein[:80] + ("…" if len(protein) > 80 else ""), language=None)

    # ═══════════════════════════════════════════════════
    #  TAB 2 — ADMET
    # ═══════════════════════════════════════════════════
    with tab_admet:
        if admet is None:
            st.error("Could not compute ADMET — invalid SMILES.")
        else:
            # Metrics
            st.markdown(
                f"""
                <div class="metric-grid">
                  <div class="metric-box">
                    <div class="metric-label">Mol. Weight (Da)</div>
                    <div class="metric-value">{admet['mw']}</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-label">LogP</div>
                    <div class="metric-value">{admet['logp']}</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-label">H-Donors</div>
                    <div class="metric-value">{admet['h_donors']}</div>
                  </div>
                  <div class="metric-box">
                    <div class="metric-label">H-Acceptors</div>
                    <div class="metric-value">{admet['h_acceptors']}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Flags
            lip_cls  = "flag-ok"  if admet["lipinski_ok"] else "flag-bad"
            lip_txt  = ("✅ Drug-Like Molecule (all Lipinski rules passed)"
                        if admet["lipinski_ok"]
                        else "❌ May have poor drug-likeness (one or more Lipinski rules violated)")
            bbb_cls  = "flag-ok"  if admet["bbb"]     else "flag-warn"
            bbb_txt  = ("🧠 Likely to cross the Blood–Brain Barrier (LogP > 2)"
                        if admet["bbb"]
                        else "🚫 Unlikely to cross the Blood–Brain Barrier (LogP ≤ 2)")
            tox_cls  = "flag-bad" if admet["toxicity"] else "flag-ok"
            tox_txt  = ("⚠️ Higher toxicity risk — molecular weight > 600 Da"
                        if admet["toxicity"]
                        else "✅ Lower toxicity risk (MW ≤ 600 Da)")

            st.markdown(
                f'<div class="{lip_cls}">{lip_txt}</div>'
                f'<div class="{bbb_cls}">{bbb_txt}</div>'
                f'<div class="{tox_cls}">{tox_txt}</div>',
                unsafe_allow_html=True,
            )

            # Lipinski table
            st.markdown("---")
            st.markdown("**Lipinski Rule of Five — Detail**")
            lip_df = pd.DataFrame([
                {"Property": "Molecular Weight (Da)", "Value": admet["mw"],         "Threshold": "< 500",  "Pass": admet["mw"] < 500},
                {"Property": "LogP",                  "Value": admet["logp"],        "Threshold": "< 5",    "Pass": admet["logp"] < 5},
                {"Property": "H-Bond Donors",          "Value": admet["h_donors"],   "Threshold": "≤ 5",    "Pass": admet["h_donors"] <= 5},
                {"Property": "H-Bond Acceptors",       "Value": admet["h_acceptors"],"Threshold": "≤ 10",   "Pass": admet["h_acceptors"] <= 10},
            ])
            lip_df["Status"] = lip_df["Pass"].map({True: "✅ Pass", False: "❌ Fail"})
            st.dataframe(
                lip_df[["Property", "Value", "Threshold", "Status"]],
                use_container_width=True,
                hide_index=True,
            )

    # ═══════════════════════════════════════════════════
    #  TAB 3 — Molecule Structure
    # ═══════════════════════════════════════════════════
    with tab_struct:
        img = mol_to_pil(mol_query, size=(560, 320))
        st.image(img, caption=f"2D Structure: {smiles}", use_container_width=False)
        st.code(smiles, language=None)

    # ═══════════════════════════════════════════════════
    #  TAB 4 — Similar Drugs
    # ═══════════════════════════════════════════════════
    with tab_similar:
        if not similar:
            st.info("No similar drugs found — check your SMILES input.")
        else:
            st.markdown(
                "Top 5 structurally similar drugs from **BindingDB** "
                "(ranked by real Morgan-fingerprint **Tanimoto similarity**)."
            )
            st.markdown("")

            # Card grid: 5 columns
            cols = st.columns(len(similar))
            for col, (sm, score, mol) in zip(cols, similar):
                with col:
                    pct = int(score * 100)
                    score_color = (
                        "#00d4aa" if pct > 60 else
                        "#ffb347" if pct > 30 else
                        "#ff4d6d"
                    )
                    # 2D molecule image
                    mol_img = mol_to_pil(mol, size=(220, 160))
                    st.image(mol_img, use_container_width=True)
                    # Score badge + bar
                    st.markdown(
                        f'<div style="text-align:center;margin-top:.3rem;">'
                        f'<span style="color:{score_color};font-weight:800;'
                        f'font-family:monospace;font-size:1.05rem;">{score:.3f}</span>'
                        f' <span style="color:#5a6286;font-size:.75rem;">Tanimoto</span></div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(pct, text=None)
                    st.code(sm, language=None)

            # Grid view
            st.markdown("---")
            st.markdown("**Grid View — All 5 Similar Drugs**")
            mols_only   = [m for _, _, m in similar]
            legends     = [f"Score: {s:.3f}" for _, s, _ in similar]
            grid_img    = mol_grid_pil(mols_only, legends=legends, per_row=5, sub_size=(240, 200))
            st.image(grid_img, use_container_width=True)

    # ═══════════════════════════════════════════════════
    #  TAB 5 — Model Metrics
    # ═══════════════════════════════════════════════════
    with tab_model:
        st.markdown("**Model Performance on Held-out Test Set**")
        met_col = st.columns(4)
        for (k, v), ccc in zip(metrics.items(), met_col):
            ccc.metric(k, v)

        st.markdown("")
        mc1, mc2, mc3 = st.columns(3)

        with mc1:
            st.markdown("**Training Loss Curve**")
            st.image(plot_loss(losses), use_container_width=True)

        with mc2:
            st.markdown("**ROC Curve**")
            st.image(plot_roc(*eval_data), use_container_width=True)

        with mc3:
            st.markdown("**Precision-Recall Curve**")
            st.image(plot_pr(*eval_data), use_container_width=True)

        st.markdown("---")
        st.markdown("**Architecture Summary**")
        arch_cols = st.columns(4)
        arch_info = [
            ("💊 Drug Branch",       "SMILES → Morgan FP (1024-bit) → Linear(1024→256) → ReLU"),
            ("🧬 Protein Branch",    "Sequence → Embedding(20,128) → Transformer(2L,4H) → Attention"),
            ("🔗 Fusion",            "Concat(256+128=384) → Linear(384→128) → Linear(128→1) → Sigmoid"),
            ("📊 ADMET",             "RDKit Descriptors: MolWt, LogP, H-donors, H-acceptors → Lipinski / BBB / Toxicity"),
        ]
        for col, (title, desc) in zip(arch_cols, arch_info):
            with col:
                st.markdown(f"**{title}**")
                st.caption(desc)
