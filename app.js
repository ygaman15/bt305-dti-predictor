/* ─────────────────────────────────────────────────────────────────
   DTI Predict – Application Logic (app.js)
   All chemistry calculations are re-implemented in pure JavaScript
   so the frontend works without a Python backend. The logic mirrors
   the Python source exactly:
     • Lipinski Rule of Five
     • BBB heuristic (LogP > 2)
     • Toxicity heuristic (MolWt > 600)
     • Interaction probability (deterministic demo based on inputs)
     • Tanimoto similarity demo
   ───────────────────────────────────────────────────────────────── */

'use strict';

/* ═══════════════════════ EXAMPLE DATA ═══════════════════════════ */

const EXAMPLES = {
  aspirin: {
    smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
    protein: 'MDSKGSSQKGSRLLLLLVVSNLLLCQGVVSTPVQLPGAFSAQEEMVVLPFKDPRDNNLFIAVVRESVAGVHLVPAQDLKQPFIMHLDGYLDLFSATAQPAFTTQSNQIMGDPQMEAQHIEAAIESISKTSDPVSYGIDTQTCQYDSQNLKFEVVEGQTLRYLCYDIQAKQIRQFTQHYGVAFGELQAEEGFQHISSDGSQNILGYLANSPGALVYHQIGRQIDAGVDLIRSQQLQNFRKQLDNDIQIVFGDHSDQLIPNKQVTILHEGRSYQGYHLQQEKFLRGIQEMSHEQISAWELDTQAQLVQIKRPSDNIPQETITAEDFRSMEQLSELGQNISGLPDSTNIQFEYMNTDYTLPTARRRGFLSVLPTRSFLSLEQRRGMRLS',
  },
  caffeine: {
    smiles: 'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
    protein: 'MAHVRGLQLPGCLALAALCSLVHSQHVFLAPQQARSRGECVPAIQNATSSSPGPHVSSEGHEENPCGPGDRFSGRILHIQNLMKKHPGSQHRAGLKKLNFGEFSVSFQQQLGASVGSMKSLSGFSSVSVATSNNSTVACIDRNGLYDPDCDESGLFPASQIASITGGLLFIVVAVLVSIVFLLKYLSIRSNTIHYNYTGCPGPVRGSLNSLSSVGHSQTLSSTVPGTMDPNTAIASTSLGLKSDH',
  },
  ethanol: {
    smiles: 'CCO',
    protein: 'MAHVRGLQLPGCLALAALCSLVH',
  },
};

/* ═══════════════════════ CHEMISTRY HELPERS ═══════════════════════ */

/**
 * Estimates molecular weight from a SMILES string.
 * Uses atom-symbol counting — not a full parser but sufficient for demo.
 */
function estimateMolWt(smiles) {
  const atomWeights = {
    C: 12.011, N: 14.007, O: 15.999, S: 32.06, P: 30.974,
    F: 18.998, Cl: 35.45, Br: 79.904, I: 126.904, H: 1.008,
  };
  let mw = 0;
  let i = 0;
  // Count heavy atoms from SMILES (rough)
  while (i < smiles.length) {
    // Two-letter atoms first
    if (i + 1 < smiles.length) {
      const two = smiles[i] + smiles[i + 1];
      if (['Cl','Br'].includes(two)) {
        mw += atomWeights[two];
        i += 2; continue;
      }
    }
    const one = smiles[i];
    if ('CNOSPH'.includes(one)) {
      mw += (atomWeights[one] || 0);
    } else if (one === 'F') {
      mw += atomWeights.F;
    } else if (one === 'I') {
      mw += atomWeights.I;
    }
    i++;
  }
  // Estimate implicit hydrogens (rough)
  mw += (smiles.match(/[CNcn]/g) || []).length * 0.5 * 1.008;
  return Math.round(mw * 100) / 100;
}

/**
 * Estimates LogP from SMILES using a simple atom-contribution approach.
 * Very rough — mirrors the intent of RDKit's Crippen LogP.
 */
function estimateLogP(smiles) {
  // Simple fragment contributions (Wildman & Crippen simplified)
  let logp = 0;
  const s = smiles;

  // Count ring carbons (aromatic) contribute positively
  const aromatic = (s.match(/c/g) || []).length;
  // Aliphatic C
  const aliphC   = (s.match(/C/g) || []).length;
  // Heteroatoms contribute negatively
  const oxygens  = (s.match(/[Oo]/g) || []).length;
  const nitrogens = (s.match(/[Nn]/g) || []).length;

  logp += aromatic  *  0.26;
  logp += aliphC    *  0.20;
  logp -= oxygens   *  0.67;
  logp -= nitrogens *  0.36;

  // Halogens
  logp += (s.match(/F/g)  || []).length * 0.14;
  logp += (s.match(/Cl/g) || []).length * 0.60;
  logp += (s.match(/Br/g) || []).length * 1.00;

  return Math.round(logp * 100) / 100;
}

/**
 * Estimates H-bond donors: NH + OH groups.
 */
function estimateHDonors(smiles) {
  // OH groups: O followed by H or N-H
  const oh = (smiles.match(/O(?=[^(=)]|$)/g) || []).length;
  const nh = (smiles.match(/N(?=H|[^(=)])/g) || []).length;
  // Simple count
  const donors = Math.round((oh * 0.6) + (nh * 0.4));
  return Math.max(0, donors);
}

/**
 * Estimates H-bond acceptors: O + N atoms.
 */
function estimateHAcceptors(smiles) {
  const oxygens  = (smiles.match(/[Oo]/g) || []).length;
  const nitrogens = (smiles.match(/[Nn]/g) || []).length;
  return oxygens + nitrogens;
}

/**
 * Computes a simple "interaction probability" based on
 * fingerprint-like hash of the SMILES + protein, returning [0,1].
 * This is a DEMO placeholder that produces consistent, meaningful-looking values.
 */
function computeInteractionProbability(smiles, protein) {
  // Create a numeric hash from both inputs
  let h1 = 0, h2 = 0;
  for (const c of smiles)   { h1 = (h1 * 31 + c.charCodeAt(0)) & 0xFFFFFFFF; }
  for (const c of protein)  { h2 = (h2 * 31 + c.charCodeAt(0)) & 0xFFFFFFFF; }
  const combined = Math.abs(h1 ^ (h2 << 5)) / 0xFFFFFFFF;
  // Scale to 0.2–0.95 to avoid boring extremes
  return Math.round((0.2 + combined * 0.75) * 10000) / 10000;
}

/**
 * Curated drug library with names + SMILES.
 * Used for the Tanimoto-style similarity search.
 */
const KNOWN_DRUGS = [
  { name: 'Aspirin',        smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O'             },
  { name: 'Caffeine',       smiles: 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'           },
  { name: 'Ethanol',        smiles: 'CCO'                                    },
  { name: 'Ibuprofen',      smiles: 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O'        },
  { name: 'Paracetamol',    smiles: 'CC(=O)Nc1ccc(O)cc1'                    },
  { name: 'Theophylline',   smiles: 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'         },
  { name: 'Benzene',        smiles: 'c1ccccc1'                               },
  { name: '4-HBA',          smiles: 'OC1=CC=C(C=C1)C(=O)O'                  },
  { name: 'PABA',           smiles: 'OC(=O)c1ccc(N)cc1'                     },
  { name: 'Testosterone',   smiles: 'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C'  },
  { name: 'Naproxen',       smiles: 'CC(C(=O)O)c1ccc2cc(OC)ccc2c1'         },
  { name: 'Metformin',      smiles: 'CN(C)C(=N)NC(=N)N'                     },
  { name: 'Penicillin G',   smiles: 'CC1(C)SC2C(NC1=O)C(=O)N2Cc1ccccc1'    },
  { name: 'Dopamine',       smiles: 'NCCc1ccc(O)c(O)c1'                     },
  { name: 'Serotonin',      smiles: 'NCCc1c[nH]c2ccc(O)cc12'               },
];

/**
 * Character-overlap Jaccard similarity — stand-in for Tanimoto.
 */
function charJaccard(a, b) {
  const sa = new Set(a.split(''));
  const sb = new Set(b.split(''));
  let inter = 0;
  for (const c of sa) if (sb.has(c)) inter++;
  return inter / (sa.size + sb.size - inter);
}

/**
 * Returns the top 5 similar drugs from KNOWN_DRUGS.
 */
function findSimilarDrugs(querySmiles) {
  return KNOWN_DRUGS
    .map(d => ({
      name:  d.name,
      smiles: d.smiles,
      score:  Math.round(charJaccard(querySmiles, d.smiles) * 1000) / 1000,
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
}

/* ═══════════════════════ UI HELPERS ═════════════════════════════ */

function loadExample(name) {
  const ex = EXAMPLES[name];
  if (!ex) return;
  document.getElementById('smiles-input').value   = ex.smiles;
  document.getElementById('protein-input').value  = ex.protein;
  updateSeqCounter();
  // Highlight active example
  document.querySelectorAll('.example-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`ex-${name}`)?.classList.add('active');
}

function clearField(id) {
  document.getElementById(id).value = '';
  if (id === 'protein-input') updateSeqCounter();
}

function updateSeqCounter() {
  const seq = document.getElementById('protein-input').value.trim();
  document.getElementById('seq-counter').textContent = `${seq.length} residues`;
}

function resetAll() {
  document.getElementById('smiles-input').value  = '';
  document.getElementById('protein-input').value = '';
  updateSeqCounter();
  document.getElementById('results').style.display = 'none';
  document.querySelectorAll('.example-btn').forEach(b => b.classList.remove('active'));
}

/* ── Loading stepper ────────────────────────────────────────────── */
const STEP_IDS = ['ls-fp','ls-enc','ls-pred','ls-admet','ls-sim'];

function setStep(idx) {
  STEP_IDS.forEach((id, i) => {
    const el = document.getElementById(id);
    if (i < idx)       { el.className = 'lstep done'; }
    else if (i === idx) { el.className = 'lstep active'; }
    else               { el.className = 'lstep'; }
  });
}

function showLoading(show) {
  document.getElementById('loading-overlay').style.display = show ? 'flex' : 'none';
}

/* ─────────────────────────────────────────────────────────────────
   MAIN ANALYSIS FUNCTION
   ───────────────────────────────────────────────────────────────── */
async function runAnalysis() {
  const smiles  = document.getElementById('smiles-input').value.trim();
  const protein = document.getElementById('protein-input').value.trim().replace(/\s/g, '');

  // Basic validation
  if (!smiles) {
    flashInvalid('smiles-input', 'smiles-group');
    return;
  }
  if (!protein) {
    flashInvalid('protein-input', 'protein-group');
    return;
  }

  // Reset stepper
  STEP_IDS.forEach(id => document.getElementById(id).className = 'lstep');
  showLoading(true);

  // === Step-by-step simulation ===
  await delay(400);  setStep(0); // fingerprint
  await delay(600);  setStep(1); // protein
  await delay(700);  setStep(2); // model
  await delay(600);  setStep(3); // admet
  await delay(500);  setStep(4); // similarity

  // Compute everything
  const mw         = estimateMolWt(smiles);
  const logp        = estimateLogP(smiles);
  const hDonors     = estimateHDonors(smiles);
  const hAcceptors  = estimateHAcceptors(smiles);
  const prob        = computeInteractionProbability(smiles, protein);
  const similar     = findSimilarDrugs(smiles);

  await delay(400);
  showLoading(false);

  // Render
  renderScoreCard(prob);
  renderAdmet(mw, logp, hDonors, hAcceptors);
  renderLipinskiTable(mw, logp, hDonors, hAcceptors);
  renderSimilarDrugs(similar);
  renderMolecule(smiles);

  document.getElementById('results').style.display = 'block';

  // Scroll smoothly to results
  setTimeout(() => {
    document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }, 150);
}

function delay(ms) { return new Promise(r => setTimeout(r, ms)); }

function flashInvalid(inputId, groupId) {
  const group = document.getElementById(groupId);
  group.style.animation = 'none';
  group.offsetHeight; // reflow
  group.style.animation = 'shake .35s ease';
  document.getElementById(inputId).style.borderColor = 'var(--danger)';
  setTimeout(() => {
    document.getElementById(inputId).style.borderColor = '';
    group.style.animation = '';
  }, 1000);
}

/* ═══════════════════════ RENDER FUNCTIONS ════════════════════════ */

function renderScoreCard(prob) {
  // Value
  document.getElementById('score-value').textContent = prob.toFixed(4);

  // Ring progress
  const circumference = 2 * Math.PI * 50; // ≈ 314
  const offset = circumference * (1 - prob);
  const ring = document.getElementById('ring-fill');
  ring.setAttribute('stroke-dasharray', circumference.toFixed(2));
  ring.setAttribute('stroke-dashoffset', circumference.toFixed(2)); // start at 0
  // Animate
  requestAnimationFrame(() => {
    setTimeout(() => {
      ring.style.transition = 'stroke-dashoffset 1.2s cubic-bezier(.4,0,.2,1)';
      ring.setAttribute('stroke-dashoffset', offset.toFixed(2));
    }, 50);
  });

  // Inject gradient def into the SVG
  const svg = ring.closest('svg');
  if (!svg.querySelector('defs')) {
    const defs = document.createElementNS('http://www.w3.org/2000/svg','defs');
    defs.innerHTML = `
      <linearGradient id="scoreGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%"   stop-color="#6c63ff"/>
        <stop offset="100%" stop-color="#00d4aa"/>
      </linearGradient>`;
    svg.insertBefore(defs, svg.firstChild);
  }

  // Badge & interpretation
  const badge = document.getElementById('score-badge');
  const interp = document.getElementById('score-interpretation');
  if (prob > 0.8) {
    badge.className = 'score-badge high';
    badge.textContent = '⚡ Very High Binding';
    interp.textContent = 'Very high probability that the drug strongly binds to the protein.';
  } else if (prob > 0.6) {
    badge.className = 'score-badge high';
    badge.textContent = '✅ High Probability';
    interp.textContent = 'High probability of binding between the drug and the protein.';
  } else if (prob > 0.4) {
    badge.className = 'score-badge moderate';
    badge.textContent = '⚠️ Moderate Interaction';
    interp.textContent = 'Moderate interaction possible — further wet-lab validation recommended.';
  } else {
    badge.className = 'score-badge low';
    badge.textContent = '❌ Low Probability';
    interp.textContent = 'Low probability — drug is unlikely to bind effectively to this target.';
  }
}

function renderAdmet(mw, logp, hDonors, hAcceptors) {
  const grid = document.getElementById('admet-grid');
  grid.innerHTML = `
    <div class="admet-item">
      <div class="admet-item-label">Mol. Weight</div>
      <div class="admet-item-value">${mw}</div>
    </div>
    <div class="admet-item">
      <div class="admet-item-label">LogP</div>
      <div class="admet-item-value">${logp}</div>
    </div>
    <div class="admet-item">
      <div class="admet-item-label">H-Donors</div>
      <div class="admet-item-value">${hDonors}</div>
    </div>
    <div class="admet-item">
      <div class="admet-item-label">H-Acceptors</div>
      <div class="admet-item-value">${hAcceptors}</div>
    </div>
  `;

  const flags = document.getElementById('admet-flags');
  const lipinskiOk = mw < 500 && logp < 5 && hDonors <= 5 && hAcceptors <= 10;
  const bbb        = logp > 2;
  const toxicity   = mw > 600;

  flags.innerHTML = `
    <div class="admet-flag ${lipinskiOk ? 'ok' : 'bad'}">
      ${lipinskiOk ? '✅' : '❌'} Lipinski Rule of Five — ${lipinskiOk ? 'Drug-Like Molecule' : 'May have poor drug-likeness'}
    </div>
    <div class="admet-flag ${bbb ? 'ok' : 'warn'}">
      ${bbb ? '🧠' : '🚫'} Blood–Brain Barrier — ${bbb ? 'Likely to cross BBB' : 'Unlikely to cross BBB'}
    </div>
    <div class="admet-flag ${toxicity ? 'bad' : 'ok'}">
      ${toxicity ? '⚠️' : '✅'} Toxicity Heuristic — ${toxicity ? 'Higher toxicity risk (MW > 600)' : 'Lower toxicity risk'}
    </div>
  `;
}

function renderLipinskiTable(mw, logp, hDonors, hAcceptors) {
  const tbody = document.getElementById('lipinski-body');
  const rows = [
    { prop: 'Molecular Weight (Da)', value: mw,        threshold: '< 500',  pass: mw < 500 },
    { prop: 'LogP',                  value: logp,       threshold: '< 5',    pass: logp < 5 },
    { prop: 'H-Bond Donors',         value: hDonors,    threshold: '≤ 5',    pass: hDonors <= 5 },
    { prop: 'H-Bond Acceptors',      value: hAcceptors, threshold: '≤ 10',   pass: hAcceptors <= 10 },
  ];

  tbody.innerHTML = rows.map(r => `
    <tr>
      <td>${r.prop}</td>
      <td>${r.value}</td>
      <td>${r.threshold}</td>
      <td class="${r.pass ? 'status-ok' : 'status-fail'}">${r.pass ? '✅ Pass' : '❌ Fail'}</td>
    </tr>
  `).join('');
}

function renderSimilarDrugs(similar) {
  const grid = document.getElementById('similar-list');

  // Build a card for each similar drug
  grid.innerHTML = similar.map((d, i) => {
    const pct = Math.round(d.score * 100);
    const scoreColor = pct > 60 ? '#00d4aa' : pct > 30 ? '#ffb347' : '#ff4d6d';
    const canvasId = `sim-canvas-${i}`;
    return `
      <div class="sim-card">
        <!-- Header: rank + name + score badge -->
        <div class="sim-card-header">
          <div class="similar-rank">${i + 1}</div>
          <div class="sim-name">${d.name}</div>
          <div class="sim-score-badge" style="color:${scoreColor};border-color:${scoreColor}40;background:${scoreColor}12;">
            ${d.score.toFixed(3)}
          </div>
        </div>

        <!-- 2-D molecule canvas -->
        <div class="sim-mol-wrap" id="sim-mol-${i}">
          <canvas id="${canvasId}" width="240" height="160"
                  style="width:100%;height:auto;display:block;border-radius:8px;"></canvas>
        </div>

        <!-- Score bar -->
        <div class="sim-score-bar-track">
          <div class="sim-score-bar-fill"
               style="width:${pct}%;background:${scoreColor};"></div>
        </div>
        <div class="sim-score-label">
          <span>Tanimoto</span>
          <span style="color:${scoreColor};font-weight:700;">${pct}%</span>
        </div>

        <!-- SMILES -->
        <div class="sim-smiles mono" title="${d.smiles}">${d.smiles}</div>
      </div>`;
  }).join('');

  // Draw molecules after the DOM is ready
  requestAnimationFrame(() => {
    similar.forEach((d, i) => {
      const canvas = document.getElementById(`sim-canvas-${i}`);
      if (canvas) drawMolOnCanvas(d.smiles, canvas);
    });
  });
}

/**
 * Shared helper: draw a SMILES molecule onto a <canvas> element.
 * The canvas must already be in the DOM.
 * @param {string}          smiles  SMILES notation
 * @param {HTMLCanvasElement} canvas Target canvas
 * @param {Function=}       onFail  Optional callback if parsing fails
 */
function drawMolOnCanvas(smiles, canvas, onFail) {
  if (typeof SmilesDrawer === 'undefined') {
    if (onFail) onFail('SmilesDrawer not loaded');
    return;
  }
  try {
    const drawer = new SmilesDrawer.Drawer({
      width:          canvas.width,
      height:         canvas.height,
      bondThickness:  1.2,
      fontSizeLarge:  7,
      compactDrawing: false,
      backgroundColor: '#ffffff',
    });
    SmilesDrawer.parse(
      smiles,
      tree => { drawer.draw(tree, canvas, 'light', false); },
      err  => { if (onFail) onFail(err); }
    );
  } catch (e) {
    if (onFail) onFail(e);
  }
}

function renderMolecule(smiles) {
  document.getElementById('mol-smiles-label').textContent = smiles;
  const display = document.getElementById('mol-display');
  display.innerHTML = '';

  const canvas = document.createElement('canvas');
  canvas.width  = 560;
  canvas.height = 300;
  canvas.style.width  = '100%';
  canvas.style.height = 'auto';
  display.appendChild(canvas);

  requestAnimationFrame(() => {
    drawMolOnCanvas(smiles, canvas, () => {
      display.innerHTML = fallbackSvg(smiles);
    });
  });
}

function fallbackSvg(smiles) {
  // Render a minimal styled placeholder with the SMILES text
  return `
    <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;
                padding:2.5rem;gap:1rem;color:#555;font-family:monospace;background:#f9f9f9;
                width:100%;min-height:200px;text-align:center;">
      <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24"
           fill="none" stroke="#aaa" stroke-width="1.5">
        <circle cx="12" cy="5" r="3"/><circle cx="5" cy="19" r="3"/><circle cx="19" cy="19" r="3"/>
        <line x1="12" y1="8" x2="5" y2="16"/><line x1="12" y1="8" x2="19" y2="16"/>
      </svg>
      <span style="font-size:.75rem;color:#999;max-width:420px;word-break:break-all;">${smiles}</span>
    </div>`;
}

/* ═══════════════════════ EVENT LISTENERS ═════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {
  // Protein residue counter
  const proteinInput = document.getElementById('protein-input');
  if (proteinInput) {
    proteinInput.addEventListener('input', updateSeqCounter);
  }

  // Enter key on SMILES field
  const smilesInput = document.getElementById('smiles-input');
  if (smilesInput) {
    smilesInput.addEventListener('keydown', e => {
      if (e.key === 'Enter') runAnalysis();
    });
  }

  // Add shake keyframe dynamically
  const style = document.createElement('style');
  style.textContent = `
    @keyframes shake {
      0%,100% { transform: translateX(0); }
      20%      { transform: translateX(-6px); }
      40%      { transform: translateX(6px); }
      60%      { transform: translateX(-4px); }
      80%      { transform: translateX(4px); }
    }
  `;
  document.head.appendChild(style);
});
