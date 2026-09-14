// ==========================================================================
// Data Constants & Models
// ==========================================================================

const PIPELINE = [
  { id: "text", label: "raw text", hint: "The Pile, instructions, preferences", sink: false },
  { id: "tokens", label: "tokens", hint: "tiktoken r50k_base → HDF5", sink: false },
  { id: "tau", label: "Transformer", hint: "MLP · Head · Block", sink: false },
  { id: "ce", label: "CE loss", hint: "next-token cross-entropy", sink: false },
  { id: "base", label: "base", hint: "a language model", sink: false },
  { id: "sft", label: "SFT", hint: "assistant tokens only", sink: false },
  { id: "rm", label: "RM", hint: "Bradley-Terry", sink: false },
  { id: "dpo", label: "DPO", hint: "no reward model", sink: false },
  { id: "ppo", label: "PPO", hint: "clipped surrogate", sink: false },
  { id: "grpo", label: "GRPO", hint: "group-relative", sink: false },
  { id: "chat", label: "chat", hint: "the sink", sink: true },
];

const THEOREMS = [
  {
    id: "T1",
    title: "τ is assembled from small pieces",
    body: "MLP, a causal Head, MultiHeadAttention, a pre-norm Block, then Embedding + N blocks + LayerNorm + lm_head. Each file is a few dozen lines of plain PyTorch. There is no transformers.Transformer.",
  },
  {
    id: "T2",
    title: "Pretraining is next-token CE",
    body: "Write logits over the vocabulary at every position, reshape, and call F.cross_entropy against the shifted tokens. The 77M run in the README falls from 11.14 train loss to 3.73 / 3.76 dev in 2 000 steps.",
  },
  {
    id: "T3",
    title: "SFT masks everything except the assistant",
    body: "Chat turns are packed with <|user|>, <|assistant|>, <|endoftext|>. A loss_mask zeros the user tokens so the model is graded only on the reply: ce * mask; return ce.sum() / mask.sum().",
  },
  {
    id: "T4",
    title: "The reward model is Bradley-Terry",
    body: "A scalar head on the same backbone scores (chosen, rejected) pairs. Loss is −log σ(chosen − rejected). The README reports 0.574 accuracy on 7 974 held-out pairs — a weak but real preference signal.",
  },
  {
    id: "T5",
    title: "DPO aligns without a reward model",
    body: "DPO: −log σ(β (π log-ratio − π_ref log-ratio)). ORPO is reference-free and folds SFT into the same step. KTO takes unpaired signal. Same small Transformer, real preference data (HH-RLHF / UltraFeedback).",
  },
  {
    id: "T6",
    title: "PPO is the clipped surrogate plus GAE",
    body: "ratio = exp(new_logp − old_logp). Loss = −min(ratio A, clip(ratio) A), with a value head and generalized advantage estimation. Reward can come from the RM or from a verifier.",
  },
  {
    id: "T7",
    title: "GRPO is group-relative and has no value head",
    body: "Sample a group of answers, set advantage to (r − mean) / (std + ε) inside the group. That is the whole baseline. The repo treats this as RLVR: reinforcement learning from a verifiable reward, not a learned critic.",
  },
  {
    id: "T8",
    title: "GSM8K greedy accuracy is the evaluation",
    body: "Generate greedily, parse the <answer> tag, compare to gold. eval_post_training.py prints one table across Base, SFT, DPO, PPO, and GRPO so the stages can be read as a single experiment.",
  },
  {
    id: "T9",
    title: "Every orbit is handwritten PyTorch",
    body: "No trl, no peft, no transformers. The same modules that pretrain are the modules that SFT, reward, PPO, DPO, and GRPO. The flow is almost everywhere the same idea: change the data, change the loss, keep the net.",
  },
];

const PIECES = [
  {
    id: "mlp",
    title: "MLP",
    file: "src/models/mlp.py",
    formula: "Linear(d, 4d) → ReLU → Linear(4d, d)",
    note: "The feed-forward expansion. One hidden layer, four times wide, then a projection back.",
    code: `class MLP(nn.Module):
    def __init__(self, n_embed):
        self.hidden = nn.Linear(n_embed, 4 * n_embed)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(4 * n_embed, n_embed)`,
  },
  {
    id: "head",
    title: "Head",
    file: "src/models/attention.py",
    formula: "softmax(q kᵀ / √d · tril) v",
    note: "One causal head. Keys, queries, and values have no bias. The future is −∞ before softmax.",
    code: `attn = q @ k.transpose(-2, -1) * (1 / sqrt(d))
attn = attn.masked_fill(tril[:T, :T] == 0, -inf)
return softmax(attn, -1) @ v`,
  },
  {
    id: "mha",
    title: "Multi-head",
    file: "src/models/attention.py",
    formula: "concat(h₁…hₙ) Wₒ",
    note: "n_head parallel Heads, concatenated, then a linear map back to n_embed.",
    code: `heads = [Head(n_embed // n_head, ...) for _ in range(n_head)]
x = torch.cat([h(x) for h in heads], dim=-1)
return self.proj(x)`,
  },
  {
    id: "block",
    title: "Block",
    file: "src/models/transformer_block.py",
    formula: "x + attn(LN(x)); x + mlp(LN(x))",
    note: "Pre-norm residuals. Attention first, MLP second. This is the atom of depth.",
    code: `x = x + self.attn(self.ln1(x))
x = x + self.mlp(self.ln2(x))
return x`,
  },
  {
    id: "tau",
    title: "Transformer",
    file: "src/models/transformer.py",
    formula: "Etok + Epos → Blockᴺ → LN → lm_head",
    note: "The whole net. generate() crops to the context window, softmaxes the last logit, and multinomials.",
    code: `x = token_embed(idx) + position_embed(pos)
for block in attn_blocks: x = block(x)
logits = lm_head(layer_norm(x))`,
  },
];

const VOCAB_SIZE = 50304;

const PRESETS = [
  { id: "13m", label: "13M", nEmbed: 128, nHead: 8, nBlocks: 1, contextLength: 256, reported: 13142656 },
  { id: "77m", label: "77M", nEmbed: 512, nHead: 8, nBlocks: 8, contextLength: 512, reported: 77031552 },
  { id: "406m", label: "406M", nEmbed: 1024, nHead: 16, nBlocks: 24, contextLength: 1024, reported: 406359168 },
  { id: "3b", label: "3B", nEmbed: 2048, nHead: 16, nBlocks: 64, contextLength: 512, reported: 3430733312 },
];

const POINTS = [
  { t: 0, train: 11.14, dev: 11.2 },
  { t: 250, train: 8.4, dev: 8.55 },
  { t: 500, train: 6.35, dev: 6.5 },
  { t: 1000, train: 4.82, dev: 4.95 },
  { t: 1500, train: 4.12, dev: 4.22 },
  { t: 2000, train: 3.73, dev: 3.76 },
];

const BASE_CORPUS = `In 1978, The park was returned to the factory-plate that
the public share to the lower of the electronic fence that
follow from the Station's cities. The Canal of ancient Western
nations were confined to the city spot. The villages were directly
linked to cities in China that revolt that the US budget and in
Odambinais is uncertain and fortune established in rural areas.
The transformer was returned to the factory of language that
the public share to the lower of the electronic fence.
Ancient Western nations were confined to the city spot and
the villages were directly linked to cities that revolt.
`;

const ORDER = 3;
function trainMarkov(corpus, order) {
  const table = new Map();
  const padded = corpus.replace(/\s+/g, " ").trim();
  for (let i = 0; i <= padded.length - order; i++) {
    const key = padded.slice(i, i + order);
    const next = padded[i + order] ?? "";
    const arr = table.get(key);
    if (arr) arr.push(next);
    else table.set(key, [next]);
  }
  return table;
}
const BASE_TABLE = trainMarkov(BASE_CORPUS, ORDER);

function sampleMarkov(table, prefix, maxChars, temperature) {
  let out = prefix;
  let guard = 0;
  while (out.length < prefix.length + maxChars && guard++ < 800) {
    const key = out.slice(-ORDER);
    const options = table.get(key);
    if (!options || options.length === 0) break;
    const pick = options[Math.floor(Math.random() * options.length)] ?? "";
    if (!pick) break;
    if (temperature < 0.4 && options.length > 2) {
      const counts = new Map();
      for (const ch of options) counts.set(ch, (counts.get(ch) ?? 0) + 1);
      let best = options[0];
      let bestN = 0;
      for (const [ch, n] of counts) {
        if (n > bestN) {
          best = ch;
          bestN = n;
        }
      }
      out += best;
    } else {
      out += pick;
    }
  }
  return out;
}

function generateBase(prompt, maxChars = 220, temperature = 0.95) {
  const seed = (prompt.trim() || "The ").slice(0, 80);
  let start = seed;
  if (start.length < ORDER) start = (BASE_CORPUS.slice(0, ORDER) + start).slice(0, ORDER);
  const body = sampleMarkov(BASE_TABLE, start.slice(-ORDER), maxChars, temperature);
  return (seed + body.slice(ORDER)).replace(/\s+/g, " ").trim();
}

const STAGE_REPLIES = {
  base: [
    {
      prompt: "What is 2 + 2?",
      reply: "In 1978, The park was returned to the factory-plate that the public share to the lower of the electronic fence that follow from the Station's cities.",
    },
    {
      prompt: "Explain attention.",
      reply: "The Canal of ancient Western nations were confined to the city spot. The villages were directly linked to cities in China that revolt that the US budget.",
    },
    {
      prompt: "A park in 1978.",
      reply: "In 1978, The park was returned to the factory-plate that the public share to the lower of the electronic fence that follow from the Station's cities.",
    },
  ],
  sft: [
    { prompt: "What is 2 + 2?", reply: "2 + 2 equals 4." },
    { prompt: "Explain attention.", reply: "Attention lets each token look at the others and decide which ones matter. In a causal language model, a token may only look at itself and what came before." },
    { prompt: "A park in 1978.", reply: "Write a short paragraph about a park that was returned to the public in 1978, with a fence, a canal, and nearby villages." },
  ],
  grpo: [
    { prompt: "What is 2 + 2?", reply: "Adding two and two gives four.\n<answer>4</answer>" },
    { prompt: "Explain attention.", reply: "Causal self-attention computes q kᵀ / √d, masks the future, softmaxes, and mixes values. That is the whole move.\n<answer>causal scaled-dot-product attention</answer>" },
    { prompt: "A park in 1978.", reply: "A public park restored in 1978, bounded by a fence and a canal, with villages linked to the city.\n<answer>1978 public park</answer>" },
  ],
};

const PROBLEM = {
  question: "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  gold: "72",
  reasoning: "April: 48. May: 48 / 2 = 24. Total: 48 + 24 = 72.",
};

// ==========================================================================
// Initialization & Logic
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
  initPipeline();
  initTokenizer();
  initModelCalculator();
  initFiveSeats();
  initLossSection();
  initTheorems();
  initEvalSection();
  initCopyBlocks();
});

// 1. Pipeline Stepper
function initPipeline() {
  const list = document.getElementById("pipeline-list");
  const currentLabel = document.getElementById("pipeline-current-label");
  const currentHint = document.getElementById("pipeline-current-hint");
  const walkBtn = document.getElementById("walk-btn");
  const resetBtn = document.getElementById("reset-walk-btn");

  let cursor = 0;
  let intervalId = null;

  function render() {
    list.innerHTML = "";
    PIPELINE.forEach((step, i) => {
      const li = document.createElement("li");
      li.className = "pipeline-item";

      const active = i === cursor;
      if (active) {
        const ring = document.createElement("span");
        ring.className = "dwell-ring";
        li.appendChild(ring);
      }

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `pipeline-btn ${active ? "active dwell-beat" : ""} ${step.sink && active ? "sink" : ""}`;
      btn.textContent = step.label;
      btn.onclick = () => {
        stopWalk();
        cursor = i;
        render();
      };
      li.appendChild(btn);
      list.appendChild(li);
    });

    const curr = PIPELINE[cursor];
    currentLabel.innerHTML = `${curr.label}${curr.sink ? '<span class="text-mark"> ∞</span>' : ""}`;
    currentHint.textContent = curr.hint;
  }

  function startWalk() {
    stopWalk();
    cursor = 0;
    render();
    intervalId = setInterval(() => {
      if (cursor >= PIPELINE.length - 1) {
        stopWalk();
      } else {
        cursor++;
        render();
      }
    }, 900);
  }

  function stopWalk() {
    if (intervalId) {
      clearInterval(intervalId);
      intervalId = null;
    }
  }

  walkBtn.onclick = () => startWalk();
  resetBtn.onclick = () => {
    stopWalk();
    cursor = 0;
    render();
  };

  render();
}

// 2. Tokenizer Demo
function initTokenizer() {
  const input = document.getElementById("tok-input");
  const countEl = document.getElementById("tok-count");
  const chipsContainer = document.getElementById("tokens-container");

  const GPT2_RE = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  function standInHash(piece) {
    let h = 2166136261;
    for (let i = 0; i < piece.length; i++) {
      h ^= piece.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h >>> 0) % 50257;
  }

  function update() {
    const text = input.value;
    const pieces = text ? (text.match(GPT2_RE) ?? []) : [];
    countEl.textContent = `${pieces.length} pieces · stand-in IDs, not r50k_base`;

    if (pieces.length === 0) {
      chipsContainer.innerHTML = '<p class="text-subtle" style="font-size: 0.875rem;">Type something to see the cut.</p>';
      return;
    }

    chipsContainer.innerHTML = "";
    pieces.forEach((piece) => {
      const chip = document.createElement("span");
      chip.className = "token-chip";
      const displayPiece = piece === " " ? "␣" : piece;
      chip.innerHTML = `<span class="text-fg">${escapeHtml(displayPiece)}</span><span class="text-subtle tabular-nums">${standInHash(piece)}</span>`;
      chipsContainer.appendChild(chip);
    });
  }

  input.addEventListener("input", update);
  update();
}

// 3. Model Architecture Parameter Calculator
function initModelCalculator() {
  const presetContainer = document.getElementById("preset-buttons");
  const nembedInput = document.getElementById("input-nembed");
  const nheadInput = document.getElementById("input-nhead");
  const nblocksInput = document.getElementById("input-nblocks");
  const contextInput = document.getElementById("input-context");

  const totalEl = document.getElementById("param-total");
  const compactEl = document.getElementById("param-compact");
  const headSizeEl = document.getElementById("param-headsize");
  const brokenMsg = document.getElementById("param-broken-msg");
  const statsDl = document.getElementById("param-stats-dl");

  let activePreset = "13m";

  function countParams(spec) {
    const { nEmbed, nHead, nBlocks, contextLength, vocabSize } = spec;
    const headSize = Math.floor(nEmbed / nHead);
    const head = 3 * nEmbed * headSize;
    const attn = nHead * head + nEmbed * nEmbed + nEmbed;
    const mlp = nEmbed * (4 * nEmbed) + 4 * nEmbed + 4 * nEmbed * nEmbed + nEmbed;
    const ln = 4 * nEmbed;
    const block = attn + mlp + ln;
    const token = vocabSize * nEmbed;
    const pos = contextLength * nEmbed;
    const finalLn = 2 * nEmbed;
    const lmHead = nEmbed * vocabSize + vocabSize;
    return {
      total: token + pos + nBlocks * block + finalLn + lmHead,
      headSize,
    };
  }

  function formatInt(n) {
    const rounded = Math.round(n);
    const sign = rounded < 0 ? "−" : "";
    return sign + Math.abs(rounded).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  }

  function formatCompact(n) {
    if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 100_000_000 ? 0 : 1)}M`;
    if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
    return formatInt(n);
  }

  function calculate() {
    const nEmbed = Number(nembedInput.value) || 0;
    const nHead = Number(nheadInput.value) || 0;
    const nBlocks = Number(nblocksInput.value) || 0;
    const context = Number(contextInput.value) || 0;

    const broken = nHead < 1 || nEmbed % nHead !== 0;

    if (broken) {
      brokenMsg.style.display = "block";
      statsDl.style.display = "none";
      return;
    }

    brokenMsg.style.display = "none";
    statsDl.style.display = "flex";

    const p = PRESETS.find((x) => x.id === activePreset);
    const counted = countParams({
      nEmbed,
      nHead,
      nBlocks,
      contextLength: context,
      vocabSize: VOCAB_SIZE,
    });

    const displayTotal = p ? p.reported : counted.total;
    totalEl.textContent = formatInt(displayTotal);
    compactEl.textContent = formatCompact(displayTotal);
    headSizeEl.textContent = counted.headSize;
  }

  function applyPreset(id) {
    activePreset = id;
    renderPresets();
    const p = PRESETS.find((x) => x.id === id);
    if (p) {
      nembedInput.value = p.nEmbed;
      nheadInput.value = p.nHead;
      nblocksInput.value = p.nBlocks;
      contextInput.value = p.contextLength;
    }
    calculate();
  }

  function renderPresets() {
    presetContainer.innerHTML = "";
    [...PRESETS, { id: "custom", label: "custom" }].forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `preset-btn ${activePreset === p.id ? "active" : ""}`;
      btn.textContent = p.label;
      btn.onclick = () => applyPreset(p.id);
      presetContainer.appendChild(btn);
    });
  }

  [nembedInput, nheadInput, nblocksInput, contextInput].forEach((el) => {
    el.addEventListener("input", () => {
      activePreset = "custom";
      renderPresets();
      calculate();
    });
  });

  renderPresets();
  calculate();
}

// 4. Five Seats Switcher
function initFiveSeats() {
  const container = document.getElementById("piece-buttons");
  const formulaEl = document.getElementById("piece-formula");
  const noteEl = document.getElementById("piece-note");
  const codeEl = document.getElementById("piece-code");

  let activeId = "block";

  function render() {
    container.innerHTML = "";
    PIECES.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `seat-btn ${activeId === p.id ? "active" : ""}`;
      btn.innerHTML = `<span class="font-display italic" style="font-size: 1.25rem;">${p.title}</span><span class="font-mono text-subtle" style="display: block; margin-top: 0.25rem; font-size: 11px;">${p.file}</span>`;
      btn.onclick = () => {
        activeId = p.id;
        render();
      };
      container.appendChild(btn);
    });

    const piece = PIECES.find((p) => p.id === activeId) || PIECES[3];
    formulaEl.textContent = piece.formula;
    noteEl.textContent = piece.note;
    codeEl.textContent = piece.code;
  }

  render();
}

// 5. Next-Token Loss Scrubber & Generation
function initLossSection() {
  const slider = document.getElementById("loss-slider");
  const stepLabel = document.getElementById("loss-step-label");
  const trainVal = document.getElementById("loss-train-val");
  const devVal = document.getElementById("loss-dev-val");
  const trainBar = document.getElementById("loss-train-bar");
  const devBar = document.getElementById("loss-dev-bar");

  const stagesContainer = document.getElementById("stage-buttons");
  const promptsContainer = document.getElementById("prompt-buttons");
  const promptBadge = document.getElementById("stage-prompt-title");
  const promptText = document.getElementById("stage-prompt-text");
  const replyPre = document.getElementById("generated-reply");
  const sampleBtn = document.getElementById("sample-btn");
  const sampleBadge = document.getElementById("sample-badge");

  let currentStage = "base";
  let promptIdx = 0;
  let customGenerated = null;

  function lerpLoss(step) {
    const first = POINTS[0];
    const last = POINTS[POINTS.length - 1];
    if (step <= 0) return first;
    if (step >= 2000) return last;
    let a = first;
    let b = last;
    for (let i = 0; i < POINTS.length - 1; i++) {
      const lo = POINTS[i];
      const hi = POINTS[i + 1];
      if (step >= lo.t && step <= hi.t) {
        a = lo;
        b = hi;
        break;
      }
    }
    const u = (step - a.t) / (b.t - a.t || 1);
    return {
      t: step,
      train: a.train + (b.train - a.train) * u,
      dev: a.dev + (b.dev - a.dev) * u,
    };
  }

  function updateLoss() {
    const step = Number(slider.value);
    const loss = lerpLoss(step);
    stepLabel.textContent = `step ${step.toLocaleString("en-US")}`;
    trainVal.textContent = loss.train.toFixed(2);
    devVal.textContent = `train · dev ${loss.dev.toFixed(2)}`;

    trainBar.style.width = `${Math.min(100, (loss.train / 11.2) * 100)}%`;
    devBar.style.width = `${Math.min(100, (loss.dev / 11.2) * 100)}%`;
  }

  function updateGenUI() {
    stagesContainer.innerHTML = "";
    ["base", "sft", "grpo"].forEach((s) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `preset-btn ${currentStage === s ? "active" : ""}`;
      btn.textContent = s.toUpperCase();
      btn.onclick = () => {
        currentStage = s;
        customGenerated = null;
        updateGenUI();
      };
      stagesContainer.appendChild(btn);
    });

    promptsContainer.innerHTML = "";
    const replies = STAGE_REPLIES[currentStage];
    replies.forEach((p, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `preset-btn ${promptIdx === i ? "active" : ""}`;
      btn.textContent = p.prompt;
      btn.onclick = () => {
        promptIdx = i;
        customGenerated = null;
        updateGenUI();
      };
      promptsContainer.appendChild(btn);
    });

    const activePair = replies[promptIdx] || replies[0];
    promptBadge.textContent = `${currentStage} · prompt`;
    promptText.textContent = activePair.prompt;
    replyPre.textContent = customGenerated ?? activePair.reply;

    if (currentStage === "base") {
      sampleBtn.style.display = "inline-flex";
      sampleBadge.style.display = "none";
    } else {
      sampleBtn.style.display = "none";
      sampleBadge.style.display = "inline-flex";
    }
  }

  sampleBtn.onclick = () => {
    const activePair = STAGE_REPLIES.base[promptIdx] || STAGE_REPLIES.base[0];
    customGenerated = generateBase(activePair.prompt, 200, 0.95);
    replyPre.textContent = customGenerated;
  };

  slider.addEventListener("input", updateLoss);
  updateLoss();
  updateGenUI();
}

// 6. Post-Training Theorems
function initTheorems() {
  const list = document.getElementById("theorems-list");
  const titleEl = document.getElementById("theorem-title");
  const idEl = document.getElementById("theorem-id");
  const bodyEl = document.getElementById("theorem-body");

  let activeId = "T9";

  function render() {
    list.innerHTML = "";
    THEOREMS.forEach((t) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `theorem-btn ${activeId === t.id ? "active" : ""}`;
      btn.innerHTML = `<span class="font-mono text-mark" style="width: 2rem; flex-shrink: 0; font-size: 0.75rem;">${t.id}</span><span>${escapeHtml(t.title)}</span>`;
      btn.onclick = () => {
        activeId = t.id;
        render();
      };
      li.appendChild(btn);
      list.appendChild(li);
    });

    const theorem = THEOREMS.find((t) => t.id === activeId) || THEOREMS[8];
    idEl.textContent = theorem.id;
    titleEl.textContent = theorem.title;
    bodyEl.textContent = theorem.body;
  }

  render();
}

// 7. GSM8K Evaluation Section
function initEvalSection() {
  const form = document.getElementById("gsm8k-form");
  const input = document.getElementById("completion-input");
  const goldBtn = document.getElementById("btn-gold");
  const missBtn = document.getElementById("btn-miss");

  const resultsDl = document.getElementById("eval-results");
  const parsedEl = document.getElementById("eval-parsed");
  const verdictEl = document.getElementById("eval-verdict");

  function parseAnswer(raw) {
    const tagged = raw.match(/<answer>\s*([^<]+?)\s*<\/answer>/i);
    if (tagged) {
      const inside = tagged[1].trim();
      const cleaned = inside.replace(/[.,;:]+$/, "").trim();
      return cleaned || inside;
    }
    const last = raw.trim().split(/\s+/).at(-1);
    if (!last) return null;
    const num = last.replace(/[^0-9.-]/g, "").replace(/[.-]+$/, "");
    return num || last;
  }

  function evaluate() {
    const raw = input.value;
    const hasTag = /<answer>\s*([^<]+?)\s*<\/answer>/i.test(raw);
    const parsed = parseAnswer(raw);
    const isGold = parsed !== null && parsed.replace(/,/g, "") === PROBLEM.gold;

    // Verifier RL reward as per Fareed Khan's src/post_training/rewards.py
    let reward = 0.0;
    if (hasTag && isGold) reward = 1.2;      // Correct AND well formatted (+0.2 tag bonus)
    else if (hasTag && !isGold) reward = 0.2; // Format bonus only
    else if (!hasTag && isGold) reward = 1.0; // Correct without tag

    resultsDl.style.display = "flex";
    parsedEl.textContent = parsed ?? "—";
    verdictEl.textContent = `${isGold ? "match" : "miss"} (reward: ${reward.toFixed(1)})`;
    verdictEl.className = `font-mono ${isGold ? "text-fg" : "text-mark"}`;
  }

  form.onsubmit = (e) => {
    e.preventDefault();
    evaluate();
  };

  goldBtn.onclick = () => {
    input.value = "<answer>72</answer>";
    resultsDl.style.display = "none";
  };

  missBtn.onclick = () => {
    input.value = "I think the total is 48.";
    resultsDl.style.display = "none";
  };
}

// 8. Clipboard Copy Blocks
function initCopyBlocks() {
  const buttons = document.querySelectorAll(".copy-btn");
  buttons.forEach((btn) => {
    btn.onclick = async () => {
      const text = btn.getAttribute("data-code");
      const label = btn.querySelector(".copy-label");
      try {
        await navigator.clipboard.writeText(text);
        btn.classList.add("copied");
        if (label) label.textContent = "copied";
        setTimeout(() => {
          btn.classList.remove("copied");
          if (label) label.textContent = "copy";
        }, 1400);
      } catch {
        // Fallback for older browsers
        const ta = document.createElement("textarea");
        ta.value = text;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        if (label) label.textContent = "copied";
        setTimeout(() => {
          if (label) label.textContent = "copy";
        }, 1400);
      }
    };
  });
}

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, (tag) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  }[tag] || tag));
}
