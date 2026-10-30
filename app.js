/* ==========================================================================
   τ — Train LLM From Scratch — Interactive Engine
   Pure vanilla JS, no dependencies. Every interactive feature on the page.
   ========================================================================== */

(function () {
  "use strict";

  /* ----------------------------------------------------------------------
     Utility helpers
     ---------------------------------------------------------------------- */

  const $ = (sel, ctx = document) => ctx.querySelector(sel);
  const $$ = (sel, ctx = document) => Array.from(ctx.querySelectorAll(sel));
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  const lerp = (a, b, t) => a + (b - a) * t;
  const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);

  function fmtInt(n) {
    if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
    if (n >= 1e6) return (n / 1e6).toFixed(2) + "M";
    if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
    return String(n);
  }

  function fmtBytes(bytes) {
    if (bytes >= 1e12) return (bytes / 1e12).toFixed(2) + " TB";
    if (bytes >= 1e9) return (bytes / 1e9).toFixed(2) + " GB";
    if (bytes >= 1e6) return (bytes / 1e6).toFixed(2) + " MB";
    return (bytes / 1e3).toFixed(2) + " KB";
  }

  // Tiny string hash → simulated token id (32-bit unsigned)
  function hashStr(s) {
    let h = 2166136261 >>> 0;
    for (let i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619);
    }
    return (h % 50257) + 1; // shift past special tokens
  }

  /* ----------------------------------------------------------------------
     Section 1: Pipeline Stepper
     ---------------------------------------------------------------------- */

  const PIPELINE_STAGES = [
    { name: "raw text", hint: "The Pile, OpenWebText, FineWeb-Edu, StarCoder — raw UTF-8 bytes collected into 22+ heterogeneous subsets. No labels, no segmentation yet." },
    { name: "tokens", hint: "tiktoken r50k_base encodes UTF-8 byte strings into integer IDs from a 50 257-token vocabulary, padded to 50 304 for GPU tensor-core alignment (multiple of 64)." },
    { name: "Transformer", hint: "Decoder-only causal architecture: N identical blocks of Multi-Head Self-Attention + Position-wise Feed-Forward MLP, with a single lower-triangular causal mask." },
    { name: "CE loss", hint: "Cross-entropy on next-token predictions. L = -(1/T) Σ log p(y_{t+1} | x_≤t). Initial loss ≈ ln(V) = 10.83 for a uniform random init." },
    { name: "base", hint: "The pretrained model. It predicts plausible next tokens but doesn't follow instructions — it just continues text in the training distribution." },
    { name: "SFT", hint: "Supervised Fine-Tuning on (instruction, response) pairs. Loss is masked so gradients flow only through assistant tokens. Cheap, stable, essential first alignment step." },
    { name: "RM", hint: "Reward Model trained on (chosen, rejected) preference pairs via Bradley-Terry ranking. Replaces LM head with a scalar score. Optional precursor to PPO; bypassed by DPO." },
    { name: "DPO", hint: "Direct Preference Optimisation. Implicit reward formulation that directly optimises the policy on preference pairs without a separate RM or unstable online rollouts." },
    { name: "PPO", hint: "Proximal Policy Optimisation. Classic RLHF with clipped surrogate objective, actor + critic networks, and Generalised Advantage Estimation. Powerful but heavy." },
    { name: "GRPO", hint: "Group Relative Policy Optimisation. Samples G candidates per prompt, normalises rewards within the group, drops the critic entirely. State-of-the-art for verifiable reasoning." },
    { name: "chat", hint: "The unique sink of the τ substitution. Every upstream stage exists to certify this terminal state. The model now answers questions, follows instructions, and reasons step-by-step." }
  ];

  const pipelineList = $("#pipeline-list");
  const pipelineLabel = $("#pipeline-current-label");
  const pipelineHint = $("#pipeline-current-hint");
  const walkBtn = $("#walk-btn");
  const resetWalkBtn = $("#reset-walk-btn");
  let walkTimer = null;
  let walkIdx = 0;
  let activePipelineIdx = 0;

  function buildPipeline() {
    if (!pipelineList) return;
    PIPELINE_STAGES.forEach((stage, i) => {
      const li = document.createElement("li");
      li.className = "pipeline-item";

      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "pipeline-btn";
      btn.dataset.idx = String(i);
      if (i === PIPELINE_STAGES.length - 1) btn.classList.add("sink");

      const idx = document.createElement("span");
      idx.className = "font-mono text-subtle";
      idx.style.fontSize = "0.7rem";
      idx.textContent = String(i + 1).padStart(2, "0") + " ";

      btn.appendChild(idx);
      btn.appendChild(document.createTextNode(stage.name));

      btn.addEventListener("click", () => selectPipeline(i, true));
      li.appendChild(btn);
      pipelineList.appendChild(li);
    });
    selectPipeline(0, false);
  }

  function selectPipeline(idx, beat) {
    activePipelineIdx = idx;
    const btns = $$(".pipeline-btn", pipelineList);
    btns.forEach((b, i) => {
      b.classList.toggle("active", i === idx);
      if (beat && i === idx) {
        b.classList.remove("dwell-beat");
        // restart animation
        void b.offsetWidth;
        b.classList.add("dwell-beat");
      }
    });
    const stage = PIPELINE_STAGES[idx];
    if (pipelineLabel) {
      pipelineLabel.textContent = stage.name;
      pipelineLabel.classList.remove("fade-up");
      void pipelineLabel.offsetWidth;
      pipelineLabel.classList.add("fade-up");
    }
    if (pipelineHint) {
      pipelineHint.textContent = stage.hint;
      pipelineHint.classList.remove("fade-up");
      void pipelineHint.offsetWidth;
      pipelineHint.classList.add("fade-up");
    }
  }

  function walkPipeline() {
    if (walkTimer) {
      clearInterval(walkTimer);
      walkTimer = null;
      if (walkBtn) walkBtn.textContent = "Walk the source-line";
      return;
    }
    walkIdx = activePipelineIdx >= PIPELINE_STAGES.length - 1 ? 0 : activePipelineIdx + 1;
    if (walkBtn) walkBtn.textContent = "Pause";
    walkTimer = setInterval(() => {
      selectPipeline(walkIdx, true);
      walkIdx = (walkIdx + 1) % PIPELINE_STAGES.length;
    }, 1800);
  }

  function resetPipeline() {
    if (walkTimer) {
      clearInterval(walkTimer);
      walkTimer = null;
      if (walkBtn) walkBtn.textContent = "Walk the source-line";
    }
    selectPipeline(0, false);
  }

  if (walkBtn) walkBtn.addEventListener("click", walkPipeline);
  if (resetWalkBtn) resetWalkBtn.addEventListener("click", resetPipeline);
  buildPipeline();

  /* ----------------------------------------------------------------------
     Section 1: BPE Tokenizer Playground
     ---------------------------------------------------------------------- */

  // GPT-2 / tiktoken r50k_base style regex (approximate, JS-friendly)
  const GPT2_REGEX = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;

  const tokInput = $("#tok-input");
  const tokCount = $("#tok-count");
  const tokensContainer = $("#tokens-container");

  function tokenizeBPE(text) {
    const matches = text.match(GPT2_REGEX) || [];
    return matches.map((piece) => ({
      text: piece,
      id: hashStr(piece)
    }));
  }

  function escapeHtml(s) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderTokens() {
    if (!tokInput || !tokensContainer) return;
    const text = tokInput.value;
    const tokens = tokenizeBPE(text);
    tokensContainer.innerHTML = "";
    tokens.forEach((tok) => {
      const chip = document.createElement("span");
      chip.className = "token-chip";
      chip.innerHTML =
        '<span class="tok-text">' + escapeHtml(tok.text.replace(/ /g, "·")) + "</span>" +
        '<span class="tok-id">' + tok.id + "</span>";
      tokensContainer.appendChild(chip);
    });
    if (tokCount) {
      tokCount.textContent = tokens.length + " token" + (tokens.length === 1 ? "" : "s");
    }
  }

  if (tokInput) {
    tokInput.addEventListener("input", renderTokens);
    renderTokens();
  }

  /* ----------------------------------------------------------------------
     Section 2: Model Parameter Calculator + Donut Chart
     ---------------------------------------------------------------------- */

  const PRESETS = [
    { label: "13M",  nEmb: 192, nHead: 6,  nBlocks: 4,  ctx: 256  },
    { label: "77M",  nEmb: 384, nHead: 6,  nBlocks: 6,  ctx: 256  },
    { label: "124M", nEmb: 768, nHead: 12, nBlocks: 12, ctx: 1024 },
    { label: "350M", nEmb: 1024, nHead: 16, nBlocks: 24, ctx: 1024 },
    { label: "774M", nEmb: 1280, nHead: 20, nBlocks: 36, ctx: 1024 },
    { label: "1.5B", nEmb: 1600, nHead: 25, nBlocks: 48, ctx: 1024 },
    { label: "3B",   nEmb: 2048, nHead: 32, nBlocks: 32, ctx: 2048 }
  ];

  const VOCAB = 50304;
  const presetButtons = $("#preset-buttons");
  const inNemb = $("#input-nembed");
  const inNhead = $("#input-nhead");
  const inNblocks = $("#input-nblocks");
  const inCtx = $("#input-context");
  const paramBroken = $("#param-broken-msg");
  const paramTotal = $("#param-total");
  const paramCompact = $("#param-compact");
  const paramHeadSize = $("#param-headsize");
  const paramVram = $("#param-vram");
  const donutSegments = $("#donut-segments");
  const donutTotal = $("#donut-total");
  const donutLegend = $("#donut-legend");

  function buildPresets() {
    if (!presetButtons) return;
    PRESETS.forEach((p, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "preset-btn";
      btn.textContent = p.label;
      btn.addEventListener("click", () => applyPreset(i));
      presetButtons.appendChild(btn);
    });
  }

  function applyPreset(i) {
    const p = PRESETS[i];
    inNemb.value = p.nEmb;
    inNhead.value = p.nHead;
    inNblocks.value = p.nBlocks;
    inCtx.value = p.ctx;
    $$(".preset-btn", presetButtons).forEach((b, j) => b.classList.toggle("active", j === i));
    recalc();
  }

  function recalc() {
    const d = parseInt(inNemb.value, 10) || 0;
    const nHead = parseInt(inNhead.value, 10) || 0;
    const N = parseInt(inNblocks.value, 10) || 0;
    const T = parseInt(inCtx.value, 10) || 0;
    const V = VOCAB;

    const divisible = d > 0 && nHead > 0 && d % nHead === 0;
    if (paramBroken) paramBroken.style.display = divisible ? "none" : "block";

    // Parameter breakdown:
    //   Token embedding: V * d
    //   Position embedding: T * d
    //   Per block attention (QKV+proj with bias): 4*d*d + 4*d
    //   Per block MLP (4d expansion): 2*d*(4d) + 4d + d = 8*d*d + 5d
    //   Per block LayerNorms: 2*(2d) = 4d  (2 LN per block, each d scale+bias)
    //   Final LN: 2d
    //   LM head (untied): V*d + V
    // Simplified per-block total: 12d^2 + 13d (close to repo's 12d^2 + 10d)
    const embParams = V * d;
    const posParams = T * d;
    const attnPerBlock = 4 * d * d + 4 * d;
    const mlpPerBlock = 8 * d * d + 5 * d;
    const lnPerBlock = 4 * d;
    const blockPerBlock = attnPerBlock + mlpPerBlock + lnPerBlock;
    const finalLn = 2 * d;
    const lmHead = V * d + V;

    const attnTotal = attnPerBlock * N;
    const mlpTotal = mlpPerBlock * N;
    const lnTotal = lnPerBlock * N + finalLn;
    const blockTotal = blockPerBlock * N;

    const total = embParams + posParams + blockTotal + finalLn + lmHead;

    if (paramTotal) paramTotal.textContent = fmtInt(total) + " params";
    if (paramCompact) paramCompact.textContent = `τ(${d}, ${N}, ${T}, ${V})`;
    if (paramHeadSize) paramHeadSize.textContent = divisible ? (d / nHead) + " dim" : "invalid";

    // VRAM estimate: fp32 weights = total*4 bytes; training ≈ 4-8x weights (gradients, optim, activations)
    const trainVramBytes = total * 4 * 6;
    if (paramVram) paramVram.textContent = fmtBytes(trainVramBytes);

    if (donutTotal) donutTotal.textContent = fmtInt(total);

    // Donut chart segments
    const segments = [
      { name: "Token Emb",   value: embParams, color: "var(--seg-emb)" },
      { name: "Position Emb", value: posParams, color: "var(--seg-pos)" },
      { name: "Attention",    value: attnTotal, color: "var(--seg-attn)" },
      { name: "MLP",          value: mlpTotal, color: "var(--seg-mlp)" },
      { name: "LayerNorm",    value: lnTotal, color: "var(--seg-ln)" },
      { name: "LM Head",      value: lmHead, color: "var(--seg-head)" }
    ].filter((s) => s.value > 0);

    drawDonut(segments, total);
  }

  function drawDonut(segments, total) {
    if (!donutSegments || !donutLegend) return;
    donutSegments.innerHTML = "";
    donutLegend.innerHTML = "";

    const r = 80;
    const c = 2 * Math.PI * r;
    const cx = 100, cy = 100;
    let offset = 0;

    segments.forEach((seg) => {
      const pct = seg.value / total;
      const len = pct * c;

      const path = document.createElementNS("http://www.w3.org/2000", "circle");
      path.setAttribute("cx", String(cx));
      path.setAttribute("cy", String(cy));
      path.setAttribute("r", String(r));
      path.setAttribute("fill", "none");
      path.setAttribute("stroke", seg.color);
      path.setAttribute("stroke-width", "28");
      path.setAttribute("stroke-dasharray", `${len} ${c - len}`);
      path.setAttribute("stroke-dashoffset", String(-offset));
      path.setAttribute("transform", `rotate(-90 ${cx} ${cy})`);
      path.setAttribute("class", "donut-segment");
      donutSegments.appendChild(path);

      // Legend
      const li = document.createElement("li");
      li.innerHTML =
        `<span class="swatch" style="background:${seg.color}"></span>` +
        `<span class="lbl">${seg.name}</span>` +
        `<span class="pct">${(pct * 100).toFixed(1)}%</span>`;
      donutLegend.appendChild(li);

      offset += len;
    });
  }

  [inNemb, inNhead, inNblocks, inCtx].forEach((el) => {
    if (el) el.addEventListener("input", () => {
      $$(".preset-btn", presetButtons).forEach((b) => b.classList.remove("active"));
      recalc();
    });
  });

  buildPresets();
  applyPreset(1); // Default to 77M

  /* ----------------------------------------------------------------------
     Section 2: Five PyTorch Building Blocks
     ---------------------------------------------------------------------- */

  const PIECES = [
    {
      name: "Attention",
      desc: "causal self-attn",
      formula: "softmax((QKᵀ / √d_k) + mask) · V",
      note: "Q, K, V are linear projections of the input: Q = xW_Q, K = xW_K, V = xW_V. The causal mask is a lower-triangular matrix with -inf above the diagonal. The scale 1/√d_k prevents the softmax from saturating for large head dims.",
      code:
`class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd
        self.head_size = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_head, self.head_size)
        q, k, v = qkv.unbind(dim=2)
        # Flash attention handles causal mask implicitly
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_drop.p
        )
        return self.resid_drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))`
    },
    {
      name: "MLP",
      desc: "feedforward block",
      formula: "FFN(x) = W₂ · GeLU(W₁ · x + b₁) + b₂",
      note: "The MLP is the workhorse of the transformer — it accounts for ~66% of total parameters per block. The 4× expansion gives the model room to compute per-token features. GeLU is preferred over ReLU in modern LLMs for smoother gradients.",
      code:
`class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))`
    },
    {
      name: "Block",
      desc: "residual stack",
      formula: "x = x + Attn(LN(x));  x = x + MLP(LN(x))",
      note: "Pre-LayerNorm: normalisations are applied inside the residual path, not on it. This preserves the gradient flow and lets early layers see unnormalised input. The residual connection is the single most important architectural choice for training stability above 100M parameters.",
      code:
`class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x`
    },
    {
      name: "Head",
      desc: "lm projection",
      formula: "logits = LN(x) · W_lmᵀ  +  b_lm",
      note: "The LM head is UNTIED from the token embedding matrix. Tying them saves ~V·d parameters (often 25%+ for small models) but reduces expressivity. Untied is the from-scratch default for cleaner gradient bookkeeping.",
      code:
`class LMHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

    def forward(self, x):
        return self.proj(self.ln(x))`
    },
    {
      name: "Transformer",
      desc: "root model",
      formula: "y = LMHead(Blockᴺ(Emb(x) + Pos))",
      note: "Token + positional embeddings are summed elementwise. The position embedding is LEARNED, not sinusoidal. The full model is just: embed → N blocks → final LN → project to vocab logits. That is the entire architecture.",
      code:
`class Transformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([
            Block(cfg) for _ in range(cfg.n_layer)
        ])
        self.head = LMHead(cfg)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
            ignore_index=-1
        )
        return logits, loss`
    }
  ];

  const pieceButtons = $("#piece-buttons");
  const pieceFormula = $("#piece-formula");
  const pieceNote = $("#piece-note");
  const pieceCode = $("#piece-code");

  function highlightPy(code) {
    // Very small tokenizer: keywords, strings, comments, numbers, fn names
    const KW = ["def", "class", "return", "if", "else", "for", "in", "self", "import", "from", "super", "None", "True", "False", "lambda", "with", "as"];
    let out = "";
    let i = 0;
    while (i < code.length) {
      const ch = code[i];
      // Comment
      if (ch === "#") {
        let j = code.indexOf("\n", i);
        if (j < 0) j = code.length;
        out += '<span class="tok-com">' + escapeHtml(code.slice(i, j)) + "</span>";
        i = j;
        continue;
      }
      // String
      if (ch === '"' || ch === "'") {
        const q = ch;
        let j = i + 1;
        while (j < code.length && code[j] !== q) {
          if (code[j] === "\\") j++;
          j++;
        }
        j++;
        out += '<span class="tok-str">' + escapeHtml(code.slice(i, j)) + "</span>";
        i = j;
        continue;
      }
      // Identifier / keyword
      if (/[A-Za-z_]/.test(ch)) {
        let j = i;
        while (j < code.length && /[A-Za-z0-9_]/.test(code[j])) j++;
        const word = code.slice(i, j);
        // Check if next non-space char is "(" → function call
        let k = j;
        while (k < code.length && code[k] === " ") k++;
        const isCall = code[k] === "(";
        if (KW.indexOf(word) >= 0) {
          out += '<span class="tok-kw">' + word + "</span>";
        } else if (isCall) {
          out += '<span class="tok-fn">' + word + "</span>";
        } else {
          out += word;
        }
        i = j;
        continue;
      }
      // Number
      if (/[0-9]/.test(ch)) {
        let j = i;
        while (j < code.length && /[0-9.]/.test(code[j])) j++;
        out += '<span class="tok-num">' + code.slice(i, j) + "</span>";
        i = j;
        continue;
      }
      // Operators
      if ("+-*/=<>!&|^~@".indexOf(ch) >= 0) {
        let j = i;
        while (j < code.length && "+-*/=<>!&|^~@".indexOf(code[j]) >= 0) j++;
        out += '<span class="tok-op">' + escapeHtml(code.slice(i, j)) + "</span>";
        i = j;
        continue;
      }
      out += escapeHtml(ch);
      i++;
    }
    return out;
  }

  function buildPieces() {
    if (!pieceButtons) return;
    PIECES.forEach((p, i) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "seat-btn";
      btn.innerHTML =
        '<span class="seat-name">' + p.name + "</span>" +
        '<span class="seat-desc">' + p.desc + "</span>";
      btn.addEventListener("click", () => selectPiece(i));
      pieceButtons.appendChild(btn);
    });
    selectPiece(0);
  }

  function selectPiece(i) {
    const p = PIECES[i];
    $$(".seat-btn", pieceButtons).forEach((b, j) => b.classList.toggle("active", j === i));
    if (pieceFormula) {
      pieceFormula.textContent = p.formula;
      pieceFormula.classList.remove("fade-up");
      void pieceFormula.offsetWidth;
      pieceFormula.classList.add("fade-up");
    }
    if (pieceNote) {
      pieceNote.textContent = p.note;
      pieceNote.classList.remove("fade-up");
      void pieceNote.offsetWidth;
      pieceNote.classList.add("fade-up");
    }
    if (pieceCode) {
      pieceCode.innerHTML = highlightPy(p.code);
    }
  }

  buildPieces();

  /* ----------------------------------------------------------------------
     Section 2: Attention Heatmap
     ---------------------------------------------------------------------- */

  const ATTN_DIM = 8;
  const ATTN_HEADS = 8;
  const attnHeadButtons = $("#attn-head-buttons");
  const attnGrid = $("#attn-grid");
  const attnXLabels = $("#attn-x-labels");
  const attnYLabels = $("#attn-y-labels");
  let activeHead = 0;
  let attnData = []; // [head][row][col]

  // Pre-compute attention weights per head. Each head has a distinct "personality":
  //   - head 0: diagonal/local (current token dominates)
  //   - head 1: previous-token heavy
  //   - head 2: broad uniform distribution
  //   - head 3: first-token sink (attention sinks to BOS)
  //   - head 4: looking back 3-4 positions
  //   - head 5: scattered, weak
  //   - head 6: alternating pattern
  //   - head 7: syntactic locality (prev 2 tokens)
  function buildAttnData() {
    attnData = [];
    for (let h = 0; h < ATTN_HEADS; h++) {
      const head = [];
      for (let i = 0; i < ATTN_DIM; i++) {
        const row = new Array(ATTN_DIM).fill(0);
        for (let j = 0; j <= i; j++) {
          // Base raw weight, then normalise per row
          let w;
          if (h === 0) {
            // Local: prefer current token
            w = j === i ? 3.0 : (i - j) === 1 ? 1.2 : 0.3;
          } else if (h === 1) {
            // Previous-token heavy
            w = j === i - 1 ? 3.5 : (j === i ? 1.5 : 0.4);
          } else if (h === 2) {
            // Broad uniform
            w = 1.0 + Math.random() * 0.2;
          } else if (h === 3) {
            // Attention sink at position 0
            w = j === 0 ? 2.8 : 0.6;
          } else if (h === 4) {
            // Look back 3-4 positions
            const dist = i - j;
            w = (dist >= 2 && dist <= 4) ? 2.2 : 0.5;
          } else if (h === 5) {
            // Weak scattered
            w = 0.8 + Math.random() * 0.4;
          } else if (h === 6) {
            // Alternating
            w = (j % 2 === 0) ? 1.8 : 0.5;
          } else {
            // Last 2 tokens
            w = (j === i || j === i - 1) ? 2.0 : 0.3;
          }
          row[j] = w;
        }
        // Softmax with mask
        let max = -Infinity;
        for (let j = 0; j <= i; j++) max = Math.max(max, row[j]);
        let sum = 0;
        for (let j = 0; j <= i; j++) {
          row[j] = Math.exp(row[j] - max);
          sum += row[j];
        }
        for (let j = 0; j <= i; j++) row[j] /= sum;
        head.push(row);
      }
      attnData.push(head);
    }
  }

  function buildAttnUI() {
    if (!attnHeadButtons || !attnGrid) return;
    // Head buttons
    attnHeadButtons.innerHTML = "";
    for (let h = 0; h < ATTN_HEADS; h++) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "attn-head-btn";
      btn.textContent = String(h + 1);
      btn.addEventListener("click", () => selectHead(h));
      attnHeadButtons.appendChild(btn);
    }

    // Grid cells
    attnGrid.innerHTML = "";
    for (let i = 0; i < ATTN_DIM; i++) {
      for (let j = 0; j < ATTN_DIM; j++) {
        const cell = document.createElement("div");
        cell.className = "attn-cell";
        cell.dataset.row = String(i);
        cell.dataset.col = String(j);
        attnGrid.appendChild(cell);
      }
    }

    // Axis labels (token symbols)
    const symbols = ["<bos>", "The", "cat", "sat", "on", "the", "mat", "."];
    if (attnXLabels) {
      attnXLabels.innerHTML = "";
      for (let j = 0; j < ATTN_DIM; j++) {
        const span = document.createElement("span");
        span.textContent = symbols[j];
        attnXLabels.appendChild(span);
      }
    }
    if (attnYLabels) {
      attnYLabels.innerHTML = "";
      for (let i = 0; i < ATTN_DIM; i++) {
        const span = document.createElement("span");
        span.textContent = symbols[i];
        attnYLabels.appendChild(span);
      }
    }
    selectHead(0);
  }

  function selectHead(h) {
    activeHead = h;
    $$(".attn-head-btn", attnHeadButtons).forEach((b, i) => b.classList.toggle("active", i === h));
    renderAttnGrid();
  }

  function renderAttnGrid() {
    const head = attnData[activeHead];
    const cells = $$(".attn-cell", attnGrid);
    cells.forEach((cell) => {
      const i = parseInt(cell.dataset.row, 10);
      const j = parseInt(cell.dataset.col, 10);
      if (j > i) {
        cell.classList.add("masked");
        cell.removeAttribute("data-w");
        cell.style.backgroundColor = "";
        return;
      }
      cell.classList.remove("masked");
      const w = head[i][j];
      cell.dataset.w = w.toFixed(2);
      // Map weight to opacity. Most rows are roughly uniform (1/8 = 0.125).
      // We want to see structure, so normalise within the visible range.
      // Use a log-scaled darkness for visibility.
      const intensity = clamp(w * 1.6, 0, 1);
      cell.style.backgroundColor = `rgba(24, 22, 20, ${intensity})`;
    });
  }

  buildAttnData();
  buildAttnUI();

  /* ----------------------------------------------------------------------
     Section 3: Loss Scrubber + SVG Line Chart
     ---------------------------------------------------------------------- */

  // Synthetic but realistic 77M training trajectory.
  // Loss starts at ~11.0, drops fast (power-law decay) to ~3.7 by step 2000.
  // Validation tracks train with a small gap that opens late.
  function generateLossCurve() {
    const pts = 41; // every 50 steps
    const train = [];
    const dev = [];
    for (let i = 0; i < pts; i++) {
      const step = i * 50;
      // Power-law-ish decay: loss = a + b / sqrt(step + c)
      const t = step / 2000;
      const base = 10.83;
      const floor = 3.6;
      const trainLoss = floor + (base - floor) * Math.pow(1 - t, 1.6) + 0.05 * Math.sin(t * 12);
      const devLoss = trainLoss + 0.04 + 0.08 * t + 0.05 * Math.sin(t * 7 + 1.2);
      train.push({ step, loss: clamp(trainLoss, 0, 12) });
      dev.push({ step, loss: clamp(devLoss, 0, 12) });
    }
    return { train, dev };
  }

  const lossCurve = generateLossCurve();
  const lossSlider = $("#loss-slider");
  const lossStepLabel = $("#loss-step-label");
  const lossTrainVal = $("#loss-train-val");
  const lossDevVal = $("#loss-dev-val");
  const lossPplVal = $("#loss-ppl-val");
  const lossTrainBar = $("#loss-train-bar");
  const lossDevBar = $("#loss-dev-bar");
  const lossTrainPath = $("#loss-train-path");
  const lossDevPath = $("#loss-dev-path");
  const lossCursor = $("#loss-cursor");
  const lossDotTrain = $("#loss-dot-train");
  const lossDotDev = $("#loss-dot-dev");

  // SVG chart bounds
  const CHART_X0 = 36, CHART_X1 = 468, CHART_Y0 = 20, CHART_Y1 = 210;
  const LOSS_MAX = 11.2;

  function lossToY(loss) {
    return CHART_Y1 - (loss / LOSS_MAX) * (CHART_Y1 - CHART_Y0);
  }
  function stepToX(step) {
    return CHART_X0 + (step / 2000) * (CHART_X1 - CHART_X0);
  }

  function buildLossChart() {
    if (!lossTrainPath) return;
    const makePath = (arr) => {
      return arr.map((p, i) => {
        const x = stepToX(p.step).toFixed(1);
        const y = lossToY(p.loss).toFixed(1);
        return (i === 0 ? "M" : "L") + x + " " + y;
      }).join(" ");
    };
    lossTrainPath.setAttribute("d", makePath(lossCurve.train));
    lossDevPath.setAttribute("d", makePath(lossCurve.dev));
  }

  function interpLoss(step, arr) {
    if (step <= arr[0].step) return arr[0].loss;
    if (step >= arr[arr.length - 1].step) return arr[arr.length - 1].loss;
    for (let i = 0; i < arr.length - 1; i++) {
      if (step >= arr[i].step && step <= arr[i + 1].step) {
        const t = (step - arr[i].step) / (arr[i + 1].step - arr[i].step);
        return lerp(arr[i].loss, arr[i + 1].loss, t);
      }
    }
    return arr[arr.length - 1].loss;
  }

  function updateLossReadout() {
    const step = parseInt(lossSlider.value, 10);
    const train = interpLoss(step, lossCurve.train);
    const dev = interpLoss(step, lossCurve.dev);
    if (lossStepLabel) lossStepLabel.textContent = "step " + step + " / 2000";
    if (lossTrainVal) lossTrainVal.textContent = train.toFixed(2);
    if (lossDevVal) lossDevVal.textContent = dev.toFixed(2);
    if (lossPplVal) lossPplVal.textContent = Math.round(Math.exp(train)) + "";

    if (lossTrainBar) lossTrainBar.style.width = clamp((train / LOSS_MAX) * 100, 0, 100) + "%";
    if (lossDevBar) lossDevBar.style.width = clamp((dev / LOSS_MAX) * 100, 0, 100) + "%";

    // Cursor + dots
    const x = stepToX(step);
    if (lossCursor) {
      lossCursor.style.display = "";
      lossCursor.setAttribute("x1", String(x));
      lossCursor.setAttribute("x2", String(x));
    }
    if (lossDotTrain) {
      lossDotTrain.style.display = "";
      lossDotTrain.setAttribute("cx", String(x));
      lossDotTrain.setAttribute("cy", String(lossToY(train)));
    }
    if (lossDotDev) {
      lossDotDev.style.display = "";
      lossDotDev.setAttribute("cx", String(x));
      lossDotDev.setAttribute("cy", String(lossToY(dev)));
    }
  }

  if (lossSlider) {
    lossSlider.addEventListener("input", updateLossReadout);
    buildLossChart();
    updateLossReadout();
  }

  /* ----------------------------------------------------------------------
     Section 3: Sample Text Inspector
     ---------------------------------------------------------------------- */

  const SAMPLE_STAGES = ["base", "sft", "grpo"];
  const SAMPLE_PROMPTS = [
    {
      key: "hello",
      label: "Hello",
      text: "Hello, who are you?"
    },
    {
      key: "math",
      label: "Math",
      text: "What is 25 * 4?"
    },
    {
      key: "code",
      label: "Code",
      text: "Write a Python function to reverse a list."
    }
  ];

  // Curated completions for each (stage × prompt). Realistic tone shifts across stages.
  const SAMPLES = {
    base: {
      hello: "Hello, who are you? I am a student of the language model. I have been reading about your work, and I was wondering if you could tell me about your research. The work is about a particular type of model that",
      math: "What is 25 * 4? The product of two numbers. In this case, the answer is 100. The number 25 is a square of 5, and 4 is the square of 2. The product of these two squares is 100. The square of 10",
      code: "Write a Python function to reverse a list. The list is a collection of items, like the one you see below. You can also use it for other purposes, such as a list of strings, or a list of numbers."
    },
    sft: {
      hello: "Hello! I'm an assistant trained to help answer questions, write text, and reason about problems. How can I help you today?",
      math: "25 * 4 = 100. Let me know if you'd like me to walk through the calculation step by step.",
      code: "Here's a Python function that reverses a list:\n\n```python\ndef reverse_list(lst):\n    return lst[::-1]\n```\n\nThis uses Python's slice notation to step through the list backwards."
    },
    grpo: {
      hello: "Reasoning: The user is greeting me and asking about my identity. I should respond as a helpful assistant.\n\nResponse: Hello! I'm a language model fine-tuned to follow instructions and reason step-by-step. I can help with questions, writing, code, and math. What would you like to work on?",
      math: "Reasoning: 25 * 4. I can compute this by breaking 25 into 20 + 5. Then (20 + 5) * 4 = 80 + 20 = 100. Verified.\n\n<answer>100</answer>\n\nExplanation: 25 * 4 = 100. I broke it as (20 + 5) * 4 = 80 + 20 = 100 for verification.",
      code: "Reasoning: The user wants a Python function to reverse a list. The idiomatic ways are: slice [::-1], reversed(), or manual. Slice is shortest.\n\nResponse: Here is a Python function that reverses a list:\n\n```python\ndef reverse_list(lst):\n    \"\"\"Return a new list with elements in reverse order.\"\"\"\n    return lst[::-1]\n```\n\nFor an in-place version, use lst.reverse()."
    }

  };

  const stageButtons = $("#stage-buttons");
  const promptButtons = $("#prompt-buttons");
  const stagePromptTitle = $("#stage-prompt-title");
  const stagePromptText = $("#stage-prompt-text");
  const generatedReply = $("#generated-reply");
  const sampleBtn = $("#sample-btn");
  const sampleBadge = $("#sample-badge");

  let activeStage = "sft";
  let activePrompt = "hello";

  function buildSampleUI() {
    if (!stageButtons) return;
    SAMPLE_STAGES.forEach((s) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample-btn";
      btn.textContent = s.toUpperCase();
      btn.addEventListener("click", () => {
        activeStage = s;
        $$(".sample-btn", stageButtons).forEach((b, j) => b.classList.toggle("active", j === SAMPLE_STAGES.indexOf(s)));
        renderSample();
      });
      stageButtons.appendChild(btn);
    });

    SAMPLE_PROMPTS.forEach((p) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sample-btn";
      btn.textContent = p.label;
      btn.addEventListener("click", () => {
        activePrompt = p.key;
        $$(".sample-btn", promptButtons).forEach((b, j) => b.classList.toggle("active", j === SAMPLE_PROMPTS.indexOf(p)));
        renderSample();
      });
      promptButtons.appendChild(btn);
    });

    // Set initial active states
    $$(".sample-btn", stageButtons)[SAMPLE_STAGES.indexOf(activeStage)].classList.add("active");
    $$(".sample-btn", promptButtons)[SAMPLE_PROMPTS.findIndex((p) => p.key === activePrompt)].classList.add("active");
    renderSample();
  }

  function renderSample() {
    const promptObj = SAMPLE_PROMPTS.find((p) => p.key === activePrompt);
    const reply = SAMPLES[activeStage][activePrompt];

    if (stagePromptTitle) {
      stagePromptTitle.textContent = `Stage: ${activeStage.toUpperCase()} · Prompt: ${promptObj.label}`;
    }
    if (stagePromptText) {
      stagePromptText.textContent = promptObj.text;
    }
    if (generatedReply) {
      generatedReply.textContent = reply;
      generatedReply.classList.remove("fade-up");
      void generatedReply.offsetWidth;
      generatedReply.classList.add("fade-up");
    }
    if (sampleBadge) sampleBadge.style.display = "inline-flex";
  }

  if (sampleBtn) {
    sampleBtn.addEventListener("click", () => {
      // Re-render with a tiny "re-sampled" effect — same content, but flash the badge.
      if (sampleBadge) {
        sampleBadge.textContent = "Re-sampled ✓";
        setTimeout(() => {
          sampleBadge.textContent = "Curated checkpoint reply";
        }, 1500);
      }
      renderSample();
    });
  }

  buildSampleUI();

  /* ----------------------------------------------------------------------
     Section 4: Theorems T1–T9
     ---------------------------------------------------------------------- */

  const THEOREMS = [
    { id: "T1", title: "Architecture Invariance", body: "The Transformer backbone — embeddings, N attention+MLP blocks, final LayerNorm, LM head — is the same architecture in every stage. SFT, DPO, PPO, GRPO change only the loss and data; the forward pass through the network is identical." },
    { id: "T2", title: "Closed-Form Parameter Count", body: "Total parameters admit an exact closed form: τ(d, N, T, V) = Vd + Td + N(12d² + 10d) + 2d + Vd + V. No symbolic enumeration of weight tensors is needed; the formula predicts the parameter count to the integer." },
    { id: "T3", title: "CE Barrier", body: "Initial cross-entropy on a uniform-output model is exactly ln(V). For V = 50 304 this is 10.83 nats. Loss cannot exceed this in expectation for a randomly-initialised softmax model; convergence below 4.0 implies the model has learned non-trivial token statistics." },
    { id: "T4", title: "Causal Mask Lemma", body: "The lower-triangular causal mask is a sufficient condition for autoregressive generation. Any token at position i is a function only of tokens at positions ≤ i. This is provable by induction over the residual stream and is preserved by every alignment stage." },
    { id: "T5", title: "Substitution Equivalence", body: "Replacing any hand-written module with an external library equivalent (e.g. HuggingFace attention ↔ local attention) yields the same forward output up to floating-point non-determinism. The wrappers do not change the math." },
    { id: "T6", title: "Loss Mask Admissibility", body: "In SFT, masking the loss over prompt tokens is admissible: it does not change the gradient through any non-prompt position, because the cross-entropy factorises per-position. Only assistant tokens contribute to the loss, but all tokens contribute to the forward pass." },
    { id: "T7", title: "Bradley-Terry Identifiability", body: "Reward models trained with L = -log σ(r(chosen) - r(rejected)) identify the reward function up to an additive constant. The constant is fixed by convention (e.g. r(empty) = 0). This makes the reward identifiable but not unique — a known degeneracy that PPO must handle with regularisation." },
    { id: "T8", title: "DPO Implicit Reward", body: "DPO optimises an implicit reward formulation: the policy itself encodes the reward via the log-ratio to the reference policy. No separate reward model is needed. The β parameter controls the strength of the KL constraint to the reference policy." },
    { id: "T9", title: "Unique Sink at Chat", body: "In the substitution τ, every upstream stage is a function of the previous, and the terminal state 'chat' is the unique sink. There is no second terminal stage; all alignment methods (SFT, DPO, PPO, GRPO) are intermediate refinements of the path to chat." }
  ];

  const theoremsList = $("#theorems-list");
  const theoremId = $("#theorem-id");
  const theoremTitle = $("#theorem-title");
  const theoremBody = $("#theorem-body");

  function buildTheorems() {
    if (!theoremsList) return;
    THEOREMS.forEach((t, i) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "theorem-btn";
      btn.innerHTML =
        '<span class="theorem-id">' + t.id + "</span>" +
        "<span>" + t.title + "</span>";
      btn.addEventListener("click", () => selectTheorem(i));
      li.appendChild(btn);
      theoremsList.appendChild(li);
    });
    selectTheorem(0);
  }

  function selectTheorem(i) {
    const t = THEOREMS[i];
    $$(".theorem-btn", theoremsList).forEach((b, j) => b.classList.toggle("active", j === i));
    if (theoremId) {
      theoremId.textContent = t.id + " · FORMAL INVARIANT";
      theoremId.classList.remove("fade-up");
      void theoremId.offsetWidth;
      theoremId.classList.add("fade-up");
    }
    if (theoremTitle) {
      theoremTitle.textContent = t.title;
      theoremTitle.classList.remove("fade-up");
      void theoremTitle.offsetWidth;
      theoremTitle.classList.add("fade-up");
    }
    if (theoremBody) {
      theoremBody.textContent = t.body;
      theoremBody.classList.remove("fade-up");
      void theoremBody.offsetWidth;
      theoremBody.classList.add("fade-up");
    }
  }

  buildTheorems();

  /* ----------------------------------------------------------------------
     Section 5: GSM8K Verifier
     ---------------------------------------------------------------------- */

  const GOLD_NUMBER = 72;
  const gsmForm = $("#gsm8k-form");
  const completionInput = $("#completion-input");
  const btnGold = $("#btn-gold");
  const btnMiss = $("#btn-miss");
  const evalResults = $("#eval-results");
  const evalParsed = $("#eval-parsed");
  const evalVerdict = $("#eval-verdict");

  function extractAnswer(text) {
    // 1. Try <answer>...</answer>
    let m = text.match(/<answer>\s*(-?\d+(?:\.\d+)?)\s*<\/answer>/i);
    if (m) return { value: parseFloat(m[1]), hasTag: true };
    // 2. Try #### N (GSM8K convention)
    m = text.match(/####\s*(-?\d+(?:\.\d+)?)/);
    if (m) return { value: parseFloat(m[1]), hasTag: false };
    // 3. Try last number in text
    const nums = text.match(/-?\d+(?:\.\d+)?/g);
    if (nums && nums.length > 0) {
      return { value: parseFloat(nums[nums.length - 1]), hasTag: false };
    }
    return { value: null, hasTag: false };
  }

  function evaluateReward() {
    const text = completionInput.value;
    const parsed = extractAnswer(text);
    let score, verdict;

    if (parsed.value === null) {
      score = 0.0;
      verdict = "0.0 — Total Miss";
    } else if (parsed.value === GOLD_NUMBER) {
      if (parsed.hasTag) {
        score = 1.2;
        verdict = "1.2 — Gold + Format";
      } else {
        score = 1.0;
        verdict = "1.0 — Gold Only";
      }
    } else if (parsed.hasTag) {
      score = 0.2;
      verdict = "0.2 — Format Bonus";
    } else {
      score = 0.0;
      verdict = "0.0 — Total Miss";
    }

    if (evalParsed) evalParsed.textContent = parsed.value === null ? "—" : String(parsed.value);
    if (evalVerdict) {
      const cls = score >= 1.0 ? "text-mark" : (score >= 0.5 ? "text-fg" : "");
      evalVerdict.innerHTML = `<span class="${cls}">${verdict}</span>`;
    }
    if (evalResults) {
      evalResults.style.display = "";
      evalResults.classList.remove("fade-up");
      void evalResults.offsetWidth;
      evalResults.classList.add("fade-up");
    }
  }

  if (gsmForm) gsmForm.addEventListener("submit", (e) => { e.preventDefault(); evaluateReward(); });
  if (btnGold) btnGold.addEventListener("click", () => {
    completionInput.value = "<answer>72</answer>";
    evaluateReward();
  });
  if (btnMiss) btnMiss.addEventListener("click", () => {
    completionInput.value = "I think the answer is 64 because 48 + 24 = 64.";
    evaluateReward();
  });

  /* ----------------------------------------------------------------------
     Section 7: Copy-to-clipboard buttons
     ---------------------------------------------------------------------- */

  $$(".copy-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const code = btn.dataset.code || "";
      navigator.clipboard.writeText(code).then(() => {
        const label = btn.querySelector(".copy-label");
        const orig = label ? label.textContent : "";
        btn.classList.add("copied");
        if (label) label.textContent = "copied ✓";
        setTimeout(() => {
          btn.classList.remove("copied");
          if (label) label.textContent = orig || "copy";
        }, 1600);
      }).catch(() => {
        // Fallback for non-secure contexts
        const ta = document.createElement("textarea");
        ta.value = code;
        document.body.appendChild(ta);
        ta.select();
        try { document.execCommand("copy"); } catch (e) {}
        document.body.removeChild(ta);
      });
    });
  });

  /* ----------------------------------------------------------------------
     Scroll spy: highlight active nav link based on visible section
     ---------------------------------------------------------------------- */

  const navLinks = $$(".nav-link");
  const sections = ["path", "model", "loss", "align", "eval", "name", "repo"]
    .map((id) => document.getElementById(id))
    .filter(Boolean);

  function updateScrollSpy() {
    const scrollY = window.scrollY + 100;
    let activeIdx = -1;
    for (let i = 0; i < sections.length; i++) {
      const top = sections[i].offsetTop;
      if (scrollY >= top) activeIdx = i;
    }
    if (activeIdx >= 0) {
      navLinks.forEach((l, i) => l.classList.toggle("active", i === activeIdx));
    } else {
      navLinks.forEach((l) => l.classList.remove("active"));
    }
  }

  /* ----------------------------------------------------------------------
     Scroll progress + back-to-top
     ---------------------------------------------------------------------- */

  const scrollProgress = $("#scroll-progress");
  const backToTop = $("#back-to-top");

  function onScroll() {
    const scrollTop = window.scrollY || document.documentElement.scrollTop;
    const docHeight = (document.documentElement.scrollHeight - window.innerHeight);
    const pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
    if (scrollProgress) scrollProgress.style.width = pct + "%";

    if (backToTop) {
      if (scrollTop > 600) {
        backToTop.classList.add("visible");
        backToTop.style.display = "inline-flex";
      } else {
        backToTop.classList.remove("visible");
        // Keep display long enough for fade out
        setTimeout(() => {
          if (!backToTop.classList.contains("visible")) backToTop.style.display = "none";
        }, 220);
      }
    }

    updateScrollSpy();
  }

  if (backToTop) {
    backToTop.addEventListener("click", () => {
      window.scrollTo({ top: 0, behavior: "smooth" });
    });
  }

  let scrollTicking = false;
  window.addEventListener("scroll", () => {
    if (!scrollTicking) {
      window.requestAnimationFrame(() => {
        onScroll();
        scrollTicking = false;
      });
      scrollTicking = true;
    }
  });
  onScroll();

  /* ----------------------------------------------------------------------
     Keyboard nav: 1-7 jump to sections
     ---------------------------------------------------------------------- */

  document.addEventListener("keydown", (e) => {
    // Ignore when typing in inputs
    const tag = (e.target.tagName || "").toLowerCase();
    if (tag === "input" || tag === "textarea") return;

    if (e.key >= "1" && e.key <= "7") {
      const target = sections[parseInt(e.key, 10) - 1];
      if (target) {
        e.preventDefault();
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    }
  });

  // Final sanity log
  console.log("%cτ Train LLM From Scratch — interactive engine ready.", "color:#b4532b;font-family:monospace;");
})();
