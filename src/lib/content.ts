export const REPO = "https://github.com/FareedKhan-dev/train-llm-from-scratch";
export const DOCS = "https://fareedkhan-dev.github.io/train-llm-from-scratch/";
export const PAPER = "https://arxiv.org/abs/1706.03762";
export const AUTHOR = "https://github.com/FareedKhan-dev";

export const PIPELINE = [
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
] as const;

export const THEOREMS = [
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
] as const;

export const PIECES = [
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
] as const;
