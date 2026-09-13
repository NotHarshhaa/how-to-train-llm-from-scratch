export type ModelSpec = {
  nEmbed: number;
  nHead: number;
  nBlocks: number;
  contextLength: number;
  vocabSize: number;
};

export const VOCAB_SIZE = 50304;
export const TIKTOKEN_VOCAB = 50257;

export const PRESETS = [
  {
    id: "13m",
    label: "13M",
    nEmbed: 128,
    nHead: 8,
    nBlocks: 1,
    contextLength: 256,
    reported: 13_142_656,
  },
  {
    id: "77m",
    label: "77M",
    nEmbed: 512,
    nHead: 8,
    nBlocks: 8,
    contextLength: 512,
    reported: 77_031_552,
  },
  {
    id: "406m",
    label: "406M",
    nEmbed: 1024,
    nHead: 16,
    nBlocks: 24,
    contextLength: 1024,
    reported: 406_359_168,
  },
] as const;

export type PresetId = (typeof PRESETS)[number]["id"];

/** Exact nn.Parameter count of the handwritten Transformer in the repo. */
export function countParams(spec: ModelSpec) {
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
    token,
    pos,
    attn: nBlocks * attn,
    mlp: nBlocks * mlp,
    ln: nBlocks * ln + finalLn,
    lmHead,
    block,
    total: token + pos + nBlocks * block + finalLn + lmHead,
    headSize,
  };
}

export function formatInt(n: number) {
  const rounded = Math.round(n);
  const sign = rounded < 0 ? "−" : "";
  return sign + Math.abs(rounded).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

export function formatCompact(n: number) {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(2)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(n >= 100_000_000 ? 0 : 1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}K`;
  return formatInt(n);
}

export const GPUS = [
  { name: "NVIDIA A100", memory: "40 GB", max: "~6B–8B", fit13: true, fit2b: true },
  { name: "NVIDIA V100", memory: "16 GB", max: "~2B", fit13: true, fit2b: false },
  { name: "NVIDIA RTX 4090", memory: "24 GB", max: "~4B", fit13: true, fit2b: true },
  { name: "NVIDIA RTX 5090", memory: "32 GB", max: "13M verified", fit13: true, fit2b: true },
  { name: "NVIDIA RTX 3090", memory: "24 GB", max: "~3.5B–4B", fit13: true, fit2b: true },
  { name: "NVIDIA RTX 4080", memory: "16 GB", max: "~2B", fit13: true, fit2b: false },
  { name: "NVIDIA RTX 4060", memory: "8 GB", max: "~1B", fit13: true, fit2b: false },
  { name: "Tesla T4", memory: "16 GB", max: "~1.5B–2B", fit13: true, fit2b: false },
] as const;

export function fitsGpu(params: number, maxLabel: string) {
  const m = maxLabel.match(/~?([\d.]+)\s*B/i);
  if (!m) return params < 50_000_000;
  return params <= Number(m[1]) * 1_000_000_000;
}
