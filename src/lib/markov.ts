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

type Table = Map<string, string[]>;

function train(corpus: string, order: number): Table {
  const table: Table = new Map();
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

const ORDER = 3;
const BASE_TABLE = train(BASE_CORPUS, ORDER);

function sampleFrom(table: Table, prefix: string, maxChars: number, temperature: number) {
  let out = prefix;
  let guard = 0;
  while (out.length < prefix.length + maxChars && guard++ < 800) {
    const key = out.slice(-ORDER);
    const options = table.get(key);
    if (!options || options.length === 0) break;
    const pick = options[Math.floor(Math.random() * options.length)] ?? "";
    if (!pick) break;
    if (temperature < 0.4 && options.length > 2) {
      const counts = new Map<string, number>();
      for (const ch of options) counts.set(ch, (counts.get(ch) ?? 0) + 1);
      let best = options[0]!;
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

export function generateBase(prompt: string, maxChars = 220, temperature = 0.9) {
  const seed = (prompt.trim() || "The ").slice(0, 80);
  let start = seed;
  if (start.length < ORDER) start = (BASE_CORPUS.slice(0, ORDER) + start).slice(0, ORDER);
  const body = sampleFrom(BASE_TABLE, start.slice(-ORDER), maxChars, temperature);
  return (seed + body.slice(ORDER)).replace(/\s+/g, " ").trim();
}

export const STAGE_REPLIES: Record<
  "base" | "sft" | "grpo",
  { prompt: string; reply: string }[]
> = {
  base: [
    {
      prompt: "What is 2 + 2?",
      reply:
        "In 1978, The park was returned to the factory-plate that the public share to the lower of the electronic fence that follow from the Station's cities.",
    },
    {
      prompt: "Explain attention.",
      reply:
        "The Canal of ancient Western nations were confined to the city spot. The villages were directly linked to cities in China that revolt that the US budget.",
    },
    {
      prompt: "A park in 1978.",
      reply:
        "In 1978, The park was returned to the factory-plate that the public share to the lower of the electronic fence that follow from the Station's cities.",
    },
  ],
  sft: [
    {
      prompt: "What is 2 + 2?",
      reply: "2 + 2 equals 4.",
    },
    {
      prompt: "Explain attention.",
      reply:
        "Attention lets each token look at the others and decide which ones matter. In a causal language model, a token may only look at itself and what came before.",
    },
    {
      prompt: "A park in 1978.",
      reply:
        "Write a short paragraph about a park that was returned to the public in 1978, with a fence, a canal, and nearby villages.",
    },
  ],
  grpo: [
    {
      prompt: "What is 2 + 2?",
      reply: "Adding two and two gives four.\n<answer>4</answer>",
    },
    {
      prompt: "Explain attention.",
      reply:
        "Causal self-attention computes q kᵀ / √d, masks the future, softmaxes, and mixes values. That is the whole move.\n<answer>causal scaled-dot-product attention</answer>",
    },
    {
      prompt: "A park in 1978.",
      reply:
        "A public park restored in 1978, bounded by a fence and a canal, with villages linked to the city.\n<answer>1978 public park</answer>",
    },
  ],
};
