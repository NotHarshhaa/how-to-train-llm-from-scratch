import { useState } from "react";
import { THEOREMS } from "@/lib/content";
import { cn } from "@/lib/cn";

const METHODS = [
  {
    id: "SFT",
    title: "Supervised fine-tune",
    formula: "ce = CE(logits, y) * mask;  return ce.sum() / mask.sum()",
    note: "Instruction data (Alpaca, Dolly, GSM8K). Train only on assistant tokens. The base model learns to answer instead of to continue the internet.",
  },
  {
    id: "RM",
    title: "Reward model",
    formula: "−log σ(r(chosen) − r(rejected))",
    note: "Bradley-Terry on preference pairs. Test accuracy in the README: 0.574 on 7 974 pairs. Weak, and enough to point PPO.",
  },
  {
    id: "DPO",
    title: "DPO / ORPO / KTO",
    formula: "−log σ(β (π log-ratio − π_ref log-ratio))",
    note: "Preference without sampling from a reward model. ORPO drops the reference. KTO takes unpaired thumbs.",
  },
  {
    id: "PPO",
    title: "PPO",
    formula: "−min(ratio A, clip(ratio) A)  with GAE",
    note: "Classic RLHF. Value head on the same backbone. Reward from the RM or from a verifier.",
  },
  {
    id: "GRPO",
    title: "GRPO / RLVR",
    formula: "A = (r − mean_G) / (std_G + ε)",
    note: "Group of answers, relative advantage, no value head. The sink of the post-training line when the task can be checked.",
  },
] as const;

export function AlignSection() {
  const [tid, setTid] = useState<(typeof THEOREMS)[number]["id"]>("T9");
  const theorem = THEOREMS.find((t) => t.id === tid) ?? THEOREMS[8];

  return (
    <section id="align" className="scroll-mt-16 border-b border-border">
      <div className="mx-auto max-w-6xl px-4 py-16 sm:px-6 lg:py-20">
        <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">Post-training</p>
        <h2 className="font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl">
          Turning a base into an assistant
        </h2>
        <p className="mt-5 max-w-2xl text-base leading-relaxed text-muted">
          After the sink of pretraining, the data changes and the loss changes. The net does not.
          Five methods, all handwritten, all on the same small Transformer.
        </p>

        <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {METHODS.map((m) => (
            <article key={m.id} className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
              <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">{m.id}</p>
              <h3 className="font-display mt-3 text-2xl font-medium tracking-[-0.02em]">{m.title}</h3>
              <p className="mt-3 font-mono text-xs leading-relaxed break-words text-fg sm:text-sm">{m.formula}</p>
              <p className="mt-3 text-sm leading-relaxed text-muted">{m.note}</p>
            </article>
          ))}
        </div>
      </div>

      <div className="mx-auto grid max-w-6xl gap-10 px-4 pb-16 sm:px-6 lg:grid-cols-[minmax(0,16rem)_1fr] lg:items-start lg:pb-20">
        <div>
          <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">Theorems</p>
          <h3 className="font-display mt-3 text-3xl font-medium tracking-[-0.03em]">T1–T9</h3>
          <ol className="mt-6 space-y-1">
            {THEOREMS.map((t) => (
              <li key={t.id}>
                <button
                  type="button"
                  onClick={() => setTid(t.id)}
                  className={cn(
                    "flex w-full items-baseline gap-3 rounded-md px-3 py-2 text-left text-sm transition-[background-color,color] duration-150 ease-out",
                    tid === t.id ? "bg-elevated text-fg" : "text-muted hover:bg-elevated/60 hover:text-fg",
                  )}
                >
                  <span className="w-8 shrink-0 font-mono text-xs text-mark">{t.id}</span>
                  <span>{t.title}</span>
                </button>
              </li>
            ))}
          </ol>
        </div>
        <article className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
          <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">{theorem.id}</p>
          <h4 className="font-display mt-3 text-2xl font-medium tracking-[-0.02em]">{theorem.title}</h4>
          <p className="mt-4 text-base leading-relaxed text-muted">{theorem.body}</p>
        </article>
      </div>
    </section>
  );
}
