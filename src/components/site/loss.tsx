import { useMemo, useState } from "react";
import { generateBase, STAGE_REPLIES } from "@/lib/markov";
import { cn } from "@/lib/cn";

const POINTS: { t: number; train: number; dev: number }[] = [
  { t: 0, train: 11.14, dev: 11.2 },
  { t: 250, train: 8.4, dev: 8.55 },
  { t: 500, train: 6.35, dev: 6.5 },
  { t: 1000, train: 4.82, dev: 4.95 },
  { t: 1500, train: 4.12, dev: 4.22 },
  { t: 2000, train: 3.73, dev: 3.76 },
];

function lerpLoss(step: number) {
  const first = POINTS[0]!;
  const last = POINTS[POINTS.length - 1]!;
  if (step <= 0) return first;
  if (step >= 2000) return last;
  let a = first;
  let b = last;
  for (let i = 0; i < POINTS.length - 1; i++) {
    const lo = POINTS[i]!;
    const hi = POINTS[i + 1]!;
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

type Stage = "base" | "sft" | "grpo";

export function LossSection() {
  const [step, setStep] = useState(2000);
  const [stage, setStage] = useState<Stage>("base");
  const [promptIdx, setPromptIdx] = useState(0);
  const [generated, setGenerated] = useState<string | null>(null);

  const loss = useMemo(() => lerpLoss(step), [step]);
  const replies = STAGE_REPLIES[stage];
  const pair = replies[promptIdx] ?? replies[0];

  function runGenerate() {
    if (stage === "base") {
      setGenerated(generateBase(pair.prompt, 200, 0.95));
    } else {
      setGenerated(pair.reply);
    }
  }

  return (
    <section id="loss" className="scroll-mt-16 border-b border-border">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20">
        <div>
          <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">Unique sink of pretraining</p>
          <h2 className="font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl">
            Next-token loss
          </h2>
          <p className="mt-5 text-base leading-relaxed text-muted">
            Every orbit of raw text reaches the same objective. Write the next token, grade the
            logits, descend. The 77 million parameter run in the README is the exhibited orbit:
            11.14 → 3.73 train, 3.76 on the held-out slice, in 2 000 steps.
          </p>
          <p className="mt-4 font-mono text-sm break-words text-fg">
            L = CE(softmax(z), y<sub>t+1</sub>)
          </p>
          <p className="mt-4 text-sm leading-relaxed text-subtle">
            T2. Independently checked in the training logs. The curve here is a reading of those two
            endpoints, not a re-run.
          </p>
        </div>

        <div className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
          <div className="flex items-baseline justify-between gap-3">
            <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">Orbit · 77M</p>
            <p className="font-mono text-xs tabular-nums text-muted">step {step.toLocaleString("en-US")}</p>
          </div>
          <p className="font-display mt-4 text-5xl leading-none font-medium tracking-[-0.04em] tabular-nums text-fg">
            {loss.train.toFixed(2)}
          </p>
          <p className="mt-2 text-sm text-muted">
            train · dev {loss.dev.toFixed(2)}
          </p>
          <label className="mt-6 block text-sm font-medium text-fg" htmlFor="step-slider">
            Training step
          </label>
          <input
            id="step-slider"
            type="range"
            min={0}
            max={2000}
            step={10}
            value={step}
            onChange={(e) => setStep(Number(e.target.value))}
            className="mt-2 w-full accent-[var(--color-mark)]"
          />
          <div className="mt-5 space-y-3">
            <div>
              <div className="mb-1 flex justify-between text-xs text-subtle">
                <span>train</span>
                <span className="font-mono tabular-nums">{loss.train.toFixed(2)}</span>
              </div>
              <div className="loss-bar">
                <span style={{ width: `${(loss.train / 11.2) * 100}%` }} />
              </div>
            </div>
            <div>
              <div className="mb-1 flex justify-between text-xs text-subtle">
                <span>dev</span>
                <span className="font-mono tabular-nums">{loss.dev.toFixed(2)}</span>
              </div>
              <div className="loss-bar">
                <span style={{ width: `${(loss.dev / 11.2) * 100}%` }} />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="mx-auto max-w-6xl px-4 pb-16 sm:px-6 lg:pb-20">
        <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">Where the small end starts</p>
        <h3 className="font-display mt-3 text-3xl font-medium tracking-[-0.03em]">Generate</h3>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted">
          Below is the output of a trained 13 million parameter LLM, just so you can see where the
          small end of this starts — and how SFT then GRPO change the mouth of the same net. Base
          generation is a character Markov reading of that sample. Later stages are staged replies,
          not a hosted checkpoint.
        </p>

        <div className="mt-6 flex flex-wrap gap-2">
          {(["base", "sft", "grpo"] as const).map((s) => (
            <button
              key={s}
              type="button"
              onClick={() => {
                setStage(s);
                setGenerated(null);
              }}
              className={cn(
                "h-10 rounded-md px-3 font-mono text-sm uppercase tracking-wide shadow-[var(--shadow-border)] transition-[background-color,color] duration-150 ease-out",
                stage === s ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg",
              )}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          {STAGE_REPLIES.base.map((p, i) => (
            <button
              key={p.prompt}
              type="button"
              onClick={() => {
                setPromptIdx(i);
                setGenerated(null);
              }}
              className={cn(
                "h-10 rounded-md px-3 text-sm shadow-[var(--shadow-border)] transition-[background-color,color] duration-150 ease-out",
                promptIdx === i ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg",
              )}
            >
              {p.prompt}
            </button>
          ))}
        </div>

        <div className="mt-5 grid gap-4 lg:grid-cols-[1fr_auto] lg:items-start">
          <article className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
            <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">
              {stage} · prompt
            </p>
            <p className="mt-3 text-sm text-fg">{pair.prompt}</p>
            <pre className="mt-4 whitespace-pre-wrap font-mono text-sm leading-relaxed text-muted">
              {generated ?? pair.reply}
            </pre>
          </article>
          <button
            type="button"
            onClick={runGenerate}
            className="inline-flex h-11 items-center justify-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]"
          >
            Sample again
          </button>
        </div>
      </div>
    </section>
  );
}
