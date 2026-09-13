import { useMemo, useState } from "react";
import { cn } from "@/lib/cn";

const PROBLEM = {
  id: "gsm8k-natalia",
  question:
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?",
  gold: "72",
  reasoning:
    "April: 48. May: 48 / 2 = 24. Total: 48 + 24 = 72.",
};

function parseAnswer(raw: string): string | null {
  const tagged = raw.match(/<answer>\s*([^<]+?)\s*<\/answer>/i);
  if (tagged) return tagged[1].trim();
  const last = raw.trim().split(/\s+/).at(-1);
  if (!last) return null;
  const num = last.replace(/[^0-9.-]/g, "");
  return num || last;
}

export function EvalSection() {
  const [draft, setDraft] = useState("<answer>72</answer>");
  const [submitted, setSubmitted] = useState<string | null>(null);

  const parsed = useMemo(() => (submitted === null ? null : parseAnswer(submitted)), [submitted]);
  const correct = parsed !== null && parsed.replace(/,/g, "") === PROBLEM.gold;

  return (
    <section id="eval" className="scroll-mt-16 border-b border-border">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20">
        <div>
          <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">The evaluation</p>
          <h2 className="font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl">
            GSM8K greedy
          </h2>
          <p className="mt-5 text-base leading-relaxed text-muted">
            T8. Generate greedily, parse the{" "}
            <span className="font-mono text-fg">{"<answer>"}</span> tag, compare to gold. That is
            the whole metric.{" "}
            <span className="font-mono text-fg">scripts/eval_post_training.py</span> prints one table
            across Base, SFT, DPO, PPO, and GRPO so the stages can be read as a single experiment.
          </p>
          <p className="mt-4 text-sm leading-relaxed text-subtle">
            The box on the right is the parser, not a model. Type a completion, including a tag or a
            bare number, and see whether the sink accepts it.
          </p>
        </div>

        <form
          className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6"
          onSubmit={(e) => {
            e.preventDefault();
            setSubmitted(draft);
          }}
        >
          <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">C1 · Problem</p>
          <p className="mt-3 text-sm leading-relaxed text-fg">{PROBLEM.question}</p>
          <p className="mt-3 text-xs leading-relaxed text-subtle">{PROBLEM.reasoning}</p>
          <label htmlFor="completion" className="mt-5 block text-sm font-medium text-fg">
            Completion
          </label>
          <textarea
            id="completion"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            rows={4}
            className="mt-2 w-full resize-y rounded-md bg-elevated px-3 py-2.5 font-mono text-sm text-fg shadow-[var(--shadow-border)] outline-none focus-visible:ring-2 focus-visible:ring-ring/70"
          />
          <div className="mt-4 flex flex-wrap gap-2">
            <button
              type="submit"
              className="inline-flex h-11 items-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]"
            >
              Evaluate
            </button>
            <button
              type="button"
              onClick={() => {
                setDraft("<answer>72</answer>");
                setSubmitted(null);
              }}
              className="inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg shadow-[var(--shadow-border)] transition-[scale,background-color] duration-150 ease-out hover:bg-elevated/60 active:scale-[0.96]"
            >
              Gold
            </button>
            <button
              type="button"
              onClick={() => {
                setDraft("I think the total is 48.");
                setSubmitted(null);
              }}
              className="inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg shadow-[var(--shadow-border)] transition-[scale,background-color] duration-150 ease-out hover:bg-elevated/60 active:scale-[0.96]"
            >
              Miss
            </button>
          </div>
          {submitted !== null ? (
            <dl className="mt-5 space-y-2 border-t border-border pt-5 text-sm">
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Parsed</dt>
                <dd className="font-mono text-fg">{parsed ?? "—"}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Gold</dt>
                <dd className="font-mono text-fg">{PROBLEM.gold}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Verdict</dt>
                <dd className={cn("font-mono", correct ? "text-fg" : "text-mark")}>
                  {correct ? "match" : "miss"}
                </dd>
              </div>
            </dl>
          ) : null}
        </form>
      </div>
    </section>
  );
}
