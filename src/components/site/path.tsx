import { useEffect, useMemo, useState } from "react";
import { PIPELINE } from "@/lib/content";
import { tokenize } from "@/lib/tokenize";
import { cn } from "@/lib/cn";

const DEFAULT_TEXT = "Attention is all you need to train an LLM from scratch.";

export function PathSection() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [cursor, setCursor] = useState(0);
  const [playing, setPlaying] = useState(false);

  const tokens = useMemo(() => tokenize(text), [text]);

  useEffect(() => {
    if (!playing) return;
    const id = window.setInterval(() => {
      setCursor((c) => {
        if (c >= PIPELINE.length - 1) {
          setPlaying(false);
          return c;
        }
        return c + 1;
      });
    }, 900);
    return () => window.clearInterval(id);
  }, [playing]);

  const current = PIPELINE[cursor] ?? PIPELINE[0];

  return (
    <section id="path" className="scroll-mt-16 border-b border-border">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20">
        <div className="flex min-w-0 flex-col gap-5">
          <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">The operator</p>
          <h2 className="font-display text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl">
            The source-line
          </h2>
          <p className="text-base leading-relaxed text-muted">
            The whole journey is one idea repeated: turn text into numbers, predict the next token,
            then keep changing the data and the loss until the model does what we want. Fareed
            Khan’s repo occupies that line from The Pile to chat, in plain PyTorch.
          </p>
          <p className="font-mono text-sm break-words text-fg">
            τ : text → tokens → Transformer → CE → base → SFT → {"{"}PPO, DPO{"}"} → GRPO → chat
          </p>
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              onClick={() => {
                setCursor(0);
                setPlaying(true);
              }}
              className="inline-flex h-11 items-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]"
            >
              Walk the source-line
            </button>
            <button
              type="button"
              onClick={() => {
                setPlaying(false);
                setCursor(0);
              }}
              className="inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg shadow-[var(--shadow-border)] transition-[scale,box-shadow,background-color] duration-150 ease-out hover:bg-elevated/60 active:scale-[0.96]"
            >
              Reset
            </button>
          </div>
        </div>

        <aside className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
          <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">FN · eleven seats</p>
          <ol className="mt-5 flex flex-wrap gap-2">
            {PIPELINE.map((step, i) => {
              const active = i === cursor;
              return (
                <li key={step.id} className="relative">
                  {active ? <span className="dwell-ring" /> : null}
                  <button
                    type="button"
                    onClick={() => {
                      setPlaying(false);
                      setCursor(i);
                    }}
                    className={cn(
                      "relative h-10 rounded-md px-3 font-mono text-sm tabular-nums shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out",
                      active
                        ? "dwell-beat bg-elevated text-fg"
                        : "text-muted hover:bg-elevated hover:text-fg",
                      step.sink && active && "text-mark",
                    )}
                  >
                    {step.label}
                  </button>
                </li>
              );
            })}
          </ol>
          <p className="mt-5 font-display text-2xl leading-snug font-medium tracking-[-0.02em] italic text-fg">
            {current.label}
            {current.sink ? <span className="text-mark"> ∞</span> : null}
          </p>
          <p className="mt-2 text-sm leading-relaxed text-muted">{current.hint}</p>
        </aside>
      </div>

      <div className="mx-auto max-w-6xl px-4 pb-16 sm:px-6 lg:pb-20">
        <div className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">Tokenize</p>
              <h3 className="font-display mt-2 text-2xl font-medium tracking-[-0.02em]">
                Write n as pieces
              </h3>
            </div>
            <p className="font-mono text-xs text-subtle">
              {tokens.length} pieces · stand-in IDs, not r50k_base
            </p>
          </div>
          <label htmlFor="tok-input" className="mt-5 block text-sm font-medium text-fg">
            Raw text
          </label>
          <textarea
            id="tok-input"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={3}
            className="mt-2 w-full resize-y rounded-md bg-elevated px-3 py-2.5 font-mono text-sm text-fg shadow-[var(--shadow-border)] outline-none placeholder:text-subtle focus-visible:ring-2 focus-visible:ring-ring/70"
          />
          <p className="mt-2 text-xs leading-relaxed text-subtle">
            The repo uses tiktoken <span className="text-fg">r50k_base</span> (50 257 tokens, padded
            to 50 304). This box only splits on the GPT-2 regex and hashes each piece so you can see
            the cut. The real BPE table lives in the training scripts.
          </p>
          <div className="mt-4 flex flex-wrap gap-1.5">
            {tokens.length === 0 ? (
              <p className="text-sm text-subtle">Type something to see the cut.</p>
            ) : (
              tokens.map((tok, i) => (
                <span key={`${tok.id}-${i}`} className="token-chip">
                  <span className="text-fg">{tok.piece === " " ? "␣" : tok.piece}</span>
                  <span className="text-subtle tabular-nums">{tok.id}</span>
                </span>
              ))
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
