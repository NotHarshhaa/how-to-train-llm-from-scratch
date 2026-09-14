import { useMemo, useState } from "react";
import { PIECES } from "@/lib/content";
import { countParams, formatCompact, formatInt, GPUS, PRESETS, VOCAB_SIZE } from "@/lib/params";
import { cn } from "@/lib/cn";

export function ModelSection() {
  const [presetId, setPresetId] = useState<(typeof PRESETS)[number]["id"] | "custom">("13m");
  const [nEmbed, setNEmbed] = useState(128);
  const [nHead, setNHead] = useState(8);
  const [nBlocks, setNBlocks] = useState(1);
  const [context, setContext] = useState(256);
  const [pieceId, setPieceId] = useState<(typeof PIECES)[number]["id"]>("block");

  const preset = PRESETS.find((p) => p.id === presetId);

  const spec = {
    nEmbed,
    nHead,
    nBlocks,
    contextLength: context,
    vocabSize: VOCAB_SIZE,
  };

  const broken = nHead < 1 || nEmbed % nHead !== 0;
  const counted = useMemo(() => (broken ? null : countParams(spec)), [broken, nEmbed, nHead, nBlocks, context]);
  const displayTotal = preset ? preset.reported : (counted?.total ?? 0);
  const piece = PIECES.find((p) => p.id === pieceId) ?? PIECES[3];

  function applyPreset(id: (typeof PRESETS)[number]["id"]) {
    const p = PRESETS.find((x) => x.id === id)!;
    setPresetId(id);
    setNEmbed(p.nEmbed);
    setNHead(p.nHead);
    setNBlocks(p.nBlocks);
    setContext(p.contextLength);
  }

  return (
    <section id="model" className="scroll-mt-16 border-b border-border">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-14 lg:py-20">
        <div className="flex min-w-0 flex-col gap-5">
          <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">The architecture</p>
          <h2 className="font-display text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl">
            Analeza, for nets
          </h2>
          <p className="text-base leading-relaxed text-muted">
            τ is the kucwenga map of this repo: write a model as (embed × 1), split every block down
            to MLP and Head, then sum every Linear the splits produce. For n_embed = d the lemma
            gives a single formula, independent of split order.
          </p>
          <p className="font-mono text-sm break-words text-fg">
            τ(d, N, T, V) = Vd + Td + N(12d² + 10d) + 2d + Vd + V
          </p>
          <p className="text-xs leading-relaxed text-subtle">
            Vocab V is 50 304. Untied lm_head. Attention keys have no bias; MLP and projections do.
            Reported 13M / 77M / 406M counts come from the README; custom uses the formula above.
          </p>
        </div>

        <form
          className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6"
          onSubmit={(e) => e.preventDefault()}
        >
          <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">Evaluate τ</p>
          <div className="mt-4 flex flex-wrap gap-2">
            {PRESETS.map((p) => (
              <button
                key={p.id}
                type="button"
                onClick={() => applyPreset(p.id)}
                className={cn(
                  "h-10 min-w-10 rounded-md px-3 font-mono text-sm tabular-nums shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out",
                  presetId === p.id ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg",
                )}
              >
                {p.label}
              </button>
            ))}
            <button
              type="button"
              onClick={() => setPresetId("custom")}
              className={cn(
                "h-10 min-w-10 rounded-md px-3 font-mono text-sm shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out",
                presetId === "custom" ? "bg-elevated text-fg" : "text-muted hover:bg-elevated hover:text-fg",
              )}
            >
              custom
            </button>
          </div>

          <div className="mt-5 grid gap-4 sm:grid-cols-2">
            <NumberField
              label="n_embed d"
              value={nEmbed}
              min={64}
              max={2048}
              step={64}
              onChange={(v) => {
                setPresetId("custom");
                setNEmbed(v);
              }}
            />
            <NumberField
              label="n_head"
              value={nHead}
              min={1}
              max={32}
              step={1}
              onChange={(v) => {
                setPresetId("custom");
                setNHead(v);
              }}
            />
            <NumberField
              label="N_BLOCKS"
              value={nBlocks}
              min={1}
              max={48}
              step={1}
              onChange={(v) => {
                setPresetId("custom");
                setNBlocks(v);
              }}
            />
            <NumberField
              label="context T"
              value={context}
              min={128}
              max={2048}
              step={128}
              onChange={(v) => {
                setPresetId("custom");
                setContext(v);
              }}
            />
          </div>

          {broken ? (
            <p className="mt-4 text-sm text-mark">d must be divisible by n_head. Head size is d / n_head.</p>
          ) : (
            <dl className="mt-5 space-y-2 border-t border-border pt-5 text-sm">
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Parameters</dt>
                <dd className="font-mono tabular-nums text-fg">{formatInt(displayTotal)}</dd>
              </div>
              <div className="flex justify-between gap-4">
                <dt className="text-muted">Compact</dt>
                <dd className="font-mono tabular-nums text-fg">{formatCompact(displayTotal)}</dd>
              </div>
              {counted ? (
                <div className="flex justify-between gap-4">
                  <dt className="text-muted">Head size</dt>
                  <dd className="font-mono tabular-nums text-fg">{counted.headSize}</dd>
                </div>
              ) : null}
            </dl>
          )}
        </form>
      </div>

      <div className="mx-auto max-w-6xl px-4 pb-10 sm:px-6">
        <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">The chamber</p>
        <h3 className="font-display mt-3 text-3xl font-medium tracking-[-0.03em] sm:text-4xl">
          Five seats, not fourteen
        </h3>
        <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted">
          L_τ = {"{"}MLP, Head, Multi-head, Block, Transformer{"}"}. Five — not chosen to match a
          paper, only noticed after the files were named. Click a seat.
        </p>
        <div className="mt-6 grid gap-2 sm:grid-cols-5">
          {PIECES.map((p) => (
            <button
              key={p.id}
              type="button"
              onClick={() => setPieceId(p.id)}
              className={cn(
                "rounded-lg px-3 py-3 text-left shadow-[var(--shadow-border)] transition-[background-color,box-shadow,color] duration-150 ease-out",
                pieceId === p.id ? "bg-elevated text-fg" : "bg-surface text-muted hover:text-fg",
              )}
            >
              <span className="font-display text-xl italic">{p.title}</span>
              <span className="mt-1 block font-mono text-[11px] tracking-wide text-subtle">{p.file}</span>
            </button>
          ))}
        </div>
        <article className="mt-4 rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
          <p className="font-mono text-sm break-words text-fg">{piece.formula}</p>
          <p className="mt-3 text-sm leading-relaxed text-muted">{piece.note}</p>
          <pre className="mt-4 overflow-x-auto rounded-lg bg-elevated p-4 font-mono text-xs leading-relaxed text-accent shadow-[var(--shadow-border)]">
            {piece.code}
          </pre>
        </article>
      </div>

      <div className="mx-auto max-w-6xl px-4 pb-16 sm:px-6 lg:pb-20">
        <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">Hardware</p>
        <h3 className="font-display mt-3 text-2xl font-medium tracking-[-0.02em]">Which GPUs hold τ</h3>
        <div className="mt-5 max-w-full overflow-x-auto">
          <table className="w-full min-w-[36rem] text-left text-sm">
            <thead>
              <tr className="border-b border-border text-xs tracking-[0.14em] text-subtle uppercase">
                <th className="py-2 pr-4 font-medium">GPU</th>
                <th className="py-2 pr-4 font-medium">Memory</th>
                <th className="py-2 pr-4 font-medium">13M</th>
                <th className="py-2 pr-4 font-medium">~2B</th>
                <th className="py-2 font-medium">Practical max</th>
              </tr>
            </thead>
            <tbody>
              {GPUS.map((g) => (
                <tr key={g.name} className="border-b border-border/80">
                  <td className="py-2.5 pr-4 text-fg">{g.name}</td>
                  <td className="py-2.5 pr-4 font-mono tabular-nums text-muted">{g.memory}</td>
                  <td className="py-2.5 pr-4 font-mono text-mark">{g.fit13 ? "✔" : "✘"}</td>
                  <td className="py-2.5 pr-4 font-mono text-muted">{g.fit2b ? "✔" : "✘"}</td>
                  <td className="py-2.5 font-mono text-muted">{g.max}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-xs leading-relaxed text-subtle">
          A free Colab or Kaggle T4 is enough for 13 million parameters. A billion-parameter run is
          not. Flags <span className="text-fg">--amp</span>,{" "}
          <span className="text-fg">--grad-checkpointing</span>,{" "}
          <span className="text-fg">--grad-accum</span> bring the memory down when a large config
          refuses.
        </p>
      </div>
    </section>
  );
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="flex flex-col gap-1.5 text-sm font-medium text-fg">
      {label}
      <input
        type="number"
        inputMode="numeric"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => {
          const raw = e.target.value;
          if (raw === "") return;
          const v = Number(raw);
          if (Number.isFinite(v) && v >= 0) onChange(v);
        }}
        className="h-11 w-full rounded-md bg-elevated px-3 font-mono text-base text-fg tabular-nums shadow-[var(--shadow-border)] outline-none focus-visible:ring-2 focus-visible:ring-ring/70"
      />
    </label>
  );
}
