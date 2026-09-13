import { PIPELINE } from "@/lib/content";

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden border-b border-border">
      <div aria-hidden="true" className="tau-watermark absolute -right-6 top-10 text-fg/5 sm:right-10 sm:top-6">
        τ
      </div>
      <div className="stagger-in mx-auto flex max-w-6xl flex-col gap-8 px-4 py-16 sm:px-6 sm:py-24 lg:py-28">
        <p className="flex items-baseline gap-3 text-xs font-medium tracking-[0.22em] text-muted uppercase">
          <span className="font-display text-2xl leading-none font-medium tracking-normal text-fg italic normal-case">
            τ
          </span>
          After Attention, without the wrappers
        </p>
        <h1 className="font-display max-w-3xl text-5xl leading-[0.92] font-medium tracking-[-0.03em] text-fg sm:text-6xl lg:text-7xl">
          Training an LLM from <span className="italic">scratch</span>
        </h1>
        <p className="max-w-xl text-base leading-relaxed text-muted sm:text-lg">
          Code proves a path, not a product. Here the path is occupied: one architecture, a first
          loss, a source-line of stages, and a unique sink at chat. From-scratch names that path —
          fitted, not forced.
        </p>
        <div className="flex flex-wrap items-center gap-3">
          <a
            href="#path"
            className="inline-flex h-11 items-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg no-underline transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]"
          >
            Walk the source-line
          </a>
          <a
            href="#model"
            className="inline-flex h-11 items-center rounded-md px-4 text-sm font-medium text-fg no-underline shadow-[var(--shadow-border)] transition-[scale,box-shadow,background-color] duration-150 ease-out hover:bg-elevated/60 hover:shadow-[var(--shadow-border-hover)] active:scale-[0.96]"
          >
            Compute τ(d)
          </a>
        </div>
        <p className="flex flex-wrap items-baseline gap-x-1 gap-y-1 font-mono text-xs leading-relaxed tracking-wide text-subtle sm:text-sm">
          {PIPELINE.map((step, i) => (
            <span key={step.id}>
              {i > 0 ? <span className="text-subtle/50"> → </span> : null}
              {step.label}
              {step.sink ? <sup className="text-mark">∞</sup> : null}
            </span>
          ))}
        </p>
      </div>
    </section>
  );
}
