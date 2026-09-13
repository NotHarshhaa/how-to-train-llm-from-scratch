import { AUTHOR, DOCS, PAPER, REPO } from "@/lib/content";

export function NameSection() {
  return (
    <section id="name" className="scroll-mt-16 border-b border-border">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-16 lg:py-20">
        <div>
          <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">Fit, not force</p>
          <h2 className="font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl">
            The name
          </h2>
          <p className="mt-5 text-base leading-relaxed text-muted">
            A substitution test shows that T1–T9 do not print the phrase “from scratch.” Replace it
            with any wrapper — Hugging Face transformers, trl, peft — and every sum of the
            handwritten modules still compiles. That test is valid and small. It shows only that
            arithmetic of tensors does not output a title.
          </p>
          <p className="mt-4 text-base leading-relaxed text-muted">
            It does not show that every title fits τ equally. “From scratch,” in the inherited
            sense, already carries content: one net, built of small files, trained on public data,
            aligned by losses you can read. τ exhibits a single architecture, a first triad of
            stages, and a unique sink for what those stages generate.
          </p>
          <p className="mt-4 text-base leading-relaxed text-muted">
            This is a semantic argument, not a tenth evaluation of CE. It stands or falls on whether
            MLP, Head, and Block carry the structural role claimed for them. The README should not
            present it as a theorem. It is the name of the path.
          </p>
        </div>
        <aside className="mt-4 rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6 lg:mt-12">
          <p className="font-display text-2xl leading-snug font-medium tracking-[-0.02em] italic text-fg">
            τ = (MLP, Head, Blockᴺ, CE, SFT → {"{"}DPO, PPO, GRPO{"}"})
          </p>
          <p className="mt-5 text-sm leading-relaxed text-muted">
            From-scratch, in this reading, names τ. Existence as a hosted billion-parameter service
            is not a line of this map. Existence as form is.
          </p>
          <dl className="mt-6 space-y-3 border-t border-border pt-5 text-sm">
            <div className="flex gap-4">
              <dt className="w-24 shrink-0 font-medium text-fg">Proved</dt>
              <dd className="text-muted">T1–T9, the formula, the CE barrier, unique sink at chat</dd>
            </div>
            <div className="flex gap-4">
              <dt className="w-24 shrink-0 font-medium text-fg">Stipulated</dt>
              <dd className="text-muted">r50k_base, ReLU not GELU, untied lm_head, pre-norm</dd>
            </div>
            <div className="flex gap-4">
              <dt className="w-24 shrink-0 font-medium text-fg">Fit</dt>
              <dd className="text-muted">The name from-scratch, pedagogical</dd>
            </div>
            <div className="flex gap-4">
              <dt className="w-24 shrink-0 font-medium text-fg">Evidence</dt>
              <dd className="text-muted">
                13M / 77M / 406M; GSM8K table; RM accuracy 0.574; Attention is All You Need
              </dd>
            </div>
            <div className="flex gap-4">
              <dt className="w-24 shrink-0 font-medium text-fg">Refused</dt>
              <dd className="text-muted">transformers, trl, peft; proof of a product</dd>
            </div>
          </dl>
          <p className="mt-6 text-sm leading-relaxed text-muted">
            Source:{" "}
            <a
              href={REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              FareedKhan-dev/train-llm-from-scratch
            </a>
            <span className="text-subtle"> · </span>
            <a
              href={DOCS}
              target="_blank"
              rel="noopener noreferrer"
              className="text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              Docs
            </a>
            <span className="text-subtle"> · </span>
            <a
              href={PAPER}
              target="_blank"
              rel="noopener noreferrer"
              className="text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              Vaswani et al.
            </a>
            <span className="text-subtle"> · </span>
            <a
              href={AUTHOR}
              target="_blank"
              rel="noopener noreferrer"
              className="text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              Fareed Khan
            </a>
          </p>
        </aside>
      </div>
    </section>
  );
}
