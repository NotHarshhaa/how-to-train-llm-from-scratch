import { useState } from "react";
import { REPO, DOCS } from "@/lib/content";
import { Check, Copy } from "lucide-react";

const CLONE = `git clone https://github.com/FareedKhan-dev/train-llm-from-scratch.git
cd train-llm-from-scratch
pip install -e .`;

const EXTRAS = `pip install -e ".[train]"   # datasets + wandb
pip install -e ".[ui]"      # streamlit control panel
pip install -e ".[docs]"    # mkdocs
pip install -e ".[all]"`;

const RUN = `NPROC=1 bash scripts/run_posttraining.sh`;

function CopyBlock({ label, code }: { label: string; code: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <div className="rounded-xl bg-surface p-5 shadow-[var(--shadow-border)] sm:p-6">
      <div className="flex items-center justify-between gap-3">
        <p className="font-mono text-xs tracking-[0.18em] text-subtle uppercase">{label}</p>
        <button
          type="button"
          onClick={async () => {
            try {
              await navigator.clipboard.writeText(code);
              setCopied(true);
              window.setTimeout(() => setCopied(false), 1400);
            } catch {
              setCopied(false);
            }
          }}
          className="inline-flex h-10 items-center gap-2 rounded-md px-3 text-sm text-muted shadow-[var(--shadow-border)] transition-[background-color,color] duration-150 ease-out hover:bg-elevated hover:text-fg"
          aria-label={`Copy ${label}`}
        >
          {copied ? <Check className="size-4" strokeWidth={2} /> : <Copy className="size-4" strokeWidth={2} />}
          <span className="font-mono text-xs">{copied ? "copied" : "copy"}</span>
        </button>
      </div>
      <pre className="mt-4 overflow-x-auto font-mono text-xs leading-relaxed text-fg sm:text-sm">{code}</pre>
    </div>
  );
}

export function RepoSection() {
  return (
    <section id="repo" className="scroll-mt-16 border-b border-border">
      <div className="mx-auto grid max-w-6xl gap-10 px-4 py-16 sm:px-6 lg:grid-cols-2 lg:items-start lg:gap-16 lg:py-20">
        <div>
          <p className="text-xs font-medium tracking-[0.22em] text-muted uppercase">The sink amount</p>
          <h2 className="font-display mt-3 text-4xl leading-tight font-medium tracking-[-0.03em] sm:text-5xl">
            The repository
          </h2>
          <p className="mt-5 text-base leading-relaxed text-muted">
            Twelve was the unique sink of An. Chat is the unique sink of τ. The asked gift is the
            same number the README already asks: clone the repo, run the smoke configs, read the
            files. Stars are optional. The MIT license is not.
          </p>
          <p className="mt-4 text-sm leading-relaxed text-subtle">
            Fareed Khan is looking for a PhD position in AI. The work is a tutorial that grew into a
            full post-training line — still one idea, repeated.
          </p>
          <div className="mt-6 flex flex-col gap-2 sm:flex-row">
            <a
              href={REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-12 flex-1 items-center justify-center rounded-md bg-accent px-4 text-sm font-medium text-accent-fg no-underline transition-[scale,background-color] duration-150 ease-out hover:bg-fg active:scale-[0.96]"
            >
              Open the GitHub repo
            </a>
            <a
              href={DOCS}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex h-12 flex-1 items-center justify-center rounded-md px-4 text-sm font-medium text-fg no-underline shadow-[var(--shadow-border)] transition-[scale,box-shadow,background-color] duration-150 ease-out hover:bg-elevated/60 hover:shadow-[var(--shadow-border-hover)] active:scale-[0.96]"
            >
              Read the docs
            </a>
          </div>
        </div>
        <div className="flex flex-col gap-4">
          <p className="font-display text-5xl leading-none font-medium tracking-[-0.04em] tabular-nums text-fg">
            9.6k
          </p>
          <p className="text-sm text-muted">stars · 1.3k forks · MIT</p>
          <CopyBlock label="Setup" code={CLONE} />
          <CopyBlock label="Extras" code={EXTRAS} />
          <CopyBlock label="Run the line" code={RUN} />
        </div>
      </div>
    </section>
  );
}
