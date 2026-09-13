import { AUTHOR, DOCS, PAPER, REPO } from "@/lib/content";

export function SiteFooter() {
  return (
    <footer className="border-t border-border">
      <div className="mx-auto flex max-w-6xl flex-col gap-8 px-4 py-10 sm:flex-row sm:items-end sm:justify-between sm:px-6">
        <div>
          <p className="font-display text-2xl italic text-fg">τ</p>
          <p className="mt-2 max-w-md text-sm leading-relaxed text-muted">
            Training an LLM from scratch — an interactive reading of Fareed Khan’s repository. A
            defined path code can certify, and a name fitted rather than free.
          </p>
          <p className="mt-4 max-w-md text-sm leading-relaxed text-muted">
            Source:{" "}
            <a
              href={REPO}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              train-llm-from-scratch
            </a>
            <span className="text-subtle"> · </span>
            <a
              href={DOCS}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              Docs
            </a>
            <span className="text-subtle"> · </span>
            <a
              href={PAPER}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              Attention is All You Need
            </a>
          </p>
          <p className="mt-1 text-sm leading-relaxed text-muted">
            Author:{" "}
            <a
              href={AUTHOR}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex min-h-11 items-center text-fg underline decoration-border underline-offset-4 transition-[color,text-decoration-color] duration-150 ease-out hover:decoration-fg"
            >
              Fareed Khan
            </a>
          </p>
        </div>
        <p className="text-xs tracking-wide text-subtle">Form, not product. Occupied, not wrapped.</p>
      </div>
    </footer>
  );
}
