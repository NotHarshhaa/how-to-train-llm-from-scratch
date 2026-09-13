const LINKS = [
  { href: "#path", label: "Path" },
  { href: "#model", label: "Model" },
  { href: "#loss", label: "Loss" },
  { href: "#align", label: "Align" },
  { href: "#eval", label: "Eval" },
  { href: "#name", label: "Name" },
  { href: "#repo", label: "Repo" },
] as const;

export function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl min-w-0 items-center justify-between gap-3 px-4 sm:h-16 sm:gap-4 sm:px-6">
        <a href="#top" className="flex min-w-0 shrink items-baseline gap-2 no-underline">
          <span className="font-display text-2xl leading-none text-fg italic">τ</span>
          <span className="truncate text-xs font-medium tracking-[0.14em] text-fg uppercase sm:text-sm sm:tracking-[0.16em]">
            From Scratch
          </span>
        </a>
        <nav aria-label="Sections" className="flex min-w-0 flex-1 items-center justify-end gap-1 overflow-x-auto sm:gap-0">
          {LINKS.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="shrink-0 rounded-md px-2.5 py-2 text-xs font-medium tracking-wide text-muted no-underline transition-[color,background-color] duration-150 ease-out hover:text-fg focus-visible:ring-2 focus-visible:ring-ring/70 sm:px-3 sm:text-sm"
            >
              {link.label}
            </a>
          ))}
        </nav>
      </div>
    </header>
  );
}
