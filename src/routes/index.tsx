import { createFileRoute } from "@tanstack/react-router";
import { AlignSection } from "@/components/site/align";
import { EvalSection } from "@/components/site/eval";
import { SiteFooter } from "@/components/site/footer";
import { SiteHeader } from "@/components/site/header";
import { Hero } from "@/components/site/hero";
import { LossSection } from "@/components/site/loss";
import { ModelSection } from "@/components/site/model";
import { NameSection } from "@/components/site/name";
import { PathSection } from "@/components/site/path";
import { RepoSection } from "@/components/site/repo";

export const Route = createFileRoute("/")({ component: Home });

function Home() {
  return (
    <div className="min-h-dvh overflow-x-clip bg-bg text-fg">
      <SiteHeader />
      <main>
        <Hero />
        <PathSection />
        <ModelSection />
        <LossSection />
        <AlignSection />
        <EvalSection />
        <NameSection />
        <RepoSection />
      </main>
      <SiteFooter />
    </div>
  );
}
