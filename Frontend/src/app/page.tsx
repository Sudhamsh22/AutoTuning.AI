import { Header } from '@/components/layout/header';
import { LandingHero } from '@/components/landing-hero';

export default function Home() {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1">
        <LandingHero />
      </main>
    </div>
  );
}
