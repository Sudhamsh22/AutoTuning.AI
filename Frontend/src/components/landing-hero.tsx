import Link from 'next/link';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { PlaceHolderImages } from '@/lib/placeholder-images';

export function LandingHero() {
  const heroImage = PlaceHolderImages.find(p => p.id === 'hero-image');

  return (
    <>
      <section className="container mx-auto px-4 md:px-6 py-12 md:py-24 lg:py-32">
        <div className="grid md:grid-cols-2 gap-8 lg:gap-16 items-center">
          <div className="space-y-6">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-headline font-black tracking-tighter text-foreground">
              Your AI-Powered Personal Mechanic
            </h1>
            <p className="text-lg md:text-xl text-muted-foreground max-w-2xl">
              Improve, maintain, and upgrade your bike or car with our autonomous multi-agent vehicle advisor. Get diagnoses, part recommendations, and clear roadmaps, all from a few photos.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Button asChild size="lg" className="bg-primary hover:bg-primary/90 text-primary-foreground shadow-lg shadow-primary/20">
                <Link href="/analyze">Analyze My Vehicle</Link>
              </Button>
              <Button asChild size="lg" variant="outline" className="border-accent text-accent hover:bg-accent/10 hover:text-accent">
                <Link href="/#features">Explore Features</Link>
              </Button>
            </div>
          </div>
          <div className="relative aspect-video rounded-xl overflow-hidden shadow-2xl shadow-primary/10">
            {heroImage && (
              <Image
                src={heroImage.imageUrl}
                alt={heroImage.description}
                fill
                priority
                className="object-cover"
                data-ai-hint={heroImage.imageHint}
              />
            )}
            <div className="absolute inset-0 bg-gradient-to-t from-background/50 to-transparent"></div>
          </div>
        </div>
      </section>
      <section id="features" className="container mx-auto px-4 md:px-6 py-12 md:py-24 lg:py-32 bg-card rounded-t-xl">
        {/* Placeholder for features section */}
         <div className="text-center space-y-4">
            <h2 className="text-3xl font-bold font-headline">Features Coming Soon</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">We are working hard to bring you a full suite of features to manage your vehicle's health.</p>
        </div>
      </section>
    </>
  );
}
