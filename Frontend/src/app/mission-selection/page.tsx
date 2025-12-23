
'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Wrench, ArrowRight, GaugeCircle } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';
import { Header } from '@/components/layout/header';

function MissionSelectionContent() {
  const searchParams = useSearchParams();

  const queryParams = new URLSearchParams(searchParams.toString());
  const repairLink = `/operational-category?${queryParams.toString()}&mission=repair`;
  const tuneLink = `/ecu-tuning?${queryParams.toString()}&mission=tune`;


  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 md:px-6 py-12 md:py-24">
        <div className="text-center max-w-3xl mx-auto mb-12">
          <h1 className="text-5xl font-bold font-headline-display italic uppercase tracking-wider">Mission Selection</h1>
          <p className="text-muted-foreground mt-2 uppercase text-sm tracking-widest">Define agent objectives for vehicle telemetry analysis</p>
        </div>
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          <Link href={repairLink}>
            <Card className="h-full bg-card/40 border-2 border-input hover:border-primary transition-all group flex flex-col p-8 rounded-2xl">
              <CardHeader className="p-0">
                <div className="mb-4 w-16 h-16 rounded-lg bg-input flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                  <Wrench className="w-8 h-8 text-primary" />
                </div>
                <CardTitle className="text-2xl font-bold font-headline-display italic uppercase">Repair & Ops</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 p-0 mt-4">
                <p className="text-muted-foreground">
                  Identify faults, diagnose anomalous sounds, and optimize maintenance intervals for maximum reliability.
                </p>
              </CardContent>
              <div className="mt-8">
                <div className="flex items-center gap-2 font-bold text-primary text-sm uppercase tracking-wider">
                  Access Diagnostics <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                </div>
              </div>
            </Card>
          </Link>
          <Link href={tuneLink}>
            <Card className="h-full bg-card/40 border-2 border-input hover:border-primary transition-all group flex flex-col p-8 rounded-2xl">
              <CardHeader className="p-0">
                <div className="mb-4 w-16 h-16 rounded-lg bg-input flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                  <GaugeCircle className="w-8 h-8 text-primary" />
                </div>
                <CardTitle className="text-2xl font-bold font-headline-display italic uppercase">Tune & Kinetic</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 p-0 mt-4">
                <p className="text-muted-foreground">
                  Explore synthetic performance upgrades, kinetic modifications, and aerodynamic synthesis to exceed factory limits.
                </p>
              </CardContent>
              <div className="mt-8">
                <div className="flex items-center gap-2 font-bold text-primary text-sm uppercase tracking-wider">
                  Execute Synthesis <ArrowRight className="w-4 h-4 transition-transform group-hover:translate-x-1" />
                </div>
              </div>
            </Card>
          </Link>
        </div>
      </main>
    </div>
  );
}


export default function MissionSelectionPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <MissionSelectionContent />
        </Suspense>
    )
}
