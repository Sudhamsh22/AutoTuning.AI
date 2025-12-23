
'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { Wrench, AlertTriangle, ArrowLeft } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';
import { Header } from '@/components/layout/header';
import { Button } from '@/components/ui/button';

function OperationalCategoryContent() {
  const searchParams = useSearchParams();

  const queryParams = new URLSearchParams(searchParams.toString());
  const sustenanceLink = `/part-identification?${queryParams.toString()}`;
  const criticalLink = `/critical-diagnosis?${queryParams.toString()}&priority=critical`;

  const backLink = `/mission-selection?${queryParams.toString()}`;

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 md:px-6 py-12 md:py-24">
        <div className="max-w-4xl mx-auto mb-12">
            <Button variant="ghost" asChild className="mb-8">
                <Link href={backLink}>
                    <ArrowLeft className="w-4 h-4 mr-2" />
                    Back to Origin
                </Link>
            </Button>
            <div className="text-center">
                <h1 className="text-5xl font-bold font-headline-display italic uppercase tracking-wider">Operational Category</h1>
                <p className="text-muted-foreground mt-2 uppercase text-sm tracking-widest">Select Maintenance Priority Level</p>
            </div>
        </div>
        
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
          <Link href={sustenanceLink}>
            <Card className="h-full bg-card/40 border-2 border-input hover:border-primary transition-all group flex flex-col p-8 rounded-2xl text-center">
              <CardHeader className="p-0 items-center">
                <div className="mb-4 w-20 h-20 rounded-lg bg-input flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                  <Wrench className="w-10 h-10 text-primary" />
                </div>
                <CardTitle className="text-2xl font-bold font-headline-display italic uppercase">Part Identification</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 p-0 mt-4">
                <p className="text-muted-foreground uppercase text-sm">
                  Identify and find parts for your vehicle.
                </p>
              </CardContent>
            </Card>
          </Link>
          <Link href={criticalLink}>
            <Card className="h-full bg-card/40 border-2 border-input hover:border-destructive transition-all group flex flex-col p-8 rounded-2xl text-center hover:border-red-500">
               <CardHeader className="p-0 items-center">
                <div className="mb-4 w-20 h-20 rounded-lg bg-input flex items-center justify-center group-hover:bg-destructive/10 transition-colors">
                  <AlertTriangle className="w-10 h-10 text-destructive" />
                </div>
                <CardTitle className="text-2xl font-bold font-headline-display italic uppercase">Diagnostics</CardTitle>
              </CardHeader>
              <CardContent className="flex-1 p-0 mt-4">
                <p className="text-muted-foreground uppercase text-sm">
                  Describe the issue and get a preliminary diagnosis from our AI.
                </p>
              </CardContent>
            </Card>
          </Link>
        </div>
      </main>
    </div>
  );
}

export default function OperationalCategoryPage() {
    return (
        <Suspense fallback={<div>Loading...</div>}>
            <OperationalCategoryContent />
        </Suspense>
    )
}
