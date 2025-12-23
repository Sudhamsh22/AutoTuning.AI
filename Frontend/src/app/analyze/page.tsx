'use client';

import { VehicleForm } from '@/components/vehicle-form';

export default function AnalyzePage() {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <main className="flex-1 container mx-auto px-4 md:px-6 py-8 md:py-12 flex items-center justify-center">
        <div className="w-full max-w-3xl">
          <VehicleForm />
        </div>
      </main>
    </div>
  );
}
