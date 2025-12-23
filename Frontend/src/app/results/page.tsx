import { Suspense } from 'react';
import { Header } from '@/components/layout/header';
import ResultsView from '@/components/results-view';
import { Skeleton } from '@/components/ui/skeleton';

function ResultsPageSkeleton() {
    return (
      <div className="container mx-auto px-4 md:px-6 py-8 md:py-12">
        <Skeleton className="h-8 w-1/3 mb-4" />
        <Skeleton className="h-4 w-2/3 mb-8" />
        <div className="space-y-6">
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
            <Skeleton className="h-24 w-full" />
        </div>
      </div>
    );
  }

export default function ResultsPage() {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1">
        <Suspense fallback={<ResultsPageSkeleton />}>
          <ResultsView />
        </Suspense>
      </main>
    </div>
  );
}
