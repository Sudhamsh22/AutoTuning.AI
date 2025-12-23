

"use client";

import { useEffect, useState, useMemo } from 'react';
import { useSearchParams } from 'next/navigation';
import { analyzeVehicleImages, AnalyzeVehicleImagesOutput } from '@/ai/flows/analyze-vehicle-images';
import { generateMaintenanceRoadmap, GenerateMaintenanceRoadmapOutput } from '@/ai/flows/generate-maintenance-roadmap';
import { discoverParts, DiscoverPartsOutput } from '@/ai/flows/discover-parts';
import { ProcessingView, ProcessingStep } from './processing-view';
import { ResultsDashboard } from './results-dashboard';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertTriangle } from 'lucide-react';
import { Button } from './ui/button';
import Link from 'next/link';

export type VehicleData = {
    vehicleType: string;
    brand: string;
    model: string;
    year: string;
    mileage: string;
    problem?: string | null;
};

export default function ResultsView() {
  const searchParams = useSearchParams();
  const { toast } = useToast();

  const [processingState, setProcessingState] = useState<'idle' | 'processing' | 'success' | 'error'>('idle');
  const [processingSteps, setProcessingSteps] = useState<ProcessingStep[]>([]);
  const [analysisResult, setAnalysisResult] = useState<AnalyzeVehicleImagesOutput | null>(null);
  const [roadmapResult, setRoadmapResult] = useState<GenerateMaintenanceRoadmapOutput | null>(null);
  const [partsResult, setPartsResult] = useState<DiscoverPartsOutput | null>(null);
  const [error, setError] = useState<string | null>(null);

  const vehicleData: VehicleData | null = useMemo(() => {
    const data = {
        vehicleType: searchParams.get('vehicleType'),
        brand: searchParams.get('brand'),
        model: searchParams.get('model'),
        year: searchParams.get('year'),
        mileage: searchParams.get('mileage'),
        problem: searchParams.get('problem'),
    };
    if (Object.values(data).some(v => v === null)) return null;
    return data as VehicleData;
  }, [searchParams]);

  useEffect(() => {
    if (processingState !== 'idle' || !vehicleData) return;

    const runAnalysis = async () => {
      setProcessingState('processing');
      try {
        setProcessingSteps(prev => [...prev, { name: 'Vision Analysis Agent', status: 'running' }]);
        const analysis = await analyzeVehicleImages({
          vehicleType: vehicleData.vehicleType,
          brand: vehicleData.brand,
          model: vehicleData.model,
          year: vehicleData.year,
          mileage: vehicleData.mileage,
        });
        setAnalysisResult(analysis);
        setProcessingSteps(prev => prev.map(s => s.name === 'Vision Analysis Agent' ? {...s, status: 'complete'} : s));
        toast({ title: 'Vehicle Analysis Complete', description: `${analysis.detectedIssues.length} potential issues found.` });

        setProcessingSteps(prev => [...prev, { name: 'Parts Discovery Agent', status: 'running' }]);
        const parts = await discoverParts({
          vehicleBrand: vehicleData.brand,
          vehicleModel: vehicleData.model,
          vehicleYear: vehicleData.year,
          parts: analysis.detectedIssues,
        });
        setPartsResult(parts);
        setProcessingSteps(prev => prev.map(s => s.name === 'Parts Discovery Agent' ? {...s, status: 'complete'} : s));
        toast({ title: 'Parts Discovery Complete', description: 'Found compatible parts and prices.' });

        setProcessingSteps(prev => [...prev, { name: 'Maintenance Planner Agent', status: 'running' }]);
        const issues = [
            ...(analysis.detectedIssues),
            ...(vehicleData.problem ? [vehicleData.problem] : [])
        ].join(', ');

        const roadmap = await generateMaintenanceRoadmap({
          vehicleType: vehicleData.vehicleType,
          detectedIssues: issues || 'No major issues detected.',
        });
        setRoadmapResult(roadmap);
        setProcessingSteps(prev => prev.map(s => s.name === 'Maintenance Planner Agent' ? {...s, status: 'complete'} : s));
        toast({ title: 'Roadmap Generated', description: 'Your maintenance plan is ready.' });

        setProcessingState('success');
      } catch (err: any) {
        console.error("Analysis failed:", err);
        setError(err.message || "An unknown error occurred during analysis.");
        setProcessingState('error');
      }
    };

    runAnalysis();
  }, [vehicleData, toast, processingState]);
  
  if (!vehicleData) {
     return (
        <div className="container mx-auto px-4 md:px-6 py-8 md:py-12 text-center">
            <Alert variant="destructive" className="max-w-lg mx-auto">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Missing Information</AlertTitle>
                <AlertDescription>
                    Vehicle data is missing. Please start the analysis process from the beginning.
                </AlertDescription>
            </Alert>
            <Button asChild className="mt-6">
                <Link href="/analyze">Start Over</Link>
            </Button>
        </div>
    );
  }

  if (processingState === 'processing') {
    return <ProcessingView steps={processingSteps} />;
  }

  if (processingState === 'error') {
    return (
        <div className="container mx-auto px-4 md:px-6 py-8 md:py-12 text-center">
            <Alert variant="destructive" className="max-w-lg mx-auto">
                <AlertTriangle className="h-4 w-4" />
                <AlertTitle>Analysis Failed</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
            </Alert>
             <Button asChild className="mt-6">
                <Link href="/analyze">Try Again</Link>
            </Button>
        </div>
    );
  }

  if (processingState === 'success' && analysisResult && roadmapResult && partsResult && vehicleData) {
    return (
      <ResultsDashboard
        analysis={analysisResult}
        roadmap={roadmapResult}
        parts={partsResult}
        vehicle={vehicleData}
      />
    );
  }

  return null;
}
