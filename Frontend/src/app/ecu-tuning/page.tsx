
'use client';

import { Suspense, useState, useEffect, useCallback } from 'react';
import { useSearchParams } from 'next/navigation';
import { Header } from '@/components/layout/header';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { AlertTriangle, ArrowLeft, ArrowRight, Bot, Cpu, Fuel, Gauge, Info, Loader2, Zap } from 'lucide-react';
import Link from 'next/link';
import { useToast } from '@/hooks/use-toast';
import { cn } from '@/lib/utils';
import { Skeleton } from '@/components/ui/skeleton';

// --- TYPE DEFINITIONS ---
type EcuSchema = {
  tunable_params: string[];
  bounds: { [key: string]: [number, number] };
  step_sizes: { [key: string]: number };
};

type EcuConfig = { [key: string]: number };

type RecommendationResponse = {
  current_score: number;
  best_score: number;
  new_config: Record<string, number>;
  deltas: Record<string, number>;
  rationale: string;
  direction: 'maximize' | 'minimize';
};


// --- HELPER FUNCTIONS & CONSTANTS ---
const PARAM_LABELS: { [key: string]: string } = {
  fuel_air_ratio: 'Fuel/Air Ratio',
  ignition_timing_advance: 'Ignition Timing Advance (°)',
  throttle_response_rate: 'Throttle Response Rate',
  boost_pressure: 'Boost Pressure (psi)',
};

const getParamLabel = (param: string) => PARAM_LABELS[param] || param.replace(/_/g, ' ').toUpperCase();

// --- MAIN COMPONENT ---
function EcuTuningContent() {
  const searchParams = useSearchParams();
  const { toast } = useToast();

  const [schema, setSchema] = useState<EcuSchema | null>(null);
  const [isLoadingSchema, setIsLoadingSchema] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [optimizationGoal, setOptimizationGoal] = useState<'maximize' | 'minimize'>('maximize');
  const [vehicleState, setVehicleState] = useState<Partial<EcuConfig>>({});
  const [ecuParams, setEcuParams] = useState<EcuConfig>({});

  const [isOptimizing, setIsOptimizing] = useState(false);
  const [results, setResults] = useState<RecommendationResponse | null>(null);

  // --- DATA FETCHING ---
  const fetchSchema = useCallback(async () => {
    setIsLoadingSchema(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/ecu/schema');
      if (!response.ok) throw new Error(`API Error: ${response.statusText}`);
      const data: EcuSchema = await response.json();
      setSchema(data);

      // Initialize ECU params with default (median) values
      const initialParams: EcuConfig = {};
      data.tunable_params.forEach(param => {
        const [min, max] = data.bounds[param];
        initialParams[param] = (min + max) / 2;
      });
      setEcuParams(initialParams);

    } catch (e: any) {
      setError('Failed to load ECU schema. The tuning module is unavailable.');
      console.error(e);
    } finally {
      setIsLoadingSchema(false);
    }
  }, []);

  useEffect(() => {
    fetchSchema();
  }, [fetchSchema]);

  const backLink = `/mission-selection?${searchParams.toString()}`;

  // --- HANDLERS ---
  const handleVehicleStateChange = (key: string, value: string) => {
    const numValue = parseFloat(value);
    setVehicleState(prev => ({ ...prev, [key]: isNaN(numValue) ? undefined : numValue }));
  };

  const handleEcuParamChange = (key: string, value: number[]) => {
    setEcuParams(prev => ({ ...prev, [key]: value[0] }));
  };

  const handleOptimize = async () => {
    if (!schema) return;
    setIsOptimizing(true);
    setError(null);
    setResults(null);

    const config: EcuConfig = {};
    Object.assign(config, vehicleState, ecuParams);
    // Remove undefined values
    Object.keys(config).forEach(key => (config[key] === undefined) && delete config[key]);

    try {
      const response = await fetch('http://localhost:8000/ecu/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config, goal: optimizationGoal }),
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Optimization request failed.');
      }
      const data: RecommendationResponse = await response.json();
      setResults(data);
      toast({
        title: 'Optimization Complete!',
        description: 'AI has generated a new ECU profile for your vehicle.',
      });
    } catch (e: any) {
      setError(e.message);
      toast({
        variant: 'destructive',
        title: 'Optimization Failed',
        description: e.message,
      });
    } finally {
      setIsOptimizing(false);
    }
  };

  // --- RENDER LOGIC ---
  if (isLoadingSchema) {
    return <SkeletonScreen />;
  }

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 md:px-6 py-12">
        <div className="max-w-4xl mx-auto">
          <Button variant="ghost" asChild className="mb-8">
            <Link href={backLink}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Mission Selection
            </Link>
          </Button>

          <div className="text-center mb-12">
            <h1 className="text-5xl font-bold font-headline-display italic uppercase tracking-wider">ECU Tuning & Optimization</h1>
            <p className="text-muted-foreground mt-2 uppercase text-sm tracking-widest">Interface with the vehicle's core logic unit</p>
          </div>
          
          {error && !schema && (
            <Alert variant="destructive">
              <AlertTriangle className="h-4 w-4" />
              <AlertTitle>Module Offline</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {schema && (
            <div className="space-y-12">
              {/* Section 1: Goal */}
              <Card className="bg-card/40 border-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Zap /> Optimization Goal</CardTitle>
                  <CardDescription>Define the primary objective for the AI tuning process.</CardDescription>
                </CardHeader>
                <CardContent>
                  <Select value={optimizationGoal} onValueChange={(v: 'maximize' | 'minimize') => setOptimizationGoal(v)}>
                    <SelectTrigger>
                      <SelectValue placeholder="Select a goal..." />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="maximize"><div className="flex items-center gap-2"><Gauge /> Maximize Performance</div></SelectItem>
                      <SelectItem value="minimize"><div className="flex items-center gap-2"><Fuel /> Maximize Fuel Efficiency</div></SelectItem>
                    </SelectContent>
                  </Select>
                </CardContent>
              </Card>

              {/* Section 2 & 3: Parameters */}
              <Card className="bg-card/40 border-border">
                <CardHeader>
                  <CardTitle className="flex items-center gap-2"><Cpu /> Parameter Configuration</CardTitle>
                  <CardDescription>Adjust vehicle parameters. All fields are optional and will be estimated by the AI if left blank.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                    <div>
                        <Label className='text-sm text-muted-foreground font-semibold'>VEHICLE STATE (OPTIONAL)</Label>
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-2">
                            <div className='space-y-2'><Label htmlFor='engine_rpm'>Engine RPM</Label><Input id='engine_rpm' type="number" placeholder="e.g., 2500" onChange={e => handleVehicleStateChange('engine_rpm', e.target.value)} /></div>
                            <div className='space-y-2'><Label htmlFor='engine_load'>Engine Load (%)</Label><Input id='engine_load' type="number" placeholder="e.g., 50" onChange={e => handleVehicleStateChange('engine_load', e.target.value)} /></div>
                            <div className='space-y-2'><Label htmlFor='intake_air_temperature'>Intake Temp (°C)</Label><Input id='intake_air_temperature' type="number" placeholder="e.g., 25" onChange={e => handleVehicleStateChange('intake_air_temperature', e.target.value)} /></div>
                        </div>
                    </div>
                    <div>
                        <Label className='text-sm text-muted-foreground font-semibold'>ECU PARAMETERS</Label>
                        <div className="space-y-6 mt-2">
                        {schema.tunable_params.map(param => (
                            <div key={param} className="space-y-2">
                                <div className="flex justify-between items-center">
                                    <Label htmlFor={param}>{getParamLabel(param)}</Label>
                                    <span className="text-sm font-mono px-2 py-1 rounded bg-input">{ecuParams[param]?.toFixed(2)}</span>
                                </div>
                                <Slider
                                    id={param}
                                    min={schema.bounds[param][0]}
                                    max={schema.bounds[param][1]}
                                    step={schema.step_sizes[param]}
                                    value={[ecuParams[param]]}
                                    onValueChange={(v) => handleEcuParamChange(param, v)}
                                />
                            </div>
                        ))}
                        </div>
                    </div>
                </CardContent>
              </Card>

              {/* Section 4: Optimize Button */}
              <div className="text-center">
                <Button size="lg" onClick={handleOptimize} disabled={isOptimizing}>
                  {isOptimizing ? <Loader2 className="mr-2 h-5 w-5 animate-spin" /> : <Bot className="mr-2 h-5 w-5" />}
                  Generate ECU Recommendation
                </Button>
              </div>

              {/* Section 5: Results */}
              {isOptimizing && (
                <div className="text-center p-8 space-y-4">
                    <Loader2 className="w-12 h-12 text-primary animate-spin mx-auto"/>
                    <p className="text-muted-foreground">AI is calculating optimal ECU configuration...</p>
                 </div>
              )}
              {error && (
                <Alert variant="destructive">
                  <AlertTriangle className="h-4 w-4" />
                  <AlertTitle>Optimization Error</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}
              {results && <ResultsDisplay results={results} />}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

// --- SUB-COMPONENTS ---
function SkeletonScreen() {
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 md:px-6 py-8 md:py-12 flex items-center justify-center">
        <div className="w-full max-w-4xl space-y-8">
          <Skeleton className="h-12 w-1/2 mx-auto" />
          <Skeleton className="h-8 w-3/4 mx-auto" />
          <Skeleton className="h-64 w-full" />
          <Skeleton className="h-96 w-full" />
        </div>
      </main>
    </div>
  );
}

function ResultsDisplay({ results }: { results: RecommendationResponse }) {
    const originalScore = results.current_score;
    const recommendedScore = results.best_score;
    const improvement = ((recommendedScore - originalScore) / Math.abs(originalScore)) * 100;
    const isImprovement = improvement >= 0;

    return (
        <Card className="bg-gradient-to-br from-card to-card/50 border-primary/50">
            <CardHeader>
                <CardTitle className="text-2xl font-headline-display italic">Optimization Results</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Score Comparison */}
                <div className="flex justify-around items-center text-center p-4 rounded-lg bg-input">
                    <div>
                        <Label className="text-muted-foreground">Original Score</Label>
                        <p className="text-4xl font-bold">{originalScore.toFixed(2)}</p>
                    </div>
                    <ArrowRight className="w-8 h-8 text-primary shrink-0"/>
                    <div>
                        <Label className={cn("font-semibold", isImprovement ? "text-green-400" : "text-red-400")}>Optimized Score</Label>
                        <p className={cn("text-4xl font-bold", isImprovement ? "text-green-400" : "text-red-400")}>
                            {recommendedScore.toFixed(2)}
                        </p>
                        <p className={cn("text-sm font-semibold", isImprovement ? "text-green-500" : "text-red-500")}>
                            ({isImprovement ? '+' : ''}{improvement.toFixed(2)}%)
                        </p>
                    </div>
                </div>

                {/* Rationale */}
                <Alert>
                    <Info className="h-4 w-4" />
                    <AlertTitle>AI Rationale</AlertTitle>
                    <AlertDescription>{results.rationale}</AlertDescription>
                </Alert>

                {/* Parameter Changes */}
                <div>
                    <h4 className="font-semibold mb-2">Recommended Changes:</h4>
                    <div className="space-y-3">
                        {Object.entries(results.deltas).map(([key, delta]) => {
                            if (delta === 0) return null;
                            const recommended = results.new_config[key];
                            const original = recommended - delta;

                            return (
                                <div key={key} className="p-3 rounded-md bg-input grid grid-cols-3 items-center gap-2 text-sm">
                                    <span className="font-semibold text-muted-foreground">{getParamLabel(key)}</span>
                                    <div className="flex items-center justify-center gap-2 font-mono">
                                        <span className="px-2 py-1 rounded bg-background/50">{original.toFixed(2)}</span>
                                        <ArrowRight className="w-4 h-4 text-muted-foreground"/>
                                        <span className="px-2 py-1 rounded bg-primary/20 text-primary font-bold">{recommended.toFixed(2)}</span>
                                    </div>
                                    <span className={cn("text-right font-bold", recommended > original ? "text-green-400" : "text-yellow-400")}>
                                        {recommended > original ? '↑' : '↓'} {Math.abs(delta).toFixed(2)}
                                    </span>
                                </div>
                            )
                        })}
                    </div>
                </div>
            </CardContent>
        </Card>
    )
}

// --- PAGE EXPORT ---
export default function EcuTuningPage() {
  return (
    <Suspense fallback={<SkeletonScreen />}>
      <EcuTuningContent />
    </Suspense>
  );
}

    