
'use client';

import { Suspense, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import { ArrowLeft, Upload, Loader, Bot, Car, Bike, ShieldCheck } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';
import { Header } from '@/components/layout/header';
import { Button } from '@/components/ui/button';
import { FileUploader, FileWithPreview } from '@/components/file-uploader';
import { IdentifyPartOutput } from '@/ai/flows/identify-part-flow';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { useToast } from '@/hooks/use-toast';
import Image from 'next/image';
import { cn } from '@/lib/utils';

async function identifyPart({ file, vehicleType }: { file: File, vehicleType: 'car' | 'bike' }): Promise<IdentifyPartOutput> {
  const formData = new FormData();
  formData.append("image", file);

  const res = await fetch(
    `http://localhost:8000/parts/identify-part?vehicle_type=${vehicleType}`,
    {
      method: "POST",
      body: formData
    }
  );

  if (!res.ok) {
    const err = await res.text();
    throw new Error(err);
  }

  return await res.json();
}


function PartIdentificationContent() {
  const searchParams = useSearchParams();
  const { toast } = useToast();

  const [file, setFile] = useState<FileWithPreview | null>(null);
  const [isIdentifying, setIsIdentifying] = useState(false);
  const [result, setResult] = useState<IdentifyPartOutput | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selectedVehicleType, setSelectedVehicleType] = useState<'car' | 'bike' | null>(null);

  const queryParams = new URLSearchParams(searchParams.toString());
  const backLink = `/operational-category?${queryParams.toString()}`;

  const handleIdentify = async () => {
    if (!file) {
      toast({
        variant: 'destructive',
        title: 'Missing Image',
        description: 'Please upload an image of the part.',
      });
      return;
    }
    if (!selectedVehicleType) {
        toast({
            variant: 'destructive',
            title: 'Missing Vehicle Type',
            description: 'Please select a vehicle type first.',
        });
        return;
    }

    setIsIdentifying(true);
    setError(null);
    setResult(null);

    try {
      const identificationResult = await identifyPart({ file: file, vehicleType: selectedVehicleType });
      setResult(identificationResult);
    } catch (err: any) {
      console.error("Part identification failed:", err);
      setError(err.message || "An unknown error occurred during identification.");
      toast({
        variant: 'destructive',
        title: 'Identification Failed',
        description: err.message || 'An unknown error occurred during identification.'
      })
    } finally {
      setIsIdentifying(false);
    }
  };
  
  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 md:px-6 py-12 md:py-16">
        <div className="max-w-2xl w-full mx-auto">
          <Button variant="ghost" asChild className="mb-8">
            <Link href={backLink}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Category
            </Link>
          </Button>

          <Card className="bg-card/40 border-border">
            <CardHeader>
              <CardTitle className="text-3xl font-bold font-headline-display italic uppercase">Part Identification Unit</CardTitle>
              <CardDescription>Upload an image of the vehicle part for AI analysis.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              {!selectedVehicleType ? (
                <div className="space-y-4">
                  <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                    <ShieldCheck className="w-5 h-5 text-primary" />
                    01. SELECT CLASSIFICATION
                  </h3>
                   <div className="grid grid-cols-2 gap-4">
                        <div 
                            onClick={() => setSelectedVehicleType('car')}
                            className="cursor-pointer rounded-lg border-2 p-6 flex flex-col items-center justify-center gap-2 transition-all border-input bg-input/50 hover:border-primary/50"
                        >
                            <Car className="w-10 h-10 text-muted-foreground" />
                            <span className="font-semibold">CAR PART</span>
                        </div>
                        <div 
                            onClick={() => setSelectedVehicleType('bike')}
                            className="cursor-pointer rounded-lg border-2 p-6 flex flex-col items-center justify-center gap-2 transition-all border-input bg-input/50 hover:border-primary/50"
                        >
                            <Bike className="w-10 h-10 text-muted-foreground" />
                            <span className="font-semibold">BIKE PART</span>
                        </div>
                    </div>
                </div>
              ) : (
                <div className="space-y-6">
                  <div>
                    <div className="flex justify-between items-center mb-2">
                        <h3 className="text-sm font-semibold uppercase tracking-wider text-muted-foreground">01. CLASSIFICATION</h3>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedVehicleType(null)}>Change</Button>
                    </div>
                    <div className={cn("cursor-pointer rounded-lg border-2 p-4 flex items-center justify-center gap-4 transition-all border-primary bg-primary/10")}>
                        {selectedVehicleType === 'car' ? <Car className="w-8 h-8 text-primary" /> : <Bike className="w-8 h-8 text-primary" />}
                        <span className="font-semibold text-lg uppercase">{selectedVehicleType} PART</span>
                    </div>
                  </div>

                  <div>
                     <h3 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground mb-4">
                        <Upload className="w-5 h-5 text-primary" />
                        02. UPLOAD IMAGE
                    </h3>
                    <FileUploader onFileChange={setFile} />
                  </div>
                  
                  <Button onClick={handleIdentify} disabled={isIdentifying || !file} className="w-full" size="lg">
                    {isIdentifying ? <Loader className="animate-spin mr-2" /> : <Bot className="mr-2" />}
                    Identify Part
                  </Button>
                </div>
              )}

              {isIdentifying && (
                 <div className="text-center p-8 space-y-4">
                    <Loader className="w-12 h-12 text-primary animate-spin mx-auto"/>
                    <p className="text-muted-foreground">Analyzing image... This may take a moment.</p>
                 </div>
              )}

              {error && (
                <Alert variant="destructive">
                  <AlertTitle>Identification Failed</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {result && (
                <Card className="mt-6 bg-input">
                    <CardHeader>
                        <CardTitle className="text-xl">Identification Complete</CardTitle>
                        <CardDescription>
                            Confidence Score: <span className="font-bold text-primary">{ (result.confidence * 100).toFixed(2) }%</span>
                        </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                            <div><strong className="block text-muted-foreground">Identified Part:</strong> <span className="capitalize font-bold text-lg">{result.part.replace(/_/g, ' ')}</span></div>
                            <div><strong className="block text-muted-foreground">Vehicle System:</strong> <span className="capitalize">{result.system}</span></div>
                        </div>
                        <div>
                            <strong className="block text-muted-foreground text-sm">Purpose:</strong>
                            <p>{result.purpose}</p>
                        </div>
                        {file?.preview && (
                            <div className="flex justify-center">
                                <Image src={file.preview} alt="Identified part" width={200} height={200} className="rounded-lg border-2 border-border"/>
                            </div>
                        )}
                        <div>
                            <strong className="block text-muted-foreground text-sm mb-2">Possible Alternatives:</strong>
                            <div className="space-y-2">
                            {result.alternatives.map((alt, index) => (
                                <div key={index} className="flex justify-between items-center text-sm p-2 bg-background/50 rounded-md">
                                    <span className="capitalize">{alt.part.replace(/_/g, ' ')}</span>
                                    <span className="text-xs text-muted-foreground">Confidence: {(alt.confidence * 100).toFixed(1)}%</span>
                                </div>
                            ))}
                            </div>
                        </div>
                    </CardContent>
                </Card>
              )}

            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}


export default function PartIdentificationPage() {
    return (
        <Suspense fallback={<div className="text-center p-8">Loading...</div>}>
            <PartIdentificationContent />
        </Suspense>
    )
}
