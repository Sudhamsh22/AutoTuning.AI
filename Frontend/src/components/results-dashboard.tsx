"use client";

import type { AnalyzeVehicleImagesOutput } from "@/ai/flows/analyze-vehicle-images";
import type { GenerateMaintenanceRoadmapOutput } from "@/ai/flows/generate-maintenance-roadmap";
import type { DiscoverPartsOutput } from "@/ai/flows/discover-parts";
import type { VehicleData } from "./results-view";
import Image from "next/image";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "./ui/button";
import { Download, Share2, RefreshCw, CheckCircle, AlertTriangle, Star, ExternalLink, Wrench } from "lucide-react";
import { PlaceHolderImages } from "@/lib/placeholder-images";
import { Separator } from "./ui/separator";

interface ResultsDashboardProps {
  analysis: AnalyzeVehicleImagesOutput;
  roadmap: GenerateMaintenanceRoadmapOutput;
  parts: DiscoverPartsOutput;
  vehicle: VehicleData;
}

const tutorials = [
    { id: 'tutorial-brake-pads', title: "How to Change Brake Pads", channel: "DIY Garage" },
    { id: 'tutorial-oil-change', title: "Easy Oil Change at Home", channel: "MechanicTips" },
    { id: 'tutorial-tire-rotation', title: "Tire Rotation Guide", channel: "AutoEd" },
];

export function ResultsDashboard({ analysis, roadmap, parts, vehicle }: ResultsDashboardProps) {
  const healthScore = 100 - analysis.detectedIssues.length * 15;

  return (
    <div className="container mx-auto px-4 md:px-6 py-8 md:py-12">
      <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
        <div>
          <h1 className="text-3xl font-bold font-headline capitalize">{`${vehicle.brand} ${vehicle.model} Analysis`}</h1>
          <p className="text-muted-foreground">{`${vehicle.year} | ${vehicle.mileage} miles`}</p>
        </div>
        <div className="flex gap-2">
            <Button variant="outline"><Download className="mr-2" /> Download PDF</Button>
            <Button variant="outline"><Share2 className="mr-2" /> Share Report</Button>
            <Button variant="secondary"><RefreshCw className="mr-2" /> Re-analyze</Button>
        </div>
      </div>
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-3 space-y-6">
            {/* Health Summary & Detected Issues in one row */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <Card className="md:col-span-1">
                    <CardHeader>
                        <CardTitle>Vehicle Health Summary</CardTitle>
                    </CardHeader>
                    <CardContent className="text-center">
                        <div className={`relative w-40 h-40 mx-auto flex items-center justify-center text-5xl font-bold ${healthScore > 70 ? 'text-green-400' : healthScore > 40 ? 'text-yellow-400' : 'text-red-400'}`}>
                            <svg className="absolute inset-0" viewBox="0 0 36 36">
                                <path className="text-muted/20" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="2" />
                                <path className="transition-all duration-500" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="currentColor" strokeWidth="2" strokeDasharray={`${healthScore}, 100`} />
                            </svg>
                            {healthScore}
                        </div>
                        <p className="text-muted-foreground mt-4">Based on provided details.</p>
                    </CardContent>
                </Card>
                <Card className="md:col-span-2">
                    <CardHeader>
                        <CardTitle>Detected Issues</CardTitle>
                        <CardDescription>AI analysis highlights potential problems.</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <ul className="space-y-2">
                            {analysis.detectedIssues.length > 0 ? (
                                analysis.detectedIssues.map((issue, i) => (
                                    <li key={i} className="flex items-start gap-2">
                                        <AlertTriangle className="w-5 h-5 text-yellow-400 flex-shrink-0 mt-1" /> 
                                        <span>{issue}</span>
                                    </li>
                                ))
                            ) : (
                                <li className="flex items-center gap-2">
                                    <CheckCircle className="w-5 h-5 text-green-500" />
                                    <span>No significant issues detected from the provided information.</span>
                                </li>
                            )}
                        </ul>
                    </CardContent>
                </Card>
            </div>

            {/* Maintenance Roadmap */}
            <Card>
                <CardHeader>
                    <CardTitle>Maintenance & Upgrade Roadmap</CardTitle>
                    <CardDescription>A step-by-step plan generated by our AI planner.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap font-code bg-muted/50 p-4 rounded-md">{roadmap.roadmap}</div>
                </CardContent>
            </Card>

             {/* Spare Parts & Tutorials */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <Card>
                    <CardHeader>
                        <CardTitle>Recommended Spare Parts</CardTitle>
                        <CardDescription>Top-rated parts for your fixes and upgrades.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {parts.parts.map(part => (
                            <div key={part.name} className="flex items-center gap-4 p-2 rounded-md hover:bg-muted/50">
                                <div className="p-2 bg-muted rounded-md"><Wrench className="text-primary" /></div>
                                <div className="flex-1">
                                    <p className="font-semibold">{part.name}</p>
                                    <p className="text-sm text-muted-foreground">{part.platform} - <span className="font-bold text-foreground">${part.price}</span></p>
                                </div>
                                <div className="flex items-center gap-1 text-yellow-400">
                                    <Star className="w-4 h-4 fill-current"/>
                                    <span className="text-sm font-bold text-foreground">{part.rating}</span>
                                </div>
                            </div>
                        ))}
                    </CardContent>
                </Card>
                <Card>
                    <CardHeader>
                        <CardTitle>Video Tutorials</CardTitle>
                        <CardDescription>Helpful guides for your DIY projects.</CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                        {tutorials.map(tut => {
                            const image = PlaceHolderImages.find(p => p.id === tut.id);
                            return image && (
                                <div key={tut.id} className="flex items-center gap-4">
                                    <Image src={image.imageUrl} alt={tut.title} width={120} height={68} className="rounded-md object-cover" data-ai-hint={image.imageHint}/>
                                    <div className="flex-1">
                                        <p className="font-semibold leading-tight">{tut.title}</p>
                                        <p className="text-sm text-muted-foreground">{tut.channel}</p>
                                    </div>
                                    <Button variant="ghost" size="icon"><ExternalLink className="w-4 h-4" /></Button>
                                </div>
                            )
                        })}
                    </CardContent>
                </Card>
            </div>
        </div>
      </div>
    </div>
  );
}

    