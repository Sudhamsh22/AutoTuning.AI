"use client";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Bot, Wrench, Search, CheckCircle, Loader } from "lucide-react";
import { motion } from "framer-motion";

export type ProcessingStep = {
  name: string;
  status: 'running' | 'complete';
};

interface ProcessingViewProps {
  steps: ProcessingStep[];
}

const agents = [
    { name: 'Vision Analysis Agent', icon: Bot, description: 'Inspecting vehicle images for damages and wear.' },
    { name: 'Parts Discovery Agent', icon: Search, description: 'Finding compatible parts and best prices.' },
    { name: 'Maintenance Planner Agent', icon: Wrench, description: 'Building a step-by-step maintenance roadmap.' },
];

export function ProcessingView({ steps }: ProcessingViewProps) {
  return (
    <div className="container mx-auto px-4 md:px-6 py-8 md:py-12">
        <div className="max-w-2xl mx-auto text-center space-y-4">
            <h1 className="text-3xl font-bold font-headline">AI Analysis in Progress...</h1>
            <p className="text-muted-foreground">
                Our AI agents are working on your vehicle. This may take a moment.
            </p>
            <div className="relative w-full h-2 bg-muted rounded-full overflow-hidden">
                <motion.div 
                    className="absolute top-0 left-0 h-full bg-primary"
                    initial={{ width: '0%' }}
                    animate={{ width: `${(steps.length / agents.length) * 100}%`}}
                    transition={{ duration: 0.5, ease: "easeInOut" }}
                />
            </div>
        </div>
        <div className="max-w-2xl mx-auto mt-8 space-y-4">
            {agents.map((agent, index) => {
                const step = steps.find(s => s.name === agent.name);
                const isVisible = steps.length >= index;

                if (!isVisible) return null;

                return (
                    <motion.div
                        key={agent.name}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.5, delay: index * 0.2 }}
                    >
                        <Card className="shadow-md hover:shadow-lg transition-shadow">
                            <CardHeader className="flex flex-row items-center justify-between">
                                <div className="flex items-center gap-3">
                                    <agent.icon className="w-6 h-6 text-primary"/>
                                    <CardTitle>{agent.name}</CardTitle>
                                </div>
                                {step?.status === 'running' && <Loader className="w-5 h-5 animate-spin text-muted-foreground"/>}
                                {step?.status === 'complete' && <CheckCircle className="w-5 h-5 text-green-500"/>}
                            </CardHeader>
                            <CardContent>
                                <p className="text-muted-foreground">{agent.description}</p>
                            </CardContent>
                        </Card>
                    </motion.div>
                )
            })}
        </div>
    </div>
  );
}
