'use server';

/**
 * @fileOverview A maintenance roadmap generator AI agent.
 *
 * - generateMaintenanceRoadmap - A function that handles the generation of a maintenance roadmap.
 * - GenerateMaintenanceRoadmapInput - The input type for the generateMaintenanceRoadmap function.
 * - GenerateMaintenanceRoadmapOutput - The return type for the generateMaintenanceRoadmap function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const GenerateMaintenanceRoadmapInputSchema = z.object({
  vehicleType: z.string().describe('The type of vehicle (e.g., car, bike).'),
  detectedIssues: z.string().describe('A description of the detected issues.'),
});
export type GenerateMaintenanceRoadmapInput = z.infer<typeof GenerateMaintenanceRoadmapInputSchema>;

const GenerateMaintenanceRoadmapOutputSchema = z.object({
  roadmap: z.string().describe('A step-by-step maintenance or upgrade roadmap.'),
});
export type GenerateMaintenanceRoadmapOutput = z.infer<typeof GenerateMaintenanceRoadmapOutputSchema>;

export async function generateMaintenanceRoadmap(input: GenerateMaintenanceRoadmapInput): Promise<GenerateMaintenanceRoadmapOutput> {
  return generateMaintenanceRoadmapFlow(input);
}

const prompt = ai.definePrompt({
  name: 'generateMaintenanceRoadmapPrompt',
  input: {schema: GenerateMaintenanceRoadmapInputSchema},
  output: {schema: GenerateMaintenanceRoadmapOutputSchema},
  prompt: `You are an expert mechanic specializing in creating maintenance roadmaps for vehicles.

  Based on the identified vehicle type and detected issues, generate a clear and actionable step-by-step maintenance roadmap.

  Vehicle Type: {{{vehicleType}}}
  Detected Issues: {{{detectedIssues}}}

  Roadmap:`,
});

const generateMaintenanceRoadmapFlow = ai.defineFlow(
  {
    name: 'generateMaintenanceRoadmapFlow',
    inputSchema: GenerateMaintenanceRoadmapInputSchema,
    outputSchema: GenerateMaintenanceRoadmapOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
