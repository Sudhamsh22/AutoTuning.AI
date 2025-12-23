'use server';

/**
 * @fileOverview Analyzes vehicle images to detect potential issues.
 *
 * - analyzeVehicleImages - A function that handles the vehicle image analysis process.
 * - AnalyzeVehicleImagesInput - The input type for the analyzeVehicleImages function.
 * - AnalyzeVehicleImagesOutput - The return type for the analyzeVehicleImages function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const AnalyzeVehicleImagesInputSchema = z.object({
  vehicleType: z.string(),
  brand: z.string(),
  model: z.string(),
  year: z.string(),
  mileage: z.string(),
});
export type AnalyzeVehicleImagesInput = z.infer<typeof AnalyzeVehicleImagesInputSchema>;

const AnalyzeVehicleImagesOutputSchema = z.object({
  detectedIssues: z
    .array(z.string())
    .describe('A list of detected issues based on the vehicle information.'),
});
export type AnalyzeVehicleImagesOutput = z.infer<typeof AnalyzeVehicleImagesOutputSchema>;

export async function analyzeVehicleImages(
  input: AnalyzeVehicleImagesInput
): Promise<AnalyzeVehicleImagesOutput> {
  return analyzeVehicleImagesFlow(input);
}

const analyzeVehicleImagesPrompt = ai.definePrompt({
  name: 'analyzeVehicleImagesPrompt',
  input: {schema: AnalyzeVehicleImagesInputSchema},
  output: {schema: AnalyzeVehicleImagesOutputSchema},
  prompt: `You are an AI that analyzes vehicle issues based on user-provided information.

  Analyze the following vehicle details to identify potential issues.

  Vehicle: {{{year}}} {{{brand}}} {{{model}}} ({{{vehicleType}}}) with {{{mileage}}} miles.

  List the potential detected issues based on common problems for this vehicle at this mileage.
  Make sure to set detectedIssues as an array of strings.
  `,
});

const analyzeVehicleImagesFlow = ai.defineFlow(
  {
    name: 'analyzeVehicleImagesFlow',
    inputSchema: AnalyzeVehicleImagesInputSchema,
    outputSchema: AnalyzeVehicleImagesOutputSchema,
  },
  async input => {
    const {output} = await analyzeVehicleImagesPrompt(input);
    return output!;
  }
);
