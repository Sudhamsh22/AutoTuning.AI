'use server';
/**
 * @fileOverview A part discovery AI agent.
 *
 * - discoverParts - A function that handles the part discovery process.
 * - DiscoverPartsInput - The input type for the discoverParts function.
 * - DiscoverPartsOutput - The return type for the discoverParts function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';
import { findParts } from '../tools/parts';

export const PartSchema = z.object({
  name: z.string().describe('The name of the part.'),
  platform: z.string().describe('The platform where the part can be found.'),
  price: z.number().describe('The price of the part.'),
  rating: z.number().describe('The rating of the part.'),
});

const DiscoverPartsInputSchema = z.object({
  vehicleBrand: z.string().describe('The brand of the vehicle.'),
  vehicleModel: z.string().describe('The model of the vehicle.'),
  vehicleYear: z.string().describe('The year of the vehicle.'),
  parts: z.array(z.string()).describe('The parts to search for.'),
});
export type DiscoverPartsInput = z.infer<typeof DiscoverPartsInputSchema>;

const DiscoverPartsOutputSchema = z.object({
  parts: z.array(PartSchema),
});
export type DiscoverPartsOutput = z.infer<typeof DiscoverPartsOutputSchema>;

export async function discoverParts(input: DiscoverPartsInput): Promise<DiscoverPartsOutput> {
  return discoverPartsFlow(input);
}

const prompt = ai.definePrompt({
  name: 'discoverPartsPrompt',
  input: {schema: DiscoverPartsInputSchema},
  output: {schema: DiscoverPartsOutputSchema},
  tools: [findParts],
  prompt: `You are an expert mechanic specializing in finding parts for vehicles.

  Based on the identified vehicle brand, model, year, and parts, find the best parts for the job.

  Vehicle Brand: {{{vehicleBrand}}}
  Vehicle Model: {{{vehicleModel}}}
  Vehicle Year: {{{vehicleYear}}}
  Parts: {{{parts}}}

  Parts:`,
});

const discoverPartsFlow = ai.defineFlow(
  {
    name: 'discoverPartsFlow',
    inputSchema: DiscoverPartsInputSchema,
    outputSchema: DiscoverPartsOutputSchema,
  },
  async input => {
    const {output} = await prompt(input);
    return output!;
  }
);
