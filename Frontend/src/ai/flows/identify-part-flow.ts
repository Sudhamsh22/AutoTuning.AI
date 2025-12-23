
'use server';
/**
 * @fileOverview An AI agent for identifying vehicle parts from images.
 *
 * - identifyPartFlow - A function that handles the part identification process.
 * - IdentifyPartInput - The input type for the identifyPartFlow function.
 * - IdentifyPartOutput - The return type for the identifyPartFlow function.
 */

import {ai} from '@/ai/genkit';
import {z} from 'genkit';

const IdentifyPartInputSchema = z.object({
  vehicle_type: z.string().describe('The type of vehicle (e.g., car, bike).'),
  image: z.string().describe("A photo of a vehicle part, as a data URI that must include a MIME type and use Base64 encoding. Expected format: 'data:<mimetype>;base64,<encoded_data>'."),
});
export type IdentifyPartInput = z.infer<typeof IdentifyPartInputSchema>;

const IdentifyPartOutputSchema = z.object({
  vehicle_type: z.string(),
  system: z.string(),
  part: z.string(),
  confidence: z.number(),
  purpose: z.string(),
  alternatives: z.array(z.object({
      part: z.string(),
      confidence: z.number(),
  })),
  method: z.string(),
});
export type IdentifyPartOutput = z.infer<typeof IdentifyPartOutputSchema>;


export async function identifyPartFlow(input: IdentifyPartInput): Promise<IdentifyPartOutput> {
  // This flow is no longer used as the client calls the API directly.
  // The file is kept for type definitions.
  throw new Error("identifyPartFlow is deprecated. The client should call the backend API directly.");
}
