
'use server';

/**
 * @fileOverview A critical diagnosis AI agent.
 *
 * - getCriticalDiagnosis - A function that handles the diagnosis process for a critical vehicle issue.
 * - GetCriticalDiagnosisInput - The input type for the getCriticalDiagnosis function.
 * - GetCriticalDiagnosisOutput - The return type for the getCriticalDiagnosis function.
 */

import { ai } from '@/ai/genkit';
import { z } from 'genkit';
import { diagnoseIssues } from '../tools/diagnostics';

const ChatMessageSchema = z.object({
    role: z.enum(['user', 'assistant']),
    content: z.string(),
});

const GetCriticalDiagnosisInputSchema = z.object({
  vehicleType: z.string().describe('The type of vehicle (e.g., car, bike).'),
  brand: z.string().describe('The brand of the vehicle.'),
  model: z.string().describe('The model of the vehicle.'),
  year: z.string().describe('The year of the vehicle.'),
  mileage: z.string().describe('The current mileage of the vehicle.'),
  problemDescription: z.string().describe('A detailed description of the problem reported by the user.'),
  chatHistory: z.array(ChatMessageSchema).describe('The history of the conversation so far.'),
});
export type GetCriticalDiagnosisInput = z.infer<typeof GetCriticalDiagnosisInputSchema>;

const GetCriticalDiagnosisOutputSchema = z.object({
  reply: z.string().describe('The AI assistant\'s diagnostic reply to the user.'),
});
export type GetCriticalDiagnosisOutput = z.infer<typeof GetCriticalDiagnosisOutputSchema>;

export async function getCriticalDiagnosis(input: GetCriticalDiagnosisInput): Promise<GetCriticalDiagnosisOutput> {
  return getCriticalDiagnosisFlow(input);
}

const getCriticalDiagnosisFlow = ai.defineFlow(
  {
    name: 'getCriticalDiagnosisFlow',
    inputSchema: GetCriticalDiagnosisInputSchema,
    outputSchema: GetCriticalDiagnosisOutputSchema,
  },
  async (input) => {
    // Call the diagnoseIssues tool which hits your API
    const diagnosisResult = await diagnoseIssues(input);
    return { reply: diagnosisResult.reply };
  }
);
