
'use server';
import { ai } from '@/ai/genkit';
import { z } from 'zod';

export const diagnoseIssues = ai.defineTool(
  {
    name: 'diagnoseIssues',
    description: 'Diagnoses issues with a vehicle based on a problem description.',
    inputSchema: z.object({
      vehicleType: z.string(),
      brand: z.string(),
      model: z.string(),
      year: z.string(),
      mileage: z.string(),
      problemDescription: z.string(),
    }),
    outputSchema: z.object({
      reply: z.string(),
    }),
  },
  async (input) => {
    const params = new URLSearchParams({
      vehicle_type: input.vehicleType.toLowerCase(),
      query: input.problemDescription,
      topk: '3',
    });

    const response = await fetch(
      `http://localhost:8000/diagnostics/probable-cause?${params.toString()}`,
      {
        method: 'GET',
        cache: 'no-store',
      }
    );

    if (!response.ok) {
      throw new Error('Diagnostics API failed');
    }

    const data = await response.json();

    if (!data.results || data.results.length === 0) {
      return {
        reply: 'No clear diagnosis could be determined from the description.',
      };
    }

    const best = data.results[0];

    const reply =
      `Possible issue detected: ${best.failure.replaceAll('_', ' ')}\n\n` +
      `Likely causes:\n` +
      best.causes.map((c: string) => `• ${c}`).join('\n') +
      `\n\nConfidence: ${(best.confidence * 100).toFixed(1)}%`;

    return { reply };
  }
);
