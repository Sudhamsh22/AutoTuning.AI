'use server';
import { ai } from '@/ai/genkit';
import { z } from 'zod';

export const identifyPart = ai.defineTool(
  {
    name: 'identifyPart',
    description: 'Identifies a vehicle part from an image.',
    inputSchema: z.object({
      vehicle_type: z.string().describe('car or bike'),
      image: z
        .string()
        .describe(
          "A photo of a vehicle part as a data URI. Format: data:<mimetype>;base64,<encoded_data>"
        ),
    }),
    outputSchema: z.object({
      vehicle_type: z.string(),
      system: z.string(),
      part: z.string(),
      confidence: z.number(),
      purpose: z.string(),
      alternatives: z.array(
        z.object({
          part: z.string(),
          confidence: z.number(),
        })
      ),
      method: z.string(),
    }),
  },
  async (input) => {
    console.log(`Identifying part for ${input.vehicle_type}...`);

    const matches = input.image.match(/^data:(.+);base64,(.+)$/);
    if (!matches) {
      throw new Error('Invalid image data URI');
    }

    const contentType = matches[1];
    const buffer = Buffer.from(matches[2], 'base64');

    const formData = new FormData();
    formData.append('image', new Blob([buffer], { type: contentType }), 'part.jpg');
    formData.append('vehicle_type', input.vehicle_type);

    const response = await fetch('http://localhost:8000/parts/identify-part', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed with status ${response.status}: ${errorText}`);
    }

    return await response.json();
  }
);

export const findParts = ai.defineTool(
  {
    name: 'findParts',
    description: 'Finds parts for a vehicle.',
    inputSchema: z.object({
      vehicleBrand: z.string().describe('The brand of the vehicle.'),
      vehicleModel: z.string().describe('The model of the vehicle.'),
      vehicleYear: z.string().describe('The year of the vehicle.'),
      parts: z.array(z.string()).describe('The parts to search for.'),
    }),
    outputSchema: z.object({
      parts: z.array(
        z.object({
          name: z.string().describe('The name of the part.'),
          platform: z.string().describe('The platform where the part can be found.'),
          price: z.number().describe('The price of the part.'),
          rating: z.number().describe('The rating of the part.'),
        })
      ),
    }),
  },
  async (input) => {
    console.log(
      `Finding parts for ${input.vehicleBrand} ${input.vehicleModel} ${input.vehicleYear}...`
    );

    const response = await fetch('http://localhost:8000/parts/find-parts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      throw new Error('Failed to fetch parts');
    }

    return await response.json();
  }
);
