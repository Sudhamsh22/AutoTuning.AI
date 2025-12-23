'use server';
import {ai} from '@/ai/genkit';
import {z} from 'zod';

export const getEcuData = ai.defineTool(
  {
    name: 'getEcuData',
    description: 'Gets ECU data for a vehicle.',
    inputSchema: z.object({
      vin: z.string().describe('The VIN of the vehicle.'),
    }),
    outputSchema: z.object({
      data: z.any().describe('The ECU data.'),
    }),
  },
  async (input) => {
    console.log(`Getting ECU data for VIN ${input.vin}...`);
    const response = await fetch(`http://localhost:8000/ecu/${input.vin}`);
    return await response.json();
  }
);
