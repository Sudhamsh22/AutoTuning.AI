# **App Name**: AutoMechanic.AI

## Core Features:

- Vehicle Input: Allow users to input vehicle details including type (Bike/Car), brand, model, year, fuel type, and current mileage.
- Image Upload: Enable users to upload vehicle images using a drag-and-drop interface for AI analysis.
- AI Analysis Orchestration: Orchestrate parallel agent processing using Vision Analysis, Parts Discovery, and Maintenance Planner agents.
- Issue Detection with Visual Aid: The Vision Analysis agent analyzes the provided vehicle images for issues, highlighting them visually for the user. The LLM uses its tool to highlight certain issues and not others. The text from this tool helps power the following step.
- Spare Part Suggestions: Display a comparison table with platform name, price, and rating for compatible spare parts identified by the Parts Discovery agent.
- Video Tutorials: Embed relevant YouTube tutorial thumbnails that users can view and follow to perform recommended fixes and upgrades.
- Maintenance Roadmap: Generate a step-by-step maintenance or upgrade roadmap as a timeline or checklist to guide the user through the process.

## Style Guidelines:

- Primary color: Electric blue (#7DF9FF) for a modern tech feel. The color reflects advanced AI capabilities and resonates well with the automotive theme.
- Background color: Dark gray (#28282B), close in hue to the primary color, desaturated to 20% for a sleek dark mode base.
- Accent color: Vivid orange (#FF9933), an analogous hue, to create a sharp contrast for CTAs and highlights.
- Body and headline font: 'Inter', a sans-serif, providing a modern and clean aesthetic suitable for both headlines and body text.
- Use sharp, vector-based icons for agents, vehicle parts, and action items.  Icons should use the primary and accent colors to highlight important information.
- Implement a card-based layout for presenting information. Use clear and consistent spacing throughout the interface, adopting a mobile-first responsive design.
- Incorporate subtle animations during the AI processing phase and when transitioning between different sections to enhance user experience without being distracting.