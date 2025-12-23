
"use client";

import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { z } from "zod";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Form,
  FormControl,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form";
import { Input } from "@/components/ui/input";
import { useToast } from "@/hooks/use-toast";
import { Car, Bike, ChevronRight, Hash, Database, ShieldCheck, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useAuth } from "@/hooks/use-auth";

const formSchema = z.object({
  vehicleType: z.enum(["car", "bike"], {
    required_error: "You need to select a vehicle type.",
  }),
  brand: z.string().min(2, { message: "Brand must be at least 2 characters." }),
  model: z.string().min(1, { message: "Model is required." }),
  year: z.coerce.number().min(1900, { message: "Year must be after 1900." }).max(new Date().getFullYear() + 1, { message: `Year cannot be in the future.`}),
  mileage: z.coerce.number().min(0, { message: "Mileage must be a positive number." }),
});

export function VehicleForm() {
  const router = useRouter();
  const { toast } = useToast();
  const { token } = useAuth();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const form = useForm<z.infer<typeof formSchema>>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      vehicleType: "car",
      brand: "",
      model: "",
      year: undefined,
      mileage: undefined,
    },
  });

  async function onSubmit(values: z.infer<typeof formSchema>) {
    setIsSubmitting(true);
    if (!token) {
        toast({
            variant: "destructive",
            title: "Authentication Error",
            description: "You must be logged in to add a vehicle.",
        });
        setIsSubmitting(false);
        return;
    }

    try {
      const response = await fetch('/api/vehicles', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
              'Authorization': `Bearer ${token}`
          },
          body: JSON.stringify({
              type: values.vehicleType,
              brand: values.brand,
              model: values.model,
              year: values.year,
              mileage: values.mileage,
          })
      });

      if (!response.ok) {
        let errorMessage = `Submission failed: ${response.statusText}`;
        try {
            const errorData = await response.json();
            errorMessage = errorData.message || errorMessage;
        } catch (e) {
            // Not a JSON response
        }
        throw new Error(errorMessage);
      }

      toast({
          title: "Vehicle Added",
          description: "Your vehicle has been successfully registered.",
      });

      const queryParams = new URLSearchParams({
        vehicleType: values.vehicleType,
        brand: values.brand,
        model: values.model,
        year: values.year.toString(),
        mileage: values.mileage.toString(),
      });
      
      router.push(`/mission-selection?${queryParams.toString()}`);

    } catch (error: any) {
      console.error("Form submission error:", error);
      toast({
        variant: "destructive",
        title: "Submission Failed",
        description: error.message || "Could not save your vehicle. Please try again.",
      });
    } finally {
        setIsSubmitting(false);
    }
  }

  return (
    <div className="bg-card/40 p-8 rounded-2xl border border-border">
      <div className="flex items-center justify-between mb-2">
        <div>
          <h1 className="text-4xl font-bold font-headline-display italic uppercase text-foreground">Machine Registration</h1>
          <p className="text-muted-foreground">AWAITING TELEMETRY DATA INPUT</p>
        </div>
        <Hash className="w-10 h-10 text-primary" />
      </div>

      <Form {...form}>
        <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-10">
          
          <FormField
            control={form.control}
            name="vehicleType"
            render={({ field }) => (
              <FormItem className="space-y-4">
                <FormLabel className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
                  <ShieldCheck className="w-5 h-5 text-primary" />
                  01. CLASSIFICATION
                </FormLabel>
                <FormControl>
                    <div className="grid grid-cols-2 gap-4">
                        <div 
                            onClick={() => field.onChange('car')}
                            className={cn(
                                "cursor-pointer rounded-lg border-2 p-6 flex flex-col items-center justify-center gap-2 transition-all",
                                field.value === 'car' ? 'border-primary bg-primary/10' : 'border-input bg-input/50 hover:border-primary/50'
                            )}
                        >
                            <Car className={cn("w-10 h-10", field.value === 'car' ? 'text-primary' : 'text-muted-foreground')} />
                            <span className="font-semibold">4-WHEEL CHASSIS</span>
                        </div>
                        <div 
                            onClick={() => field.onChange('bike')}
                            className={cn(
                                "cursor-pointer rounded-lg border-2 p-6 flex flex-col items-center justify-center gap-2 transition-all",
                                field.value === 'bike' ? 'border-primary bg-primary/10' : 'border-input bg-input/50 hover:border-primary/50'
                            )}
                        >
                            <Bike className={cn("w-10 h-10", field.value === 'bike' ? 'text-primary' : 'text-muted-foreground')} />
                            <span className="font-semibold">2-WHEEL CHASSIS</span>
                        </div>
                    </div>
                </FormControl>
                <FormMessage />
              </FormItem>
            )}
          />

          <div className="space-y-4">
            <h2 className="flex items-center gap-2 text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              <Database className="w-5 h-5 text-primary" />
              02. TELEMETRY LOG
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
              <FormField
                control={form.control}
                name="brand"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs uppercase tracking-wider text-muted-foreground">Manufacturer</FormLabel>
                    <FormControl>
                      <Input placeholder="BRD-700" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="model"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs uppercase tracking-wider text-muted-foreground">Model Series</FormLabel>
                    <FormControl>
                      <Input placeholder="GT-X" {...field} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
               <FormField
                control={form.control}
                name="year"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs uppercase tracking-wider text-muted-foreground">Production Year</FormLabel>
                    <FormControl>
                      <Input type="number" placeholder="YYYY" {...field} value={field.value ?? ''} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
              <FormField
                control={form.control}
                name="mileage"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel className="text-xs uppercase tracking-wider text-muted-foreground">Mileage Log (KM)</FormLabel>
                    <FormControl>
                      <Input type="number" placeholder="TOTAL KM" {...field} value={field.value ?? ''} />
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>
          </div>

          <Button type="submit" size="lg" className="w-full text-lg !mt-12 font-bold" disabled={isSubmitting}>
              {isSubmitting && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              SUBMIT <ChevronRight className="ml-2"/>
          </Button>
        </form>
      </Form>
    </div>
  );
}
