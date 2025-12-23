
'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { ArrowLeft, Send, Bot, User as UserIcon } from 'lucide-react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import Link from 'next/link';
import { Header } from '@/components/layout/header';
import { useState, useMemo, useRef, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { cn } from '@/lib/utils';
import { getCriticalDiagnosis } from '@/ai/flows/get-critical-diagnosis';
import type { VehicleData } from '@/components/results-view';

type ChatMessage = {
  role: 'user' | 'assistant';
  content: string;
};

function CriticalDiagnosisContent() {
  const searchParams = useSearchParams();
  const [problemDescription, setProblemDescription] = useState('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isReplying, setIsReplying] = useState(false);
  const scrollAreaRef = useRef<HTMLDivElement>(null);

  const vehicleData: VehicleData | null = useMemo(() => {
    const data = {
      vehicleType: searchParams.get('vehicleType'),
      brand: searchParams.get('brand'),
      model: searchParams.get('model'),
      year: searchParams.get('year'),
      mileage: searchParams.get('mileage'),
    };
    if (Object.values(data).some(v => !v)) return null;
    return data as VehicleData;
  }, [searchParams]);

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTo({
        top: scrollAreaRef.current.scrollHeight,
        behavior: 'smooth',
      });
    }
  }, [chatHistory]);

  const queryParams = new URLSearchParams(searchParams.toString());
  const backLink = `/operational-category?${queryParams.toString()}`;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!problemDescription.trim() || !vehicleData) return;

    const newHistory: ChatMessage[] = [...chatHistory, { role: 'user', content: problemDescription }];
    setChatHistory(newHistory);
    setProblemDescription('');
    setIsReplying(true);

    try {
      const diagnosis = await getCriticalDiagnosis({
        ...vehicleData,
        problemDescription,
        chatHistory: newHistory.slice(0, -1), // Send history without current message
      });
      setChatHistory(prev => [...prev, { role: 'assistant', content: diagnosis.reply }]);
    } catch (error) {
      console.error("Diagnosis failed:", error);
      setChatHistory(prev => [...prev, { role: 'assistant', content: "I'm sorry, I encountered an error and can't provide a diagnosis right now." }]);
    } finally {
      setIsReplying(false);
    }
  };

  return (
    <div className="flex flex-col min-h-dvh bg-background">
      <Header />
      <main className="flex-1 container mx-auto px-4 md:px-6 py-12 md:py-16 flex flex-col">
        <div className="max-w-4xl w-full mx-auto">
          <Button variant="ghost" asChild className="mb-8">
            <Link href={backLink}>
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back to Category
            </Link>
          </Button>
          <Card className="bg-card/40 border-border flex-1 flex flex-col">
            <CardHeader>
              <CardTitle className="text-3xl font-bold font-headline-display italic uppercase">Critical Diagnosis Terminal</CardTitle>
              <CardDescription>Describe the issue in detail. Our AI will provide a preliminary diagnosis.</CardDescription>
            </CardHeader>
            <CardContent className="flex-1 flex flex-col gap-4">
              <ScrollArea className="h-96 w-full pr-4" ref={scrollAreaRef}>
                <div className="space-y-6">
                  {chatHistory.map((message, index) => (
                    <div key={index} className={cn("flex items-start gap-3", message.role === 'user' ? 'justify-end' : '')}>
                      {message.role === 'assistant' && (
                        <Avatar className="w-8 h-8 bg-primary/20 text-primary">
                          <AvatarFallback><Bot size={20} /></AvatarFallback>
                        </Avatar>
                      )}
                      <div className={cn("rounded-lg p-3 max-w-lg", message.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-input')}>
                        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
                      </div>
                      {message.role === 'user' && (
                         <Avatar className="w-8 h-8">
                          <AvatarFallback><UserIcon size={20} /></AvatarFallback>
                        </Avatar>
                      )}
                    </div>
                  ))}
                  {isReplying && (
                    <div className="flex items-start gap-3">
                       <Avatar className="w-8 h-8 bg-primary/20 text-primary">
                          <AvatarFallback><Bot size={20} /></AvatarFallback>
                        </Avatar>
                      <div className="rounded-lg p-3 bg-input animate-pulse">
                        <p className="text-sm text-muted-foreground">Thinking...</p>
                      </div>
                    </div>
                  )}
                </div>
              </ScrollArea>
              <form onSubmit={handleSubmit} className="flex items-start gap-3 pt-4 border-t">
                <Textarea
                  value={problemDescription}
                  onChange={(e) => setProblemDescription(e.target.value)}
                  placeholder="e.g., I hear a loud grinding noise from the front right wheel when I turn left..."
                  className="flex-1 resize-none"
                  rows={2}
                  disabled={isReplying}
                />
                <Button type="submit" size="icon" disabled={!problemDescription.trim() || isReplying}>
                  <Send className="w-5 h-5" />
                </Button>
              </form>
            </CardContent>
          </Card>
        </div>
      </main>
    </div>
  );
}

export default function CriticalDiagnosisPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <CriticalDiagnosisContent />
    </Suspense>
  )
}
