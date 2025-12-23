
'use client';

import Link from 'next/link';
import { Button } from '@/components/ui/button';
import { useAuth } from '@/hooks/use-auth';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar';
import { User as UserIcon, LogOut } from 'lucide-react';

function CarIcon(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg
      {...props}
      xmlns="http://www.w3.org/2000/svg"
      width="24"
      height="24"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M19 17h2c.6 0 1-.4 1-1v-3c0-.9-.7-1.7-1.5-1.9C18.7 10.6 16 10 16 10s-1.3-1.4-2.2-2.3c-.5-.4-1.1-.7-1.8-.7H5c-.6 0-1.1.4-1.4.9L2 12v9h2" />
      <path d="M7 19h10" />
      <path d="M5.8 19.3A2 2 0 1 0 4 17.2V12" />
      <circle cx="7" cy="17" r="2" />
      <circle cx="17" cy="17" r="2" />
    </svg>
  );
}

export function Header() {
  const { user, isUserLoading, logout } = useAuth();

  const handleLogout = () => {
    logout();
  };

  return (
    <header className="py-4 px-4 md:px-6 sticky top-0 z-50 w-full border-b border-border/40 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container mx-auto flex items-center justify-between">
        <Link href="/" className="flex items-center gap-2" prefetch={false}>
          <CarIcon className="h-6 w-6 text-primary" />
          <span className="text-xl font-display font-bold text-foreground">AUTOTUNING.AI</span>
        </Link>
        <nav className="hidden items-center gap-4 md:flex">
          <Button variant="ghost" asChild>
            <Link href="/#features" prefetch={false}>Features</Link>
          </Button>
          <Button variant="ghost" asChild>
            <Link href="/analyze" prefetch={false}>Analyze</Link>
          </Button>
           <Button variant="ghost" asChild>
            <Link href="/chart" prefetch={false}>Chart</Link>
          </Button>
        </nav>
        
        <div className="flex items-center gap-4">
            {!isUserLoading && (
                user ? (
                    <DropdownMenu>
                        <DropdownMenuTrigger asChild>
                            <Button variant="ghost" className="relative h-8 w-8 rounded-full">
                                <Avatar className="h-8 w-8">
                                    <AvatarFallback>
                                        <UserIcon />
                                    </AvatarFallback>
                                </Avatar>
                            </Button>
                        </DropdownMenuTrigger>
                        <DropdownMenuContent className="w-56" align="end" forceMount>
                            <DropdownMenuItem onClick={handleLogout}>
                                <LogOut className="mr-2 h-4 w-4" />
                                <span>Log out</span>
                            </DropdownMenuItem>
                        </DropdownMenuContent>
                    </DropdownMenu>
                ) : (
                    <div className='flex items-center gap-2'>
                        <Button variant="ghost" asChild>
                            <Link href="/login" prefetch={false}>Login</Link>
                        </Button>
                        <Button asChild>
                            <Link href="/signup" prefetch={false}>Sign Up</Link>
                        </Button>
                    </div>
                )
            )}
        </div>
      </div>
    </header>
  );
}
