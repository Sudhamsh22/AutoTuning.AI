'use client';

import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { UploadCloud, X } from 'lucide-react';
import Image from 'next/image';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';

export interface FileWithPreview extends File {
  preview: string;
}

interface FileUploaderProps {
  onFileChange: (file: FileWithPreview | null) => void;
  className?: string;
}

export function FileUploader({ onFileChange, className }: FileUploaderProps) {
  const [file, setFile] = useState<FileWithPreview | null>(null);
  const { toast } = useToast();

  const onDrop = useCallback((acceptedFiles: File[], fileRejections: any[]) => {
    if (fileRejections.length > 0) {
      toast({
        variant: 'destructive',
        title: 'File upload error',
        description: fileRejections[0].errors[0].message,
      });
      return;
    }

    const selectedFile = acceptedFiles[0];
    if (selectedFile) {
      const fileWithPreview = Object.assign(selectedFile, {
        preview: URL.createObjectURL(selectedFile),
      });
      setFile(fileWithPreview);
      onFileChange(fileWithPreview);
    }
  }, [onFileChange, toast]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': ['.jpeg', '.png', '.jpg', '.webp'] },
    maxFiles: 1,
    maxSize: 10 * 1024 * 1024, // 10MB
  });

  const removeFile = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (file) {
      URL.revokeObjectURL(file.preview);
    }
    setFile(null);
    onFileChange(null);
  };

  if (file) {
    return (
      <div className={cn("relative p-2 border rounded-lg", className)}>
        <div className="flex items-center gap-4">
          <Image src={file.preview} alt="Preview" width={80} height={80} className="rounded-md object-cover aspect-square" />
          <div className="flex-1 space-y-1">
            <p className="text-sm font-medium truncate">{file.name}</p>
            <p className="text-xs text-muted-foreground">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        </div>
        <Button variant="ghost" size="icon" className="absolute top-2 right-2 h-7 w-7" onClick={removeFile}>
          <X className="w-4 h-4" />
          <span className="sr-only">Remove file</span>
        </Button>
      </div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={cn(
        'relative flex flex-col items-center justify-center w-full p-8 border-2 border-dashed rounded-lg cursor-pointer transition-colors',
        isDragActive ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50',
        className
      )}
    >
      <input {...getInputProps()} />
      <div className="text-center">
        <UploadCloud className="w-12 h-12 text-muted-foreground mx-auto" />
        <p className="mt-4 text-center text-muted-foreground">
          {isDragActive ? 'Drop the image here...' : "Drag 'n' drop an image, or click to select"}
        </p>
        <p className="text-xs text-muted-foreground/80">PNG, JPG, WEBP (max 10MB)</p>
      </div>
    </div>
  );
}
