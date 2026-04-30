
import React, { useState, useEffect } from "react";
import { Loader2, CheckCircle2, Circle, Activity, Cpu, Scan, ShieldCheck, Binary, RefreshCw } from "lucide-react";
import GlassCard from "./GlassCard";

interface Step {
  label: string;
  icon: React.ReactNode;
}

interface ProcessingOverlayProps {
  isOpen: boolean;
  type: "diagnosis" | "segmentation";
}

const DIAGNOSIS_STEPS: Step[] = [
  { label: "Resizing Image", icon: <Scan className="w-5 h-5" /> },
  { label: "Input to Diagnosis mode", icon: <Cpu className="w-5 h-5" /> },
  { label: "Region identified", icon: <Activity className="w-5 h-5" /> },
  { label: "Status determined", icon: <ShieldCheck className="w-5 h-5" /> },
  { label: "Stage diagnosed", icon: <Binary className="w-5 h-5" /> },
];

const SEGMENTATION_STEPS: Step[] = [
  { label: "Image resizing", icon: <Scan className="w-5 h-5" /> },
  { label: "Input to model", icon: <Cpu className="w-5 h-5" /> },
  { label: "Reconstructing image and then segmenting cells", icon: <Activity className="w-5 h-5" /> },
  { label: "Lastly cleaning the mask", icon: <Binary className="w-5 h-5" /> },
];

const ProcessingOverlay: React.FC<ProcessingOverlayProps> = ({ isOpen, type }) => {
  const [currentStep, setCurrentStep] = useState(0);
  const steps = type === "diagnosis" ? DIAGNOSIS_STEPS : SEGMENTATION_STEPS;

  useEffect(() => {
    if (!isOpen) {
      setCurrentStep(0);
      return;
    }

    const interval = setInterval(() => {
      setCurrentStep((prev) => {
        if (prev < steps.length - 1) return prev + 1;
        return prev;
      });
    }, 2000); // Change step every 2 seconds

    return () => clearInterval(interval);
  }, [isOpen, steps.length]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-md animate-in fade-in duration-300">
      <GlassCard variant="bordered" className="w-full max-w-md p-8 flex flex-col items-center gap-8 shadow-2xl border-primary/20 bg-card/50">
        {/* Main Loader Icon */}
        <div className="relative">
          <div className="absolute inset-0 rounded-full bg-primary/20 animate-ping" />
          <div className="relative w-20 h-20 rounded-full bg-primary/10 flex items-center justify-center border border-primary/30">
            <Loader2 className="w-10 h-10 text-primary animate-spin" />
          </div>
        </div>

        <div className="text-center">
          <h2 className="text-2xl font-bold text-foreground mb-2">
            {type === "diagnosis" ? "Running Diagnosis" : "Generating Segmentation"}
          </h2>
          <p className="text-muted-foreground text-sm">
            Please wait while our AI models process your sample...
          </p>
        </div>

        {/* Steps List */}
        <div className="w-full flex flex-col gap-4">
          {steps.map((step, index) => {
            const isCompleted = index < currentStep;
            const isCurrent = index === currentStep;
            
            return (
              <div 
                key={index}
                className={`flex items-center gap-4 p-3 rounded-lg transition-all duration-500 ${
                  isCurrent ? "bg-primary/10 border border-primary/20 scale-105" : "opacity-50"
                } ${isCompleted ? "opacity-80" : ""}`}
              >
                <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                  isCompleted ? "bg-success/20 text-success" : 
                  isCurrent ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
                }`}>
                  {isCompleted ? <CheckCircle2 className="w-5 h-5" /> : step.icon}
                </div>
                
                <div className="flex-1 text-left">
                  <p className={`font-medium ${isCurrent ? "text-primary" : "text-foreground"}`}>
                    {step.label}
                  </p>
                </div>

                {isCurrent && <Loader2 className="w-4 h-4 text-primary animate-spin" />}
              </div>
            );
          })}
        </div>
      </GlassCard>
    </div>
  );
};

export default ProcessingOverlay;
