import { cn } from "@/lib/utils";

interface CellularBackgroundProps {
  children: React.ReactNode;
  className?: string;
}

const CellularBackground = ({ children, className }: CellularBackgroundProps) => {
  return (
    <div className={cn("min-h-screen bg-background relative overflow-hidden", className)}>
      {/* Animated floating orbs */}
      <div 
        className="floating-orb w-72 h-72 bg-primary/30"
        style={{ top: '5%', left: '5%', animationDelay: '0s' }}
      />
      <div 
        className="floating-orb w-48 h-48 bg-success/20"
        style={{ top: '60%', right: '10%', animationDelay: '2s' }}
      />
      <div 
        className="floating-orb w-56 h-56 bg-warning/15"
        style={{ bottom: '10%', left: '20%', animationDelay: '4s' }}
      />
      <div 
        className="floating-orb w-40 h-40 bg-primary/20"
        style={{ top: '30%', right: '20%', animationDelay: '1s' }}
      />
      <div 
        className="floating-orb w-32 h-32 bg-secondary/25"
        style={{ bottom: '30%', right: '5%', animationDelay: '3s' }}
      />

      {/* Cellular pattern overlay */}
      <div className="absolute inset-0 cellular-bg" />

      {/* Content */}
      <div className="relative z-10">
        {children}
      </div>
    </div>
  );
};

export default CellularBackground;
