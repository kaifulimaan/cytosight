import { cn } from "@/lib/utils";
import logo from "@/components/ui/logo.png";

interface CytoSightLogoProps {
  size?: "sm" | "md" | "lg" | "xl";
  align?: "left" | "center" | "right";
  className?: string;
}

const CytoSightLogo = ({ size = "xl", align = "center", className }: CytoSightLogoProps) => {
  const sizeClasses = {
    sm: "h-10",
    md: "h-16",
    lg: "h-28",
    xl: "h-40",
  };

  const alignClasses = {
    left: "justify-start",
    center: "justify-center",
    right: "justify-end",
  };

  return (
    <div className={cn("flex items-center w-full", alignClasses[align], className)}>
      <img 
        src={logo} 
        alt="CytoSight Logo" 
        className={cn("object-contain w-auto", sizeClasses[size])}
      />
    </div>
  );
};

export default CytoSightLogo;