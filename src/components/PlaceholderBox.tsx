import { cn } from "@/lib/utils";
import { ImageIcon } from "lucide-react";

interface PlaceholderBoxProps {
  label: string;
  height?: string;
  icon?: React.ReactNode;
  className?: string;
}

const PlaceholderBox = ({ 
  label, 
  height = "h-48", 
  icon,
  className 
}: PlaceholderBoxProps) => {
  return (
    <div 
      className={cn(
        "placeholder-box rounded-lg",
        height,
        className
      )}
    >
      <div className="flex flex-col items-center gap-3 p-4 text-center">
        {icon || <ImageIcon className="w-10 h-10 opacity-50" />}
        <span className="text-sm">{label}</span>
      </div>
    </div>
  );
};

export default PlaceholderBox;
