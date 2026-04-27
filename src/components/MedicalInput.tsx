import { cn } from "@/lib/utils";
import { forwardRef } from "react";
import { LucideIcon } from "lucide-react";

interface MedicalInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  icon?: LucideIcon;
  error?: string;
}

const MedicalInput = forwardRef<HTMLInputElement, MedicalInputProps>(
  ({ className, label, icon: Icon, error, type = "text", ...props }, ref) => {
    return (
      <div className="space-y-2">
        {label && (
          <label className="block text-sm font-medium text-primary">
            {label}
          </label>
        )}
        <div className="relative">
          {Icon && (
            <div className="absolute left-4 top-1/2 -translate-y-1/2 text-muted-foreground">
              <Icon className="w-5 h-5" />
            </div>
          )}
          <input
            ref={ref}
            type={type}
            className={cn(
              "w-full input-medical rounded-lg py-3 px-4 text-foreground",
              Icon && "pl-12",
              error && "border-destructive focus:border-destructive focus:ring-destructive/20",
              className
            )}
            {...props}
          />
        </div>
        {error && (
          <p className="text-sm text-destructive">{error}</p>
        )}
      </div>
    );
  }
);

MedicalInput.displayName = "MedicalInput";

export default MedicalInput;
