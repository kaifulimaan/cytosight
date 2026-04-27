import { useNavigate, useLocation } from "react-router-dom";
import { useEffect } from "react";
import CellularBackground from "@/components/CellularBackground";
import GlassCard from "@/components/GlassCard";
import MedicalButton from "@/components/MedicalButton";
import { Home, AlertTriangle } from "lucide-react";

const NotFound = () => {
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    console.error("404 Error: User attempted to access non-existent route:", location.pathname);
  }, [location.pathname]);

  return (
    <CellularBackground>
      <div className="min-h-screen flex items-center justify-center px-4">
        <GlassCard variant="bordered" className="text-center max-w-md">
          <div className="flex justify-center mb-6">
            <div className="w-20 h-20 rounded-full bg-warning/20 flex items-center justify-center">
              <AlertTriangle className="w-10 h-10 text-warning" />
            </div>
          </div>

          <h1 className="text-6xl font-bold text-primary mb-4">404</h1>
          <h2 className="text-xl font-semibold text-foreground mb-2">
            Page Not Found
          </h2>
          <p className="text-muted-foreground mb-8">
            The page you're looking for doesn't exist or has been moved.
          </p>

          <MedicalButton
            variant="primary"
            size="lg"
            onClick={() => navigate("/dashboard")}
          >
            <Home className="w-5 h-5 mr-2" />
            Back to Dashboard
          </MedicalButton>
        </GlassCard>
      </div>
    </CellularBackground>
  );
};

export default NotFound;
