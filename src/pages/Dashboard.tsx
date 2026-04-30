import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { User, ChevronDown, LogOut, History, Trash2 } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import CytoSightLogo from "@/components/CytoSightLogo";
import GlassCard from "@/components/GlassCard";
import MedicalButton from "@/components/MedicalButton";
import { getCurrentUser, logout as logoutApi, isLoggedIn, clearAllHistory, getToken } from "@/lib/api";
import { toast } from "sonner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

const Dashboard = () => {
  const navigate = useNavigate();
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);
  const [user, setUser] = useState<any>(null);

  useEffect(() => {
    // Check if user is logged in
    if (!isLoggedIn()) {
      navigate("/login");
      return;
    }
    
    // Load user data from localStorage
    const currentUser = getCurrentUser();
    if (currentUser) {
      setUser(currentUser);
    }
  }, [navigate]);

  const handleLogout = () => {
    // Clear auth state
    logoutApi();
    console.log("[LOGOUT] User logged out");
    navigate("/login");
  };

  const handleRemoveHistory = async () => {
    const token = getToken();
    if (!token) return;

    setShowConfirmDelete(true);
    try {
      await clearAllHistory(token);
      toast.success("All history and associated images removed successfully");
      console.log("[DELETE] History cleared");
    } catch (error: any) {
      toast.error(error.message || "Failed to remove history");
      console.error("[DELETE] Error:", error);
    } finally {
      setShowConfirmDelete(false);
    }
  };

  // Default user if not loaded
  const displayUser = user || {
    name: "User",
    email: "user@example.com",
  };

  return (
    <CellularBackground>
      <div className="min-h-screen flex flex-col">
        {/* Header with Profile */}
        <header className="p-4 flex justify-end">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button className="flex items-center gap-2 px-4 py-2 rounded-full glass-card hover:border-primary/50 transition-all duration-300 border border-border/30">
                <div className="w-8 h-8 rounded-full bg-primary/20 flex items-center justify-center">
                  <User className="w-4 h-4 text-primary" />
                </div>
                <span className="text-sm font-medium text-foreground">Profile</span>
                <ChevronDown className="w-4 h-4 text-muted-foreground" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56 bg-card border-border">
              <div className="px-3 py-2 border-b border-border">
                <p className="text-sm font-medium text-foreground">
                  {displayUser.full_name || displayUser.name}
                </p>
                <p className="text-xs text-muted-foreground">{displayUser.email}</p>
              </div>
              <DropdownMenuItem 
                className="cursor-pointer text-destructive focus:text-destructive"
                onClick={handleLogout}
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </header>

        {/* Main Content */}
        <main className="flex-1 flex items-center justify-center px-4 pb-16">
          <GlassCard variant="bordered" className="w-full max-w-2xl text-center">
            {/* Logo */}
            <div className="flex justify-center mb-6">
              <CytoSightLogo size="lg" />
            </div>

            {/* Tagline */}
            <p className="text-lg md:text-xl text-muted-foreground mb-8 max-w-2xl mx-auto leading-relaxed">
              Unified Platform for Blood and Tissue Disease Diagnosis and Cell Morphology Analysis Using Microscopic Imaging
            </p>

            {/* Divider */}
            <div className="h-px bg-gradient-to-r from-transparent via-border to-transparent mb-10" />

            {/* Welcome Message */}
            <p className="text-xl md:text-2xl text-foreground mb-8">
              Welcome back, <span className="text-primary font-semibold">
                {displayUser.full_name || displayUser.name || "User"}
              </span>
            </p>

            {/* Action Buttons */}
            <div className="space-y-4 max-w-md mx-auto">
              <MedicalButton
                variant="primary"
                size="xl"
                fullWidth
                onClick={() => navigate("/upload")}
              >
                Go to Diagnosis or Segmentation
              </MedicalButton>

              <MedicalButton
                variant="secondary"
                size="xl"
                fullWidth
                onClick={() => navigate("/history")}
              >
                <History className="w-5 h-5 mr-2 inline" />
                View Past Diagnosis
              </MedicalButton>

              <MedicalButton
                variant="danger"
                size="xl"
                fullWidth
                onClick={handleRemoveHistory}
                isLoading={showConfirmDelete}
              >
                <Trash2 className="w-5 h-5 mr-2 inline" />
                Remove All History
              </MedicalButton>
            </div>

            {/* About Link */}
            <button 
              className="mt-10 text-muted-foreground hover:text-primary transition-colors text-sm"
              onClick={() => navigate("/about")}
            >
              About CytoSight
            </button>
          </GlassCard>
        </main>
      </div>
    </CellularBackground>
  );
};

export default Dashboard;