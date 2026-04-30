import { useNavigate } from "react-router-dom";
import { ArrowLeft } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import CytoSightLogo from "@/components/CytoSightLogo";
import GlassCard from "@/components/GlassCard";

const About = () => {
  const navigate = useNavigate();

  return (
    <CellularBackground>
      <div className="min-h-screen flex flex-col p-4 md:p-8">
        <header className="flex items-center mb-8">
          <button 
            onClick={() => navigate(-1)}
            className="flex items-center text-muted-foreground hover:text-foreground transition-colors glass-card px-4 py-2 rounded-full border border-border/30"
          >
            <ArrowLeft className="w-4 h-4 mr-2" />
            Back
          </button>
        </header>

        <main className="flex-1 flex flex-col items-center justify-center max-w-4xl mx-auto w-full">
          <GlassCard variant="bordered" className="w-full text-center py-16 px-8 md:px-16 animate-fade-in">
            {/* Logo */}
            <div className="flex justify-center mb-10">
              <CytoSightLogo size="xl" />
            </div>

            {/* Description Text */}
            <div className="max-w-3xl mx-auto">
              <h1 className="text-2xl md:text-3xl font-bold text-foreground leading-relaxed mb-6">
                Welcome to CytoSight: An Explainable Unified Framework for Heterogeneous Cellular Disease Diagnosis and unsupervised segmentation mask generation for blood smears
              </h1>
            </div>
            
            <div className="mt-12 h-px bg-gradient-to-r from-transparent via-border to-transparent w-full" />
            
            <div className="mt-8 text-muted-foreground text-sm">
              <p>© {new Date().getFullYear()} CytoSight. All rights reserved.</p>
            </div>
          </GlassCard>
        </main>
      </div>
    </CellularBackground>
  );
};

export default About;
