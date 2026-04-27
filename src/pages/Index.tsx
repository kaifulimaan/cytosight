import { useNavigate } from "react-router-dom";
import CellularBackground from "@/components/CellularBackground";
import CytoSightLogo from "@/components/CytoSightLogo";
import MedicalButton from "@/components/MedicalButton";
import { ArrowRight } from "lucide-react";

const Index = () => {
  const navigate = useNavigate();

  return (
    <CellularBackground>
      <div className="min-h-screen flex items-center justify-center px-4">
        <div className="text-center max-w-2xl">
          {/* Logo */}
          <div className="flex justify-center mb-8">
            <CytoSightLogo size="lg" />
          </div>

          {/* Tagline */}
          <p className="text-muted-foreground text-lg mb-12 max-w-xl mx-auto">
            Unified Platform for Blood and Tissue Disease Diagnosis and Cell Morphology Analysis Using Microscopic Imaging
          </p>

          {/* CTA Button */}
          <MedicalButton
            variant="primary"
            size="xl"
            onClick={() => navigate("/login")}
          >
            Get Started
            <ArrowRight className="w-5 h-5 ml-2 inline" />
          </MedicalButton>
        </div>
      </div>
    </CellularBackground>
  );
};

export default Index;
