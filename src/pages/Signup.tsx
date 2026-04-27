import { useState } from "react";
import { signup } from "@/lib/api";
import { Link, useNavigate } from "react-router-dom";
import { User, Mail, Lock, CheckCircle, AlertCircle } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import CytoSightLogo from "@/components/CytoSightLogo";
import GlassCard from "@/components/GlassCard";
import MedicalInput from "@/components/MedicalInput";
import MedicalButton from "@/components/MedicalButton";

const Signup = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    fullName: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [isLoading, setIsLoading] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiError(null);
    setErrors({});
    
    // Validation
    const newErrors: Record<string, string> = {};
    if (formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
    }
    if (formData.password.length < 8) {
      newErrors.password = "Password must be at least 8 characters";
    }
    if (!formData.fullName.trim()) {
      newErrors.fullName = "Full name is required";
    }
    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email.trim())) {
      newErrors.email = "Please enter a valid email address";
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    setIsLoading(true);
    try {
      console.log("[SIGNUP] Attempting signup with:", formData.email);
      const data = await signup({
        fullName: formData.fullName,
        email: formData.email,
        password: formData.password,
      });
      console.log("[SIGNUP] Success - tokens stored, redirecting to dashboard");
      setIsLoading(false);
      navigate("/dashboard");
    } catch (err: any) {
      setIsLoading(false);
      const errorMsg = err.message || "Signup failed";
      console.error("[SIGNUP] Error:", errorMsg);
      setApiError(errorMsg);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === "email" ? value.trim() : value,  // Trim email on input
    }));
    // Clear error when user types
    if (errors[name]) {
      setErrors(prev => ({ ...prev, [name]: "" }));
    }
  };

  return (
    <CellularBackground>
      <div className="min-h-screen flex items-center justify-center px-4 py-8">
        <GlassCard variant="bordered" className="w-full max-w-md">
          {/* Logo */}
          <div className="flex justify-center mb-6">
            <CytoSightLogo size="lg" />
          </div>

          {/* Subtitle */}
          <p className="text-center text-muted-foreground mb-8">
            Create your account
          </p>

          {/* Error Alert */}
          {apiError && (
            <div className="mb-6 p-3 bg-red-900/20 border border-red-500/50 rounded-lg flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
              <div>
                <p className="text-sm text-red-300">{apiError}</p>
                <p className="text-xs text-red-300 mt-1">
                  Make sure backend is running and ngrok tunnel is active
                </p>
              </div>
            </div>
          )}

          {/* Signup Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            <MedicalInput
              label="Full Name"
              name="fullName"
              type="text"
              placeholder="Enter your full name"
              icon={User}
              value={formData.fullName}
              onChange={handleChange}
              error={errors.fullName}
              required
            />

            <MedicalInput
              label="Email"
              name="email"
              type="email"
              placeholder="Enter your email"
              icon={Mail}
              value={formData.email}
              onChange={handleChange}
              error={errors.email}
              required
            />

            <MedicalInput
              label="Password"
              name="password"
              type="password"
              placeholder="Create a password"
              icon={Lock}
              value={formData.password}
              onChange={handleChange}
              error={errors.password}
              required
            />

            <MedicalInput
              label="Confirm Password"
              name="confirmPassword"
              type="password"
              placeholder="Confirm your password"
              icon={CheckCircle}
              value={formData.confirmPassword}
              onChange={handleChange}
              error={errors.confirmPassword}
              required
            />

            <div className="pt-4">
              <MedicalButton
                type="submit"
                variant="primary"
                size="lg"
                fullWidth
                isLoading={isLoading}
              >
                Sign Up
              </MedicalButton>
            </div>
          </form>

          {/* Login Link */}
          <p className="text-center mt-6 text-muted-foreground">
            Already have an account?{" "}
            <Link 
              to="/login" 
              className="text-primary hover:text-primary-glow transition-colors font-medium"
            >
              Login
            </Link>
          </p>
        </GlassCard>
      </div>

      {/* Tagline at bottom */}
      <div className="absolute bottom-8 left-0 right-0 text-center px-4">
        <p className="text-muted-foreground text-sm max-w-2xl mx-auto">
          Unified Platform for Blood and Tissue Disease Diagnosis and Cell Morphology Analysis Using Microscopic Imaging
        </p>
      </div>
    </CellularBackground>
  );
};

export default Signup;