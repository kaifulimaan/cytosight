import React, { useState } from "react";
import { login } from "@/lib/api";
import { Link, useNavigate } from "react-router-dom";
import { User, Lock, AlertCircle } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import CytoSightLogo from "@/components/CytoSightLogo";
import GlassCard from "@/components/GlassCard";
import MedicalInput from "@/components/MedicalInput";
import MedicalButton from "@/components/MedicalButton";

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    email: "",
    password: "",
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
    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email.trim())) {
      newErrors.email = "Please enter a valid email address";
    }
    if (!formData.password) {
      newErrors.password = "Password is required";
    }
    
    if (Object.keys(newErrors).length > 0) {
      setErrors(newErrors);
      return;
    }
    
    setIsLoading(true);
    try {
      console.log("[LOGIN] Attempting login with:", formData.email);
      const data = await login({
        email: formData.email,
        password: formData.password,
      });
      console.log("[LOGIN] Success - tokens stored, redirecting to dashboard");
      setIsLoading(false);
      navigate("/dashboard");
    } catch (err: any) {
      setIsLoading(false);
      const errorMsg = err.message || "Login failed";
      console.error("[LOGIN] Error:", errorMsg);
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
          {/* Logo - Right Aligned */}
          <div className="w-full mb-8 justify-end">
            <CytoSightLogo size="lg" />
          </div>

          {/* Subtitle */}
          <p className="text-center text-muted-foreground mb-8">
            Please sign in to continue
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

          {/* Login Form */}
          <form onSubmit={handleSubmit} className="space-y-5">
            <MedicalInput
              label="Email"
              name="email"
              type="email"
              placeholder="Enter your email"
              icon={User}
              value={formData.email}
              onChange={handleChange}
              error={errors.email}
              required
            />

            <MedicalInput
              label="Password"
              name="password"
              type="password"
              placeholder="Enter your password"
              icon={Lock}
              value={formData.password}
              onChange={handleChange}
              error={errors.password}
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
                Login
              </MedicalButton>
            </div>
          </form>

          {/* Sign Up Link */}
          <p className="text-center mt-6 text-muted-foreground">
            Don't have an account?{" "}
            <Link 
              to="/signup" 
              className="text-primary hover:text-primary-glow transition-colors font-medium"
            >
              Sign Up
            </Link>
          </p>
        </GlassCard>
      </div>
    </CellularBackground>
  );
};

export default Login;