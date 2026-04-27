import { useState, useRef, useCallback, useEffect } from "react";
import { uploadImage, runDiagnosis, runSegmentation, isLoggedIn } from "@/lib/api";
import { useNavigate } from "react-router-dom";
import { Upload, ArrowLeft, X, FileImage, AlertCircle } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import CytoSightLogo from "@/components/CytoSightLogo";
import GlassCard from "@/components/GlassCard";
import MedicalButton from "@/components/MedicalButton";

const MAX_FILE_SIZE = 5 * 1024 * 1024; // 5MB

const UploadScreen = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isLoadingDiagnosis, setIsLoadingDiagnosis] = useState(false);
  const [isLoadingSegmentation, setIsLoadingSegmentation] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);

  // Check authentication on mount
  useEffect(() => {
    if (!isLoggedIn()) {
      navigate("/login");
    }
  }, [navigate]);

  const validateFile = (file: File): string | null => {
    if (file.size > MAX_FILE_SIZE) {
      return `File size exceeds 5MB limit. Current size: ${(file.size / 1024 / 1024).toFixed(2)}MB`;
    }
    
    const validTypes = ['image/jpeg', 'image/png', 'image/tiff'];
    const validExtensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif', '.svs'];
    const extension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    
    if (!validTypes.includes(file.type) && !validExtensions.includes(extension)) {
      return 'Invalid file type. Supported formats: JPG, PNG, TIFF, SVS';
    }
    
    return null;
  };

  const handleFileSelect = useCallback((file: File) => {
    setError(null);
    
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      setSelectedFile(null);
      setPreview(null);
      return;
    }
    
    setSelectedFile(file);
    
    // Generate preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreview(reader.result as string);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFileSelect(file);
    }
  }, [handleFileSelect]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreview(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const getToken = () => {
    return localStorage.getItem("access_token") || "";
  };

  const handleDiagnosis = async () => {
    if (!selectedFile || !preview) return;
    setIsLoadingDiagnosis(true);
    setUploadError(null);
    setUploadResult(null);
    try {
      const token = getToken();
      if (!token) {
        throw new Error("Not authenticated. Please login.");
      }
      
      // Step 1: Upload image to Supabase
      console.log("[UPLOAD] Step 1: Uploading image...");
      const uploadResult = await uploadImage({ file: selectedFile, token });
      console.log("[UPLOAD] Success:", uploadResult);
      
      // Step 2: Run diagnosis on uploaded image
      console.log("[DIAGNOSIS] Step 2: Running diagnosis...");
      const diagnosisResult = await runDiagnosis({
        imagePath: uploadResult.file_path,
        imageUrl: uploadResult.public_url,  // Fallback URL if path-based download fails
        token
      });
      console.log("[DIAGNOSIS] Success:", diagnosisResult);
      
      setUploadResult(diagnosisResult);
      setIsLoadingDiagnosis(false);
      
      // Step 3: Pass diagnosis data AND image to results page
      navigate("/diagnosis-results", { 
        state: { 
          diagnosisData: diagnosisResult,
          imageUrl: uploadResult.public_url,
          imagePreview: preview
        } 
      });
    } catch (err: any) {
      setIsLoadingDiagnosis(false);
      const errorMsg = err.message || "Image upload or diagnosis failed";
      console.error("[UPLOAD] Error:", errorMsg);
      setUploadError(errorMsg);
    }
  };

  const handleSegmentation = async () => {
    if (!selectedFile || !preview) return;
    setIsLoadingSegmentation(true);
    setUploadError(null);
    setUploadResult(null);

    try {
      const token = getToken();
      if (!token) {
        throw new Error("Not authenticated. Please login.");
      }

      // Step 1: Upload image to Supabase (same flow as diagnosis).
      const uploaded = await uploadImage({ file: selectedFile, token });

      // Step 2: Run segmentation with storage path and fallback URL.
      const segmentationResult = await runSegmentation({
        imagePath: uploaded.file_path,
        imageUrl: uploaded.public_url,  // Fallback URL if path-based download fails
        token,
      });

      setUploadResult(segmentationResult);
      setIsLoadingSegmentation(false);

      navigate("/segmentation-results", {
        state: {
          segmentationData: segmentationResult,
          imagePreview: preview,
          originalImageUrl: uploaded.public_url,
        },
      });
    } catch (err: any) {
      setIsLoadingSegmentation(false);
      const errorMsg = err.message || "Image upload or segmentation failed";
      setUploadError(errorMsg);
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
  };

  return (
    <CellularBackground>
      <div className="min-h-screen flex flex-col">
        {/* Header */}
        <header className="p-4 flex items-center">
          <MedicalButton
            variant="ghost"
            size="sm"
            onClick={() => navigate("/dashboard")}
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Dashboard
          </MedicalButton>
        </header>

        {/* Main Content */}
        <main className="flex-1 flex items-center justify-center px-4 pb-8">
          <GlassCard variant="bordered" className="w-full max-w-2xl text-center">
            {/* Logo */}
            <div className="flex justify-center mb-4">
              <CytoSightLogo size="md" />
            </div>

            {/* Tagline */}
            <p className="text-muted-foreground mb-8 text-sm max-w-lg mx-auto">
              Unified Platform for Blood and Tissue Disease Diagnosis and Cell Morphology Analysis Using Microscopic Imaging
            </p>

            {/* Upload Zone */}
            <div
              className={`upload-zone rounded-xl p-8 mb-6 cursor-pointer transition-all duration-300 ${
                isDragging ? "drag-over border-primary bg-primary/5" : ""
              } ${error ? "border-destructive" : ""}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".jpg,.jpeg,.png,.tiff,.tif,.svs"
                onChange={handleFileInput}
                className="hidden"
              />

              {!selectedFile ? (
                <div className="flex flex-col items-center gap-4">
                  <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
                    <Upload className="w-8 h-8 text-primary" />
                  </div>
                  <div>
                    <p className="text-primary font-semibold text-lg">
                      Drag & Drop Your Image Here
                    </p>
                    <p className="text-muted-foreground text-sm mt-1">
                      or click to browse
                    </p>
                  </div>
                  <p className="text-muted-foreground text-xs italic">
                    Supported formats: JPG, PNG, TIFF, SVS (Max 5MB)
                  </p>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-4">
                  {/* Preview Thumbnail */}
                  {preview && (
                    <div className="relative">
                      <img
                        src={preview}
                        alt="Preview"
                        className="max-w-48 max-h-48 rounded-lg border border-border object-contain"
                      />
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          clearFile();
                        }}
                        className="absolute -top-2 -right-2 w-6 h-6 rounded-full bg-destructive text-destructive-foreground flex items-center justify-center hover:bg-destructive/80 transition-colors"
                      >
                        <X className="w-4 h-4" />
                      </button>
                    </div>
                  )}
                  <div className="flex items-center gap-2">
                    <FileImage className="w-5 h-5 text-primary" />
                    <span className="text-foreground font-medium">{selectedFile.name}</span>
                    <span className="text-muted-foreground text-sm">
                      ({formatFileSize(selectedFile.size)})
                    </span>
                  </div>
                  <p className="text-success text-sm">File selected - choose an action below</p>
                </div>
              )}
            </div>

            {/* Error Message */}
            {error && (
              <div className="mb-4 p-3 bg-red-900/20 border border-red-500/50 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                <p className="text-sm text-red-300">{error}</p>
              </div>
            )}

            {/* Action Buttons */}
            <div className="flex gap-4 justify-center mb-4">
              <MedicalButton
                variant="primary"
                size="lg"
                onClick={handleDiagnosis}
                disabled={!selectedFile || isLoadingDiagnosis || isLoadingSegmentation}
                isLoading={isLoadingDiagnosis}
                className="min-w-48"
              >
                Disease Diagnosis
              </MedicalButton>

              <MedicalButton
                variant="secondary"
                size="lg"
                onClick={handleSegmentation}
                disabled={!selectedFile || isLoadingDiagnosis || isLoadingSegmentation}
                isLoading={isLoadingSegmentation}
                className="min-w-48"
              >
                Binary Segmentation
              </MedicalButton>
            </div>

            {uploadError && (
              <div className="p-3 bg-red-900/20 border border-red-500/50 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm text-red-300">{uploadError}</p>
                  <p className="text-xs text-red-300 mt-1">
                    Make sure backend is running and ngrok tunnel is active
                  </p>
                </div>
              </div>
            )}

            {uploadResult && (
              <div className="text-success text-center text-sm mt-2">
                Image uploaded successfully!
              </div>
            )}
          </GlassCard>
        </main>
      </div>
    </CellularBackground>
  );
};

export default UploadScreen;