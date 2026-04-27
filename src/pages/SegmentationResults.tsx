import { useMemo } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { ArrowLeft, Download, AlertTriangle } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import GlassCard from "@/components/GlassCard";
import MedicalButton from "@/components/MedicalButton";

interface SegmentationData {
  original_image_path: string;
  original_image_url: string;
  segmented_mask_path: string;
  segmented_mask_url: string;
  mask_download_name: string;
  width: number;
  height: number;
  tile_size: number;
  tiling_used: boolean;
}

const SegmentationResults = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const segmentationData: SegmentationData | null = location.state?.segmentationData || null;
  const imagePreview: string | null = location.state?.imagePreview || null;
  const originalImageUrlFromUpload: string | null = location.state?.originalImageUrl || null;

  const originalImageSrc = useMemo(
    () => segmentationData?.original_image_url || originalImageUrlFromUpload || imagePreview,
    [segmentationData?.original_image_url, originalImageUrlFromUpload, imagePreview]
  );

  const maskImageSrc = segmentationData?.segmented_mask_url || null;

  const handleSaveMask = () => {
    if (!maskImageSrc) return;
    const anchor = document.createElement("a");
    anchor.href = maskImageSrc;
    anchor.download = segmentationData?.mask_download_name || "segmented_mask.png";
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
  };

  if (!segmentationData || !maskImageSrc) {
    return (
      <CellularBackground>
        <div className="min-h-screen flex items-center justify-center px-4">
          <GlassCard variant="bordered" className="max-w-md text-center p-8">
            <div className="text-destructive mb-4">
              <AlertTriangle className="w-12 h-12 mx-auto" />
            </div>
            <h2 className="text-xl font-bold text-foreground mb-2">No Segmentation Result Found</h2>
            <p className="text-muted-foreground mb-6">
              Upload an image and run binary segmentation to view the result.
            </p>
            <MedicalButton variant="primary" onClick={() => navigate("/upload")}>
              Go to Upload
            </MedicalButton>
          </GlassCard>
        </div>
      </CellularBackground>
    );
  }

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
          <div className="w-full max-w-3xl space-y-6">
            
            {/* Title */}
            <h1 className="text-3xl font-bold text-success text-center">
              Segmentation Results
            </h1>

            {/* Original Image Card */}
            <GlassCard variant="bordered">
              <h2 className="text-lg font-semibold text-primary mb-4">
                Original Blood Smear Image:
              </h2>
              {originalImageSrc ? (
                <img
                  src={originalImageSrc}
                  alt="Original uploaded slide"
                  className="w-full max-h-[480px] object-contain rounded-lg border border-border"
                />
              ) : (
                <div className="h-64 rounded-lg border border-border bg-muted/20 flex items-center justify-center text-muted-foreground">
                  Original image unavailable
                </div>
              )}
            </GlassCard>

            {/* Segmented Cells Card */}
            <GlassCard variant="bordered">
              <h2 className="text-lg font-semibold text-primary mb-4">
                Segmented Cells:
              </h2>
              <img
                src={maskImageSrc}
                alt="Segmented mask"
                className="w-full max-h-[480px] object-contain rounded-lg border border-border bg-black"
              />
            </GlassCard>

            <div className="text-center text-sm text-muted-foreground">
              {segmentationData.width} x {segmentationData.height} | Tile size: {segmentationData.tile_size} | {segmentationData.tiling_used ? "Tiled inference" : "Single-pass inference"}
            </div>

            {/* Save Button */}
            <div className="flex justify-center pt-4">
              <MedicalButton
                variant="primary"
                size="lg"
                onClick={handleSaveMask}
                className="min-w-64"
              >
                <Download className="w-5 h-5 mr-2" />
                Download segmentation mask
              </MedicalButton>
            </div>
          </div>
        </main>
      </div>
    </CellularBackground>
  );
};

export default SegmentationResults;
