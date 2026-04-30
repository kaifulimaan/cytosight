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
  reconstructed_image_url: string;
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
  const reconstructedImageSrc = segmentationData?.reconstructed_image_url || null;

  const handleSaveMask = async () => {
    if (!maskImageSrc) return;
    try {
      // Fetch the image to handle potential cross-origin download issues
      const response = await fetch(maskImageSrc);
      const blob = await response.blob();
      const blobUrl = window.URL.createObjectURL(blob);
      
      const anchor = document.createElement("a");
      anchor.href = blobUrl;
      anchor.download = segmentationData?.mask_download_name || "segmented_mask.png";
      document.body.appendChild(anchor);
      anchor.click();
      document.body.removeChild(anchor);
      window.URL.revokeObjectURL(blobUrl);
    } catch (err) {
      console.error("Download failed:", err);
      // Fallback to simple anchor click
      const anchor = document.createElement("a");
      anchor.href = maskImageSrc;
      anchor.target = "_blank";
      anchor.download = segmentationData?.mask_download_name || "segmented_mask.png";
      anchor.click();
    }
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
        <main className="flex-1 flex flex-col items-center justify-center px-4 pb-8">
          <div className="w-full max-w-6xl space-y-8">
            
            {/* Title */}
            <h1 className="text-3xl font-bold text-success text-center">
              Segmentation Results
            </h1>

            {/* Top Row: Original and Reconstructed */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <GlassCard variant="bordered" className="flex flex-col h-full">
                <h2 className="text-lg font-semibold text-primary mb-4 text-center">
                  Original Image:
                </h2>
                <div className="flex-1 flex items-center justify-center bg-black/20 rounded-lg overflow-hidden border border-border">
                  {originalImageSrc ? (
                    <img
                      src={originalImageSrc}
                      alt="Original"
                      className="max-w-full max-h-[400px] object-contain"
                    />
                  ) : (
                    <p className="text-muted-foreground">Original unavailable</p>
                  )}
                </div>
              </GlassCard>

              <GlassCard variant="bordered" className="flex flex-col h-full">
                <h2 className="text-lg font-semibold text-primary mb-4 text-center">
                  Reconstructed Image:
                </h2>
                <div className="flex-1 flex items-center justify-center bg-black/20 rounded-lg overflow-hidden border border-border">
                  {reconstructedImageSrc ? (
                    <img
                      src={reconstructedImageSrc}
                      alt="Reconstructed"
                      className="max-w-full max-h-[400px] object-contain"
                    />
                  ) : (
                    <p className="text-muted-foreground">Reconstruction unavailable</p>
                  )}
                </div>
              </GlassCard>
            </div>

            {/* Bottom Row: Segmented Cells (Centered) */}
            <div className="flex justify-center">
              <GlassCard variant="bordered" className="w-full md:w-2/3 lg:w-1/2 flex flex-col">
                <h2 className="text-lg font-semibold text-primary mb-4 text-center">
                  Segmented Cells:
                </h2>
                <div className="flex-1 flex items-center justify-center bg-black rounded-lg overflow-hidden border border-border">
                  <img
                    src={maskImageSrc}
                    alt="Segmented Mask"
                    className="max-w-full max-h-[400px] object-contain"
                  />
                </div>
              </GlassCard>
            </div>

            {/* Stats */}
            <div className="text-center text-sm text-muted-foreground bg-muted/10 p-2 rounded-full max-w-lg mx-auto border border-border/50">
              {segmentationData.width} x {segmentationData.height} | Tile size: {segmentationData.tile_size} | {segmentationData.tiling_used ? "Tiled inference" : "Single-pass inference"}
            </div>

            {/* Save Button */}
            <div className="flex justify-center pt-4">
              <MedicalButton
                variant="primary"
                size="lg"
                onClick={handleSaveMask}
                className="min-w-64 shadow-lg shadow-primary/20"
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
