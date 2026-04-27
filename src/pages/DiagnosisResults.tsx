import { useState, useEffect } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { ArrowLeft, AlertTriangle, CheckCircle2, Loader2, Eye, Activity, BarChart3, Info } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import GlassCard from "@/components/GlassCard";
import MedicalButton from "@/components/MedicalButton";
import PlaceholderBox from "@/components/PlaceholderBox";

interface DiagnosisData {
  region: string;
  status: "normal" | "abnormal";
  stage?: {
    index: number;
    name: string;
    description: string;
  } | string | null;
  disease: {
    name: string;
    key: string;
    description: string;
    confidence: number;
  };
  severity: {
    name: string;
    description: string;
    level: number;
    confidence: number;
  };
  confidence_score: number;
  diagnosis: string;
  timestamp?: string;
}

const DiagnosisResults = () => {
  const navigate = useNavigate();
  const location = useLocation();

  // Get diagnosis data, image, and preview from location state
  const diagnosisData: DiagnosisData | null = location.state?.diagnosisData || null;
  const imageUrl: string | null = location.state?.imageUrl || null;
  const imagePreview: string | null = location.state?.imagePreview || null;
  const stageName =
    typeof diagnosisData?.stage === "string"
      ? diagnosisData.stage
      : diagnosisData?.stage?.name ?? null;
  const stageDescription =
    typeof diagnosisData?.stage === "object" && diagnosisData.stage
      ? diagnosisData.stage.description
      : null;

  const [explainData, setExplainData] = useState<{
    attention_heatmap_base64: string;
    gradcam_heatmap_base64: string;
    gpt_statement: string;
  } | null>(null);
  const [explainLoading, setExplainLoading] = useState(true);
  const [explainError, setExplainError] = useState<string | null>(null);

  useEffect(() => {
    if (!diagnosisData || !imageUrl) return;
    
    const getExplainability = async () => {
      try {
        const { fetchExplainability } = await import("@/lib/api");
        const data = await fetchExplainability({
          imageUrl,
          diagnosisData,
        });
        setExplainData(data);
      } catch (err) {
        console.error("Error fetching explainability:", err);
        setExplainError("Explainability generation failed. This might happen if the image could not be loaded or the model is busy.");
      } finally {
        setExplainLoading(false);
      }
    };
    
    getExplainability();
  }, [diagnosisData, imageUrl]);

  // If no data, show error and redirect
  if (!diagnosisData) {
    return (
      <CellularBackground>
        <div className="min-h-screen flex items-center justify-center">
          <GlassCard variant="bordered" className="max-w-md text-center p-8">
            <div className="text-destructive mb-4">
              <AlertTriangle className="w-12 h-12 mx-auto" />
            </div>
            <h2 className="text-xl font-bold text-foreground mb-2">No Results Found</h2>
            <p className="text-muted-foreground mb-6">
              Please upload an image and run diagnosis to see results.
            </p>
            <MedicalButton
              variant="primary"
              onClick={() => navigate("/upload")}
            >
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
        <header className="p-4 flex items-center justify-start">
          <MedicalButton
            variant="ghost"
            size="sm"
            onClick={() => navigate("/dashboard")}
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Back to Dashboard
          </MedicalButton>
        </header>

        {/* Main Content - Scrollable */}
        <main className="flex-1 overflow-y-auto px-4 pb-8">
          <div className="max-w-5xl mx-auto space-y-6">
            
            {/* Diagnosis Completed Header */}
            <GlassCard variant="bordered">
              <div className="flex items-center justify-between mb-6">
                <h1 className="text-3xl font-bold text-primary">
                  Diagnosis Completed
                </h1>
                {diagnosisData.status === "abnormal" ? (
                  <AlertTriangle className="w-8 h-8 text-warning animate-pulse" />
                ) : (
                  <CheckCircle2 className="w-8 h-8 text-success" />
                )}
              </div>
              <p className="text-lg text-muted-foreground">
                {diagnosisData.diagnosis}
              </p>
            </GlassCard>

            {/* Results Cards Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              
              {/* Region Card */}
              <GlassCard variant="bordered" className="relative overflow-hidden">
                <div className="absolute top-0 right-0 w-24 h-24 bg-blue-500/10 rounded-full -mr-12 -mt-12"></div>
                <div className="relative z-10">
                  <div className="text-blue-400 text-sm font-semibold uppercase tracking-widest mb-2">
                    Region Identified
                  </div>
                  <div className="text-2xl font-bold text-primary mb-2">
                    {diagnosisData.region}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    Area of tissue being analyzed
                  </p>
                </div>
              </GlassCard>

              {/* Status Card */}
              <GlassCard 
                variant="bordered" 
                className={`relative overflow-hidden ${
                  diagnosisData.status === "abnormal" ? "border-warning/50" : "border-success/50"
                }`}
              >
                <div className={`absolute top-0 right-0 w-24 h-24 rounded-full -mr-12 -mt-12 ${
                  diagnosisData.status === "abnormal" ? "bg-warning/10" : "bg-success/10"
                }`}></div>
                <div className="relative z-10">
                  <div className={`${
                    diagnosisData.status === "abnormal" ? "text-warning" : "text-success"
                  } text-sm font-semibold uppercase tracking-widest mb-2`}>
                    Status
                  </div>
                  <div className={`text-2xl font-bold mb-2 ${
                    diagnosisData.status === "abnormal" ? "text-warning" : "text-success"
                  }`}>
                    {diagnosisData.status === "abnormal" ? "⚠️  Abnormal" : "✓ Normal"}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {diagnosisData.severity.description}
                  </p>
                </div>
              </GlassCard>

              {/* Stage Card (if abnormal) */}
              {diagnosisData.status === "abnormal" && stageName && (
                <GlassCard variant="bordered" className="relative overflow-hidden md:col-span-2">
                  <div className="absolute top-0 right-0 w-24 h-24 bg-orange-500/10 rounded-full -mr-12 -mt-12"></div>
                  <div className="relative z-10">
                    <div className="text-orange-400 text-sm font-semibold uppercase tracking-widest mb-2">
                      Abnormality Stage/Type
                    </div>
                    <div className="text-2xl font-bold text-primary mb-2">
                      {stageName}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {stageDescription || "Classification of identified abnormality"}
                    </p>
                  </div>
                </GlassCard>
              )}

            </div>

            {/* Explainable AI Report Placeholder */}
            <GlassCard variant="bordered">
              <h2 className="text-xl font-bold text-primary mb-6">
                Explainable Report
              </h2>
              
              {explainLoading ? (
                <div className="flex flex-col items-center justify-center h-48 space-y-4">
                  <Loader2 className="w-10 h-10 text-primary animate-spin" />
                  <p className="text-muted-foreground animate-pulse">Generating attention heatmaps and GPT explanation...</p>
                </div>
              ) : explainError ? (
                <div className="flex flex-col items-center justify-center h-48 space-y-2 text-destructive">
                  <AlertTriangle className="w-8 h-8" />
                  <p>{explainError}</p>
                </div>
              ) : explainData ? (
                <div className="space-y-6">
                  {/* GPT Statement - Aesthetic Sectioned Layout */}
                  <div className="grid grid-cols-1 gap-6">
                    {(() => {
                      const text = explainData.gpt_statement;
                      // Split by [HEADER] or ### Header
                      const sections = text.split(/\[([A-Z\s]+)\]|###\s+(.+)/g).filter(Boolean);
                      
                      const formattedSections: { title: string, content: string }[] = [];
                      for (let i = 0; i < sections.length; i += 2) {
                        if (sections[i + 1]) {
                          formattedSections.push({
                            title: sections[i].trim(),
                            content: sections[i+1].trim()
                          });
                        }
                      }

                      // If parsing failed or was too simple, just show the text
                      if (formattedSections.length === 0) {
                        return (
                          <div className="p-6 bg-primary/5 rounded-2xl border border-primary/10 shadow-sm">
                            <p className="text-foreground leading-relaxed whitespace-pre-wrap">
                              {text}
                            </p>
                          </div>
                        );
                      }

                      const getIcon = (title: string) => {
                        const t = title.toLowerCase();
                        if (t.includes('decision')) return <CheckCircle2 className="w-5 h-5 text-success" />;
                        if (t.includes('looked')) return <Eye className="w-5 h-5 text-blue-400" />;
                        if (t.includes('gradcam')) return <Activity className="w-5 h-5 text-orange-400" />;
                        if (t.includes('attention')) return <BarChart3 className="w-5 h-5 text-purple-400" />;
                        if (t.includes('visual')) return <Info className="w-5 h-5 text-primary" />;
                        return <Info className="w-5 h-5 text-primary" />;
                      };

                      return formattedSections.map((section, idx) => (
                        <div key={idx} className="group p-5 bg-muted/20 hover:bg-muted/30 rounded-2xl border border-border/50 transition-all duration-300">
                          <div className="flex items-center gap-3 mb-3">
                            <div className="p-2 bg-background rounded-lg shadow-sm group-hover:scale-110 transition-transform duration-300">
                              {getIcon(section.title)}
                            </div>
                            <h3 className="text-sm font-bold text-foreground uppercase tracking-wider">
                              {section.title.replace(/[\[\]]/g, '')}
                            </h3>
                          </div>
                          <p className="text-muted-foreground leading-relaxed pl-12 italic border-l-2 border-primary/10 ml-5">
                            {section.content}
                          </p>
                        </div>
                      ));
                    })()}
                  </div>
                  
                  {/* Heatmaps */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-4">
                    <div className="group space-y-3">
                      <div className="flex items-center gap-2 px-1">
                        <BarChart3 className="w-4 h-4 text-purple-400" />
                        <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Attention Heatmap</h3>
                      </div>
                      <div className="relative rounded-2xl overflow-hidden border border-border/50 shadow-lg group-hover:border-primary/30 transition-colors">
                        <img src={explainData.attention_heatmap_base64} alt="Attention Heatmap" className="w-full h-auto object-contain" />
                        <div className="absolute inset-0 bg-gradient-to-t from-background/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </div>
                    
                    <div className="group space-y-3">
                      <div className="flex items-center gap-2 px-1">
                        <Activity className="w-4 h-4 text-orange-400" />
                        <h3 className="text-xs font-bold text-muted-foreground uppercase tracking-widest">Grad-CAM Heatmap</h3>
                      </div>
                      <div className="relative rounded-2xl overflow-hidden border border-border/50 shadow-lg group-hover:border-primary/30 transition-colors">
                        <img src={explainData.gradcam_heatmap_base64} alt="Grad-CAM Heatmap" className="w-full h-auto object-contain" />
                        <div className="absolute inset-0 bg-gradient-to-t from-background/20 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                <PlaceholderBox 
                  label="Detailed AI explanation and recommendations will be available here" 
                  height="h-48"
                />
              )}
            </GlassCard>
          </div>
        </main>
      </div>
    </CellularBackground>
  );
};

export default DiagnosisResults;
