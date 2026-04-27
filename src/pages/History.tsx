import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, ImageIcon, Loader2, AlertTriangle, Trash2, AlertCircle } from "lucide-react";
import CellularBackground from "@/components/CellularBackground";
import GlassCard from "@/components/GlassCard";
import MedicalButton from "@/components/MedicalButton";
import { useToast } from "@/hooks/use-toast";
import { apiCall, isLoggedIn } from "@/lib/api";

interface DiagnosisHistoryItem {
  diagnosis_id: string;
  region?: string;
  disease_name: string;
  severity: string;
  stage: string | null;
  confidence_disease: number;
  confidence_severity: number;
  confidence_stage?: number;
  original_image_url: string;
  created_at: string;
  status: string;
}

interface DiagnosisHistoryResponse {
  total_diagnoses: number;
  diagnoses: DiagnosisHistoryItem[];
}

const History = () => {
  const navigate = useNavigate();
  const { toast } = useToast();
  const [diagnoses, setDiagnoses] = useState<DiagnosisHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [clearingAll, setClearingAll] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if user is logged in
    if (!isLoggedIn()) {
      navigate("/login");
      return;
    }

    const fetchHistory = async () => {
      try {
        setLoading(true);
        setError(null);
        console.log("[HISTORY] Fetching diagnosis history...");
        const response = await apiCall<DiagnosisHistoryResponse>(
          "GET",
          "/diagnosis/history"
        );
        console.log("[HISTORY] Success, loaded", response.diagnoses?.length || 0, "diagnoses");
        setDiagnoses(response.diagnoses || []);
      } catch (error) {
        const errorMsg = error instanceof Error ? error.message : "Failed to load diagnosis history";
        console.error("[HISTORY] Error:", errorMsg);
        setError(errorMsg);
        toast({
          title: "Error",
          description: errorMsg,
          variant: "destructive"
        });
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, [navigate, toast]);

  const handleDeleteDiagnosis = async (diagnosisId: string) => {
    if (!window.confirm("Are you sure you want to delete this diagnosis?")) {
      return;
    }

    try {
      setDeleting(diagnosisId);
      console.log("[HISTORY] Deleting diagnosis:", diagnosisId);
      await apiCall("DELETE", `/diagnosis/history/${diagnosisId}`);
      setDiagnoses((prev) => prev.filter((d) => d.diagnosis_id !== diagnosisId));
      toast({
        title: "Success",
        description: "Diagnosis deleted successfully"
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Failed to delete diagnosis";
      console.error("[HISTORY] Delete error:", errorMsg);
      toast({
        title: "Error",
        description: errorMsg,
        variant: "destructive"
      });
    } finally {
      setDeleting(null);
    }
  };

  const handleClearAllHistory = async () => {
    // Double confirmation for destructive action
    if (!window.confirm("Are you sure you want to delete ALL diagnoses? This cannot be undone.")) {
      return;
    }
    
    if (!window.confirm("This will permanently delete all your diagnosis history and associated images. Continue?")) {
      return;
    }

    try {
      setClearingAll(true);
      console.log("[HISTORY] Clearing all diagnosis history...");
      await apiCall("DELETE", "/diagnosis/history");
      setDiagnoses([]);
      setError(null);
      toast({
        title: "Success",
        description: "All diagnosis history cleared successfully"
      });
    } catch (error) {
      const errorMsg = error instanceof Error ? error.message : "Failed to clear history";
      console.error("[HISTORY] Clear error:", errorMsg);
      toast({
        title: "Error",
        description: errorMsg,
        variant: "destructive"
      });
    } finally {
      setClearingAll(false);
    }
  };

  if (loading) {
    return (
      <CellularBackground>
        <div className="min-h-screen flex items-center justify-center">
          <GlassCard variant="bordered" className="p-8">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
              <p className="text-muted-foreground">Loading diagnosis history...</p>
            </div>
          </GlassCard>
        </div>
      </CellularBackground>
    );
  }

  return (
    <CellularBackground>
      <div className="min-h-screen flex flex-col">
        {/* Header */}
        <header className="p-4 flex items-center justify-between">
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
        <main className="flex-1 px-4 pb-8">
          <div className="max-w-6xl mx-auto">
            <div className="flex items-center justify-between mb-8">
              <h1 className="text-3xl font-bold text-primary">Diagnosis History</h1>
              {diagnoses.length > 0 && (
                <MedicalButton
                  variant="danger"
                  size="sm"
                  onClick={handleClearAllHistory}
                  disabled={clearingAll || loading}
                  className="ml-4"
                >
                  {clearingAll ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Clearing...
                    </>
                  ) : (
                    <>
                      <Trash2 className="w-4 h-4 mr-2" />
                      Remove All History
                    </>
                  )}
                </MedicalButton>
              )}
            </div>

            {/* Error Alert */}
            {error && (
              <div className="mb-6 p-4 bg-red-900/20 border border-red-500/50 rounded-lg flex items-start gap-3">
                <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-red-400 mb-1">Connection Error</h3>
                  <p className="text-sm text-red-300">{error}</p>
                  <p className="text-xs text-red-300 mt-2">
                    Make sure the backend is running at localhost:8000 or use ngrok for hybrid deployment
                  </p>
                  <MedicalButton
                    variant="ghost"
                    size="sm"
                    onClick={() => window.location.reload()}
                    className="mt-2"
                  >
                    Retry
                  </MedicalButton>
                </div>
              </div>
            )}

            {diagnoses.length === 0 ? (
              <GlassCard variant="bordered" className="text-center py-16">
                <div className="flex flex-col items-center gap-4">
                  <div className="w-20 h-20 rounded-full bg-muted/30 flex items-center justify-center">
                    <ImageIcon className="w-10 h-10 text-muted-foreground" />
                  </div>
                  <h2 className="text-xl font-semibold text-foreground">
                    No diagnosis history yet
                  </h2>
                  <p className="text-muted-foreground text-sm max-w-md">
                    Your diagnosis history will appear here once you analyze
                    microscopic images. Start by uploading an image to perform disease diagnosis.
                  </p>
                  <MedicalButton
                    variant="primary"
                    onClick={() => navigate("/upload")}
                    className="mt-4"
                  >
                    Upload Image for Diagnosis
                  </MedicalButton>
                </div>
              </GlassCard>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                {diagnoses.map((diagnosis) => (
                  <GlassCard
                    key={diagnosis.diagnosis_id}
                    variant="bordered"
                    className="relative overflow-hidden hover:border-primary/50 transition-colors"
                  >
                    {/* Image Preview */}
                    <div className="mb-4 bg-muted/30 rounded-lg overflow-hidden h-48 flex items-center justify-center">
                      {diagnosis.original_image_url ? (
                        <img
                          src={diagnosis.original_image_url}
                          alt={diagnosis.disease_name}
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <ImageIcon className="w-12 h-12 text-muted-foreground" />
                      )}
                    </div>

                    {/* Diagnosis Info */}
                    <div className="space-y-4">
                      {/* Region */}
                      <div>
                        <div className="text-xs font-semibold text-blue-400 uppercase tracking-wider mb-1">
                          Region
                        </div>
                        <div className="text-lg font-bold text-primary">
                          {diagnosis.region || diagnosis.disease_name}
                        </div>
                      </div>

                      {/* Status */}
                      <div>
                        <div
                          className={`text-xs font-semibold uppercase tracking-wider mb-1 ${
                            diagnosis.status === "abnormal"
                              ? "text-warning"
                              : "text-success"
                          }`}
                        >
                          Status
                        </div>
                        <div
                          className={`text-lg font-bold ${
                            diagnosis.status === "abnormal"
                              ? "text-warning"
                              : "text-success"
                          }`}
                        >
                          {diagnosis.severity}
                        </div>
                      </div>

                      {/* Stage (if abnormal) */}
                      {diagnosis.status === "abnormal" && diagnosis.stage && (
                        <div>
                          <div className="text-xs font-semibold text-orange-400 uppercase tracking-wider mb-1">
                            Stage/Type
                          </div>
                          <div className="text-sm font-semibold text-primary">
                            {diagnosis.stage}
                          </div>
                        </div>
                      )}

                      {/* Confidence Scores */}
                      <div className="pt-2 border-t border-border/30 space-y-2">
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">Disease Confidence:</span>
                          <span className="font-semibold text-primary">
                            {(diagnosis.confidence_disease * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-xs">
                          <span className="text-muted-foreground">Date:</span>
                          <span className="font-semibold text-primary">
                            {new Date(diagnosis.created_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>

                      {/* Delete Button */}
                      <MedicalButton
                        variant="danger"
                        size="sm"
                        className="w-full"
                        onClick={() => handleDeleteDiagnosis(diagnosis.diagnosis_id)}
                        disabled={deleting === diagnosis.diagnosis_id}
                      >
                        {deleting === diagnosis.diagnosis_id ? (
                          <>
                            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                            Deleting...
                          </>
                        ) : (
                          <>
                            <Trash2 className="w-4 h-4 mr-2" />
                            Delete
                          </>
                        )}
                      </MedicalButton>
                    </div>
                  </GlassCard>
                ))}
              </div>
            )}
          </div>
        </main>
      </div>
    </CellularBackground>
  );
};

export default History;