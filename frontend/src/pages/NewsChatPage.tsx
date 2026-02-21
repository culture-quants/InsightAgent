import { useState, useRef, useEffect } from "react";
import {
  Send, ArrowUpRight, ArrowDownRight, Minus, Clock, TrendingUp,
  BarChart3, AlertTriangle, Loader2, ChevronDown, ChevronUp,
} from "lucide-react";
import {
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell, ReferenceLine,
} from "recharts";
import { DashboardLayout } from "@/components/DashboardLayout";
import { MarkdownRenderer } from "@/components/MarkdownRenderer";
import { predictEvent, chatAnalysis, type PredictionResponse } from "@/lib/api";
import { cn } from "@/lib/utils";

const directionConfig = {
  up: { icon: ArrowUpRight, label: "Upward", className: "text-direction-up" },
  down: { icon: ArrowDownRight, label: "Downward", className: "text-direction-down" },
  flat: { icon: Minus, label: "Flat", className: "text-direction-flat" },
};

const riskColors: Record<string, string> = {
  low: "text-viz-emerald",
  medium: "text-viz-amber",
  high: "text-viz-rose",
};

const riskBg: Record<string, string> = {
  low: "bg-viz-emerald/10 border-viz-emerald/20",
  medium: "bg-viz-amber/10 border-viz-amber/20",
  high: "bg-viz-rose/10 border-viz-rose/20",
};

const tooltipStyle = {
  background: "hsl(0, 0%, 100%)",
  border: "1px solid hsl(220, 8%, 90%)",
  borderRadius: "4px",
  fontSize: "11px",
  boxShadow: "0 1px 4px rgba(0,0,0,0.06)",
};

const palette = [
  "hsl(210, 30%, 48%)", "hsl(152, 28%, 40%)", "hsl(255, 18%, 50%)",
  "hsl(30, 35%, 46%)", "hsl(340, 30%, 48%)",
];

interface ChatEntry {
  id: string;
  event_text: string;
  prediction: PredictionResponse;
  analysis: string | null;
  timestamp: string;
}

/* ── Charts ────────────────────────────────────────── */

function ClassProbChart({ probs }: { probs: Record<string, number> }) {
  const data = [
    { label: "Down", value: (probs.down || 0) * 100, fill: "hsl(0, 45%, 55%)" },
    { label: "Flat", value: (probs.flat || 0) * 100, fill: "hsl(220, 8%, 46%)" },
    { label: "Up", value: (probs.up || 0) * 100, fill: "hsl(152, 28%, 40%)" },
  ];
  return (
    <div className="mt-4">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3">
        Class Probabilities
      </p>
      <ResponsiveContainer width="100%" height={80}>
        <BarChart data={data} layout="vertical" margin={{ top: 0, right: 10, bottom: 0, left: 32 }}>
          <XAxis
            type="number" domain={[0, 100]}
            tick={{ fontSize: 9, fill: "hsl(220, 8%, 46%)" }}
            axisLine={false} tickLine={false}
            tickFormatter={(v: number) => `${v}%`}
          />
          <YAxis
            type="category" dataKey="label"
            tick={{ fontSize: 10, fill: "hsl(220, 8%, 46%)" }}
            axisLine={false} tickLine={false} width={30}
          />
          <Bar dataKey="value" radius={[0, 2, 2, 0]} barSize={14}>
            {data.map((d, i) => (
              <Cell key={i} fill={d.fill} fillOpacity={0.7} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function TrendChart({ trendData }: { trendData: PredictionResponse["trend_data"] }) {
  if (!trendData?.length) return null;

  const windows = [...new Set(trendData.flatMap((t) => t.data.map((d) => d.time_window)))].sort();
  const rows = windows.map((tw) => {
    const row: Record<string, string | number> = { time_window: tw.slice(0, 7) };
    trendData.forEach((t) => {
      const pt = t.data.find((d) => d.time_window === tw);
      if (pt) row[`c${t.cluster_id}`] = pt.post_count;
    });
    return row;
  });

  return (
    <div className="mt-5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-1.5">
        <TrendingUp className="h-[11px] w-[11px]" strokeWidth={1.75} />
        Historical Trends — Activated Clusters
      </p>
      <ResponsiveContainer width="100%" height={180}>
        <AreaChart data={rows} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
          <XAxis dataKey="time_window" tick={{ fontSize: 9, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} interval="preserveStartEnd" />
          <YAxis tick={{ fontSize: 9, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} width={36} />
          <Tooltip contentStyle={tooltipStyle} />
          {trendData.map((t, i) => (
            <Area key={t.cluster_id} type="monotone" dataKey={`c${t.cluster_id}`}
              name={t.cluster_label}
              stroke={palette[i % palette.length]} strokeWidth={1.5}
              fill={palette[i % palette.length]} fillOpacity={0.06}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-3 mt-2">
        {trendData.map((t, i) => (
          <span key={t.cluster_id} className="flex items-center gap-1.5 text-[9px] text-muted-foreground">
            <span className="h-[6px] w-[6px] rounded-full" style={{ backgroundColor: palette[i % palette.length] }} />
            {t.cluster_label}
          </span>
        ))}
      </div>
    </div>
  );
}

function MomentumChart({ trendData }: { trendData: PredictionResponse["trend_data"] }) {
  if (!trendData?.length) return null;
  const rows = trendData.map((t) => {
    const last = t.data[t.data.length - 1];
    return {
      label: t.cluster_label.length > 16 ? t.cluster_label.slice(0, 14) + "\u2026" : t.cluster_label,
      volatility: last?.volatility ?? 0,
      momentum: last?.momentum ?? 0,
    };
  });

  return (
    <div className="mt-5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3 flex items-center gap-1.5">
        <BarChart3 className="h-[11px] w-[11px]" strokeWidth={1.75} />
        Latest Momentum & Volatility
      </p>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={rows} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
          <XAxis dataKey="label" tick={{ fontSize: 8, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} />
          <YAxis tick={{ fontSize: 9, fill: "hsl(220, 8%, 46%)" }} axisLine={false} tickLine={false} width={30} />
          <Tooltip contentStyle={tooltipStyle} />
          <ReferenceLine y={0} stroke="hsl(220, 8%, 80%)" strokeDasharray="3 3" />
          <Bar dataKey="momentum" name="Momentum" fill="hsl(210, 30%, 48%)" fillOpacity={0.6} radius={[2, 2, 0, 0]} barSize={20} />
          <Bar dataKey="volatility" name="Volatility" fill="hsl(30, 35%, 46%)" fillOpacity={0.5} radius={[2, 2, 0, 0]} barSize={20} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Subcomponents ─────────────────────────────────── */

function AnalysisPanel({ analysis }: { analysis: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mt-5 rounded border border-border bg-accent/20">
      <button onClick={() => setOpen(!open)} className="w-full flex items-center justify-between px-5 py-3 text-left">
        <span className="text-[10px] text-muted-foreground uppercase tracking-widest font-medium">AI Analysis</span>
        {open ? <ChevronUp className="h-3 w-3 text-muted-foreground" /> : <ChevronDown className="h-3 w-3 text-muted-foreground" />}
      </button>
      {open && (
        <div className="px-5 pb-5">
          <MarkdownRenderer text={analysis} />
        </div>
      )}
    </div>
  );
}

function ThemeTable({ themes }: { themes: PredictionResponse["activated_themes"] }) {
  if (!themes?.length) return null;
  return (
    <div className="mt-5">
      <p className="text-[10px] text-muted-foreground uppercase tracking-widest mb-3">Activated Theme Details</p>
      <div className="overflow-x-auto">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="border-b border-border text-[8px] uppercase tracking-widest text-muted-foreground">
              <th className="px-3 py-2 text-left">Theme</th>
              <th className="px-3 py-2 text-right">Score</th>
              <th className="px-3 py-2 text-right">Weight</th>
              <th className="px-3 py-2 text-center">Dir</th>
              <th className="px-3 py-2 text-right">Vol</th>
              <th className="px-3 py-2 text-right">Mkt %</th>
              <th className="px-3 py-2 text-right">Mom</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border/50">
            {themes.map((t) => {
              const d = directionConfig[t.direction_label];
              const I = d.icon;
              return (
                <tr key={t.theme_id}>
                  <td className="px-3 py-2 text-foreground">{t.theme_name}</td>
                  <td className="px-3 py-2 text-right font-mono-data text-foreground">{t.score.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-mono-data text-muted-foreground">{(t.weight * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2 text-center">
                    <span className={cn("inline-flex items-center gap-1", d.className)}>
                      <I className="h-[10px] w-[10px]" strokeWidth={1.75} />
                      {d.label}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono-data text-muted-foreground">{t.avg_volatility.toFixed(2)}</td>
                  <td className="px-3 py-2 text-right font-mono-data text-muted-foreground">{(t.avg_market_share * 100).toFixed(2)}%</td>
                  <td className="px-3 py-2 text-right font-mono-data text-muted-foreground">{t.recent_momentum.toFixed(1)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Prediction Card ───────────────────────────────── */

function PredictionCard({ entry }: { entry: ChatEntry }) {
  const { prediction, analysis } = entry;
  const dir = directionConfig[prediction.predicted_direction];
  const DirIcon = dir.icon;

  return (
    <div className="glass-card px-6 py-6 animate-fade-in-up">
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className={cn("flex items-center justify-center h-10 w-10 rounded border border-border", dir.className)}>
            <DirIcon className="h-[18px] w-[18px]" strokeWidth={1.75} />
          </div>
          <div>
            <p className={cn("text-[14px] font-semibold", dir.className)}>{dir.label}</p>
            <p className="text-[10px] text-muted-foreground">
              {prediction.method} model{prediction.model_name ? ` \u00b7 ${prediction.model_name}` : ""}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className="font-mono-data text-[22px] font-semibold text-foreground leading-none">
            {(prediction.confidence * 100).toFixed(0)}%
          </p>
          <p className="text-[10px] text-muted-foreground mt-1">confidence</p>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3 mb-5">
        <div className="rounded border border-border px-4 py-3 text-center">
          <p className="text-[9px] text-muted-foreground uppercase tracking-widest">Volatility</p>
          <p className="mt-1.5 font-mono-data text-[14px] font-semibold text-foreground">
            {prediction.predicted_volatility.toFixed(2)}
          </p>
        </div>
        <div className={cn("rounded border px-4 py-3 text-center", riskBg[prediction.risk_band])}>
          <p className="text-[9px] text-muted-foreground uppercase tracking-widest">Risk</p>
          <p className={cn("mt-1.5 text-[14px] font-semibold capitalize", riskColors[prediction.risk_band])}>
            {prediction.risk_band}
          </p>
        </div>
        <div className="rounded border border-border px-4 py-3 text-center">
          <p className="text-[9px] text-muted-foreground uppercase tracking-widest">Themes</p>
          <p className="mt-1.5 font-mono-data text-[14px] font-semibold text-foreground">
            {prediction.activated_theme_count}
          </p>
        </div>
        <div className="rounded border border-border px-4 py-3 text-center">
          <p className="text-[9px] text-muted-foreground uppercase tracking-widest">Threshold</p>
          <p className="mt-1.5 font-mono-data text-[14px] font-semibold text-foreground">
            {prediction.activation_threshold}
          </p>
        </div>
      </div>

      {prediction.class_probabilities && Object.keys(prediction.class_probabilities).length > 0 && (
        <ClassProbChart probs={prediction.class_probabilities} />
      )}

      <ThemeTable themes={prediction.activated_themes} />
      <TrendChart trendData={prediction.trend_data} />
      <MomentumChart trendData={prediction.trend_data} />

      {analysis && <AnalysisPanel analysis={analysis} />}

      {!analysis && prediction.activated_themes?.length > 0 && (
        <div className="mt-5 rounded border border-border bg-accent/30 px-5 py-4">
          <p className="text-[11px] leading-[1.7] text-muted-foreground">
            Predicted{" "}
            <span className={cn("font-medium", dir.className)}>{dir.label.toLowerCase()}</span>
            {" "}from{" "}
            {prediction.activated_themes.slice(0, 2).map((t) => t.theme_name).join(" and ")}
            {" "}(combined score{" "}
            {prediction.activated_themes.reduce((s, t) => s + t.score, 0).toFixed(2)}).
            Risk is{" "}
            <span className={cn("font-medium", riskColors[prediction.risk_band])}>{prediction.risk_band}</span>
            {" "}with volatility {prediction.predicted_volatility.toFixed(2)}.
          </p>
        </div>
      )}

      {prediction.activated_theme_count === 0 && (
        <div className="mt-5 rounded border border-border bg-accent/30 px-5 py-4 flex items-center gap-3">
          <AlertTriangle className="h-[14px] w-[14px] text-viz-amber shrink-0" strokeWidth={1.75} />
          <p className="text-[11px] leading-[1.7] text-muted-foreground">
            No themes activated above threshold ({prediction.activation_threshold}).
            The event may not match known patterns. Try more specific keywords.
          </p>
        </div>
      )}

      <p className="mt-4 text-[9px] text-muted-foreground flex items-center gap-1.5">
        <Clock className="h-[10px] w-[10px]" strokeWidth={1.75} />
        {new Date(entry.timestamp).toLocaleTimeString()}
      </p>
    </div>
  );
}

/* ── Page ──────────────────────────────────────────── */

const NewsChatPage = () => {
  const [input, setInput] = useState("");
  const [history, setHistory] = useState<ChatEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [stage, setStage] = useState<"predicting" | "analyzing" | "">("");
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [history, loading]);

  const handleSubmit = async () => {
    if (!input.trim() || loading) return;
    setLoading(true);
    setError(null);
    const text = input.trim();
    setInput("");

    try {
      setStage("predicting");
      const prediction = await predictEvent(text);

      const entryId = crypto.randomUUID();
      const entry: ChatEntry = {
        id: entryId,
        event_text: text,
        prediction: { ...prediction, id: entryId, timestamp: new Date().toISOString() },
        analysis: null,
        timestamp: new Date().toISOString(),
      };
      setHistory((prev) => [...prev, entry]);

      setStage("analyzing");
      try {
        const prev = history.map((h) => h.prediction);
        const chat = await chatAnalysis(text, prediction, prev);
        setHistory((h) =>
          h.map((e) => (e.id === entryId ? { ...e, analysis: chat.analysis } : e))
        );
      } catch {
        setHistory((h) =>
          h.map((e) =>
            e.id === entryId
              ? { ...e, analysis: "AI analysis unavailable \u2014 the prediction data above is still valid." }
              : e
          )
        );
      }
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Prediction failed. Is the backend running on port 8000?"
      );
    } finally {
      setLoading(false);
      setStage("");
    }
  };

  return (
    <DashboardLayout title="News Impact Chat">
      <div className="flex flex-col h-[calc(100vh-8rem)] max-w-4xl mx-auto">
        {/* Scroll area */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto scrollbar-thin space-y-6 pb-6">
          {history.length === 0 && !loading && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center max-w-md">
                <div className="h-12 w-12 rounded border border-border flex items-center justify-center mx-auto mb-4">
                  <TrendingUp className="h-5 w-5 text-muted-foreground" strokeWidth={1.5} />
                </div>
                <p className="text-[13px] text-foreground font-medium">Predict News Impact</p>
                <p className="text-[11px] text-muted-foreground mt-2 leading-relaxed">
                  Enter a news event or headline to predict its impact on social media trends.
                  The ML pipeline analyzes theme activations and Gemini provides detailed trend analysis.
                </p>
                <div className="mt-5 space-y-2">
                  {[
                    "Google launches new Gemini robotics model",
                    "Fed announces surprise rate hike",
                    "AlphaFold 3 solves protein interactions",
                  ].map((ex) => (
                    <button
                      key={ex}
                      onClick={() => setInput(ex)}
                      className="block w-full text-left px-4 py-2.5 rounded border border-border text-[11px] text-muted-foreground hover:text-foreground hover:bg-accent/50 transition-colors"
                    >
                      {ex}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          )}

          {history.map((entry) => (
            <div key={entry.id} className="space-y-4">
              <div className="flex justify-end">
                <div className="rounded border border-border bg-accent/30 px-5 py-3 max-w-lg">
                  <p className="text-[12px] text-foreground">{entry.event_text}</p>
                </div>
              </div>
              <PredictionCard entry={entry} />
            </div>
          ))}

          {loading && (
            <div className="glass-card px-6 py-6">
              <div className="flex items-center gap-3">
                <Loader2 className="h-5 w-5 text-muted-foreground animate-spin" strokeWidth={1.75} />
                <div>
                  <p className="text-[12px] text-foreground font-medium">
                    {stage === "predicting" ? "Running prediction pipeline\u2026" : "Generating AI analysis\u2026"}
                  </p>
                  <p className="text-[10px] text-muted-foreground mt-0.5">
                    {stage === "predicting"
                      ? "Scoring themes, matching clusters, running trained model"
                      : "Gemini is analyzing the prediction and generating trend insights"}
                  </p>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="rounded border border-viz-rose/30 bg-viz-rose/5 px-5 py-4">
              <p className="text-[11px] text-viz-rose font-medium">Error</p>
              <p className="text-[10px] text-muted-foreground mt-1">{error}</p>
            </div>
          )}
        </div>

        {/* Input bar */}
        <div className="pt-5 border-t border-border">
          <div className="flex items-center gap-3 rounded border border-border bg-card px-5 py-3">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              placeholder="Describe a news event or headline\u2026"
              className="flex-1 bg-transparent text-[12px] text-foreground placeholder:text-muted-foreground focus:outline-none"
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || loading}
              className={cn(
                "flex items-center justify-center h-8 w-8 rounded transition-colors",
                input.trim() && !loading
                  ? "bg-foreground text-background"
                  : "bg-accent text-muted-foreground"
              )}
            >
              {loading
                ? <Loader2 className="h-[14px] w-[14px] animate-spin" strokeWidth={1.75} />
                : <Send className="h-[14px] w-[14px]" strokeWidth={1.75} />}
            </button>
          </div>
          <p className="mt-3 text-[9px] text-muted-foreground text-center">
            Predictions via trained ML model + Gemini LLM analysis
            {" \u00b7 "}{history.length} prediction{history.length !== 1 ? "s" : ""} this session
          </p>
        </div>
      </div>
    </DashboardLayout>
  );
};

export default NewsChatPage;
