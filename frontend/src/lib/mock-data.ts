// ── Mock data for InsightAgent dashboard ──

export const mockKPIs = {
  totalPosts: 48_762,
  totalEngagement: 2_847_391,
  activeClusters: 47,
  highVolatilityTopics: 8,
  avgSentiment: 0.64,
  postsChange: 12.4,
  engagementChange: -3.2,
  clustersChange: 5,
  volatilityChange: 23.1,
  emergingCount: 12,
  decliningCount: 6,
  avgVolatility: 0.42,
  eventSpikes: 8,
};

const clusterLabels = [
  { keywords: ["climate", "policy", "vote", "carbon"], label: "Climate Policy" },
  { keywords: ["AI", "regulation", "safety", "tech"], label: "AI Regulation" },
  { keywords: ["crypto", "bitcoin", "market", "trading"], label: "Crypto Markets" },
  { keywords: ["health", "vaccine", "study", "FDA"], label: "Health & Pharma" },
  { keywords: ["election", "polls", "debate", "campaign"], label: "Election Cycle" },
  { keywords: ["remote", "work", "office", "hybrid"], label: "Future of Work" },
  { keywords: ["inflation", "rates", "fed", "economy"], label: "Macro Economy" },
  { keywords: ["streaming", "content", "netflix", "media"], label: "Media & Streaming" },
  { keywords: ["space", "NASA", "launch", "mars"], label: "Space & Aero" },
  { keywords: ["education", "student", "loans", "college"], label: "Education Reform" },
];

export const mockClusters = Array.from({ length: 50 }, (_, i) => {
  const template = clusterLabels[i % 10];
  return {
    id: `cluster-${i}`,
    x: Math.random() * 100 - 50,
    y: Math.random() * 100 - 50,
    size: Math.floor(Math.random() * 800) + 100,
    keywords: template.keywords,
    cluster_label: template.label,
    engagement: Math.floor(Math.random() * 50000) + 5000,
    lifecycle: (["Emerging", "Trending", "Stable", "Declining", "Dormant"] as const)[Math.floor(Math.random() * 5)],
    momentum: Math.random() * 200 - 50,
    market_share: parseFloat((Math.random() * 8 + 0.5).toFixed(2)),
    volume_pct_change: parseFloat((Math.random() * 60 - 20).toFixed(1)),
    volume_volatility: parseFloat((Math.random() * 1).toFixed(3)),
    anomaly_score: parseFloat((Math.random() * 5).toFixed(2)),
    platforms: {
      x: Math.floor(Math.random() * 20000),
      facebook: Math.floor(Math.random() * 15000),
      linkedin: Math.floor(Math.random() * 10000),
    },
  };
});

// Monthly snapshots for trend chart
const timeWindows = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12", "2025-01"];
export const mockClusterSnapshots = mockClusters.slice(0, 15).flatMap((c) =>
  timeWindows.map((tw) => ({
    cluster_id: c.id,
    cluster_label: c.cluster_label,
    time_window: tw,
    post_count: Math.floor(Math.random() * 800) + 100,
    market_share: parseFloat((Math.random() * 8 + 0.5).toFixed(2)),
    momentum: parseFloat((Math.random() * 4 - 1).toFixed(2)),
    volatility: parseFloat((Math.random() * 1).toFixed(3)),
  }))
);

export const mockTimeSeriesData = Array.from({ length: 30 }, (_, i) => ({
  date: new Date(2025, 0, i + 1).toISOString().slice(0, 10),
  posts: Math.floor(Math.random() * 2000) + 800,
  engagement: Math.floor(Math.random() * 100000) + 40000,
  clusters: Math.floor(Math.random() * 10) + 40,
  anomaly: i === 12 || i === 23,
}));

export const mockHeatmapData = Array.from({ length: 10 }, (_, clusterIdx) => ({
  cluster: mockClusters[clusterIdx].keywords.slice(0, 2).join(", "),
  data: Array.from({ length: 7 }, (_, dayIdx) => ({
    day: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][dayIdx],
    value: Math.floor(Math.random() * 100),
  })),
}));

export const mockLifecycleData = mockClusters.slice(0, 15).map((c) => ({
  ...c,
  sparkline: Array.from({ length: 7 }, () => Math.floor(Math.random() * 100)),
  growthRate: (Math.random() * 40 - 10).toFixed(1),
}));

export const mockEventTimeline = mockTimeSeriesData.map((d) => ({
  ...d,
  events: d.anomaly
    ? [{ type: "spike" as const, label: "Engagement surge detected", sigma: (Math.random() * 2 + 2).toFixed(1) }]
    : [],
}));

// Temporal events for event timeline panel
export const mockTemporalEvents = [
  { date: "2025-01-05", cluster_label: "AI Regulation", event_type: "spike", description: "EU AI Act enforcement discussion surge", sigma: 3.2 },
  { date: "2025-01-08", cluster_label: "Crypto Markets", event_type: "spike", description: "Bitcoin ETF approval speculation", sigma: 4.1 },
  { date: "2025-01-12", cluster_label: "Climate Policy", event_type: "spike", description: "Carbon tax vote drives 340% engagement", sigma: 3.8 },
  { date: "2025-01-18", cluster_label: "Health & Pharma", event_type: "drop", description: "FDA approval delay for key drug", sigma: 2.4 },
  { date: "2025-01-22", cluster_label: "Election Cycle", event_type: "spike", description: "Primary debate night social surge", sigma: 2.9 },
  { date: "2025-01-25", cluster_label: "Macro Economy", event_type: "spike", description: "Fed rate decision triggers volatility", sigma: 3.5 },
];

export interface PredictionResult {
  id: string;
  event_text: string;
  predicted_direction: "up" | "down" | "flat";
  confidence: number;
  predicted_volatility: number;
  risk_band: "low" | "medium" | "high";
  method: "heuristic" | "trained";
  activated_theme_count: number;
  activated_themes: Array<{
    theme_id: number;
    theme_name: string;
    score: number;
    weight: number;
    direction_label: "up" | "down" | "flat";
  }>;
  timestamp: string;
  class_probabilities?: Record<string, number>;
  trend_data?: Array<{
    cluster_id: number;
    cluster_label: string;
    data: Array<{
      time_window: string;
      post_count: number;
      market_share: number;
      momentum: number;
      volatility: number;
    }>;
  }>;
  analysis?: string;
}

export function generateMockPrediction(eventText: string): PredictionResult {
  const directions: Array<"up" | "down" | "flat"> = ["up", "down", "flat"];
  const dir = directions[Math.floor(Math.random() * 3)];
  const themes = [
    { theme_id: 0, theme_name: "AI Product Launch", score: 3.21, weight: 0.44, direction_label: "up" as const },
    { theme_id: 1, theme_name: "Regulatory Shift", score: 2.88, weight: 0.31, direction_label: "down" as const },
    { theme_id: 2, theme_name: "Market Sentiment", score: 1.95, weight: 0.22, direction_label: dir },
    { theme_id: 3, theme_name: "Consumer Adoption", score: 1.42, weight: 0.18, direction_label: "up" as const },
    { theme_id: 4, theme_name: "Supply Chain", score: 0.98, weight: 0.12, direction_label: "flat" as const },
  ];
  const count = Math.floor(Math.random() * 3) + 2;
  return {
    id: crypto.randomUUID(),
    event_text: eventText,
    predicted_direction: dir,
    confidence: parseFloat((Math.random() * 0.4 + 0.55).toFixed(2)),
    predicted_volatility: parseFloat((Math.random() * 0.8 + 0.1).toFixed(2)),
    risk_band: (["low", "medium", "high"] as const)[Math.floor(Math.random() * 3)],
    method: Math.random() > 0.3 ? "trained" : "heuristic",
    activated_theme_count: count,
    activated_themes: themes.slice(0, count),
    timestamp: new Date().toISOString(),
  };
}

// Lifecycle distribution
export const mockLifecycleDistribution = [
  { state: "Emerging", count: 12 },
  { state: "Trending", count: 9 },
  { state: "Stable", count: 15 },
  { state: "Declining", count: 6 },
  { state: "Dormant", count: 5 },
];
