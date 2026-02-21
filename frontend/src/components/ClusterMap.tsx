import { useState } from "react";
import { ScatterChart, Scatter, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { mockClusters } from "@/lib/mock-data";

const lifecycleColors: Record<string, string> = {
  Emerging: "hsl(210, 30%, 48%)",
  Trending: "hsl(152, 28%, 40%)",
  Stable: "hsl(215, 25%, 50%)",
  Declining: "hsl(30, 35%, 46%)",
  Dormant: "hsl(220, 6%, 55%)",
};

interface ClusterMapProps {
  onClusterClick?: (cluster: typeof mockClusters[0]) => void;
}

export function ClusterMap({ onClusterClick }: ClusterMapProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null);

  return (
    <div className="glass-card px-6 py-6">
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h3 className="text-[11px] font-semibold text-foreground uppercase tracking-[0.1em]">Cluster Map</h3>
          <p className="mt-1 text-[11px] text-muted-foreground">{mockClusters.length} clusters · UMAP projection</p>
        </div>
        <div className="flex items-center gap-5">
          {Object.entries(lifecycleColors).map(([label, color]) => (
            <div key={label} className="flex items-center gap-1.5">
              <div className="h-[6px] w-[6px] rounded-full" style={{ backgroundColor: color }} />
              <span className="text-[10px] text-muted-foreground">{label}</span>
            </div>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 10, left: 10 }}>
          <XAxis type="number" dataKey="x" hide />
          <YAxis type="number" dataKey="y" hide />
          <Tooltip
            cursor={false}
            content={({ payload }) => {
              if (!payload?.length) return null;
              const d = payload[0].payload;
              return (
                <div className="rounded border border-border bg-card px-3 py-2.5 shadow-sm">
                  <p className="text-[11px] font-medium text-foreground">{d.cluster_label}</p>
                  <p className="mt-0.5 text-[10px] text-muted-foreground">{d.keywords.join(", ")}</p>
                  <p className="mt-1 text-[10px] text-muted-foreground">
                    {d.size} posts · {d.engagement.toLocaleString()} eng.
                  </p>
                </div>
              );
            }}
          />
          <Scatter data={mockClusters} onClick={(data) => onClusterClick?.(data)} className="cursor-pointer">
            {mockClusters.map((c) => (
              <Cell
                key={c.id}
                fill={lifecycleColors[c.lifecycle]}
                fillOpacity={hoveredId === c.id ? 0.85 : 0.4}
                r={Math.sqrt(c.size) / 5 + 3}
                onMouseEnter={() => setHoveredId(c.id)}
                onMouseLeave={() => setHoveredId(null)}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
