import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { fetchKPIs, type KPIs } from "@/lib/api";
import { mockKPIs } from "@/lib/mock-data";

const fallback: KPIs = {
  activeClusters: mockKPIs.activeClusters,
  emergingCount: mockKPIs.emergingCount,
  decliningCount: mockKPIs.decliningCount,
  avgVolatility: mockKPIs.avgVolatility,
  eventSpikes: mockKPIs.eventSpikes,
};

export function KPICards() {
  const [data, setData] = useState<KPIs>(fallback);

  useEffect(() => {
    fetchKPIs().then(setData).catch(() => {});
  }, []);

  const kpis = [
    { label: "Active Clusters", value: data.activeClusters.toString(), sub: `${data.emergingCount} emerging` },
    { label: "Emerging", value: data.emergingCount.toString(), sub: "lifecycle state" },
    { label: "Declining", value: data.decliningCount.toString(), sub: "lifecycle state" },
    { label: "Avg Volatility", value: data.avgVolatility.toFixed(2), sub: "across clusters" },
    { label: "Event Spikes", value: data.eventSpikes.toString(), sub: ">2\u03C3 anomalies" },
  ];

  return (
    <div className="grid grid-cols-5 gap-px rounded-md border border-border overflow-hidden">
      {kpis.map((kpi, i) => (
        <div key={kpi.label} className={cn("bg-card px-5 py-5", i > 0 && "border-l border-border")}>
          <p className="text-[10px] font-medium text-muted-foreground uppercase tracking-[0.1em]">{kpi.label}</p>
          <p className="mt-3 text-[22px] font-semibold font-mono-data text-foreground tracking-tight leading-none">{kpi.value}</p>
          <p className="mt-2 text-[10px] text-muted-foreground">{kpi.sub}</p>
        </div>
      ))}
    </div>
  );
}
