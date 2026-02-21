import { useState } from "react";
import { Calendar, ChevronDown } from "lucide-react";
import { cn } from "@/lib/utils";

const timeWindows = ["Daily", "Weekly", "Monthly"] as const;

interface TopBarProps {
  title?: string;
  onTimeWindowChange?: (window: string) => void;
}

export function TopBar({ title = "Executive Dashboard", onTimeWindowChange }: TopBarProps) {
  const [activeWindow, setActiveWindow] = useState<string>("Monthly");

  const handleChange = (w: string) => {
    setActiveWindow(w);
    onTimeWindowChange?.(w);
  };

  return (
    <header className="flex h-14 items-center justify-between border-b border-border px-8 bg-background">
      <h1 className="text-[11px] font-medium text-muted-foreground uppercase tracking-[0.12em]">{title}</h1>

      <div className="flex items-center gap-3">
        <div className="flex items-center rounded border border-border overflow-hidden">
          {timeWindows.map((w, i) => (
            <button
              key={w}
              onClick={() => handleChange(w)}
              className={cn(
                "px-3.5 py-1.5 text-[11px] font-medium transition-all",
                i > 0 && "border-l border-border",
                activeWindow === w
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {w}
            </button>
          ))}
        </div>

        <button className="flex items-center gap-2 rounded border border-border px-3 py-1.5 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
          <Calendar className="h-[13px] w-[13px]" strokeWidth={1.75} />
          <span>Jul 2024 – Jan 2025</span>
          <ChevronDown className="h-[11px] w-[11px]" strokeWidth={1.75} />
        </button>
      </div>
    </header>
  );
}
