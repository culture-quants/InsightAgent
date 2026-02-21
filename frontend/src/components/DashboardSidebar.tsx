import { useState } from "react";
import { NavLink } from "react-router-dom";
import {
  LayoutDashboard,
  Search,
  MessageSquare,
  FlaskConical,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { cn } from "@/lib/utils";

const navItems = [
  { title: "Dashboard", path: "/", icon: LayoutDashboard },
  { title: "Clusters", path: "/clusters", icon: Search },
  { title: "News Chat", path: "/news-chat", icon: MessageSquare },
  { title: "Scenarios", path: "/scenarios", icon: FlaskConical },
];

export function DashboardSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside
      className={cn(
        "flex flex-col border-r border-border bg-sidebar transition-all duration-200",
        collapsed ? "w-[56px]" : "w-[200px]"
      )}
    >
      {/* Logo */}
      <div className="flex h-14 items-center gap-2.5 border-b border-border px-4">
        <div className="flex h-6 w-6 items-center justify-center rounded bg-foreground shrink-0">
          <span className="text-[9px] font-bold text-background tracking-tight">IA</span>
        </div>
        {!collapsed && (
          <span className="text-[13px] font-semibold tracking-tight text-foreground">
            InsightAgent
          </span>
        )}
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-1 px-2.5 pt-6">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.path === "/"}
            className={({ isActive }) =>
              cn(
                "flex items-center gap-3 rounded px-3 py-2.5 text-[12px] font-medium transition-colors",
                isActive
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:bg-accent hover:text-foreground"
              )
            }
          >
            <item.icon className="h-[15px] w-[15px] shrink-0" strokeWidth={1.75} />
            {!collapsed && <span>{item.title}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Collapse */}
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex h-12 items-center justify-center border-t border-border text-muted-foreground hover:text-foreground transition-colors"
      >
        {collapsed ? <ChevronRight className="h-[14px] w-[14px]" strokeWidth={1.75} /> : <ChevronLeft className="h-[14px] w-[14px]" strokeWidth={1.75} />}
      </button>
    </aside>
  );
}
