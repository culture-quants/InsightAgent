import React from "react";

function InlineFormat({ text }: { text: string }) {
  const parts: React.ReactNode[] = [];
  const matches = Array.from(text.matchAll(/\*\*(.+?)\*\*|\*(.+?)\*|`(.+?)`/g));
  let lastIndex = 0;

  for (const match of matches) {
    const idx = match.index ?? 0;
    if (idx > lastIndex) {
      parts.push(text.slice(lastIndex, idx));
    }
    if (match[1]) {
      parts.push(<strong key={idx} className="font-semibold text-foreground">{match[1]}</strong>);
    } else if (match[2]) {
      parts.push(<em key={idx}>{match[2]}</em>);
    } else if (match[3]) {
      parts.push(<code key={idx} className="px-1 py-0.5 rounded bg-accent text-[10px] font-mono text-foreground">{match[3]}</code>);
    }
    lastIndex = idx + match[0].length;
  }
  if (lastIndex < text.length) {
    parts.push(text.slice(lastIndex));
  }
  return <>{parts}</>;
}

export function MarkdownRenderer({ text }: { text: string }) {
  const lines = text.split("\n");
  const elements: React.ReactNode[] = [];
  let listItems: string[] = [];

  const flushList = () => {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`ul-${elements.length}`} className="list-disc pl-4 space-y-1 my-2">
          {listItems.map((item, i) => (
            <li key={i} className="text-[11px] leading-[1.7] text-muted-foreground">
              <InlineFormat text={item} />
            </li>
          ))}
        </ul>
      );
      listItems = [];
    }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (trimmed.startsWith("### ")) {
      flushList();
      elements.push(<h3 key={i} className="text-[12px] font-semibold text-foreground mt-4 mb-1"><InlineFormat text={trimmed.slice(4)} /></h3>);
    } else if (trimmed.startsWith("## ")) {
      flushList();
      elements.push(<h2 key={i} className="text-[13px] font-semibold text-foreground mt-4 mb-1"><InlineFormat text={trimmed.slice(3)} /></h2>);
    } else if (trimmed.startsWith("# ")) {
      flushList();
      elements.push(<h1 key={i} className="text-[14px] font-semibold text-foreground mt-4 mb-1"><InlineFormat text={trimmed.slice(2)} /></h1>);
    } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
      listItems.push(trimmed.slice(2));
    } else if (/^\d+\.\s/.test(trimmed)) {
      listItems.push(trimmed.replace(/^\d+\.\s/, ""));
    } else if (trimmed === "") {
      flushList();
    } else {
      flushList();
      elements.push(<p key={i} className="text-[11px] leading-[1.7] text-muted-foreground my-1"><InlineFormat text={trimmed} /></p>);
    }
  }
  flushList();

  return <div>{elements}</div>;
}
