"use client";

import { motion } from "framer-motion";

const config: Record<string, { bg: string; text: string; glow: string }> = {
  high: { bg: "bg-emerald-500/15", text: "text-emerald-400", glow: "shadow-emerald-500/20" },
  medium: { bg: "bg-amber-500/15", text: "text-amber-400", glow: "shadow-amber-500/20" },
  low: { bg: "bg-red-500/15", text: "text-red-400", glow: "shadow-red-500/20" },
};

export default function ConfidenceBadge({ level }: { level: string }) {
  const c = config[level] || config.low;
  return (
    <motion.span
      initial={{ scale: 0.8, opacity: 0 }}
      animate={{ scale: 1, opacity: 1 }}
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold uppercase tracking-wider ${c.bg} ${c.text} shadow-lg ${c.glow}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current animate-pulse" />
      {level} confidence
    </motion.span>
  );
}
