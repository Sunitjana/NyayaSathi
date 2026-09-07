"use client";

import { motion } from "framer-motion";
import { FileText } from "lucide-react";

export default function SourceChips({ sources }: { sources: string[] }) {
  if (!sources.length) return null;

  return (
    <div className="flex flex-wrap gap-2">
      {sources.map((src, i) => (
        <motion.span
          key={src}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: i * 0.05 }}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-medium bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 hover:border-indigo-400/40 transition-colors cursor-default"
        >
          <FileText size={12} />
          {src}
        </motion.span>
      ))}
    </div>
  );
}
