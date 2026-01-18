import { motion } from "framer-motion";
import type { Citation as CitationType } from "@/types/api";

interface CitationProps {
  citation: CitationType;
  index: number;
}

export function CitationCard({ citation, index }: CitationProps) {
  const scoreColor = citation.score > 0.7 ? "border-neon-cyan/50" : "border-glass-border";

  return (
    <motion.a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
      whileHover={{ scale: 1.02, y: -2 }}
      className={`glass-panel p-4 rounded-xl block group transition-all duration-300 hover:border-neon-cyan/30 ${scoreColor}`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          <h4 className="text-sm font-medium text-text-primary truncate group-hover:text-neon-cyan transition-colors">
            {citation.title || "Untitled Document"}
          </h4>
          <p className="text-xs text-text-muted mt-1 truncate font-mono opacity-70">
            {new URL(citation.url).hostname}{new URL(citation.url).pathname}
          </p>
        </div>
        <div className="shrink-0 bg-white/5 px-2 py-1 rounded text-[10px] font-mono text-text-muted border border-white/5">
          #{index + 1}
        </div>
      </div>
      
      {/* Snippet Tooltip - simplified for now as inline expand */}
      <div className="mt-3 text-xs text-text-secondary line-clamp-2 group-hover:line-clamp-none transition-all duration-300 border-t border-white/5 pt-2 opacity-60 group-hover:opacity-100">
        {citation.text}
      </div>
    </motion.a>
  );
}
