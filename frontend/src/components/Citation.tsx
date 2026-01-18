import { motion } from "framer-motion";
import type { Citation as CitationType } from "@/types/api";

interface CitationProps {
  citation: CitationType;
  index: number;
}

export function Citation({ citation, index }: CitationProps) {
  return (
    <motion.a
      href={citation.url}
      target="_blank"
      rel="noopener noreferrer"
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: index * 0.1, duration: 0.4 }}
      className="group block p-4 rounded-lg border border-[var(--color-midnight-600)] bg-gradient-to-br from-[var(--color-midnight-800)]/50 to-[var(--color-midnight-900)]/30 backdrop-blur-sm hover:border-[var(--color-cyan-500)] transition-all duration-300 hover:shadow-lg hover:shadow-[var(--color-cyan-500)]/20"
    >
      <div className="flex items-start gap-3">
        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-[var(--color-cyan-500)]/20 text-[var(--color-cyan-400)] flex items-center justify-center text-sm font-medium">
          {index + 1}
        </span>
        <div className="flex-1 min-w-0">
          {citation.title && (
            <h4 className="text-sm font-semibold text-[var(--color-cyan-400)] mb-1 group-hover:text-[var(--color-cyan-300)] transition-colors">
              {citation.title}
            </h4>
          )}
          <p className="text-sm text-slate-300 line-clamp-2 mb-2">{citation.text}</p>
          <div className="flex items-center gap-2">
            <svg
              className="w-4 h-4 text-slate-500"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1"
              />
            </svg>
            <span className="text-xs text-slate-500 truncate group-hover:text-[var(--color-cyan-400)] transition-colors">
              {citation.source_id}
            </span>
          </div>
        </div>
      </div>
    </motion.a>
  );
}
