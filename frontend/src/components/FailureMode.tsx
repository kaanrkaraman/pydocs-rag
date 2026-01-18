import { motion } from "framer-motion";
import type { FailureMode as FailureModeType } from "@/types/api";

interface FailureModeProps {
  failureMode: FailureModeType;
  index: number;
}

const severityConfig = {
  high: {
    bg: "from-[var(--color-rose-500)]/10 to-[var(--color-rose-600)]/5",
    border: "border-[var(--color-rose-500)]/30",
    icon: "text-[var(--color-rose-400)]",
    text: "text-[var(--color-rose-300)]",
  },
  medium: {
    bg: "from-[var(--color-amber-500)]/10 to-[var(--color-amber-600)]/5",
    border: "border-[var(--color-amber-500)]/30",
    icon: "text-[var(--color-amber-400)]",
    text: "text-[var(--color-amber-300)]",
  },
  low: {
    bg: "from-[var(--color-cyan-500)]/10 to-[var(--color-cyan-600)]/5",
    border: "border-[var(--color-cyan-500)]/30",
    icon: "text-[var(--color-cyan-400)]",
    text: "text-[var(--color-cyan-300)]",
  },
};

export function FailureMode({ failureMode, index }: FailureModeProps) {
  const severity = failureMode.severity.toLowerCase() as keyof typeof severityConfig;
  const config = severityConfig[severity] || severityConfig.low;

  const getIcon = () => {
    switch (severity) {
      case "high":
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z"
              clipRule="evenodd"
            />
          </svg>
        );
      case "medium":
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
              clipRule="evenodd"
            />
          </svg>
        );
      default:
        return (
          <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
            <path
              fillRule="evenodd"
              d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z"
              clipRule="evenodd"
            />
          </svg>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1, duration: 0.4 }}
      className={`p-4 rounded-lg border ${config.border} bg-gradient-to-br ${config.bg} backdrop-blur-sm`}
    >
      <div className="flex items-start gap-3">
        <div className={config.icon}>{getIcon()}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className={`text-sm font-semibold ${config.text}`}>
              {failureMode.type.replace(/_/g, " ").toUpperCase()}
            </span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-white/5 text-slate-400 uppercase tracking-wide">
              {severity}
            </span>
          </div>
          <p className="text-sm text-slate-300">{failureMode.message}</p>
          {failureMode.metadata && Object.keys(failureMode.metadata).length > 0 && (
            <div className="mt-2 pt-2 border-t border-white/5">
              <details className="text-xs text-slate-400">
                <summary className="cursor-pointer hover:text-slate-300 transition-colors">
                  View details
                </summary>
                <pre className="mt-2 p-2 rounded bg-black/20 overflow-x-auto">
                  {JSON.stringify(failureMode.metadata, null, 2)}
                </pre>
              </details>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
