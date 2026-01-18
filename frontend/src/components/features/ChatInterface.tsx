import { useState, useRef, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CitationCard } from "./CitationCard";
import { MarkdownRenderer } from "@/components/ui/MarkdownRenderer";
import { apiClient } from "@/lib/api";
import type { QueryResponse } from "@/types/api";

interface Message {
  id: string;
  type: 'user' | 'assistant';
  content: string;
  response?: QueryResponse;
  timestamp: Date;
}

export function ChatInterface() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const inputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, isLoading]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || isLoading) return;

    const currentQuery = query;
    const userMessage: Message = {
      id: `user-${Date.now()}`,
      type: 'user',
      content: currentQuery,
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setQuery("");
    setIsLoading(true);

    try {
      const result = await apiClient.query(currentQuery);
      const assistantMessage: Message = {
        id: `assistant-${Date.now()}`,
        type: 'assistant',
        content: result.answer,
        response: result,
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error(error);
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        type: 'assistant',
        content: "Sorry, I encountered an error processing your request. Please try again.",
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-full relative">
      {/* Scrollable Content Area */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto pb-32 pt-12 px-4 md:px-6 space-y-6 no-scrollbar">
        {/* Empty State / Welcome */}
        {messages.length === 0 && !isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="flex flex-col items-center justify-center h-[50vh] text-center space-y-6"
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
              className="w-16 h-16 rounded-2xl glass-panel flex items-center justify-center animate-float"
            >
               <svg className="w-8 h-8 text-neon-cyan" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </motion.div>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.3 }}
            >
              <h1 className="text-3xl font-light tracking-tight text-white mb-2">
                Python <span className="text-neon-cyan font-normal">Docs RAG</span>
              </h1>
              <p className="text-text-muted max-w-md mx-auto leading-relaxed">
                Failure-aware retrieval engine for Python documentation.
                Ask about libraries, syntax, or concepts.
              </p>
            </motion.div>
          </motion.div>
        )}

        {/* Message History */}
        <AnimatePresence mode="popLayout">
          {messages.map((message, index) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20, scale: 0.95 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              exit={{ opacity: 0, scale: 0.95 }}
              transition={{
                type: "spring",
                stiffness: 300,
                damping: 30,
                delay: index === messages.length - 1 ? 0 : 0
              }}
              className="max-w-3xl mx-auto w-full"
            >
              {message.type === 'user' ? (
                <motion.div
                  className="flex justify-end mb-4"
                  initial={{ x: 20 }}
                  animate={{ x: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                >
                  <div className="bg-neon-cyan/20 backdrop-blur-sm border border-neon-cyan/30 rounded-2xl px-5 py-3 max-w-[85%] shadow-lg">
                    <p className="text-white text-base leading-relaxed whitespace-pre-wrap wrap-break-word">{message.content}</p>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  className="space-y-4"
                  initial={{ x: -20 }}
                  animate={{ x: 0 }}
                  transition={{ type: "spring", stiffness: 300, damping: 25 }}
                >
                  {/* Failure Modes / Warnings */}
                  {message.response?.failure_modes && message.response.failure_modes.length > 0 && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      transition={{ delay: 0.2 }}
                      className="border-l-2 border-amber-500/50 pl-4 py-2"
                    >
                      <h3 className="text-amber-500 text-sm font-medium mb-1">Retrieval Warnings</h3>
                      <ul className="text-sm text-text-secondary space-y-1">
                        {message.response.failure_modes.map((mode, i) => (
                          <li key={i}>{mode.description}</li>
                        ))}
                      </ul>
                    </motion.div>
                  )}

                  {/* Answer */}
                  <motion.div
                    className="space-y-3"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.1 }}
                  >
                    {message.response && (
                      <div className="flex items-center justify-between">
                        <div className="text-xs font-mono text-text-muted border border-white/10 px-2 py-1 rounded">
                          {Math.round(message.response.confidence * 100)}% Confidence
                        </div>
                      </div>
                    )}
                    <motion.div
                      className="glass-panel p-6 rounded-2xl text-base leading-relaxed text-text-primary shadow-xl"
                      initial={{ scale: 0.98 }}
                      animate={{ scale: 1 }}
                      transition={{ type: "spring", stiffness: 200 }}
                    >
                      <MarkdownRenderer content={message.content} />
                    </motion.div>
                  </motion.div>

                  {/* Citations */}
                  {message.response?.citations && message.response.citations.length > 0 && (
                    <motion.section
                      className="space-y-3"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: 0.3 }}
                    >
                       <h2 className="text-xs font-mono text-text-muted tracking-wider uppercase opacity-50">Sources</h2>
                       <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                         {message.response.citations.map((cit, i) => (
                           <motion.div
                             key={i}
                             initial={{ opacity: 0, y: 10 }}
                             animate={{ opacity: 1, y: 0 }}
                             transition={{ delay: 0.4 + i * 0.1 }}
                           >
                             <CitationCard citation={cit} index={i} />
                           </motion.div>
                         ))}
                       </div>
                    </motion.section>
                  )}
                </motion.div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading State */}
        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="max-w-3xl mx-auto w-full"
          >
            <div className="space-y-3">
              <motion.div
                className="glass-panel rounded-xl p-6"
                animate={{ opacity: [0.4, 0.7, 0.4] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
              >
                <div className="flex items-center space-x-3">
                  <div className="flex space-x-1">
                    <motion.div
                      className="w-2 h-2 bg-neon-cyan rounded-full"
                      animate={{ y: [0, -8, 0] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: 0 }}
                    />
                    <motion.div
                      className="w-2 h-2 bg-neon-cyan rounded-full"
                      animate={{ y: [0, -8, 0] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: 0.2 }}
                    />
                    <motion.div
                      className="w-2 h-2 bg-neon-cyan rounded-full"
                      animate={{ y: [0, -8, 0] }}
                      transition={{ duration: 0.6, repeat: Infinity, delay: 0.4 }}
                    />
                  </div>
                  <span className="text-text-muted text-sm">Searching documentation...</span>
                </div>
              </motion.div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Floating Input Area */}
      <div className="fixed bottom-0 left-0 right-0 p-6 bg-linear-to-t from-void via-void to-transparent z-50">
        <div className="max-w-3xl mx-auto relative">
          <form onSubmit={handleSubmit} className="relative group">
            <motion.div
              whileFocus={{ scale: 1.01 }}
              transition={{ type: "spring", stiffness: 300 }}
            >
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Ask a question..."
                className="w-full glass-input h-14 pl-6 pr-14 rounded-full text-white placeholder-text-muted text-lg shadow-2xl focus:ring-2 focus:ring-neon-cyan/50 transition-all"
                disabled={isLoading}
              />
            </motion.div>
            <motion.button
              type="submit"
              disabled={!query.trim() || isLoading}
              className="absolute right-2 top-2 h-10 w-10 bg-white/10 rounded-full flex items-center justify-center hover:bg-neon-cyan hover:text-black transition-all disabled:opacity-30 disabled:hover:bg-white/10 disabled:hover:text-white"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.95 }}
            >
              {isLoading ? (
                <motion.div
                  className="w-4 h-4 border-2 border-current border-t-transparent rounded-full"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                />
              ) : (
                <motion.svg
                  className="w-5 h-5 transform -rotate-45 ml-0.5 mt-0.5"
                  fill="none"
                  viewBox="0 0 24 24"
                  stroke="currentColor"
                  initial={{ scale: 1 }}
                  whileHover={{ scale: 1.1 }}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                </motion.svg>
              )}
            </motion.button>
          </form>
          <div className="text-center mt-3">
             <p className="text-[10px] text-text-muted font-mono opacity-40">
               Powered by Python Docs RAG • v0.1.0
             </p>
          </div>
        </div>
      </div>
    </div>
  );
}
