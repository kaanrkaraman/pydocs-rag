
import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { dracula } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface MarkdownRendererProps {
  content: string;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({ content }) => {
  return (
    <div className="prose prose-invert max-w-none prose-p:leading-relaxed prose-pre:bg-[#282a36] prose-pre:border prose-pre:border-white/10 prose-pre:rounded-xl overflow-hidden wrap-break-word">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code({ node, inline, className, children, ...props }: any) {
            const match = /language-(\w+)/.exec(className || '');
            // In react-markdown v9+, children is just the text content for code blocks
            const codeContent = String(children || '').replace(/\n$/, '');
            
            return !inline && match ? (
              <div className="relative group rounded-xl overflow-hidden my-4 border border-white/10 shadow-lg">
                 <div className="absolute top-0 right-0 left-0 bg-[#282a36] px-4 py-2 border-b border-white/5 flex items-center justify-between">
                    <div className="flex space-x-1.5">
                      <div className="w-3 h-3 rounded-full bg-red-500/20 border border-red-500/50"></div>
                      <div className="w-3 h-3 rounded-full bg-yellow-500/20 border border-yellow-500/50"></div>
                      <div className="w-3 h-3 rounded-full bg-green-500/20 border border-green-500/50"></div>
                    </div>
                    <div className="text-xs text-text-muted font-mono uppercase tracking-wider">{match[1]}</div>
                 </div>
                 <div className="pt-10">
                    <SyntaxHighlighter
                      style={dracula}
                      language={match[1]}
                      PreTag="div"
                      customStyle={{ margin: 0, padding: '1.5rem', background: 'transparent' }}
                      {...props}
                    >
                      {codeContent}
                    </SyntaxHighlighter>
                 </div>
              </div>
            ) : (
              <code className="bg-white/10 px-1.5 py-0.5 rounded text-neon-cyan font-mono text-sm break-all" {...props}>
                {children}
              </code>
            );
          },
          a: ({ node, ...props }) => (
            <a {...props} className="text-neon-cyan hover:underline decoration-neon-cyan/50 underline-offset-4 transition-all" target="_blank" rel="noopener noreferrer" />
          ),
          blockquote: ({ node, ...props }) => (
            <blockquote {...props} className="border-l-4 border-neon-cyan/50 pl-4 py-1 italic text-text-muted bg-white/5 rounded-r-lg my-4" />
          ),
          p: ({ node, ...props }) => (
             <p {...props} className="mb-4 leading-7 text-gray-200 wrap-break-word" />
          ),
          ul: ({ node, ...props }) => (
             <ul {...props} className="list-disc pl-6 mb-4 space-y-2 text-gray-200" />
          ),
          ol: ({ node, ...props }) => (
             <ol {...props} className="list-decimal pl-6 mb-4 space-y-2 text-gray-200" />
          )

        }}
      >
        {replaceCitationsWithLinks(content)}
      </ReactMarkdown>
    </div>
  );
};

// Helper to preprocess text and turn [n] into visible markdown links/badges if needed.
function replaceCitationsWithLinks(text: string): string {
  // Regex to match [n] patterns
  return text.replace(/\[(\d+)\]/g, ' **[$1]** '); 
}
