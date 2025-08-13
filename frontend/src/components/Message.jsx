
// frontend/src/components/Message.jsx
import React, { useState, useCallback, useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import Citation from './Citation';
import { UserIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';

// Helper to preprocess citations for markdown rendering
const preprocessCitationsForMarkdown = (text) => {
  // Replace {{CITATION_n}} with [[CITATION_n]] for easier tokenization
  return text ? text.replace(/\{\{CITATION_(\d+)\}\}/g, '[[CITATION_$1]]') : '';
};

const Message = React.memo(({ message, darkMode }) => {
  const { sender, text, sources, clarification_questions } = message;
  const [activeCitation, setActiveCitation] = useState(null);

  // Memoize preprocessing to avoid unnecessary work
  const preprocessMarkdown = useCallback((text) => {
      if (!text || !sources || sources.length === 0) return text;
      
      // Make a copy of the original text to work with
      let processedText = text;
      
      // First, handle consecutive citation markers like [2][3][4]
      processedText = processedText.replace(/((\[\d+\])+)([.,;:!?])?/g, (match, markers, _, punct) => {
        // Split markers: [2][3][4] => ['[2]', '[3]', '[4]']
        const markerArr = markers.match(/\[\d+\]/g) || [];
        // Convert each marker to {{CITATION_n}}
        const converted = markerArr.map(m => {
          const n = m.match(/\d+/)[0];
          return `{{CITATION_${n}}}`;
        }).join(' ');
        // Add punctuation if present
        return punct ? `${converted}${punct}` : converted;
      });
      
      // Then handle any remaining single citation markers [n] followed by optional punctuation
      // This catches standalone citations like [1] or [2].
      processedText = processedText.replace(/\[(\d+)\]([.,;:!?])?/g, (match, num, punct) => {
        // Only replace if the number is within the range of available sources
        if (parseInt(num) <= sources.length) {
          return punct ? `{{CITATION_${num}}}${punct}` : `{{CITATION_${num}}}`;
        }
        // If not a valid citation number, return the original text
        return match;
      });
      
      return processedText;
  }, [sources]);

  const renderMarkdownWithCitations = useCallback((text) => {
    // Step 1: Preprocess markdown citations
    const afterPreprocess = preprocessMarkdown(text);
    const processed = preprocessCitationsForMarkdown(afterPreprocess);

    // Function to process text and replace citations with buttons
    const processCitations = (content) => {
      if (!content) return null;
      
      const str = Array.isArray(content) ? content.join('') : String(content);
      const regex = /\[\[CITATION_(\d+)\]\]/g;
      
      // Quick check if there are any citations
      if (!regex.test(str)) {
        return content;
      }
      
      // Reset regex
      regex.lastIndex = 0;
      
      let parts = [];
      let lastIndex = 0;
      let match;
      let key = 0;
      
      while ((match = regex.exec(str)) !== null) {
        if (match.index > lastIndex) {
          parts.push(str.slice(lastIndex, match.index));
        }
        
        const idx = parseInt(match[1], 10) - 1;
        const source = sources && idx >= 0 && sources[idx];
        
        if (source) {
          parts.push(
            <button
              key={`citation-${key++}`}
              className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-950 hover:bg-blue-200 dark:hover:bg-blue-900 border border-blue-300 dark:border-blue-700 text-xs font-semibold shadow-md cursor-pointer"
              title={`View source: ${source.title} - ${source.jurisdiction}`}
              onClick={() => setActiveCitation(idx)}
              style={{ textDecoration: 'underline', fontWeight: 600 }}
            >
              [{match[1]}] {source.title} - {source.jurisdiction}
            </button>
          );
        } else {
          parts.push(match[0]);
        }
        
        lastIndex = regex.lastIndex;
      }
      
      if (lastIndex < str.length) {
        parts.push(str.slice(lastIndex));
      }
      
      return parts;
    };

    // Step 2: Use ReactMarkdown with custom renderers that handle citations
    return (
      <ReactMarkdown
        components={{
          p: ({ node, children, ...props }) => {
            const processedChildren = processCitations(children);
            return <p {...props}>{processedChildren}</p>;
          },
          li: ({ node, children, ...props }) => {
            const processedChildren = processCitations(children);
            return <li {...props}>{processedChildren}</li>;
          },
          h1: ({ node, children, ...props }) => {
            const processedChildren = processCitations(children);
            return <h1 {...props}>{processedChildren}</h1>;
          },
          h2: ({ node, children, ...props }) => {
            const processedChildren = processCitations(children);
            return <h2 {...props}>{processedChildren}</h2>;
          },
          h3: ({ node, children, ...props }) => {
            const processedChildren = processCitations(children);
            return <h3 {...props}>{processedChildren}</h3>;
          }
        }}
      >
        {processed}
      </ReactMarkdown>
    );
  }, [preprocessMarkdown, sources, setActiveCitation]);

  const isUser = sender === 'user';

  return (
    <motion.div 
      whileHover={{ y: -1 }}
      className={`flex gap-4 px-6 py-8 transition-all duration-300 ${
        isUser 
          ? 'bg-gradient-to-r from-blue-50/50 to-purple-50/50 dark:from-blue-950/30 dark:to-purple-950/30' 
          : 'bg-white/60 dark:bg-gray-800/60 backdrop-blur-sm'
      } border-b border-gray-100/50 dark:border-gray-700/50`}
    >
      {/* Enhanced Avatar */}
      <div className="flex-shrink-0">
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          className={`w-10 h-10 rounded-xl shadow-lg flex items-center justify-center ${
            isUser
              ? 'bg-gradient-to-br from-gray-400 to-gray-600 ring-2 ring-gray-200 dark:ring-gray-600'
              : 'bg-gradient-to-br from-blue-500 via-purple-600 to-pink-500 ring-2 ring-white dark:ring-gray-700'
          }`}
        >
          {isUser ? (
            <UserIcon className="w-5 h-5 text-white" />
          ) : (
            <ShieldCheckIcon className="w-5 h-5 text-white" />
          )}
        </motion.div>
      </div>
      {/* Enhanced Message Content */}
      <div className="flex-1 max-w-4xl">
        {isUser && (
          <div className="mb-2">
            <span className="text-sm font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wider">
              You
            </span>
          </div>
        )}
        {!isUser && (
          <div className="mb-2">
            <span className="text-sm font-semibold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent uppercase tracking-wider">
              AI Assistant
            </span>
          </div>
        )}
        <div className="prose prose-sm max-w-none text-gray-800 dark:text-gray-200 leading-relaxed">
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className="whitespace-pre-wrap text-base leading-7"
          >
            {typeof text === 'string' ? renderMarkdownWithCitations(text) : text}
          </motion.div>
          {/* Inline citation sidebar */}
          {activeCitation !== null && sources && sources[activeCitation] && (
            <Citation
              source={sources[activeCitation].title}
              chunkId={sources[activeCitation].chunk_id}
              section={sources[activeCitation].section}
              documentType={sources[activeCitation].document_type}
              jurisdiction={sources[activeCitation].jurisdiction}
              onClose={() => setActiveCitation(null)}
            />
          )}
          {clarification_questions && clarification_questions.length > 0 && (
            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <p className="font-semibold text-gray-700 dark:text-gray-300">Please clarify the following points:</p>
              <ul className="list-disc list-inside mt-2 space-y-2 text-gray-600 dark:text-gray-400">
                {clarification_questions.map((q, i) => (
                  <li key={i}>{q}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
});

export default Message;
