import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { DocumentTextIcon, XMarkIcon, BookOpenIcon } from '@heroicons/react/24/outline';
import { SparklesIcon } from '@heroicons/react/24/solid';

const Citation = ({ source, filename, chunk_id, chunkId, section, jurisdiction, onClose }) => {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Log all available props for debugging
    console.log("Citation component loaded with props:", { 
      source, 
      filename,
      chunk_id, // This is likely the correct property name based on backend code
      chunkId, 
      section, 
      jurisdiction 
    });
    
    // Determine the effective chunk ID - we need to check all possible property names
    // Since the citation object is spread as props, the chunk ID could be directly available
    let effectiveChunkId = chunk_id || chunkId;
    
    // Log the ID for debugging
    console.log("Direct chunk_id/chunkId prop:", effectiveChunkId);
    
    if (!effectiveChunkId && source) {
      // If we have a source object, check there too
      effectiveChunkId = source.chunk_id || source.chunkId;
      console.log("Found chunk ID in source object:", effectiveChunkId);
    }
    
    console.log("Final effective chunk ID for API call:", effectiveChunkId);
    
    // If we have a chunk ID, make the API call
    if (effectiveChunkId) {
      setLoading(true);
      console.log("Fetching chunk content for chunkId:", effectiveChunkId);
      fetch(`http://localhost:8000/api/v1/chat/chunk/${effectiveChunkId}`)
        .then(response => {
          console.log("Chunk API response status:", response.status);
          if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
          }
          return response.json();
        })
        .then(data => {
          console.log("Chunk API response data:", data);
          setContent(data.text);
          setLoading(false);
        })
        .catch(error => {
          console.error("Error fetching chunk content:", error);
          setLoading(false);
        });
    } else {
      // If we don't have a chunk ID, check if we need to look at other properties
      console.warn("No chunk ID found in props. Looking at other potential sources...");
      
      // Check if there's a filename that might contain the chunk ID
      // This handles cases where the ID is embedded in a filename like: "000051_f3cec8b054129bdb09baa5df34200320f9eebd50.txt"
      const sourceFilename = filename || source?.filename || '';
      if (sourceFilename && sourceFilename.includes('_')) {
        const parts = sourceFilename.split('_');
        if (parts.length > 1) {
          const extractedId = parts[1].replace('.txt', '');
          console.log("Extracted chunk ID from filename:", extractedId);
          
          setLoading(true);
          fetch(`http://localhost:8000/api/v1/chat/chunk/${extractedId}`)
            .then(response => {
              console.log("Chunk API response status (from filename):", response.status);
              if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
              }
              return response.json();
            })
            .then(data => {
              console.log("Chunk API response data (from filename):", data);
              setContent(data.text);
              setLoading(false);
            })
            .catch(error => {
              console.error("Error fetching chunk content from filename:", error);
              setLoading(false);
            });
          return;
        }
      }
      
      // If we have a content property directly, use that
      if (source?.content) {
        console.log("Using content directly from source object");
        setContent(source.content);
        setLoading(false);
        return;
      }
      
      // If we get here, we couldn't find any content
      console.error("Could not find chunk ID or content in any property", { 
        source, filename, chunk_id, chunkId, section, jurisdiction
      });
      
      // Set a useful message for the user instead of leaving the panel empty
      setContent("Source details available, but content could not be retrieved. This could be due to a missing chunk identifier.");
      setLoading(false);
    }
  }, [chunkId, chunk_id, source, filename, section, jurisdiction]);

  const closeSidebar = () => {
    if (onClose) onClose();
  };

  return (
    <motion.div 
      initial={{ x: '100%', opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: '100%', opacity: 0 }}
      transition={{ type: "spring", damping: 25, stiffness: 200 }}
      className="fixed top-0 right-0 h-full w-[550px] max-w-[90vw] bg-white/95 dark:bg-gray-900/95 backdrop-blur-xl shadow-2xl z-[9999] overflow-hidden flex flex-col border-l border-gray-200 dark:border-gray-700"
    >
      {/* Overlay to capture clicks outside the sidebar */}
      <motion.div 
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black/40 backdrop-blur-sm z-[9998]" 
        onClick={closeSidebar}
        style={{ zIndex: -1 }}
      />
      
      {/* Enhanced Header */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/50 dark:to-purple-950/50">
        <div className="flex justify-between items-start">
          <div className="flex items-start gap-4">
            <motion.div 
              whileHover={{ scale: 1.1, rotate: 10 }}
              className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg"
            >
              <BookOpenIcon className="w-6 h-6 text-white" />
            </motion.div>
            <div>
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="text-xs text-blue-600 dark:text-blue-400 font-bold uppercase tracking-wider mb-2 flex items-center gap-1"
              >
                <SparklesIcon className="w-3 h-3" />
                Source Document
              </motion.div>
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="font-bold text-gray-900 dark:text-gray-100 text-lg leading-tight bg-gradient-to-r from-gray-900 to-gray-700 dark:from-gray-100 dark:to-gray-300 bg-clip-text text-transparent" 
                title={filename || source}
              >
                {filename || source || 'Unknown Document'}
              </motion.div>
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-sm text-gray-600 dark:text-gray-400 mt-2 flex items-center gap-1"
              >
                <DocumentTextIcon className="w-4 h-4" />
                {jurisdiction && <span className="ml-1">{jurisdiction}</span>}
                {section && <span className="ml-1">• {section}</span>}

              </motion.div>
            </div>
          </div>
          <motion.button 
            whileHover={{ scale: 1.1, rotate: 90 }}
            whileTap={{ scale: 0.9 }}
            onClick={closeSidebar}
            className="p-3 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-xl transition-all duration-200 shadow-sm"
          >
            <XMarkIcon className="w-5 h-5" />
          </motion.button>
        </div>
      </div>

      {/* Enhanced Content */}
      <div className="flex-1 p-6 overflow-y-auto bg-gray-50/30 dark:bg-gray-900/30">
        {loading ? (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center h-full text-center py-12"
          >
            <motion.div 
              animate={{ 
                rotate: 360,
              }}
              transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
              className="w-12 h-12 rounded-full border-4 border-gray-200 dark:border-gray-700 border-t-blue-500 dark:border-t-blue-400 mb-6"
            />
            <div className="text-gray-500 dark:text-gray-400 text-lg font-semibold mb-3">
              Loading content...
            </div>
          </motion.div>
        ) : content ? (
          <motion.div 
            initial={{ opacity: 0, y: 20, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            transition={{ delay: 0.1, type: "spring", stiffness: 100 }}
            className="group bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm rounded-2xl p-6 border border-gray-200/50 dark:border-gray-600/50 shadow-sm"
          >
            <div className="flex items-center gap-3 mb-4">
              <motion.div 
                whileHover={{ scale: 1.2, rotate: 360 }}
                className="w-8 h-8 rounded-xl bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-sm"
              >
                <DocumentTextIcon className="w-4 h-4 text-white" />
              </motion.div>
              <div className="text-sm font-bold text-gray-700 dark:text-gray-300 bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
                {section || 'Referenced Section'}
              </div>
            </div>
            <div className="text-sm text-gray-800 dark:text-gray-200 leading-relaxed font-medium whitespace-pre-wrap">
              {content}
            </div>
          </motion.div>
        ) : (
          <motion.div 
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center h-full text-center py-12"
          >
            <motion.div 
              animate={{ 
                scale: [1, 1.1, 1],
                rotate: [0, 10, -10, 0]
              }}
              transition={{ duration: 2, repeat: Infinity }}
              className="w-16 h-16 rounded-2xl bg-gradient-to-br from-gray-100 to-gray-200 dark:from-gray-700 dark:to-gray-800 flex items-center justify-center mb-6 shadow-lg"
            >
              <DocumentTextIcon className="w-8 h-8 text-gray-400 dark:text-gray-500" />
            </motion.div>
            <div className="text-gray-500 dark:text-gray-400 text-lg font-semibold mb-3">
              No content available
            </div>
            <div className="text-gray-400 dark:text-gray-500 text-sm max-w-xs">
              This source doesn't have detailed content to display at this time
            </div>
          </motion.div>
        )}
      </div>
      
      {/* Enhanced Footer */}
      <motion.div 
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-950/50 dark:to-purple-950/50 border-t border-gray-200 dark:border-gray-700 text-center"
      >
        <div className="text-xs text-gray-600 dark:text-gray-400 flex items-center justify-center gap-1">
          <SparklesIcon className="w-3 h-3" />
          This content was used to generate the AI response
          <SparklesIcon className="w-3 h-3" />
        </div>
      </motion.div>
    </motion.div>
  );
};

export default Citation;