import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { DocumentTextIcon, XMarkIcon } from '@heroicons/react/24/outline';

const Citation = ({ source, chunks }) => {
  const [showSidebar, setShowSidebar] = useState(false);

  const handleClick = (e) => {
    e.preventDefault();
    setShowSidebar(true);
  };

  const closeSidebar = () => {
    setShowSidebar(false);
  };

  return (
    <>
      <motion.button 
        whileHover={{ scale: 1.02 }}
        whileTap={{ scale: 0.98 }}
        className="inline-flex items-center gap-1 cursor-pointer text-blue-600 hover:text-blue-700 transition-colors duration-200 mx-1 px-2 py-1 rounded bg-blue-50 hover:bg-blue-100 border border-blue-200 text-xs font-medium"
        onClick={handleClick}
        title={`View source: ${source}`}
      >
        <DocumentTextIcon className="w-3 h-3" />
        <span>{source}</span>
      </motion.button>
      
      <AnimatePresence>
        {showSidebar && (
          <>
            {/* Overlay */}
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/20 z-[9998]" 
              onClick={closeSidebar}
            />
            
            {/* Sidebar */}
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              transition={{ type: "spring", damping: 30, stiffness: 300 }}
              className="fixed top-0 right-0 h-full w-[500px] max-w-[90vw] bg-white shadow-2xl z-[9999] overflow-hidden flex flex-col border-l border-gray-200"
            >
              {/* Header */}
              <div className="p-6 border-b border-gray-200 bg-gray-50">
                <div className="flex justify-between items-start">
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-blue-100 flex items-center justify-center">
                      <DocumentTextIcon className="w-5 h-5 text-blue-600" />
                    </div>
                    <div>
                      <div className="text-xs text-blue-600 font-semibold uppercase tracking-wider mb-1">
                        Source Document
                      </div>
                      <div className="font-semibold text-gray-900 text-lg leading-tight" title={source}>
                        {source}
                      </div>
                      <div className="text-sm text-gray-500 mt-1">
                        {chunks?.length || 0} content section{chunks?.length !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </div>
                  <motion.button 
                    whileHover={{ scale: 1.1 }}
                    whileTap={{ scale: 0.9 }}
                    onClick={closeSidebar}
                    className="p-2 text-gray-400 hover:text-gray-600 hover:bg-gray-100 rounded-lg transition-all duration-200"
                  >
                    <XMarkIcon className="w-5 h-5" />
                  </motion.button>
                </div>
              </div>

              {/* Content */}
              <div className="flex-1 p-6 overflow-y-auto">
                {chunks && chunks.length > 0 ? (
                  <div className="space-y-6">
                    {chunks.map((chunk, index) => (
                      <motion.div 
                        key={index} 
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.1 * index }}
                        className="bg-gray-50 rounded-lg p-5 border border-gray-200"
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-6 h-6 rounded-full bg-blue-600 flex items-center justify-center">
                            <span className="text-xs font-bold text-white">{index + 1}</span>
                          </div>
                          <div className="text-sm font-semibold text-gray-700">
                            Section {index + 1}
                          </div>
                        </div>
                        <div className="text-sm text-gray-800 leading-relaxed">
                          {chunk}
                        </div>
                      </motion.div>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center py-12">
                    <div className="w-12 h-12 rounded-full bg-gray-100 flex items-center justify-center mb-4">
                      <DocumentTextIcon className="w-6 h-6 text-gray-400" />
                    </div>
                    <div className="text-gray-500 text-lg font-medium mb-2">
                      No content available
                    </div>
                    <div className="text-gray-400 text-sm">
                      This source doesn't have detailed content to display
                    </div>
                  </div>
                )}
              </div>
              
              {/* Footer */}
              <div className="p-4 bg-gray-50 border-t border-gray-200 text-center">
                <div className="text-xs text-gray-500">
                  This content was used to generate the AI response
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};

export default Citation;
