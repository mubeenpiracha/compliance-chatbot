// Simplified Citation component
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
      <button 
        className="inline-flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300 mx-1 px-2 py-1 rounded-lg bg-blue-50 dark:bg-blue-950 hover:bg-blue-100 dark:hover:bg-blue-900 border border-blue-200 dark:border-blue-700 text-xs font-medium shadow-sm"
        onClick={handleClick}
        title={`View source: ${source}`}
      >
        <DocumentTextIcon className="w-3 h-3" />
        <span>{source}</span>
      </button>
      
      <AnimatePresence>
        {showSidebar && (
          <>
            <div 
              className="fixed inset-0 bg-black/40 z-50" 
              onClick={closeSidebar}
            />
            
            <div className="fixed top-0 right-0 h-full w-[500px] max-w-[90vw] bg-white dark:bg-gray-900 shadow-2xl z-50 overflow-hidden flex flex-col border-l border-gray-200 dark:border-gray-700">
              
              <div className="p-4 border-b border-gray-200 dark:border-gray-700 bg-blue-50 dark:bg-blue-950">
                <div className="flex justify-between items-start">
                  <div className="flex items-start gap-3">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-500 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg">
                      <DocumentTextIcon className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <div className="text-xs text-blue-600 dark:text-blue-400 font-bold uppercase tracking-wider mb-1">
                        Source Document
                      </div>
                      <div className="font-bold text-gray-900 dark:text-gray-100 text-base leading-tight" title={source}>
                        {source}
                      </div>
                      <div className="text-xs text-gray-600 dark:text-gray-400 mt-1 flex items-center gap-1">
                        <DocumentTextIcon className="w-3 h-3" />
                        {chunks?.length || 0} section{chunks?.length !== 1 ? 's' : ''}
                      </div>
                    </div>
                  </div>
                  <button 
                    onClick={closeSidebar}
                    className="p-2 text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg"
                  >
                    <XMarkIcon className="w-4 h-4" />
                  </button>
                </div>
              </div>

              <div className="flex-1 p-4 overflow-y-auto bg-gray-50 dark:bg-gray-900">
                {chunks && chunks.length > 0 ? (
                  <div className="space-y-4">
                    {chunks.map((chunk, index) => (
                      <div 
                        key={index} 
                        className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-600 shadow-sm"
                      >
                        <div className="flex items-center gap-2 mb-3">
                          <div className="w-6 h-6 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-sm">
                            <span className="text-xs font-bold text-white">{index + 1}</span>
                          </div>
                          <div className="text-xs font-bold text-blue-600 dark:text-blue-400">
                            Section {index + 1}
                          </div>
                        </div>
                        <div className="text-xs text-gray-800 dark:text-gray-200 leading-relaxed">
                          {chunk}
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center h-full text-center py-8">
                    <div className="w-12 h-12 rounded-lg bg-gray-100 dark:bg-gray-700 flex items-center justify-center mb-4 shadow-lg">
                      <DocumentTextIcon className="w-6 h-6 text-gray-400 dark:text-gray-500" />
                    </div>
                    <div className="text-gray-500 dark:text-gray-400 text-base font-semibold mb-2">
                      No content available
                    </div>
                    <div className="text-gray-400 dark:text-gray-500 text-xs max-w-xs">
                      This source doesn't have detailed content to display
                    </div>
                  </div>
                )}
              </div>
              
              <div className="p-3 bg-blue-50 dark:bg-blue-950 border-t border-gray-200 dark:border-gray-700 text-center">
                <div className="text-xs text-gray-600 dark:text-gray-400">
                  Content used for AI response generation
                </div>
              </div>
            </div>
          </>
        )}
      </AnimatePresence>
    </>
  );
};

export default Citation;
