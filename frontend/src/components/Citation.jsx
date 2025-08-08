import React, { useState } from 'react';

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
      <span 
        className="citation-link inline-block cursor-pointer text-cyan-400 underline hover:text-cyan-300 transition-colors duration-200 mx-1 px-1 py-0.5 rounded bg-cyan-900/20 hover:bg-cyan-900/40"
        onClick={handleClick}
        title={`Click to see source content from ${source}`}
      >
        [Source: {source}]
      </span>
      
      {showSidebar && (
        <>
          {/* Overlay */}
          <div 
            className="fixed inset-0 bg-black bg-opacity-30 z-[9998]" 
            onClick={closeSidebar}
          />
          
          {/* Sidebar */}
          <div className="fixed top-0 right-0 h-full w-[500px] max-w-[80vw] bg-white shadow-2xl z-[9999] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="p-4 border-b border-gray-300 bg-gray-50 flex justify-between items-center">
              <div>
                <div className="font-semibold text-cyan-600 text-xs uppercase tracking-wide mb-1">
                  Source Document
                </div>
                <div className="font-medium text-gray-900" title={source}>
                  {source}
                </div>
              </div>
              <button 
                onClick={closeSidebar}
                className="text-gray-400 hover:text-gray-600 text-2xl font-bold w-8 h-8 flex items-center justify-center rounded hover:bg-gray-200 transition-colors"
              >
                ×
              </button>
            </div>
            
            {/* Content */}
            <div className="flex-1 p-4 overflow-y-auto">
              {chunks && chunks.length > 0 ? (
                chunks.map((chunk, index) => (
                  <div key={index} className="mb-6 last:mb-0">
                    <div className="text-xs text-gray-500 mb-2 font-mono">
                      Chunk {index + 1}:
                    </div>
                    <div className="text-sm bg-white p-4 rounded border-l-4 border-cyan-500 text-gray-900 border border-gray-200 leading-relaxed">
                      {chunk}
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-gray-500 italic">
                  No source content available for this citation.
                </div>
              )}
            </div>
            
            {/* Footer */}
            <div className="p-4 bg-gray-50 text-xs text-gray-500 text-center border-t border-gray-300">
              This content was used to generate the AI response
            </div>
          </div>
        </>
      )}
    </>
  );
};

export default Citation;
