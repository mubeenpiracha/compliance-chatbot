// Simplified Message component
import React from 'react';
import { UserIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';

function Message({ sender, text, darkMode }) {
  const isUser = sender === 'user';

  return (
    <div className={`flex gap-3 px-4 py-4 ${
      isUser 
        ? 'bg-blue-50 dark:bg-blue-950' 
        : 'bg-white dark:bg-gray-800'
    } border-b border-gray-100 dark:border-gray-700`}>
      
      <div className="flex-shrink-0">
        <div className={`w-8 h-8 rounded-lg shadow-sm flex items-center justify-center ${
          isUser
            ? 'bg-gray-500'
            : 'bg-gradient-to-br from-blue-500 via-purple-600 to-pink-500'
        }`}>
          {isUser ? (
            <UserIcon className="w-4 h-4 text-white" />
          ) : (
            <ShieldCheckIcon className="w-4 h-4 text-white" />
          )}
        </div>
      </div>
      
      <div className="flex-1">
        {isUser && (
          <div className="mb-1">
            <span className="text-xs font-semibold text-gray-600 dark:text-gray-400 uppercase tracking-wide">
              You
            </span>
          </div>
        )}
        {!isUser && (
          <div className="mb-1">
            <span className="text-xs font-semibold text-blue-600 dark:text-blue-400 uppercase tracking-wide">
              AI Assistant
            </span>
          </div>
        )}
        
        <div className="text-gray-800 dark:text-gray-200 leading-relaxed">
          {typeof text === 'string' ? (
            <div className="whitespace-pre-wrap text-sm leading-6">
              {text}
            </div>
          ) : (
            <div className="leading-6 text-sm">
              {text}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Message;
