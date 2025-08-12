// frontend/src/components/Message.jsx
import React from 'react';
import { UserIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';

// This component receives 'message', and 'darkMode' as props.
function Message({ message, darkMode }) {
  const { sender, text, sources, clarification_questions } = message;
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
          {typeof text === 'string' ? (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="whitespace-pre-wrap text-base leading-7"
            >
              {text}
            </motion.div>
          ) : (
            <motion.div
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="leading-7 text-base"
            >
              {text}
            </motion.div>
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
}

export default Message;
