// frontend/src/components/Message.jsx
import React from 'react';
import { UserIcon, ShieldCheckIcon } from '@heroicons/react/24/outline';
import { motion } from 'framer-motion';

// This component receives 'sender' and 'text' as props.
function Message({ sender, text }) {
  const isUser = sender === 'user';

  return (
    <div className={`flex gap-4 px-4 py-6 ${isUser ? 'bg-gray-50' : 'bg-white'}`}>
      {/* Avatar */}
      <div className="flex-shrink-0">
        {isUser ? (
          <div className="w-8 h-8 rounded-full bg-slate-300 flex items-center justify-center">
            <UserIcon className="w-5 h-5 text-slate-600" />
          </div>
        ) : (
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-sm">
            <ShieldCheckIcon className="w-5 h-5 text-white" />
          </div>
        )}
      </div>
      
      {/* Message Content */}
      <div className="flex-1 max-w-3xl">
        <div className="prose prose-sm max-w-none text-slate-900 dark:text-slate-100">
          {typeof text === 'string' ? (
            <p className="whitespace-pre-wrap m-0 leading-7">
              {text}
            </p>
          ) : (
            <div className="leading-7">
              {text}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Message;
