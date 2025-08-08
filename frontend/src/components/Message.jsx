// frontend/src/components/Message.jsx
import React from 'react';
import { User, Bot } from 'lucide-react'; // Import icons

// This component receives 'sender' and 'text' as props.
function Message({ sender, text }) {
  const isUser = sender === 'user';

  return (
    <div className={`flex items-start gap-4 my-4 ${isUser ? 'justify-end' : ''}`}>
      {/* Icon */}
      <div className={`flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center ${isUser ? 'bg-blue-500' : 'bg-cyan-500'}`}>
        {isUser ? <User size={24} /> : <Bot size={24} />}
      </div>

      {/* Message Bubble */}
      <div
        className={`p-4 rounded-lg max-w-lg ${
          isUser
            ? 'bg-blue-600 rounded-br-none'
            : 'bg-gray-700 rounded-bl-none'
        }`}
      >
        <p className="text-white">{text}</p>
      </div>
    </div>
  );
}

export default Message;
