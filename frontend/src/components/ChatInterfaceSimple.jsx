// Simplified ChatInterface for debugging
import React, { useState, useEffect, useRef } from 'react';
import Message from './MessageSimple';
import Citation from './CitationSimple';
import { PaperAirplaneIcon, Squares2X2Icon, MoonIcon, SunIcon } from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import axios from 'axios';

function ChatInterface({ darkMode, setDarkMode }) {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hello! How can I help you with DIFC or ADGM regulations today?' }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [jurisdiction, setJurisdiction] = useState('DIFC');

  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (inputValue.trim() === '' || isLoading) return;

    const userMessage = { sender: 'user', text: inputValue };
    const newMessages = [...messages, userMessage];
    setMessages(newMessages);
    setIsLoading(true);
    setInputValue('');

    try {
      const validHistory = messages.slice(1).filter(msg => msg.sender && typeof msg.text === 'string');
      const response = await axios.post('http://localhost:8000/api/v1/chat', {
        message: userMessage.text,
        jurisdiction: jurisdiction,
        history: validHistory
      });

      let aiResponse;
      if (typeof response.data === 'object' && response.data !== null) {
        aiResponse = {
          sender: 'ai',
          text: response.data.answer || response.data.text || JSON.stringify(response.data),
          sources: response.data.sources || []
        };
      } else {
        aiResponse = {
          sender: 'ai',
          text: String(response.data),
          sources: []
        };
      }

      setMessages(prev => [...prev, aiResponse]);
      toast.success('Response received!');
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { sender: 'ai', text: 'Sorry, I encountered an error. Please try again.' };
      setMessages(prev => [...prev, errorMessage]);
      toast.error('Failed to get response. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const renderMessageText = (message) => {
    if (message.sender === 'user' || !message.sources || message.sources.length === 0) {
      return message.text;
    }

    const parts = message.text.split(/(\s\[Source: [^\]]+\])/g);

    return parts.map((part, index) => {
      const match = part.match(/\s\[Source: (.*?)\]/);
      if (match) {
        const sourceName = match[1];
        const relevantChunks = message.sources
          .filter(s => s.metadata && s.metadata.file_name === sourceName)
          .map(s => s.page_content);

        if (relevantChunks.length > 0) {
          return <Citation key={index} source={sourceName} chunks={relevantChunks} />;
        }
      }
      return part;
    });
  };

  return (
    <div className="flex flex-col h-screen w-full">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg">
            <Squares2X2Icon className="w-4 h-4 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-blue-600 dark:text-blue-400">
              Compliance Assistant
            </h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">AI-powered regulatory guidance</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="p-1.5 rounded-md bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700"
          >
            {darkMode ? (
              <SunIcon className="w-4 h-4" />
            ) : (
              <MoonIcon className="w-4 h-4" />
            )}
          </button>
          
          <div className="flex items-center gap-2">
            <span className="text-xs font-medium text-gray-600 dark:text-gray-400">Jurisdiction:</span>
            <div className="flex bg-gray-100 dark:bg-gray-800 rounded-lg p-0.5">
              {['DIFC', 'ADGM', 'Both'].map((j) => (
                <button
                  key={j}
                  onClick={() => {
                    setJurisdiction(j);
                    console.log('Jurisdiction changed to:', j); // Debug log
                    toast.success(`Switched to ${j} jurisdiction`);
                  }}
                  className={`px-2 py-1 text-xs font-semibold rounded-md transition-all duration-300 ${
                    jurisdiction === j
                      ? 'bg-blue-500 text-white shadow-sm'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                  }`}
                >
                  {j}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900">
        <div className="w-full">
          {messages.map((msg, index) => (
            <div key={index}>
              <Message sender={msg.sender} text={renderMessageText(msg)} darkMode={darkMode} />
            </div>
          ))}
          
          {isLoading && (
            <div className="flex items-center px-4 py-4 bg-white dark:bg-gray-800 mx-4 my-2 rounded-xl shadow-sm">
              <div className="flex items-center justify-center mr-4">
                <div className="animate-spin w-6 h-6 border-2 border-blue-500 border-t-transparent rounded-full"></div>
              </div>
              <div className="flex-1">
                <span className="text-sm font-semibold text-gray-800 dark:text-gray-200">
                  AI is thinking...
                </span>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Analyzing {jurisdiction} regulatory documents...
                </p>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area */}
      <div className="px-4 py-3 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
        <div className="w-full">
          <form onSubmit={handleSendMessage}>
            <div className="relative">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSendMessage(e);
                  }
                }}
                placeholder={`Ask me anything about ${jurisdiction} regulations...`}
                className="w-full resize-none rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-3 pr-16 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                rows="1"
                disabled={isLoading}
                required
              />
              <button
                type="submit"
                disabled={isLoading || !inputValue.trim()}
                className={`absolute bottom-2 right-2 p-2 rounded-lg font-semibold transition-all duration-300 ${
                  isLoading || !inputValue.trim()
                    ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                    : 'bg-blue-500 hover:bg-blue-600 text-white shadow-sm'
                }`}
              >
                {isLoading ? (
                  <div className="animate-spin w-4 h-4 border-2 border-current border-t-transparent rounded-full" />
                ) : (
                  <PaperAirplaneIcon className="w-4 h-4" />
                )}
              </button>
            </div>
          </form>
          
          <div className="text-center mt-2">
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Powered by AI • Current jurisdiction: {jurisdiction}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ChatInterface;
