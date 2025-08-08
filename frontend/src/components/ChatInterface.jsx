// frontend/src/components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import Citation from './Citation';
import { PaperAirplaneIcon, ArrowPathIcon, Squares2X2Icon } from '@heroicons/react/24/outline';
import { motion, AnimatePresence } from 'framer-motion';
import toast from 'react-hot-toast';
import axios from 'axios';

function ChatInterface() {
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
      // Only send valid history messages to backend
      const validHistory = messages.slice(1).filter(msg => msg.sender && typeof msg.text === 'string');
      const response = await axios.post('http://localhost:8000/api/v1/chat', {
        message: userMessage.text,
        jurisdiction: jurisdiction,
        history: validHistory
      });

      // Log the raw backend response for debugging
      console.log('AI backend response:', response.data);

      // Fallback: If response is missing expected fields, show raw JSON
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

      console.log('Final aiResponse:', aiResponse);

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

  // A function to parse the message text and render citations
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
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between px-6 py-4 border-b border-gray-200 bg-white"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
            <Squares2X2Icon className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Compliance Assistant</h1>
          </div>
        </div>
        
        {/* Jurisdiction Selector - Perplexity style */}
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-500">Jurisdiction:</span>
          <div className="flex bg-gray-100 rounded-lg p-1">
            {['DIFC', 'ADGM', 'Both'].map((j) => (
              <motion.button
                key={j}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setJurisdiction(j)}
                className={`px-3 py-1.5 text-sm font-medium rounded-md transition-all duration-200 ${
                  jurisdiction === j
                    ? 'bg-white text-gray-900 shadow-sm'
                    : 'text-gray-600 hover:text-gray-900'
                }`}
              >
                {j}
              </motion.button>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Messages Area - LangUI container style */}
      <div className="flex-1 overflow-y-auto bg-slate-300 text-sm leading-6 text-slate-900 shadow-md dark:bg-slate-800 dark:text-slate-300 sm:text-base sm:leading-7">
        <div className="max-w-3xl mx-auto">
          <AnimatePresence mode="popLayout">
            {messages.map((msg, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                transition={{ duration: 0.3 }}
              >
                <Message sender={msg.sender} text={renderMessageText(msg)} />
              </motion.div>
            ))}
          </AnimatePresence>
          
          {/* Loading indicator */}
          <AnimatePresence>
            {isLoading && (
              <motion.div 
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="flex items-center px-4 py-6 bg-slate-100 dark:bg-slate-900"
              >
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center shadow-sm mr-4">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  >
                    <ArrowPathIcon className="w-5 h-5 text-white" />
                  </motion.div>
                </div>
                <div className="flex-1">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-slate-600 rounded-full animate-pulse"></div>
                    <div className="w-2 h-2 bg-slate-600 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                    <div className="w-2 h-2 bg-slate-600 rounded-full animate-pulse" style={{animationDelay: '0.4s'}}></div>
                  </div>
                  <span className="text-sm text-slate-600 mt-2 block">Analyzing regulations...</span>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Input Area - LangUI style */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="px-6 py-4 bg-white border-t border-gray-200"
      >
        <div className="max-w-3xl mx-auto">
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
                placeholder="Ask me anything about DIFC or ADGM regulations..."
                className="block w-full resize-none rounded-xl border-none bg-slate-200 p-4 pr-20 text-sm text-slate-900 shadow-md focus:outline-none focus:ring-2 focus:ring-blue-600 dark:bg-slate-800 dark:text-slate-200 dark:placeholder-slate-400 dark:focus:ring-blue-600 sm:text-base"
                rows="1"
                disabled={isLoading}
                style={{
                  scrollbarWidth: 'none',
                  msOverflowStyle: 'none',
                }}
                required
              />
              <motion.button
                type="submit"
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                disabled={isLoading || !inputValue.trim()}
                className={`absolute bottom-2 right-2.5 rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
                  isLoading || !inputValue.trim()
                    ? 'bg-gray-400 text-gray-600 cursor-not-allowed'
                    : 'bg-blue-700 text-slate-200 hover:bg-blue-800 focus:outline-none focus:ring-4 focus:ring-blue-300 dark:bg-blue-600 dark:hover:bg-blue-700 dark:focus:ring-blue-800'
                } sm:text-base`}
              >
                <AnimatePresence mode="wait">
                  {isLoading ? (
                    <motion.div
                      key="loading"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex items-center gap-2"
                    >
                      <motion.div
                        animate={{ rotate: 360 }}
                        transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                      >
                        <ArrowPathIcon className="w-4 h-4" />
                      </motion.div>
                      Sending...
                    </motion.div>
                  ) : (
                    <motion.div
                      key="send"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      Send
                    </motion.div>
                  )}
                </AnimatePresence>
                <span className="sr-only">Send message</span>
              </motion.button>
            </div>
          </form>
        </div>
      </motion.div>
    </div>
  );
}

export default ChatInterface;

