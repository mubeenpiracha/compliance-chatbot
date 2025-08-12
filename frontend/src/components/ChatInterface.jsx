// frontend/src/components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import Citation from './Citation';
import LoadingAnimation from './LoadingAnimation';
import { PaperAirplaneIcon, ArrowPathIcon, Squares2X2Icon, MoonIcon, SunIcon } from '@heroicons/react/24/outline';
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
      // Only send valid history messages to backend
      const validHistory = messages.slice(1).filter(msg => msg.sender && typeof msg.text === 'string');
      const response = await axios.post('http://localhost:8000/api/v1/chat', {
        message: userMessage.text,
        jurisdiction: jurisdiction,
        history: validHistory
      });

            // The backend now sends a response that perfectly matches our message format.
      // We can use it directly without any transformation.
      const aiResponse = response.data;

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

  return (
    <div className="flex flex-col h-screen max-w-5xl mx-auto">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg"
      >
        <div className="flex items-center gap-4">
          <motion.div 
            whileHover={{ scale: 1.05, rotate: 5 }}
            className="w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500 via-purple-600 to-pink-500 flex items-center justify-center shadow-lg"
          >
            <Squares2X2Icon className="w-6 h-6 text-white" />
          </motion.div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
              Compliance Assistant
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">AI-powered regulatory guidance</p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          {/* Dark Mode Toggle */}
          <motion.button
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            onClick={() => setDarkMode(!darkMode)}
            className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-all duration-200"
          >
            <AnimatePresence mode="wait">
              {darkMode ? (
                <motion.div
                  key="sun"
                  initial={{ rotate: -90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: 90, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <SunIcon className="w-5 h-5" />
                </motion.div>
              ) : (
                <motion.div
                  key="moon"
                  initial={{ rotate: 90, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  exit={{ rotate: -90, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                >
                  <MoonIcon className="w-5 h-5" />
                </motion.div>
              )}
            </AnimatePresence>
          </motion.button>
          
          {/* Jurisdiction Selector */}
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-600 dark:text-gray-400">Jurisdiction:</span>
            <div className="flex bg-gray-100 dark:bg-gray-800 rounded-xl p-1 shadow-inner">
              {['DIFC', 'ADGM', 'Both'].map((j) => (
                <motion.button
                  key={j}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={() => setJurisdiction(j)}
                  className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all duration-300 ${
                    jurisdiction === j
                      ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white shadow-lg transform scale-105'
                      : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                >
                  {j}
                </motion.button>
              ))}
            </div>
          </div>
        </div>
      </motion.div>

      {/* Messages Area */}
      <div className="flex-1 overflow-y-auto bg-gradient-to-br from-gray-50 via-blue-50/30 to-purple-50/30 dark:from-gray-900 dark:via-blue-950/30 dark:to-purple-950/30 text-sm leading-6 text-slate-900 dark:text-slate-100 sm:text-base sm:leading-7 relative">
        {/* Enhanced animated background elements */}
        <div className="absolute inset-0 overflow-hidden pointer-events-none">
          <motion.div
            animate={{ 
              scale: [1, 1.3, 1],
              rotate: [0, 120, 240, 360],
              opacity: [0.05, 0.15, 0.05]
            }}
            transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
            className="absolute top-10 right-16 w-40 h-40 bg-gradient-to-r from-blue-400/20 via-purple-500/20 to-pink-400/20 rounded-full blur-2xl"
          />
          <motion.div
            animate={{ 
              scale: [1.2, 1, 1.2],
              rotate: [360, 180, 0],
              opacity: [0.05, 0.2, 0.05]
            }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="absolute bottom-16 left-16 w-32 h-32 bg-gradient-to-r from-pink-400/20 via-blue-500/20 to-purple-400/20 rounded-full blur-2xl"
          />
          <motion.div
            animate={{ 
              scale: [1, 1.4, 1],
              rotate: [0, 180, 360],
              opacity: [0.03, 0.12, 0.03]
            }}
            transition={{ duration: 30, repeat: Infinity, ease: "linear" }}
            className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-48 h-48 bg-gradient-to-r from-indigo-400/15 via-cyan-400/15 to-teal-400/15 rounded-full blur-3xl"
          />
        </div>
        
        <div className="max-w-4xl mx-auto relative z-10">
          <AnimatePresence mode="popLayout">
            {messages.map((msg, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: -20, scale: 0.95 }}
                transition={{ duration: 0.4, type: "spring", stiffness: 100 }}
                className="relative z-10"
              >
                <Message message={msg} darkMode={darkMode} />
              </motion.div>
            ))}
          </AnimatePresence>
          
          {/* Enhanced Loading indicator */}
          <AnimatePresence>
            {isLoading && (
              <LoadingAnimation darkMode={darkMode} />
            )}
          </AnimatePresence>
          
          <div ref={messagesEndRef} />
        </div>
      </div>

      {/* Enhanced Input Area */}
      <motion.div 
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="px-6 py-6 bg-white/80 dark:bg-gray-900/80 backdrop-blur-lg border-t border-gray-200 dark:border-gray-700"
      >
        <div className="max-w-4xl mx-auto">
          <form onSubmit={handleSendMessage}>
            <div className="relative group">
              <motion.div
                whileFocus={{ scale: 1.02 }}
                className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-gray-50 to-gray-100 dark:from-gray-800 dark:to-gray-900 p-1 shadow-lg"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 opacity-20 dark:opacity-30 rounded-2xl blur-sm group-focus-within:opacity-40 transition-opacity duration-300" />
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
                  className="relative block w-full resize-none rounded-xl border-none bg-white/90 dark:bg-gray-800/90 backdrop-blur-sm p-4 pr-20 text-sm text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-purple-500 transition-all duration-300 sm:text-base"
                  rows="1"
                  disabled={isLoading}
                  style={{
                    scrollbarWidth: 'none',
                    msOverflowStyle: 'none',
                  }}
                  required
                />
              </motion.div>
              
              <motion.button
                type="submit"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                disabled={isLoading || !inputValue.trim()}
                className={`absolute bottom-3 right-3 p-3 rounded-xl text-sm font-semibold transition-all duration-300 shadow-lg ${
                  isLoading || !inputValue.trim()
                    ? 'bg-gray-300 dark:bg-gray-600 text-gray-500 dark:text-gray-400 cursor-not-allowed'
                    : 'bg-gradient-to-r from-blue-500 via-purple-600 to-blue-600 hover:from-blue-600 hover:via-purple-700 hover:to-blue-700 text-white shadow-xl hover:shadow-2xl transform hover:-translate-y-0.5'
                }`}
              >
                <AnimatePresence mode="wait">
                  {isLoading ? (
                    <motion.div
                      key="loading"
                      initial={{ opacity: 0, rotate: -90 }}
                      animate={{ opacity: 1, rotate: 0 }}
                      exit={{ opacity: 0, rotate: 90 }}
                      className="flex items-center justify-center"
                    >
                      <ArrowPathIcon className="w-5 h-5 animate-spin" />
                    </motion.div>
                  ) : (
                    <motion.div
                      key="send"
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0, x: -10 }}
                      className="flex items-center justify-center"
                    >
                      <PaperAirplaneIcon className="w-5 h-5" />
                    </motion.div>
                  )}
                </AnimatePresence>
                <span className="sr-only">Send message</span>
              </motion.button>
            </div>
          </form>
          
          {/* Modern Footer */}
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            className="text-center mt-4"
          >
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Powered by AI • Regulatory compliance made simple
            </p>
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
}

export default ChatInterface;

