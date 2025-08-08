// Enhanced loading animation component
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { RingLoader, DotLoader } from 'react-spinners';

const LoadingAnimation = ({ darkMode, type = 'thinking' }) => {
  const messages = [
    "Analyzing regulatory documents...",
    "Processing compliance requirements...",
    "Searching through legal frameworks...",
    "Generating comprehensive response...",
    "Reviewing jurisdictional differences...",
  ];

  const [currentMessage, setCurrentMessage] = React.useState(0);

  React.useEffect(() => {
    const interval = setInterval(() => {
      setCurrentMessage((prev) => (prev + 1) % messages.length);
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <motion.div 
      initial={{ opacity: 0, y: 20, scale: 0.9 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.9 }}
      className="flex items-center px-6 py-8 bg-white/60 dark:bg-gray-800/60 backdrop-blur-xl border border-gray-200/50 dark:border-gray-600/50 rounded-3xl mx-4 my-4 shadow-2xl relative overflow-hidden"
    >
      {/* Animated background gradient */}
      <motion.div
        animate={{
          background: [
            'linear-gradient(45deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1))',
            'linear-gradient(45deg, rgba(147, 51, 234, 0.1), rgba(236, 72, 153, 0.1))',
            'linear-gradient(45deg, rgba(236, 72, 153, 0.1), rgba(59, 130, 246, 0.1))',
          ]
        }}
        transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
        className="absolute inset-0 rounded-3xl"
      />
      
      <div className="flex items-center justify-center mr-6 relative z-10">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
          className="relative"
        >
          <RingLoader 
            color={darkMode ? "#8b5cf6" : "#7c3aed"} 
            size={50}
            speedMultiplier={0.8}
          />
          <motion.div
            animate={{ scale: [1, 1.2, 1] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="absolute inset-0 flex items-center justify-center"
          >
            <div className="w-6 h-6 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full shadow-lg" />
          </motion.div>
        </motion.div>
      </div>
      
      <div className="flex-1 relative z-10">
        <motion.div 
          className="flex items-center gap-3 mb-3"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
        >
          <span className="text-lg font-bold bg-gradient-to-r from-blue-600 to-purple-600 dark:from-blue-400 dark:to-purple-400 bg-clip-text text-transparent">
            AI is thinking
          </span>
          <DotLoader 
            color={darkMode ? "#6366f1" : "#4f46e5"} 
            size={20}
            speedMultiplier={1.2}
          />
        </motion.div>
        
        <AnimatePresence mode="wait">
          <motion.p 
            key={currentMessage}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.5 }}
            className="text-sm text-gray-600 dark:text-gray-400 font-medium"
          >
            {messages[currentMessage]}
          </motion.p>
        </AnimatePresence>
        
        {/* Progress indicator */}
        <motion.div 
          initial={{ width: 0 }}
          animate={{ width: "100%" }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
          className="h-1 bg-gradient-to-r from-blue-500 via-purple-500 to-pink-500 rounded-full mt-4 opacity-60"
        />
      </div>
    </motion.div>
  );
};

export default LoadingAnimation;
