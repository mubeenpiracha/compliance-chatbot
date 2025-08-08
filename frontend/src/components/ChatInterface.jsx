// frontend/src/components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import Citation from './Citation'; // Import the Citation component
import { Send, Loader } from 'lucide-react';
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
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { sender: 'ai', text: 'Sorry, I encountered an error. Please try again.' };
      setMessages(prev => [...prev, errorMessage]);
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
     <div className="flex flex-col h-[90vh] w-full max-w-4xl mx-auto bg-gray-800 rounded-lg shadow-xl">
      {/* Jurisdiction Selector */}
      <div className="p-2 bg-gray-900 rounded-t-lg flex justify-center items-center gap-4">
        <span className="text-sm font-medium text-gray-400">Jurisdiction:</span>
        <div>
          {['DIFC', 'ADGM', 'Both'].map((j) => (
            <button
              key={j}
              onClick={() => setJurisdiction(j)}
              className={`px-3 py-1 text-sm font-semibold rounded-md transition-colors ${ 
                jurisdiction === j
                  ? 'bg-cyan-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {j}
            </button>
          ))}
        </div>
      </div>
      {/* Message Display Area */}
      <div className="flex-grow p-6 overflow-y-auto">
        {messages.map((msg, index) => (
          <Message key={index} sender={msg.sender} text={renderMessageText(msg)} />
        ))}
        {/* Show a loading indicator while waiting for AI response */}
        {isLoading && (
          <div className="flex justify-start">
              <div className="flex items-center gap-4 my-4">
                  <div className="flex-shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-cyan-500">
                      <Loader className="animate-spin" />
                  </div>
                  <div className="p-4 rounded-lg bg-gray-700 rounded-bl-none">
                      <p className="text-white italic">Thinking...</p>
                  </div>
              </div>
          </div>
        )}
        {/* Empty div to act as a scroll target */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Form Area */}
      <div className="p-4 bg-gray-900 rounded-b-lg">
        <form onSubmit={handleSendMessage} className="flex items-center gap-4">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question..."
            className="flex-grow p-3 bg-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500 text-white disabled:opacity-50"
            disabled={isLoading} // Disable input while loading
          />
          <button
            type="submit"
            className="bg-cyan-600 hover:bg-cyan-700 text-white font-bold p-3 rounded-lg transition-colors duration-200 flex items-center justify-center disabled:opacity-50"
            disabled={isLoading} // Disable button while loading
          >
            <Send size={24} />
          </button>
        </form>
      </div>
    </div>
  );
}

export default ChatInterface;

