// frontend/src/components/ChatInterface.jsx
import React, { useState, useEffect, useRef } from 'react';
import Message from './Message';
import { Send, Loader } from 'lucide-react';
import axios from 'axios';

function ChatInterface() {
  const [messages, setMessages] = useState([
    { sender: 'ai', text: 'Hello! How can I help you with DIFC or ADGM regulations today?' }
  ]);
  const [inputValue, setInputValue] = useState('');
  // New state to track when the AI is "thinking"
  const [isLoading, setIsLoading] = useState(false);

  // A ref to the message container div
  const messagesEndRef = useRef(null);

  // This function scrolls the message container to the bottom
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Scroll to bottom whenever messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (inputValue.trim() === '' || isLoading) return;

    const userMessage = { sender: 'user', text: inputValue };
    // Add user message to the list and set loading state to true
    setMessages(prevMessages => [...prevMessages, userMessage]);
    setIsLoading(true);
    setInputValue('');

    try {
      // Send the user's message to the backend
      const response = await axios.post('http://localhost:8000/api/v1/chat', {
        message: userMessage.text
      });

      // Add the AI's response to the message list
      setMessages(prevMessages => [...prevMessages, response.data]);

    } catch (error) {
      console.error("Error fetching AI response:", error);
      // Add an error message to the chat if the API call fails
      const errorMessage = { sender: 'ai', text: 'Sorry, I encountered an error. Please try again.' };
      setMessages(prevMessages => [...prevMessages, errorMessage]);
    } finally {
      // Set loading state back to false
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[90vh] w-full max-w-4xl mx-auto bg-gray-800 rounded-lg shadow-xl">
      {/* Message Display Area */}
      <div className="flex-grow p-6 overflow-y-auto">
        {messages.map((msg, index) => (
          <Message key={index} message={msg} />
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
