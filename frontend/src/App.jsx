// frontend/src/App.jsx
import React from 'react';
import ChatInterface from './components/ChatInterface'; // 1. Import the new component

function App() {
  return (
    <div className="bg-gray-900 text-white min-h-screen flex flex-col p-4 items-center justify-center">
      {/* We've simplified the main App component. It now centers the chat interface. */}
      <ChatInterface />
    </div>
  );
}

export default App;