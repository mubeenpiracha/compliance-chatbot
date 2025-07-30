// frontend/src/components/DocumentList.jsx
import React, { useState, useEffect } from 'react';
import axios from 'axios'; // Import axios to make API calls

function DocumentList() {
  // --- React Hooks ---
  // 1. 'useState' is a hook to create state variables.
  //    'documents' will hold our array of data. 'setDocuments' is the function to update it.
  const [documents, setDocuments] = useState([]);
  //    We also create state for loading and error messages.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  // Learn more about useState: https://react.dev/reference/react/useState

  // 2. 'useEffect' is a hook to perform "side effects" like fetching data.
  //    The code inside runs after the component renders.
  // Learn more about useEffect: https://react.dev/reference/react/useEffect
  useEffect(() => {
    // We define an async function to fetch the data
    const fetchDocuments = async () => {
      try {
        // Use axios to send a GET request to our backend endpoint
        const response = await axios.get('http://localhost:8000/api/v1/documents');
        // If successful, update our state with the data from the response
        setDocuments(response.data);
        setError(null); // Clear any previous errors
      } catch (err) {
        // If an error occurs, update the error state
        setError('Failed to fetch documents. Is the backend server running?');
        console.error(err);
      } finally {
        // No matter what, set loading to false once the request is complete
        setLoading(false);
      }
    };

    fetchDocuments(); // Call the function to start the fetch
  }, []); // The empty array [] means this effect runs only ONCE, when the component first loads.

  // --- Conditional Rendering ---
  // Show a loading message while the data is being fetched
  if (loading) {
    return <div className="text-center p-8">Loading documents...</div>;
  }

  // Show an error message if the fetch failed
  if (error) {
    return <div className="text-center p-8 text-red-500">{error}</div>;
  }

  // --- JSX to Render the List ---
  // If loading is false and there's no error, display the list of documents.
  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <h2 className="text-3xl font-bold text-cyan-300 mb-6 text-center">
        Available Regulatory Documents
      </h2>
      <div className="bg-gray-800 rounded-lg shadow-lg overflow-hidden">
        <ul className="divide-y divide-gray-700">
          {documents.length > 0 ? (
            documents.map((doc) => (
              <li key={doc.doc_id} className="p-4 hover:bg-gray-700 transition-colors duration-200">
                <h3 className="font-semibold text-lg">{doc.title}</h3>
                <p className="text-sm text-gray-400">Jurisdiction: {doc.jurisdiction}</p>
                {doc.source_url && (
                  <a
                    href={doc.source_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-sm text-cyan-400 hover:underline"
                  >
                    Source Link
                  </a>
                )}
              </li>
            ))
          ) : (
            <li className="p-4 text-center text-gray-500">
              No documents found. Use the API docs to add one.
            </li>
          )}
        </ul>
      </div>
    </div>
  );
}

export default DocumentList;