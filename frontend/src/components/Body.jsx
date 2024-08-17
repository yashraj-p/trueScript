import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './Body.css';

function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const API_BASE_URL = 'http://127.0.0.1:5000';

  const api = axios.create({
    baseURL: API_BASE_URL,
  });

  useEffect(() => {
    const initializeModel = async (modelZipPath, modelExtractPath) => {
      await api.post('/initialize_model', {
        model_zip_path: modelZipPath,
        model_extract_path: modelExtractPath,
      });
    };
    initializeModel('ai_txt_detection_bertModel_epoch10EarlyFeatures.zip', 'extracted_model');
  }, []);

  const ClassifyText = async (text) => {
    const response = await api.post('/classify_text', { text });
    return response.data.classification;
  };

  const handleInitializeAndClassifyText = async (e) => {
    e.preventDefault();
    try {
      if(text.length > 0){
        setResult('Detecting..');
        const classification = await ClassifyText(text);
      setResult(classification);
      }
      else {
        setResult('');
        alert("Enter text to see results");
      }
    } catch (error) {
      console.error('Error initializing model and classifying text:', error);
    }
  };

  return (
    <div className="container">
      <h1>Enter your text here</h1>
      <textarea
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Enter text to classify"
      ></textarea>
      <button onClick={handleInitializeAndClassifyText}>Classify Text</button>
      {result && <p className="result">Classification Result: {result}</p>}
    </div>
  );
}

export default App;