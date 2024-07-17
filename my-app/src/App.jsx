import React, { useState } from 'react';
import styled from 'styled-components';
import axios from 'axios'; // Import Axios for making HTTP requests

const Container = styled.div`
  width: 800px;
  margin: 0 auto;
  padding: 20px;
`;

const Navbar = styled.div`
  background-color: #f8f9fa;
  padding: 10px 20px;
  text-align: center;
  font-size: 1.5em;
  font-weight: bold;
`;

const InputOutputContainer = styled.div`
  display: flex;
  justify-content: space-between;
  margin-top: 20px;
  gap: 20px;
`;

const TextArea = styled.textarea`
  width: 60%;
  height: 600px;
  padding: 10px;
  font-size: 1em;
  border: 1px solid #ced4da;
  border-radius: 4px;
  resize: none;
`;

const OutputBox = styled.div`
  width: 45%;
  height: 300px;
  padding: 10px;
  font-size: 1em;
  border: 1px solid #ced4da;
  border-radius: 4px;
  background-color: #e9ecef;
  overflow-y: auto;
`;

const SummarizeButton = styled.button`
  margin-top: 20px;
  padding: 10px 20px;
  font-size: 1em;
  border: none;
  border-radius: 4px;
  background-color: #007bff;
  color: white;
  cursor: pointer;
`;

function App() {
  const [inputText, setInputText] = useState('');
  const [summary, setSummary] = useState('');

  const handleSummarize = () => {
    // Make API request to your backend
    axios.post('http://127.0.0.1:5000/summarize', { text: inputText })
      .then(response => {
        setSummary(response.data.summary);
      })
      .catch(error => {
        console.error('Error fetching data:', error);
      });
  };

  return (
    <Container>
      <Navbar>Text Summarizer</Navbar>
      <InputOutputContainer>
        <TextArea
          placeholder="Enter or paste your text here..."
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
        />
        <OutputBox>
          {summary}
        </OutputBox>
      </InputOutputContainer>
      <SummarizeButton onClick={handleSummarize}>Summarize</SummarizeButton>
    </Container>
  );
}

export default App;
