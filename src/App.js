import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';  
import ContactUsPage from './pages/ContactUsPage'; 
import MainPage from './pages/MainPage';
import InputPage from './pages/InputPage';
import AnalysisPage from './pages/AnalysisPage';
import AboutUsPage from './pages/AboutUsPage';

const App = () => {
    return (
      <Router>
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/contact" element={<ContactUsPage />} />
          <Route path="/input" element={<InputPage />} />
          <Route path="/result" element={<AnalysisPage />} /> 
          <Route path="/aboutus" element={<AboutUsPage />} /> 
        </Routes>
      </Router>
    );
  };
  

export default App;
