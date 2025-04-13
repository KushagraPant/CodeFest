import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';  // Make sure 'Routes' is imported
import ContactUsPage from './pages/ContactUsPage'; 
import MainPage from './pages/MainPage';

const App = () => {
    return (
      <Router>
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/contact" element={<ContactUsPage />} />
        </Routes>
      </Router>
    );
  };
  

export default App;
