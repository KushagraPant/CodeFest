import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../styles/FrontPage.css'; 

const images = [
  '/images/slide1.jpg',
  '/images/slide2.jpg',
  '/images/slide3.jpg',
];

const FrontPage = () => {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentIndex(prev => (prev + 1) % images.length);
    }, 4000); // 4 seconds

    return () => clearInterval(interval);
  }, []);

  return (
    <section className="front-page">
      <div className="slideshow-container">
        {images.map((image, index) => (
          <img
            key={index}
            src={image}
            className={`slide ${index === currentIndex ? 'active' : ''}`}
            alt={`Slide ${index + 1}`}
          />
        ))}
      </div>
       <div className="overlay-content">
    <h1 className="main-title">Welcome to TrendScope</h1>
    <p className="tagline">AI-Powered Research. Real-World Results.</p>
    <Link to="/input">
      <button className="get-started-btn">Get Started</button>
    </Link>
  </div>
</section>

  );
};

export default FrontPage;
