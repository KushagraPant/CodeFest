import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import '../styles/Header.css';

const Header = () => {
  const [showHeader, setShowHeader] = useState(true);
  let lastScrollY = 0;

  const handleScroll = () => {
    if (window.scrollY > lastScrollY) {
      setShowHeader(false);
    } else {
      setShowHeader(true);
    }
    lastScrollY = window.scrollY > 0 ? window.scrollY : 0;
  };

  useEffect(() => {
    window.addEventListener('scroll', handleScroll);
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <header className={`header ${showHeader ? 'show' : 'hide'}`}>
      <div className="left-section">
        <img src="/images/logo.jpg" alt="Logo" className="logo" />
        <h1 className="site-name">TrendScope</h1>
      </div>
      <div className="right-section">
      <Link to="/" className="nav-link">Home</Link>
        <Link to="/aboutus" className="nav-link">About Us</Link>
        <Link to="/contact" className="nav-link">Contact Us</Link>
      </div>
    </header>
  );
};

export default Header;
