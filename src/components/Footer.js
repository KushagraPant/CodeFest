import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Footer.css';

const Footer = () => {

  const handleWatchDemo = () => {
    window.open('https://www.youtube.com/watch?v=a2Oxv18GOPs', '_blank');
  };

  return (
    <footer className="footer">
      <div className="footer-top">
        <div className="footer-left">
          <div className="logo" />
          <h2>TrendScope</h2>
          <div className="contact-info">
            <p><strong>Email</strong><br />hello@trendscope.com</p>
            <p><strong>Phone Number</strong><br />+91 1234567890</p>
          </div>
        </div>
        <div className="footer-right">
          <h3>Get started with your personal<br />AI Market Research assistant today</h3>
          <div className="cta-buttons">
            <Link to="/input">
              <button className="btn btn-green">Get Started</button>
            </Link>
            <button 
              className="btn btn-outline" 
              onClick={handleWatchDemo}
            >
              Watch Demo
            </button>
          </div>
        </div>
      </div>

      <hr />

      <div className="footer-bottom">
        <p>© 2025 TrendScope. All rights reserved.</p>
      </div>
    </footer>
  );
};

export default Footer;
