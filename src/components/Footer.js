import React from 'react';
import '../styles/Footer.css';

const Footer = () => {
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
                <button className="btn btn-green">Get Started</button>
                <button className="btn btn-outline">Watch Demo</button>
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
