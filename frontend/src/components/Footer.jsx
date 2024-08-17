import React from 'react';
import './Footer.css';
import tslogo from '../assets/tslogo.png';

const Footer = () => {
  return (
    <footer className="footer">
      <div className="footer-logo">
        <span>truescript</span>
      </div>
      <div className="footer-links">
        <a href="/privacy">Privacy Policy</a>
        <a href="/terms">Terms of Service</a>
        <a href="/support">Support</a>
      </div>
    </footer>
  );
};

export default Footer;
