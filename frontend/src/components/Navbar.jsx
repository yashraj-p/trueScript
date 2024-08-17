import React from 'react';
import './Navbar.css';
import tslogo from '../assets/tslogo.png';
const Navbar = () => {
  return (
    <nav className="navbar">
      <div className="navbar-logo">
        <img style={{width:'3rem'}} src={tslogo} alt="Truescript Logo" />
        <span>truescript</span>
      </div>
      <ul className="navbar-links">
       
      </ul>
    </nav>
  );
};

export default Navbar;