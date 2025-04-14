import React, { useState } from 'react';
import '../styles/AboutUs.css';

const ValueCard = ({ title, image, shortText, longText }) => {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className={`about-value-card ${expanded ? 'about-expanded' : ''}`}>
      <img src={image} alt={title} />
      <div className="about-card-content">
        <h3>{title}</h3>
        <p>
          {shortText}
          <span
            className="about-see-more"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? ' See Less' : ' See More'}
          </span>
        </p>
        {expanded && <p className="about-hidden-text">{longText}</p>}
      </div>
    </div>
  );
};

const AboutUs = () => {
  return (
    <div className="about-container">
      <div className="about-section">
        <div className="about-image-container">
          <img
            src="https://images.unsplash.com/photo-1522071820081-009f0129c71c?auto=format&fit=crop&w=1170&q=80"
            alt="Our Team"
          />
        </div>
        <div className="about-content">
          <h1>About Us</h1>
          <p className="about-subheading">
            We're a passionate team dedicated to innovation and excellence.
          </p>
          <p>
            Founded in 2015, our company has grown from a small startup to an industry leader, driven by our commitment to quality and customer satisfaction.
          </p>
          <p>
            Our diverse team brings together expertise from various backgrounds, allowing us to approach challenges with fresh perspectives and creative solutions.
          </p>
        </div>
      </div>

      <div className="about-values-grid">
        <ValueCard
          title="Innovation"
          image="https://images.unsplash.com/photo-1507679799987-c73779587ccf?auto=format&fit=crop&w=1171&q=80"
          shortText="We constantly push boundaries and explore new possibilities."
          longText="Our innovation process involves continuous research, development, and implementation of cutting-edge technologies. We foster a culture of creativity and experimentation."
        />
        <ValueCard
          title="Integrity"
          image="https://images.unsplash.com/photo-1552664730-d307ca884978?auto=format&fit=crop&w=1170&q=80"
          shortText="We believe in transparency, honesty, and ethical practices."
          longText="We maintain high ethical standards in all aspects of our operations, from decision-making to daily interactions. This commitment builds lasting trust with our stakeholders."
        />
        <ValueCard
          title="Excellence"
          image="https://images.unsplash.com/photo-1552664730-d307ca884978?auto=format&fit=crop&w=1170&q=80"
          shortText="We strive for the highest standards in our work."
          longText="Our pursuit of excellence is reflected in our attention to detail, quality assurance processes, and continuous improvement initiatives."
        />
      </div>

      <div className="about-team-section">
        <div className="about-team-heading">
          <h2>Our Mission</h2>
          <p>
            To create innovative solutions that empower businesses and individuals to achieve their goals. We're committed to sustainable practices and making a positive impact on our community.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AboutUs;
