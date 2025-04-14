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
            src="/images/aboutus.jpg"
            alt="Our Team"
          />
        </div>
        <div className="about-content">
          <h1>About Us</h1>
          <p className="about-subheading">
          We're a dynamic team driven by innovation, harnessing the power of AI to deliver smart, data-backed market insights with precision and purpose.
          </p>
          <p className="about-subheading2">
            TrendScope is your intelligent window into tomorrow’s markets. Powered by AI, we analyze real-time data, track emerging trends, and deliver actionable insights—helping businesses stay ahead of the curve with confidence and clarity.
          </p>
          <br></br>
          <p className="about-subheading2">
            Our diverse team brings together expertise from various backgrounds, allowing us to approach challenges with fresh perspectives and creative solutions. 
          </p>
          <br></br>
          <p className="about-subheading3">Our platform transforms complex data into clear, actionable insights—helping brands, startups, and decision-makers stay ahead in fast-moving markets. We don’t just follow trends—we forecast them.</p>
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
          image="/images/abtus.jpg"
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
