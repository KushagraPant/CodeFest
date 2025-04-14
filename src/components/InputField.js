import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import '../styles/InputField.css';

const InputField = () => {
  const [dropdownValue, setDropdownValue] = useState('');
  const [textValue, setTextValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    console.log('Dropdown:', dropdownValue);
    console.log('Text:', textValue);
  };

  return (
    <div className="container">
      <div className="image-container">
        <img src="/images/form.jpg" alt="Business Info" />
      </div>
      <form onSubmit={handleSubmit} className="form-box">
        <h2 className="form-title">Write your Business Info</h2>
        <label>
          Choose an option:
          <select
            value={dropdownValue}
            onChange={(e) => setDropdownValue(e.target.value)}
            className="input"
          >
            <option value="">-- Select --</option>
            <option value="Health & Wellness">Health & Wellness</option>
            <option value="Technology & Gadgets">Technology & Gadgets</option>
            <option value="Education & E-learning">Education & E-learning</option>
            <option value="Eco-Friendly & Sustainable Products">Eco-Friendly & Sustainable Products</option>
            <option value="Personal Finance & Investing">Personal Finance & Investing</option>
            <option value="Beauty & Skincare">Beauty & Skincare</option>
            <option value="Fashion & Apparel">Fashion & Apparel</option>
            <option value="Food & Beverage">Food & Beverage</option>
            <option value="Pet Products & Services">Pet Products & Services</option>
            <option value="Travel & Adventure">Travel & Adventure</option>
            <option value="Home Decor & DIY">Home Decor & DIY</option>
            <option value="Gaming & Entertainment">Gaming & Entertainment</option>
          </select>
        </label>

        <label>
          Enter text:
          <textarea
            value={textValue}
            onChange={(e) => setTextValue(e.target.value)}
            className="input large-textbox"
            placeholder="Type something..."
          />
        </label>
        <Link to="/result">
        <button className="submit-button">Submit</button>
        </Link>
      </form>
    </div>
  );
};

export default InputField;
