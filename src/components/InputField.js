import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import '../styles/InputField.css';

const InputField = () => {
  const [dropdownValue, setDropdownValue] = useState('');
  const [textValue, setTextValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!dropdownValue || !textValue) {
      setError('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('https://backend5-production.up.railway.app/analyze/niche', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          business_keyword: dropdownValue,
          business_description: textValue
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      localStorage.setItem('analysisResult', JSON.stringify(data));
      navigate('/result');
    } catch (error) {
      console.error('Error submitting form:', error);
      setError('Failed to analyze. Please try again later.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="input-form-container">
      <div className="input-form-image-container">
        <img src="/images/form.jpg" alt="Business Info" />
      </div>
      <form onSubmit={handleSubmit} className="input-form-box">
        <h2 className="input-form-title">Write your Business Info</h2>

        {error && <div className="input-form-error-message">{error}</div>}

        <label>
          Choose your business niche:
          <select
            value={dropdownValue}
            onChange={(e) => setDropdownValue(e.target.value)}
            className="input-form-input"
            required
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
          Describe your business:
          <textarea
            value={textValue}
            onChange={(e) => setTextValue(e.target.value)}
            className="input-form-input input-form-large-textbox"
            placeholder="Describe your business model, target audience, and unique value proposition..."
            required
            rows={6}
          />
        </label>

        <button
          type="submit"
          className="input-form-submit-button"
          disabled={isLoading}
        >
          {isLoading ? 'Analyzing...' : 'Analyze My Niche'}
        </button>
      </form>
    </div>
  );
};

export default InputField;
