import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/Result.css';

const Result = () => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const storedData = localStorage.getItem('analysisResult');
    if (storedData) {
      try {
        setAnalysisData(JSON.parse(storedData));
      } catch (e) {
        setError('Failed to parse analysis results');
      }
    } else {
      setError('No analysis data found. Please submit the form first.');
    }
    setLoading(false);
  }, []);

  const handleNewAnalysis = () => {
    navigate('/');
  };

  if (loading) {
    return (
      <div className="result-container">
        <div className="result-loading">Loading analysis results...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="result-container">
        <div className="result-error-message">{error}</div>
        <button onClick={handleNewAnalysis} className="result-submit-button">
          Start New Analysis
        </button>
      </div>
    );
  }

  return (
    <div className="result-container">
      <h1>Market Niche Analysis Results</h1>
      <div className="result-search-query">
        <h3>Analysis for: {analysisData?.search_query || 'N/A'}</h3>
      </div>

      <div className="result-section">
        <div className="result-card">
          <h2>Market Analysis</h2>
          {analysisData?.market_analysis ? (
            <div className="result-content">
              <p><strong>Market Size:</strong> {analysisData.market_analysis.total_market_size || 'N/A'} billion</p>
              <p><strong>Growth Rate:</strong> {analysisData.market_analysis.growth_rate || 'N/A'}%</p>
              {analysisData.market_analysis.trends?.length > 0 && (
                <>
                  <h4>Key Trends:</h4>
                  <ul>
                    {analysisData.market_analysis.trends.map((trend, index) => (
                      <li key={index}>{trend}</li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          ) : (
            <p>No market analysis data available</p>
          )}
        </div>

        <div className="result-card">
          <h2>Market Trends</h2>
          {analysisData?.trends?.length > 0 ? (
            <div className="result-content">
              {analysisData.trends.map((trend, index) => (
                <div key={index} className="trend-item">
                  <h3>{trend.title}</h3>
                  <p>{trend.snippet}</p>
                  {trend.url && (
                    <a href={trend.url} target="_blank" rel="noopener noreferrer">
                      Read more
                    </a>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p>No trends data available</p>
          )}
        </div>

        <div className="result-card">
          <h2>Key Competitors</h2>
          {analysisData?.competitors?.length > 0 ? (
            <div className="result-content">
              {analysisData.competitors.map((competitor, index) => (
                <div key={index} className="competitor-item">
                  <h3>{competitor.name}</h3>
                  <p><strong>Market Valuation:</strong> {competitor.market_valuation ? `$${competitor.market_valuation.toLocaleString()}` : 'N/A'}</p>
                  <p><strong>Market Share:</strong> {competitor.market_share ? `${competitor.market_share.toFixed(2)}%` : 'N/A'}</p>
                  <p><small>Source: {competitor.source}</small></p>
                </div>
              ))}
            </div>
          ) : (
            <p>No competitor data available</p>
          )}
        </div>

        <div className="result-card">
          <h2>Sentiment Analysis</h2>
          {analysisData?.sentiment_analysis?.length > 0 ? (
            <div className="result-content">
              {analysisData.sentiment_analysis.map((sentiment, index) => (
                <div key={index} className="sentiment-item">
                  <p>{sentiment.snippet}</p>
                  <p className={`sentiment-${sentiment.sentiment.label.toLowerCase()}`}>
                    <strong>Sentiment:</strong> {sentiment.sentiment.label} ({(sentiment.sentiment.score * 100).toFixed(1)}%)
                  </p>
                </div>
              ))}
            </div>
          ) : (
            <p>No sentiment analysis available</p>
          )}
        </div>

        <div className="result-card">
          <h2>Named Entities</h2>
          {analysisData?.named_entities?.length > 0 ? (
            <div className="result-content">
              {analysisData.named_entities.map((entity, index) => (
                <div key={index} className="entity-item">
                  <h3>From: {entity.source || 'Unknown source'}</h3>
                  {entity.entities?.length > 0 ? (
                    <div className="entity-tags">
                      {entity.entities.map((ent, idx) => (
                        <span key={idx} className={`entity-tag entity-${ent.entity_type.toLowerCase()}`}>
                          {ent.entity} <small>({ent.entity_type})</small>
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p>No entities found in this snippet</p>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <p>No named entity data available</p>
          )}
        </div>

        <button onClick={handleNewAnalysis} className="result-submit-button">
          Start New Analysis
        </button>
      </div>
    </div>
  );
};

export default Result;
