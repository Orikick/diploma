// RLPlotsComponent.jsx - Компонент для відображення графіків R_avg(L) та Rw_avg(L)
import React, { useState } from 'react';
import { Card, Row, Col, Button, Spinner, Alert, Nav } from 'react-bootstrap';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

const RLPlotsComponent = ({ corpusResults }) => {
  const [plots, setPlots] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('r_avg_linear');

  const generatePlots = async () => {
    if (!corpusResults || corpusResults.length === 0) {
      setError('Немає результатів корпусного аналізу для побудови графіків');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const response = await axios.post(`${API_URL}/rl-plots`, {
        corpus_results: corpusResults
      });

      if (response.data.success) {
        setPlots(response.data.plots);
      } else {
        setError('Помилка генерації графіків: ' + response.data.error);
      }
      
      setLoading(false);
    } catch (err) {
      setError('Помилка генерації графіків: ' + err.message);
      setLoading(false);
    }
  };

  return (
    <Card>
      <Card.Header>
        <div className="d-flex justify-content-between align-items-center">
          <h5 className="mb-0">Залежності R_avg(L) та Rw_avg(L)</h5>
          <Button 
            variant="primary" 
            onClick={generatePlots}
          >
            {loading ? <Spinner size="sm" animation="border" /> : 'Згенерувати графіки'}
          </Button>
        </div>
      </Card.Header>
      <Card.Body>
        {error && (
          <Alert variant="danger" onClose={() => setError(null)} dismissible>
            {error}
          </Alert>
        )}
        
        {plots ? (
          <>
            <Nav variant="tabs" activeKey={activeTab} onSelect={k => setActiveTab(k)} className="mb-3">
              <Nav.Item>
                <Nav.Link eventKey="r_avg_linear">R_avg (лінійний)</Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="r_avg_log">R_avg (лог.)</Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="rw_avg_linear">Rw_avg (лінійний)</Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="rw_avg_log">Rw_avg (лог.)</Nav.Link>
              </Nav.Item>
              <Nav.Item>
                <Nav.Link eventKey="both_log">Обидва (лог.)</Nav.Link>
              </Nav.Item>
            </Nav>
            
            <div className="text-center">
              {plots[activeTab] && (
                <img 
                  src={`data:image/png;base64,${plots[activeTab]}`} 
                  alt="R_L plot" 
                  style={{ maxWidth: '100%', maxHeight: '500px' }}
                />
              )}
            </div>
          </>
        ) : (
          <div className="text-center p-5">
            {loading ? (
              <>
                <Spinner animation="border" />
                <p className="mt-3">Генерація графіків...</p>
              </>
            ) : (
              <p>Натисніть "Згенерувати графіки" для відображення залежностей R_avg(L) та Rw_avg(L)</p>
            )}
          </div>
        )}
      </Card.Body>
    </Card>
  );
};

export default RLPlotsComponent;