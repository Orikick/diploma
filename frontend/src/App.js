// src/App.js
import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Container, Form, Button, Card, Row, Col, Spinner,
  Table, Nav, Tab, Alert, ProgressBar
} from 'react-bootstrap';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter
} from 'recharts';
import ForceGraph2D from 'react-force-graph-2d';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';

const API_URL = 'http://localhost:5000/api';

function App() {
  // State variables
  const [file, setFile] = useState(null);
  const [fileContent, setFileContent] = useState('');
  const [corpusP, setCorpusP] = useState(false);
  const [config, setConfig] = useState({
    n_size: 1,
    split: 'word',
    condition: 'no',
    f_min: 0,
    w: 10,
    wh: 10,
    we: 10,
    wm: 100,
    definition: 'static',
    fmin_for_lmin: 1,
    fmin_for_lmax: 5,
    min_type: 1
  });
  const [analysisMode, setAnalysisMode] = useState('single'); // 'single' or 'corpus'
  const [corpusFiles, setCorpusFiles] = useState([]);
  const [zipFile, setZipFile] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [corpusResult, setCorpusResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('data');
  const [selectedNgram, setSelectedNgram] = useState(null);
  const [graphScale, setGraphScale] = useState('linear');
  const [windowParams, setWindowParams] = useState(null);
  const [markovGraph, setMarkovGraph] = useState(null);
  const [processingProgress, setProcessingProgress] = useState(0);

  // Handle single file upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    setFile(file);
    setAnalysisMode('single');
    
    const reader = new FileReader();
    reader.onload = (e) => {
      setFileContent(e.target.result);
      calculateWindowParams(e.target.result);
    };
    reader.readAsText(file);
  };

  // Handle corpus file uploads
  const handleCorpusUpload = (event) => {
    const files = Array.from(event.target.files);
    setCorpusFiles(files);
    setAnalysisMode('corpus');
    
    // For corpus, we can estimate window parameters based on the first file
    if (files.length > 0) {
      const reader = new FileReader();
      reader.onload = (e) => {
        calculateWindowParams(e.target.result);
      };
      reader.readAsText(files[0]);
    }
  };

  // Handle zip file upload
  const handleZipUpload = (event) => {
    const file = event.target.files[0];
    setZipFile(file);
    setAnalysisMode('corpus');
  };

  // Calculate window parameters based on text length
  const calculateWindowParams = async (text) => {
    try {
      setLoading(true);
      const response = await axios.post(`${API_URL}/calculate-windows`, {
        text: text,
        n_size: config.n_size,
        split: config.split
      });
      
      setWindowParams(response.data);
      setConfig(prev => ({
        ...prev,
        w: response.data.w,
        wh: response.data.w,
        we: response.data.w,
        wm: response.data.wm
      }));
      setLoading(false);
    } catch (err) {
      setError('Failed to calculate window parameters');
      setLoading(false);
    }
  };

  // Process corpus
  const processCorpus = async () => {
    try {
      setLoading(true);
      setError(null);
      setProcessingProgress(0);
      
      const formData = new FormData();
      
      // Add configuration parameters
      Object.keys(config).forEach(key => {
        formData.append(key, config[key]);
      });
      
      if (zipFile) {
        // Upload as zip file
        formData.append('zip_file', zipFile);
        setCorpusP(true);
      } else if (corpusFiles.length > 0) {
        // Upload as multiple files
        corpusFiles.forEach(file => {
          formData.append('files[]', file);
        });
        setCorpusP(true);
      } else {
        setError('No corpus files selected');
        setLoading(false);
        return;
      }
      
      const response = await axios.post(`${API_URL}/upload-corpus`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: progressEvent => {
          const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setProcessingProgress(percentCompleted);
        }
      });
      
      setCorpusResult(response.data);
      setLoading(false);
      setProcessingProgress(100);
      
    } catch (err) {
      setError('Corpus processing failed: ' + (err.response?.data?.error || err.message));
      setLoading(false);
    }
  };

  // Handle form input change
  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setConfig(prev => ({
      ...prev,
      [name]: name === 'n_size' || name === 'f_min' || 
              name === 'w' || name === 'wh' || 
              name === 'we' || name === 'wm' 
              ? parseInt(value) 
              : value
    }));
  };

  // Handle analysis submission
  const handleAnalyze = async () => {
    if (analysisMode === 'single') {
      if (!file) {
        setError('Please upload a file first');
        return;
      }

      try {
        setLoading(true);
        setError(null);
        
        // Make sure we're passing the min_type parameter
        const response = await axios.post(`${API_URL}/analyze`, {
          text: fileContent,
          ...config,
          min_type: parseInt(config.min_type, 10) // Ensure it's an integer
        });
        
        setAnalysisResult(response.data);
        setLoading(false);
        
        // Get Markov graph data if in static mode
        if (config.definition === 'static') {
          getMarkovGraph();
        }
      } catch (err) {
        setError('Analysis failed: ' + (err.response?.data?.error || err.message));
        setLoading(false);
      }
    } else {
      // Corpus mode
      await processCorpus();
    }
  };

  // Get Markov graph data
  const getMarkovGraph = async () => {
    try {
      const response = await axios.post(`${API_URL}/markov-graph`, {
        n_size: config.n_size
      });
      
      if (response.data.error) {
        setError('Failed to generate Markov graph: ' + response.data.error);
      } else {
        setMarkovGraph({
          nodes: response.data.nodes.map(node => ({ 
            id: node.id,
            connections: node.connections,
            val: node.connections
          })),
          links: response.data.edges.map(edge => ({
            source: edge.source,
            target: edge.target,
            value: edge.weight
          }))
        });
      }
    } catch (err) {
      setError('Failed to generate Markov graph: ' + err.message);
    }
  };

  // Handle ngram selection
  const handleNgramSelect = async (ngram) => {
    setSelectedNgram(ngram);
  };

  // Get data for distribution graph
  const getDistributionData = () => {
    if (!selectedNgram || !analysisResult?.ngram_data) return [];
    
    const ngramData = analysisResult.ngram_data[selectedNgram];
    if (!ngramData || !ngramData.bool) return [];
    
    return ngramData.bool.map((value, index) => ({
      position: index,
      value: value
    }));
  };

  // Get data for fluctuation graph
  const getFluctuationData = () => {
    if (!selectedNgram || !analysisResult?.ngram_data) return [];
    
    const ngramData = analysisResult.ngram_data[selectedNgram];
    if (!ngramData || !ngramData.fa) return [];
    
    const faKeys = Object.keys(ngramData.fa).map(Number);
    return faKeys.map(key => ({
      window: key,
      value: ngramData.fa[key],
      fit: ngramData.temp_fa ? ngramData.temp_fa[faKeys.indexOf(key)] : 0
    }));
  };

  // Get data for R/alpha scatter plot
  const getRAlphaData = () => {
    if (!analysisResult?.dataframe) return [];
    
    return analysisResult.dataframe.map(row => ({
      ngram: row.ngram,
      R: row.R,
      b: row.b,
      selected: row.ngram === selectedNgram
    }));
  };

  return (
    <Container fluid>
      {error && (
        <Alert variant="danger" onClose={() => setError(null)} dismissible>
          {error}
        </Alert>
      )}
      
      <Row className="mb-3">
        <Col md={3}>
          <Card>
            <Card.Header>Configuration</Card.Header>
            <Card.Body>
              <Form>
                <Form.Group>
                  <Form.Label>Analysis Mode:</Form.Label>
                  <div className="mb-3">
                    <Form.Check
                      inline
                      type="radio"
                      label="Single File"
                      name="analysisMode"
                      id="singleMode"
                      checked={analysisMode === 'single'}
                      onChange={() => setAnalysisMode('single')}
                    />
                    <Form.Check
                      inline
                      type="radio"
                      label="Corpus"
                      name="analysisMode"
                      id="corpusMode"
                      checked={analysisMode === 'corpus'}
                      onChange={() => setAnalysisMode('corpus')}
                    />
                  </div>
                </Form.Group>

                {analysisMode === 'single' ? (
                  <Form.Group>
                    <Form.Label>Select a txt file for processing:</Form.Label>
                    <Form.Control 
                      type="file" 
                      accept=".txt"
                      onChange={handleFileUpload}
                    />
                    {file && <div className="mt-2">Uploaded file: {file.name}</div>}
                  </Form.Group>
                ) : (
                  <Form.Group>
                    <Form.Label>Select corpus files:</Form.Label>
                    <div className="mb-2">
                      <Form.Control 
                        type="file" 
                        accept=".txt"
                        multiple
                        onChange={handleCorpusUpload}
                        className="mb-2"
                      />
                      <small className="text-muted d-block mb-2">
                        Select multiple .txt files
                      </small>
                      <hr />
                      <Form.Label>Or upload a zip archive of .txt files:</Form.Label>
                      <Form.Control 
                        type="file" 
                        accept=".zip"
                        onChange={handleZipUpload}
                      />
                    </div>
                    
                    {corpusFiles.length > 0 && (
                      <div className="mt-2">
                        <strong>Selected files: {corpusFiles.length}</strong>
                        <ul className="small" style={{maxHeight: '100px', overflowY: 'auto'}}>
                          {corpusFiles.map((file, index) => (
                            <li key={index}>{file.name}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    
                    {zipFile && (
                      <div className="mt-2">
                        <strong>Uploaded zip: {zipFile.name}</strong>
                      </div>
                    )}
                  </Form.Group>
                )}

                <Form.Group className="mt-3">
                  <Form.Label>Size of ngram:</Form.Label>
                  <Form.Control
                    type="number"
                    name="n_size"
                    value={config.n_size}
                    onChange={handleInputChange}
                    min={1}
                  />
                </Form.Group>

                <Form.Group className="mt-3">
                  <Form.Label>Split by:</Form.Label>
                  <Form.Select
                    name="split"
                    value={config.split}
                    onChange={handleInputChange}
                  >
                    <option value="symbol">symbol</option>
                    <option value="word">word</option>
                    <option value="letter">letter</option>
                  </Form.Select>
                </Form.Group>

                <Form.Group className="mt-3">
                  <Form.Label>Boundary Condition:</Form.Label>
                  <Form.Select
                    name="condition"
                    value={config.condition}
                    onChange={handleInputChange}
                  >
                    <option value="no">no</option>
                    <option value="periodic">periodic</option>
                    <option value="ordinary">ordinary</option>
                  </Form.Select>
                </Form.Group>

                <Form.Group className="mt-3">
                  <Form.Label>Filter:</Form.Label>
                  <Form.Control
                    type="number"
                    name="f_min"
                    value={config.f_min}
                    onChange={handleInputChange}
                    min={0}
                  />
                </Form.Group>
                
                {analysisMode === 'corpus' && (
                  <>
                    <Form.Group className="mt-3">
                      <Form.Label>Fmin for Lmin:</Form.Label>
                      <Form.Control
                        type="number"
                        name="fmin_for_lmin"
                        value={config.fmin_for_lmin}
                        onChange={handleInputChange}
                        min={0}
                      />
                    </Form.Group>
                    
                    <Form.Group className="mt-3">
                      <Form.Label>Fmin for Lmax:</Form.Label>
                      <Form.Control
                        type="number"
                        name="fmin_for_lmax"
                        value={config.fmin_for_lmax}
                        onChange={handleInputChange}
                        min={0}
                      />
                    </Form.Group>
                    
                    <Form.Group className="mt-3">
                      <Form.Label>Min shift:</Form.Label>
                      <Form.Select
                        name="min_type"
                        value={config.min_type}
                        onChange={handleInputChange}
                      >
                        <option value={0}>min=0</option>
                        <option value={1}>min=1</option>
                      </Form.Select>
                    </Form.Group>
                  </>
                )}

                <Form.Group className="mt-3">
                  <Form.Label>Sliding window:</Form.Label>
                  <Row>
                    <Col>
                      <Form.Label>Min window:</Form.Label>
                      <Form.Control
                        type="number"
                        name="w"
                        value={config.w}
                        onChange={handleInputChange}
                      />
                    </Col>
                    <Col>
                      <Form.Label>Window shift:</Form.Label>
                      <Form.Control
                        type="number"
                        name="wh"
                        value={config.wh}
                        onChange={handleInputChange}
                      />
                    </Col>
                  </Row>
                  <Row className="mt-2">
                    <Col>
                      <Form.Label>Window expansion:</Form.Label>
                      <Form.Control
                        type="number"
                        name="we"
                        value={config.we}
                        onChange={handleInputChange}
                      />
                    </Col>
                    <Col>
                      <Form.Label>Max window:</Form.Label>
                      <Form.Control
                        type="number"
                        name="wm"
                        value={config.wm}
                        onChange={handleInputChange}
                      />
                    </Col>
                  </Row>
                </Form.Group>

                <Form.Group className="mt-3">
                  <Form.Label>Definition:</Form.Label>
                  <Form.Select
                    name="definition"
                    value={config.definition}
                    onChange={handleInputChange}
                  >
                    <option value="static">static</option>
                    <option value="dynamic">dynamic</option>
                  </Form.Select>
                </Form.Group>

                <Button 
                  variant="primary" 
                  className="mt-3 me-2"
                  onClick={handleAnalyze}

                >
                  {loading ? <Spinner size="sm" animation="border" /> : 'Analyze'}
                </Button>
              </Form>
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={9}>
          <Card>
            <Card.Header>
              <Nav variant="tabs" activeKey={activeTab} onSelect={k => setActiveTab(k)}>
                <Nav.Item>
                  <Nav.Link eventKey="data">Data Table</Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="markov">Markov Chain</Nav.Link>
                </Nav.Item>
              </Nav>
            </Card.Header>
            <Card.Body style={{ height: 'auto', maxHeight: '900px', overflowY: 'auto' }}>
              {loading ? (
                <div className="text-center">
                  <Spinner animation="border" />
                  <p>Processing data...</p>
                  {analysisMode === 'corpus' && (
                    <div className="mt-3">
                      <ProgressBar 
                        now={processingProgress} 
                        label={`${processingProgress}%`} 
                        variant="info" 
                      />
                    </div>
                  )}
                </div>
              ) : (
                <>
                  {analysisMode === 'single' ? (
                    <>
                      {activeTab === 'data' && analysisResult && (
                        <Table striped bordered hover responsive>
                          <thead>
                            <tr>
                              <th>Rank</th>
                              <th>Ngram</th>
                              <th>ƒ</th>
                              <th>R</th>
                              <th>a</th>
                              <th>b</th>
                              <th>Goodness</th>
                            </tr>
                          </thead>
                          <tbody>
                            {analysisResult.dataframe.map((row, index) => (
                              <tr 
                                key={index} 
                                onClick={() => handleNgramSelect(row.ngram)}
                                className={selectedNgram === row.ngram ? 'table-primary' : ''}
                              >
                                <td>{row.rank}</td>
                                <td>{row.ngram}</td>
                                <td>{row.ƒ}</td>
                                <td>{row.R}</td>
                                <td>{row.a}</td>
                                <td>{row.b}</td>
                                <td>{row.goodness}</td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                      )}
                      
                      {activeTab === 'markov' && markovGraph && (
                        <div style={{ height: '380px' }}>
                          <ForceGraph2D
                            graphData={markovGraph}
                            nodeLabel={node => `${node.id} (Connections: ${node.connections})`}
                            linkWidth={link => Math.sqrt(link.value)}
                            nodeAutoColorBy="connections"
                            onNodeClick={node => handleNgramSelect(node.id)}
                          />
                        </div>
                      )}
                    </>
                  ) : (
                    // Corpus mode results
                    corpusResult && (
                      <Table striped bordered hover responsive>
                        <thead>
                          <tr>
                            <th>№</th>
                            <th>File</th>
                            <th>F_min</th>
                            <th>L</th>
                            <th>V</th>
                            <th>Time</th>
                            <th>R_avg</th>
                            <th>dR</th>
                            <th>Rw_avg</th>
                            <th>dRw</th>
                            <th>b_avg</th>
                            <th>db</th>
                            <th>bw_avg</th>
                            <th>dbw</th>
                          </tr>
                        </thead>
                        <tbody>
                          {corpusResult.corpus_results.map((row, index) => (
                            <tr key={index}>
                              <td>{row['№']}</td>
                              <td>{row.file}</td>
                              <td>{row.F_min}</td>
                              <td>{row.L}</td>
                              <td>{row.V}</td>
                              <td>{row.time}</td>
                              <td>{Number(row.R_avg).toFixed(4)}</td>
                              <td>{Number(row.dR).toFixed(4)}</td>
                              <td>{Number(row.Rw_avg).toFixed(4)}</td>
                              <td>{Number(row.dRw).toFixed(4)}</td>
                              <td>{Number(row.b_avg).toFixed(4)}</td>
                              <td>{Number(row.db).toFixed(4)}</td>
                              <td>{Number(row.bw_avg).toFixed(4)}</td>
                              <td>{Number(row.dbw).toFixed(4)}</td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    )
                  )}
                </>
              )}
            </Card.Body>
            <Card.Footer>
              {analysisMode === 'single' && analysisResult && (
                <Row>
                  <Col>Length: {analysisResult.length}</Col>
                  <Col>Vocabulary: {analysisResult.vocabulary}</Col>
                  <Col>Time: {analysisResult.time}s</Col>
                </Row>
              )}
              {analysisMode === 'corpus' && corpusResult && (
                <Row>
                  <Col>Files processed: {corpusResult.file_count}</Col>
                  <Col>Total time: {corpusResult.total_time}s</Col>
                </Row>
              )}
            </Card.Footer>
          </Card>
        </Col>
      </Row>
      
      <Row>
        <Col md={6}>
          <Card>
            <Card.Header>Distribution</Card.Header>
            <Card.Body style={{ height: '400px' }}>
              {selectedNgram && (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={getDistributionData()}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="position" />
                    <YAxis />
                    <Tooltip />
                    <Legend />
                    <Line type="stepAfter" dataKey="value" stroke="#8884d8" />
                  </LineChart>
                </ResponsiveContainer>
              )}
              {!selectedNgram && <div className="text-center mt-5">Select an ngram to view distribution</div>}
            </Card.Body>
          </Card>
        </Col>
        
        <Col md={6}>
          <Card>
            <Card.Header>
              <div className="d-flex justify-content-between align-items-center">
                <div>Fluctuation Analysis</div>
                <Form.Group>
                  <Form.Select 
                    size="sm" 
                    value={graphScale}
                    onChange={(e) => setGraphScale(e.target.value)}
                  >
                    <option value="linear">linear</option>
                    <option value="log">log</option>
                  </Form.Select>
                </Form.Group>
              </div>
            </Card.Header>
            <Card.Body style={{ height: '400px' }}>
              <Tab.Container defaultActiveKey="fluctuation">
                <Row>
                  <Col sm={12}>
                    <Nav variant="tabs">
                      <Nav.Item>
                        <Nav.Link eventKey="fluctuation">Fluctuation</Nav.Link>
                      </Nav.Item>
                      <Nav.Item>
                        <Nav.Link eventKey="ralpha">R / Alpha</Nav.Link>
                      </Nav.Item>
                    </Nav>
                  </Col>
                  <Col sm={12}>
                    <Tab.Content>
                      <Tab.Pane eventKey="fluctuation">
                        {selectedNgram && (
                          <ResponsiveContainer width="100%" height="350px">
                            <LineChart data={getFluctuationData()}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis 
                                dataKey="window" 
                                scale={graphScale}
                                domain={graphScale === 'log' ? ['auto', 'auto'] : [0, 'auto']}
                              />
                              <YAxis 
                                scale={graphScale}
                                domain={graphScale === 'log' ? ['auto', 'auto'] : [0, 'auto']}
                              />
                              <Tooltip />
                              <Legend />
                              <Line type="scatter" dataKey="value" stroke="#8884d8" name="ΔF" />
                              <Line type="line" dataKey="fit" stroke="#82ca9d" name="fit" />
                            </LineChart>
                          </ResponsiveContainer>
                        )}
                        {!selectedNgram && <div className="text-center mt-5">Select an ngram to view fluctuation</div>}
                      </Tab.Pane>
                      <Tab.Pane eventKey="ralpha">
                        {analysisResult && (
                          <ResponsiveContainer width="100%" height="350px">
                            <ScatterChart>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis 
                                type="number" 
                                dataKey="R" 
                                name="R"
                                scale={graphScale}
                                domain={graphScale === 'log' ? ['auto', 'auto'] : [0, 'auto']}
                              />
                              <YAxis 
                                type="number" 
                                dataKey="b" 
                                name="Alpha"
                                scale={graphScale}
                                domain={graphScale === 'log' ? ['auto', 'auto'] : [0, 'auto']}
                              />
                              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                              <Scatter 
                                name="Ngrams" 
                                data={getRAlphaData().filter(d => !d.selected)} 
                                fill="#8884d8"
                              />
                              <Scatter 
                                name="Selected" 
                                data={getRAlphaData().filter(d => d.selected)} 
                                fill="#ff7300"
                              />
                            </ScatterChart>
                          </ResponsiveContainer>
                        )}
                        {!analysisResult && <div className="text-center mt-5">Analyze text to view R/Alpha plot</div>}
                      </Tab.Pane>
                    </Tab.Content>
                  </Col>
                </Row>
              </Tab.Container>
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
}

export default App;