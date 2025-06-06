import React, { useState, useEffect } from 'react';
import axios from 'axios';
import JSZip from 'jszip';
import {
  Container, Form, Button, Card, Row, Col, Spinner,
  Table, Nav, Tab, Alert, ProgressBar, Modal
} from 'react-bootstrap';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter
} from 'recharts';
import ForceGraph2D from 'react-force-graph-2d';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.scss';
import RLPlotsComponent from './components/RLPlotsComponent/RLPlotsComponent';
import FourierAnalysisComponent from './components/FourierAnalysisComponent/FourierAnalysisComponent';

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
  const [corpusFilesContent, setCorpusFilesContent] = useState({});
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

  // Text preprocessing state variables
  const [doPreprocess, setDoPreprocess] = useState(false);
  const [doSyllables, setDoSyllables] = useState(false);
  const [doCV, setDoCV] = useState(false);
  const [preprocessedText, setPreprocessedText] = useState('');
  const [showPreview, setShowPreview] = useState(false);

  // Custom error alert component
  const ErrorAlert = ({ error, onClose }) => {
    const hasMultipleLines = error && error.includes('\n');
    return (
      <Alert variant="danger" onClose={onClose} dismissible>
        {hasMultipleLines ? (
          <div>
            <strong>Error:</strong>
            <pre className="mt-2 mb-0" style={{
              whiteSpace: 'pre-wrap',
              backgroundColor: 'rgba(0,0,0,0.05)',
              padding: '8px',
              borderRadius: '4px',
              fontSize: '0.9rem',
              maxHeight: '200px',
              overflowY: 'auto'
            }}>
              {error}
            </pre>
          </div>
        ) : (
          error
        )}
      </Alert>
    );
  };

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

    const fileContents = {};
    let filesProcessed = 0;

    files.forEach(file => {
      const reader = new FileReader();
      reader.onload = (e) => {
        fileContents[file.name] = e.target.result;
        filesProcessed++;
        if (filesProcessed === files.length) {
          setCorpusFilesContent(fileContents);
        }
      };
      reader.readAsText(file);
    });

    if (files.length > 0) {
      const reader = new FileReader();
      reader.onload = (e) => {
        calculateWindowParams(e.target.result);
      };
      reader.readAsText(files[0]);
    }
  };

  // Handle zip file upload
  const handleZipUpload = async (event) => {
    const file = event.target.files[0];
    try {
      const zip = await JSZip.loadAsync(file);
      const fileObjects = [];
      const fileContents = {};
      const txtFiles = Object.keys(zip.files).filter(name => name.endsWith('.txt'));
      for (const name of txtFiles) {
        const content = await zip.file(name).async('string');
        const fileObj = new File([content], name, { type: 'text/plain' });
        fileObjects.push(fileObj);
        fileContents[name] = content;
      }
      setCorpusFiles(fileObjects);
      setCorpusFilesContent(fileContents);
      setAnalysisMode('corpus');
      setZipFile(file);
    } catch (err) {
      setError('Failed to read zip file: ' + err.message);
    }
  };

  // Calculate window parameters based on text length
  const calculateWindowParams = async (text) => {
    try {
      setLoading(true);
      setError(null);
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
      const errorData = err.response?.data;
      let errorMessage = 'Failed to calculate window parameters';
      if (errorData) {
        if (errorData.context && errorData.error) {
          errorMessage = `${errorData.error} (Context: ${errorData.context})`;
        } else if (errorData.error) {
          errorMessage = errorData.error;
        }
        if (errorData.details) {
          errorMessage += `\nDetails: ${errorData.details}`;
        }
        if (errorData.parameters) {
          const params = errorData.parameters;
          errorMessage += `\nParameters: n_size=${params.n_size}, split=${params.split}, text length=${params.text_length}`;
        }
      }
      setError(errorMessage);
      setLoading(false);
      if (errorData && errorData.trace) {
        console.error('Server error trace:', errorData.trace);
      }
    }
  };

  // Preview text processing
  const handlePreviewText = async () => {
    if (!fileContent) {
      setError('Please upload a file first');
      return;
    }

    try {
      setLoading(true);
      setError(null);
      const response = await axios.post(`${API_URL}/preprocess`, {
        text: fileContent,
        do_preprocess: doPreprocess,
        do_syllables: doSyllables,
        do_cv: doCV
      });
      setPreprocessedText(response.data.processed_text);
      setShowPreview(true);
      setLoading(false);
    } catch (err) {
      const errorData = err.response?.data;
      let errorMessage = 'Preview processing failed';
      if (errorData) {
        if (errorData.context && errorData.error) {
          errorMessage = `${errorData.error} (Context: ${errorData.context})`;
        } else if (errorData.error) {
          errorMessage = errorData.error;
        }
      } else if (err.message) {
        errorMessage += `: ${err.message}`;
      }
      setError(errorMessage);
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
      Object.keys(config).forEach(key => {
        formData.append(key, config[key]);
      });
      formData.append('do_preprocess', doPreprocess);
      formData.append('do_syllables', doSyllables);
      formData.append('do_cv', doCV);

      if (zipFile) {
        formData.append('zip_file', zipFile);
        setCorpusP(true);
      } else if (corpusFiles.length > 0) {
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
      const errorData = err.response?.data;
      let errorMessage = 'Corpus processing failed';
      if (errorData) {
        if (errorData.context && errorData.error) {
          errorMessage = `${errorData.error} (Context: ${errorData.context})`;
        } else if (errorData.error) {
          errorMessage = errorData.error;
        }
      } else if (err.message) {
        errorMessage += `: ${err.message}`;
      }
      setError(errorMessage);
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
        const response = await axios.post(`${API_URL}/analyze`, {
          text: fileContent,
          ...config,
          min_type: parseInt(config.min_type, 10),
          do_preprocess: doPreprocess,
          do_syllables: doSyllables,
          do_cv: doCV
        });
        setAnalysisResult(response.data);
        setLoading(false);
        if (config.definition === 'static') {
          getMarkovGraph();
        }
      } catch (err) {
        const errorData = err.response?.data;
        let errorMessage = 'Analysis failed';
        if (errorData) {
          if (errorData.context && errorData.error) {
            errorMessage = `${errorData.error} (Context: ${errorData.context})`;
          } else if (errorData.error) {
            errorMessage = errorData.error;
          }
          if (errorData.details) {
            errorMessage += `\nDetails: ${errorData.details}`;
          }
        } else if (err.message) {
          errorMessage += `: ${err.message}`;
        }
        setError(errorMessage);
        setLoading(false);
        if (errorData && errorData.trace) {
          console.error('Server error trace:', errorData.trace);
        }
      }
    } else {
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

  // Handle corpus row click
  const handleCorpusRowClick = async (fileName) => {
    let content = corpusFilesContent[fileName];
    if (!content) {
      const fileObj = corpusFiles.find(f => f.name === fileName);
      if (fileObj) {
        content = await new Promise((resolve) => {
          const reader = new FileReader();
          reader.onload = (e) => resolve(e.target.result);
          reader.readAsText(fileObj);
        });
      } else {
        setError(`Не вдалося знайти файл ${fileName} для аналізу`);
        return;
      }
    }
    setAnalysisMode('single');
    const fileObj = corpusFiles.find(f => f.name === fileName) || new File([content], fileName, { type: 'text/plain' });
    setFile(fileObj);
    setFileContent(content);
    setAnalysisResult(null);
    setSelectedNgram(null);
    setMarkovGraph(null);
    await calculateWindowParams(content);
    try {
      setLoading(true);
      const response = await axios.post(`${API_URL}/analyze`, {
        text: content,
        ...config,
        min_type: parseInt(config.min_type, 10),
        do_preprocess: doPreprocess,
        do_syllables: doSyllables,
        do_cv: doCV
      });
      setAnalysisResult(response.data);
      if (config.definition === 'static') {
        await getMarkovGraph();
      }
      setLoading(false);
    } catch (err) {
      const errorData = err.response?.data;
      let errorMessage = 'Analysis failed';
      if (errorData && errorData.error) {
        errorMessage = errorData.error;
      } else if (err.message) {
        errorMessage += `: ${err.message}`;
      }
      setError(errorMessage);
      setLoading(false);
    }
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

  // Download processed text for single file
  const handleDownloadProcessedText = async () => {
    try {
      const response = await axios.post(`${API_URL}/preprocess`, {
        text: fileContent,
        do_preprocess: doPreprocess,
        do_syllables: doSyllables,
        do_cv: doCV
      });
      const processedText = response.data.processed_text;
      const blob = new Blob([processedText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `processed_${file.name}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download processed text: ' + err.message);
    }
  };

  // Download processed text for a single corpus file
  const handleDownloadCorpusProcessedText = async (fileName) => {
    try {
      const content = corpusFilesContent[fileName];
      if (!content) {
        setError(`Content for ${fileName} not found`);
        return;
      }
      const response = await axios.post(`${API_URL}/preprocess`, {
        text: content,
        do_preprocess: doPreprocess,
        do_syllables: doSyllables,
        do_cv: doCV
      });
      const processedText = response.data.processed_text;
      const blob = new Blob([processedText], { type: 'text/plain' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `processed_${fileName}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(`Failed to download processed text for ${fileName}: ` + err.message);
    }
  };

  // Download all processed texts for corpus
  const handleDownloadAllProcessedTexts = async () => {
    try {
      const zip = new JSZip();
      const promises = corpusResult.corpus_results.map(async (row) => {
        if (!row.has_error) {
          const fileName = row.file;
          const content = corpusFilesContent[fileName];
          if (content) {
            const response = await axios.post(`${API_URL}/preprocess`, {
              text: content,
              do_preprocess: doPreprocess,
              do_syllables: doSyllables,
              do_cv: doCV
            });
            const processedText = response.data.processed_text;
            zip.file(`processed_${fileName}`, processedText);
          }
        }
      });
      await Promise.all(promises);
      const zipBlob = await zip.generateAsync({ type: 'blob' });
      const url = URL.createObjectURL(zipBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'processed_texts.zip';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    } catch (err) {
      setError('Failed to download all processed texts: ' + err.message);
    }
  };

  return (
    <Container fluid>
      {error && <ErrorAlert error={error} onClose={() => setError(null)} />}

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
                        <ul className="small" style={{ maxHeight: '100px', overflowY: 'auto' }}>
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

                <Form.Group className="mt-3">
                  <Form.Label>Text Preprocessing:</Form.Label>
                  <div>
                    <Form.Check
                      type="checkbox"
                      label="Apply Ukrainian text preprocessing"
                      id="do-preprocess"
                      checked={doPreprocess}
                      onChange={(e) => setDoPreprocess(e.target.checked)}
                      className="mb-2"
                    />
                    <Form.Text className="text-muted mb-2" style={{ display: 'block' }}>
                      Normalizes text: lowercase, replaces special Ukrainian letters with phonetic equivalents,
                      removes punctuation and other non-letter characters.
                    </Form.Text>

                    <Form.Check
                      type="checkbox"
                      label="Split text into syllables"
                      id="do-syllables"
                      checked={doSyllables}
                      onChange={(e) => setDoSyllables(e.target.checked)}
                      className="mb-2"
                    />
                    <Form.Text className="text-muted mb-2" style={{ display: 'block' }}>
                      Splits Ukrainian words into syllables based on phonetic rules.
                    </Form.Text>

                    <Form.Check
                      type="checkbox"
                      label="Convert to Consonant-Vowel (CV) sequences"
                      id="do-cv"
                      checked={doCV}
                      onChange={(e) => setDoCV(e.target.checked)}
                      className="mb-2"
                    />
                    <Form.Text className="text-muted mb-2" style={{ display: 'block' }}>
                      Converts Ukrainian text to CV sequences: vowels become 'v', consonants become 'c'.
                      For example, "привіт" becomes "ccvcvc".
                    </Form.Text>
                  </div>
                  {analysisMode === 'single' && file && (doPreprocess || doSyllables || doCV) && (
                    <Button
                      variant="outline-secondary"
                      size="sm"
                      onClick={handlePreviewText}
                      className="mt-2"
                    >
                      Preview processed text
                    </Button>
                  )}
                </Form.Group>

                <Button
                  variant="primary"
                  className="mt-3 me-2"
                  onClick={handleAnalyze}
                  disabled={loading}
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
                <Nav.Item>
                  <Nav.Link eventKey="rl-plots">R-L Plots</Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="fourier">Fourier Analysis</Nav.Link>
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
                  {activeTab === 'rl-plots' && (
                    <RLPlotsComponent corpusResults={corpusResult?.corpus_results || []} />
                  )}
                  {activeTab === 'fourier' && (
                    <FourierAnalysisComponent />
                  )}
                  {analysisMode === 'single' ? (
                    <>
                      {activeTab === 'data' && analysisResult && (
                        <>
                          <Table striped bordered hover responsive>
                            <thead>
                              <tr>
                                <th>Rank</th>
                                <th>Ngram</th>
                                <th>F</th>
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
                                  onClick={!row.has_error ? () => handleNgramSelect(row.ngram) : undefined}
                                  className={row.has_error ? "table-danger" : (selectedNgram === row.ngram ? 'table-primary' : '')}
                                  style={{ cursor: row.has_error ? 'not-allowed' : 'pointer' }}
                                  title={row.has_error ?
                                    `Помилка при обробці n-грами ${row.ngram}` :
                                    `Клікніть для детального аналізу n-грами ${row.ngram}`}
                                >
                                  <td>{row.rank}</td>
                                  <td>{row.ngram}</td>
                                  <td>{row.F}</td>
                                  <td>{row.has_error ? "-" : row.R}</td>
                                  <td>{row.has_error ? "-" : row.a}</td>
                                  <td>{row.has_error ? "-" : row.b}</td>
                                  <td>{row.has_error ? "-" : row.goodness}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                          <div className="mt-3">
                            <Button variant="success" onClick={handleDownloadProcessedText}>
                              Download processed text
                            </Button>
                          </div>
                        </>
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
                    corpusResult && activeTab === 'data' && (
                      <>
                        <div className="mb-3">
                          <Button variant="success" onClick={handleDownloadAllProcessedTexts}>
                            Download all processed texts
                          </Button>
                        </div>
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
                              <th>Download</th>
                            </tr>
                          </thead>
                          <tbody>
                            {corpusResult.corpus_results.map((row, index) => (
                              <tr
                                key={index}
                                onClick={!row.has_error ? () => handleCorpusRowClick(row.file) : undefined}
                                className={row.has_error ? "table-danger" : "cursor-pointer"}
                                style={{ cursor: row.has_error ? 'not-allowed' : 'pointer' }}
                                title={row.has_error ?
                                  `Помилка при обробці файлу ${row.file}` :
                                  `Клікніть для детального аналізу файлу ${row.file}`}
                              >
                                <td>{row['№']}</td>
                                <td>{row.file}</td>
                                <td>{row.has_error ? "-" : row.F_min}</td>
                                <td>{row.has_error ? "-" : row.L}</td>
                                <td>{row.has_error ? "-" : row.V}</td>
                                <td>{row.has_error ? "-" : row.time}</td>
                                <td>{row.has_error ? "-" : Number(row.R_avg).toFixed(4)}</td>
                                <td>{row.has_error ? "-" : Number(row.dR).toFixed(4)}</td>
                                <td>{row.has_error ? "-" : Number(row.Rw_avg).toFixed(4)}</td>
                                <td>{row.has_error ? "-" : Number(row.dRw).toFixed(4)}</td>
                                <td>{row.has_error ? "-" : Number(row.b_avg).toFixed(4)}</td>
                                <td>{row.has_error ? "-" : Number(row.db).toFixed(4)}</td>
                                <td>{row.has_error ? "-" : Number(row.bw_avg).toFixed(4)}</td>
                                <td>{row.has_error ? "-" : Number(row.dbw).toFixed(4)}</td>
                                <td>
                                  {!row.has_error && (
                                    <Button
                                      variant="link"
                                      size="sm"
                                      onClick={() => handleDownloadCorpusProcessedText(row.file)}
                                    >
                                      Download
                                    </Button>
                                  )}
                                </td>
                              </tr>
                            ))}
                          </tbody>
                        </Table>
                      </>
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
                  {analysisResult.preprocessing && (
                    <Col>
                      Preprocessing: {[
                        analysisResult.preprocessing.text_preprocessed ? "Text normalization" : "",
                        analysisResult.preprocessing.text_syllabled ? "Syllable splitting" : "",
                        analysisResult.preprocessing.text_cv_converted ? "CV conversion" : ""
                      ].filter(Boolean).join(", ") || "None"}
                    </Col>
                  )}
                </Row>
              )}
              {analysisMode === 'corpus' && corpusResult && (
                <Row>
                  <Col>Files processed: {corpusResult.file_count}</Col>
                  <Col>Total time: {corpusResult.total_time}s</Col>
                  {corpusResult.preprocessing && (
                    <Col>
                      Preprocessing: {[
                        corpusResult.preprocessing.text_preprocessed ? "Text normalization" : "",
                        corpusResult.preprocessing.text_syllabled ? "Syllable splitting" : "",
                        corpusResult.preprocessing.text_cv_converted ? "CV conversion" : ""
                      ].filter(Boolean).join(", ") || "None"}
                    </Col>
                  )}
                </Row>
              )}
            </Card.Footer>
          </Card>
        </Col>
      </Row>

      <Row>
        <Col md={12}>
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
      </Row>

      {/* Preview Modal */}
      <Modal
        show={showPreview}
        onHide={() => setShowPreview(false)}
        size="lg"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title>Processed Text Preview</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
            <p><strong>Original text length:</strong> {fileContent?.length || 0} characters</p>
            <p><strong>Processed text length:</strong> {preprocessedText?.length || 0} characters</p>
            <p>
              <strong>Applied processing:</strong> {[
                doPreprocess ? "Text normalization" : "",
                doSyllables ? "Syllable splitting" : "",
                doCV ? "CV conversion" : ""
              ].filter(Boolean).join(", ")}
            </p>
            <hr />
            <div style={{ whiteSpace: 'pre-wrap' }}>
              {preprocessedText || 'No preview available'}
            </div>
          </div>
        </Modal.Body>
        <Modal.Footer>
          <Button variant="secondary" onClick={() => setShowPreview(false)}>
            Close
          </Button>
        </Modal.Footer>
      </Modal>
    </Container>
  );
}

export default App;