import React, { useState } from 'react';
import { Card, Row, Col, Button, Spinner, Alert, Nav, Table, Form } from 'react-bootstrap';
import axios from 'axios';

const API_URL = 'http://localhost:5000/api';

const FourierAnalysisComponent = () => {
  const [files, setFiles] = useState([]);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('upload');
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [plotType, setPlotType] = useState('fourier_signal');

  // Завантаження файлів
  const handleFileUpload = (event) => {
    const uploadedFiles = Array.from(event.target.files);
    setFiles(uploadedFiles);
  };

  // Очищення файлів
  const clearFiles = () => {
    setFiles([]);
    setAnalysisResults(null);
    setSelectedAnalysis(null);
  };

  // Запуск аналізу
  const startAnalysis = async () => {
    if (files.length === 0) {
      setError('Будь ласка, завантажте текстові файли для аналізу');
      return;
    }

    try {
      setLoading(true);
      setError(null);

      const formData = new FormData();
      files.forEach(file => {
        formData.append('files[]', file);
      });

      const response = await axios.post(`${API_URL}/fourier-analysis`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });

      setAnalysisResults(response.data);
      setActiveTab('results');
      setLoading(false);
    } catch (err) {
      setError('Помилка аналізу: ' + (err.response?.data?.error || err.message));
      setLoading(false);
    }
  };

  // Вибір аналізу для перегляду графіків
  const selectAnalysis = (fileAnalysis, syllableType, isWaitingTimes = false) => {
    let plotsToUse;
    
    if (isWaitingTimes) {
      // Графіки для часів очікування
      plotsToUse = syllableType === 'second_frequent' 
        ? fileAnalysis.fourier_plots_waiting_times_second 
        : fileAnalysis.fourier_plots_waiting_times;
    } else {
      // Графіки для позицій
      plotsToUse = syllableType === 'second_frequent' 
        ? fileAnalysis.fourier_plots_second 
        : fileAnalysis.fourier_plots;
    }
    
    setSelectedAnalysis({
      ...fileAnalysis,
      syllableType: syllableType,
      fourier_plots: plotsToUse,
      isWaitingTimes: isWaitingTimes
    });
    setActiveTab('plots');
  };

  return (
    <div>
      <Nav variant="tabs" activeKey={activeTab} onSelect={k => setActiveTab(k)} className="mb-3">
        <Nav.Item>
          <Nav.Link eventKey="upload">Завантаження файлів</Nav.Link>
        </Nav.Item>
        <Nav.Item>
          <Nav.Link eventKey="results">Результати аналізу</Nav.Link>
        </Nav.Item>
        <Nav.Item>
          <Nav.Link eventKey="plots">Графіки Фур'є</Nav.Link>
        </Nav.Item>
      </Nav>

      {error && (
        <Alert variant="danger" onClose={() => setError(null)} dismissible>
          {error}
        </Alert>
      )}

      {/* Таб завантаження файлів */}
      {activeTab === 'upload' && (
        <Card>
          <Card.Header>Аналіз Фур'є для типів складів</Card.Header>
          <Card.Body>
            
            
            <div className="alert alert-info">
              <h6>Типи аналізу:</h6>
              <ul className="mb-0">
                <li><strong>Позиції:</strong> Бінарні послідовності позицій типів складів</li>
                <li><strong>Часи очікування:</strong> Інтервали між входженнями типів складів з застосуванням формули Kobayashi-Musha для перетворення в рівномірний ряд</li>
              </ul>
            </div>
            
            <Form.Group className="mb-3">
              <Form.Label>Виберіть текстові файли (.txt):</Form.Label>
              <Form.Control
                type="file"
                multiple
                accept=".txt"
                onChange={handleFileUpload}
              />
              
            </Form.Group>

            {files.length > 0 && (
              <div className="mb-3">
                <h6>Завантажені файли ({files.length}):</h6>
                <ul className="list-unstyled">
                  {files.map((file, index) => (
                    <li key={index} className="py-1">
                      <span className="badge bg-secondary me-2">{index + 1}</span>
                      {file.name} ({(file.size / 1024).toFixed(1)} KB)
                    </li>
                  ))}
                </ul>
                <Button variant="outline-danger" size="sm" onClick={clearFiles}>
                  Очистити файли
                </Button>
              </div>
            )}

            <div className="d-grid">
              <Button
                variant="primary"
                onClick={startAnalysis}
                disabled={loading || files.length === 0}
                size="lg"
              >
                {loading ? (
                  <>
                    <Spinner size="sm" animation="border" className="me-2" />
                    Аналіз в процесі...
                  </>
                ) : (
                  'Почати аналіз Фур\'є'
                )}
              </Button>
            </div>
          </Card.Body>
        </Card>
      )}

      {/* Таб результатів аналізу */}
      {activeTab === 'results' && (
        <Card>
          <Card.Header>Результати аналізу типів складів</Card.Header>
          <Card.Body>
            {analysisResults ? (
              <>
                <p className="text-muted">
                  Клікніть на кнопку для перегляду графіків Фур'є для відповідного типу складу
                </p>

                <Table striped bordered hover responsive>
                  <thead>
                    <tr>
                      <th>Файл</th>
                      <th>Довжина</th>
                      <th>Кількість складів</th>
                      <th>Найчастіший тип складу</th>
                      <th>Частота</th>
                      <th>Другий найчастіший</th>
                      <th>Частота</th>
                      <th>Аналіз позицій</th>
                      <th>Аналіз часів очікування</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysisResults.file_analyses.map((fileAnalysis, index) => (
                      <tr key={index} className={fileAnalysis.error ? 'table-danger' : ''}>
                        <td>{fileAnalysis.filename}</td>
                        <td>{fileAnalysis.text_length}</td>
                        <td>{fileAnalysis.syllables_count}</td>
                        <td>{fileAnalysis.most_frequent_syllable?.type || '-'}</td>
                        <td>{fileAnalysis.most_frequent_syllable?.count || 0}</td>
                        <td>{fileAnalysis.second_frequent_syllable?.type || '-'}</td>
                        <td>{fileAnalysis.second_frequent_syllable?.count || 0}</td>
                        <td>
                          {!fileAnalysis.error ? (
                            <div className="d-flex gap-1">
                              <Button
                                size="sm"
                                variant="outline-primary"
                                onClick={() => selectAnalysis(fileAnalysis, 'most_frequent', false)}
                                disabled={!fileAnalysis.fourier_plots}
                                title="Аналіз позицій найчастішого типу складу"
                              >
                                1-й
                              </Button>
                              <Button
                                size="sm"
                                variant="outline-secondary"
                                onClick={() => selectAnalysis(fileAnalysis, 'second_frequent', false)}
                                disabled={!fileAnalysis.fourier_plots_second}
                                title="Аналіз позицій другого найчастішого типу складу"
                              >
                                2-й
                              </Button>
                            </div>
                          ) : (
                            <span className="text-danger">Помилка</span>
                          )}
                        </td>
                        <td>
                          {!fileAnalysis.error ? (
                            <div className="d-flex gap-1">
                              <Button
                                size="sm"
                                variant="outline-info"
                                onClick={() => selectAnalysis(fileAnalysis, 'most_frequent', true)}
                                disabled={!fileAnalysis.fourier_plots_waiting_times}
                                title="Аналіз часів очікування найчастішого типу складу"
                              >
                                ЧО-1
                              </Button>
                              <Button
                                size="sm"
                                variant="outline-warning"
                                onClick={() => selectAnalysis(fileAnalysis, 'second_frequent', true)}
                                disabled={!fileAnalysis.fourier_plots_waiting_times_second}
                                title="Аналіз часів очікування другого найчастішого типу складу"
                              >
                                ЧО-2
                              </Button>
                            </div>
                          ) : (
                            <span className="text-danger">Помилка</span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </Table>

                <div className="mt-3">
                  <h6>Загальна статистика:</h6>
                  <Row>
                    <Col md={4}>
                      <div className="text-center p-3 bg-light rounded">
                        <h5>{analysisResults.file_analyses.length}</h5>
                        <small>Проаналізовано файлів</small>
                      </div>
                    </Col>
                    <Col md={4}>
                      <div className="text-center p-3 bg-light rounded">
                        <h5>{analysisResults.total_syllables_analyzed}</h5>
                        <small>Загальна кількість складів</small>
                      </div>
                    </Col>
                    <Col md={4}>
                      <div className="text-center p-3 bg-light rounded">
                        <h5>{Math.round(analysisResults.processing_time)} сек</h5>
                        <small>Час обробки</small>
                      </div>
                    </Col>
                  </Row>
                </div>
              </>
            ) : (
              <div className="text-center p-5">
                <p className="text-muted">Немає результатів аналізу. Спочатку завантажте файли та запустіть аналіз.</p>
              </div>
            )}
          </Card.Body>
        </Card>
      )}

      {/* Таб графіків Фур'є */}
      {activeTab === 'plots' && (
        <Card>
          <Card.Header>Графіки Фур'є для типу складу</Card.Header>
          <Card.Body>
            {selectedAnalysis ? (
              <>
                <div className="mb-3">
                  <strong>Файл:</strong> {selectedAnalysis.filename}<br />
                  <strong>Тип складу:</strong> {
                    selectedAnalysis.syllableType === 'most_frequent' 
                      ? selectedAnalysis.most_frequent_syllable?.type
                      : selectedAnalysis.second_frequent_syllable?.type
                  }<br />
                  <strong>Частота:</strong> {
                    selectedAnalysis.syllableType === 'most_frequent' 
                      ? selectedAnalysis.most_frequent_syllable?.count
                      : selectedAnalysis.second_frequent_syllable?.count
                  }<br />
                  <strong>Тип аналізу:</strong> {
                    selectedAnalysis.isWaitingTimes ? 'Часи очікування' : 'Позиції'
                  }<br />
                  {selectedAnalysis.isWaitingTimes ? (
                    selectedAnalysis.waiting_times_count && (
                      <><strong>Кількість часів очікування:</strong> {selectedAnalysis.waiting_times_count}</>
                    )
                  ) : (
                    selectedAnalysis.target_syllable_occurrences && (
                      <>
                        <strong>Входжень у тексті:</strong> {selectedAnalysis.target_syllable_occurrences}<br />
                        <strong>Довжина послідовності:</strong> {selectedAnalysis.binary_sequence_length}
                      </>
                    )
                  )}
                </div>

                <Nav variant="pills" activeKey={plotType} onSelect={k => setPlotType(k)} className="mb-3">
                  <Nav.Item>
                    <Nav.Link eventKey="fourier_signal">
                      Фур'є-спектр (сигнал)
                    </Nav.Link>
                  </Nav.Item>
                  <Nav.Item>
                    <Nav.Link eventKey="fourier_noise">
                      Фур'є-спектр (шум)
                    </Nav.Link>
                  </Nav.Item>
                  <Nav.Item>
                    <Nav.Link eventKey="signal_noise">
                      Відношення сигнал/шум
                    </Nav.Link>
                  </Nav.Item>
                  <Nav.Item>
                    <Nav.Link eventKey="comparison">
                      Порівняння
                    </Nav.Link>
                  </Nav.Item>
                </Nav>

                <div className="text-center">
                  {selectedAnalysis.fourier_plots && selectedAnalysis.fourier_plots[plotType] && (
                    <img
                      src={`data:image/png;base64,${selectedAnalysis.fourier_plots[plotType]}`}
                      alt={`Fourier analysis plot - ${plotType}`}
                      style={{ maxWidth: '100%', maxHeight: '600px' }}
                      className="border rounded"
                    />
                  )}
                  {(!selectedAnalysis.fourier_plots || !selectedAnalysis.fourier_plots[plotType]) && (
                    <div className="p-5 text-muted">
                      Графік недоступний
                    </div>
                  )}
                </div>

                <div className="mt-4">
                  <h6>Інформація про аналіз:</h6>
                  {selectedAnalysis.isWaitingTimes ? (
                    <ul>
                      <li>
                        <strong>Часи очікування:</strong> Інтервали між входженнями типу складу у тексті
                      </li>
                      <li>
                        <strong>Формула Kobayashi-Musha:</strong> Перетворення нерівномірних інтервалів у рівномірний ряд
                      </li>
                      <li>
                        <strong>Сигнал:</strong> Фур'є-спектр рівномірного ряду часів очікування
                      </li>
                      <li>
                        <strong>Шум:</strong> Фур'є-спектр перемішаного рівномірного ряду
                      </li>
                      <li>
                        <strong>Відношення сигнал/шум:</strong> Показує спектральні характеристики структури часів очікування
                      </li>
                    </ul>
                  ) : (
                    <ul>
                      <li>
                        <strong>Бінарна послідовність:</strong> Позиції цього типу складу відмічені як 1, інші як 0
                      </li>
                      <li>
                        <strong>Сигнал:</strong> Фур'є-спектр оригінальної бінарної послідовності
                      </li>
                      <li>
                        <strong>Шум:</strong> Фур'є-спектр перемішаної бінарної послідовності
                      </li>
                      <li>
                        <strong>Відношення сигнал/шум:</strong> Показує, наскільки структурована послідовність порівняно зі випадковою
                      </li>
                    </ul>
                  )}
                </div>
              </>
            ) : (
              <div className="text-center p-5">
                <p className="text-muted">Виберіть аналіз з таблиці результатів для перегляду графіків.</p>
              </div>
            )}
          </Card.Body>
        </Card>
      )}
    </div>
  );
};

export default FourierAnalysisComponent;
