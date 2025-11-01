import React, { useState, useEffect, useCallback } from 'react';
import './App.css';
import FileUpload from './components/FileUpload';
import ModelResults from './components/ModelResults';
import DatasetSelector from './components/DatasetSelector';
import Toast from './components/Toast';
import PipelineSteps from './components/PipelineSteps';
import aLogo from './a.png';
import heroVisual from './hyperspectral-1406x1536.webp';

function App() {
  const [result, setResult] = useState(null);
  const [classificationResult, setClassificationResult] = useState(null);
  const [selectedDataset, setSelectedDataset] = useState('');
  const [backendStatus, setBackendStatus] = useState('Checking backend status...');
  const [toast, setToast] = useState(null);
  const [darkMode, setDarkMode] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const [activeSection, setActiveSection] = useState('home');

  const showToast = useCallback((message, type = 'info') => {
    setToast({ message, type });
  }, []);

  const handleToastClose = useCallback(() => {
    setToast(null);
  }, []);

  const toggleHelp = useCallback(() => {
    setShowHelp(prev => !prev);
    showToast(showHelp ? 'Help panel closed' : 'Help panel opened', 'info');
  }, [showHelp, showToast]);

  const toggleDarkMode = useCallback(() => {
    setDarkMode(prev => !prev);
    showToast(`Switched to ${darkMode ? 'light' : 'dark'} mode`, 'info');
  }, [darkMode, showToast]);

  useEffect(() => {
    const savedTheme = localStorage.getItem('darkMode');
    if (savedTheme !== null) {
      setDarkMode(JSON.parse(savedTheme));
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('darkMode', JSON.stringify(darkMode));
    document.body.classList.toggle('dark-mode', darkMode);
  }, [darkMode]);

  useEffect(() => {
    const handleKeyPress = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key === 'd') {
        event.preventDefault();
        toggleDarkMode();
      }
      if (event.key === 'Escape' && toast) {
        handleToastClose();
      }
      if ((event.ctrlKey || event.metaKey) && event.key === 'h') {
        event.preventDefault();
        toggleHelp();
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, [toast, toggleDarkMode, handleToastClose, toggleHelp]);

  const announceStatus = useCallback((message) => {
    const announcement = document.createElement('div');
    announcement.setAttribute('aria-live', 'polite');
    announcement.setAttribute('aria-atomic', 'true');
    announcement.className = 'sr-only';
    announcement.textContent = message;
    document.body.appendChild(announcement);
    setTimeout(() => announcement.remove(), 1000);
  }, []);

  const checkBackendStatus = useCallback(async () => {
    let retries = 3;
    while (retries > 0) {
      try {
        const response = await fetch('http://127.0.0.1:5000/ping');
        const data = await response.json();
        if (response.ok) {
          setBackendStatus(data.message);
          announceStatus('Backend connection successful');
          return;
        }
      } catch (error) {
        retries--;
        if (retries === 0) {
          setBackendStatus('Error connecting to backend');
          announceStatus('Backend connection failed');
          showToast('Backend connection failed. Please check your connection.', 'error');
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    }
  }, [showToast, announceStatus]);

  useEffect(() => {
    checkBackendStatus();
    const intervalId = setInterval(checkBackendStatus, 30000);
    return () => clearInterval(intervalId);
  }, [checkBackendStatus]);

  const callClassifyEndpoint = useCallback(async (hsi_path, gt_path, dataset_name) => {
    try {
      const response = await fetch('http://127.0.0.1:5000/classify', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ hsi_path, gt_path, dataset_name })
      });
      const data = await response.json();
      if (response.ok) {
        setClassificationResult(data);
        showToast('Classification completed successfully', 'success');
      } else {
        showToast(data.error || 'Classification failed', 'error');
      }
    } catch (error) {
      showToast(error.message || 'Classification request failed', 'error');
    }
  }, [showToast]);

  const handleUploadSuccess = useCallback((results) => {
    if (results && results.images && selectedDataset) {
      const updatedImages = results.images.map(img => {
        if (img.name) {
          if (img.name.toLowerCase().includes('confusion matrix') || img.name.toLowerCase().includes('anomaly map')) {
            return {
              ...img,
              name: `${img.name} - ${selectedDataset}`
            };
          }
        }
        return img;
      });
      results.images = updatedImages;
    }
    setResult(results);
    showToast('Upload successful! Results are ready.', 'success');
    
    // Clear any previous classification results - classification should only run when button is clicked
    setClassificationResult(null);
  }, [selectedDataset, callClassifyEndpoint, showToast]);

  const handleUploadFailure = useCallback((error) => {
    showToast(error?.message || 'Upload failed. Please try again.', 'error');
  }, [showToast]);

  const handleDatasetChange = useCallback((dataset) => {
    setSelectedDataset(dataset);
    // Clear classification results when dataset changes
    setClassificationResult(null);
    // Also clear upload results to prevent showing wrong dataset images
    setResult(null);
    showToast(`Dataset changed to: ${dataset}`, 'info');
  }, [showToast]);

  const goToHome = () => setActiveSection('home');
  const goToHowItWorks = () => setActiveSection('howItWorks');
  const goToProcessing = () => setActiveSection('processing');
  const backendHasError = backendStatus?.toLowerCase().includes('error');
  const lastCheckTimestamp = new Date().toLocaleTimeString();

  return (
    <div className={`App ${darkMode ? 'dark' : 'light'}`}>
      <div className="app-backdrop app-backdrop--primary" aria-hidden="true" />
      <div className="app-backdrop app-backdrop--secondary" aria-hidden="true" />

      <header className="app-header" role="banner">
        <div className="brand" onClick={goToHome} role="button" tabIndex={0} onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            goToHome();
          }
        }}>
          <img src={aLogo} alt="AnomVisor logo" className="brand-logo" />
          <div className="brand-meta">
            <span className="brand-title">AnomVisor</span>
            <span className="brand-tagline">Hyperspectral anomaly studio</span>
          </div>
        </div>
        <nav className="primary-nav" aria-label="Primary navigation">
          <button type="button" onClick={goToHome} className={`nav-link ${activeSection === 'home' ? 'is-active' : ''}`}>
            Overview
          </button>
          <button type="button" onClick={goToHowItWorks} className={`nav-link ${activeSection === 'howItWorks' ? 'is-active' : ''}`}>
            Pipeline
          </button>
          <button type="button" onClick={toggleHelp} className={`nav-link ${showHelp ? 'is-active' : ''}`}>
            Help
          </button>
        </nav>
        <div className="header-actions">
          <button
            type="button"
            className="icon-button"
            onClick={toggleDarkMode}
            aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
            title="Toggle dark mode (Ctrl/Cmd + D)"
          >
            <span aria-hidden="true">{darkMode ? '☀' : '🌙'}</span>
            <span>{darkMode ? 'Light mode' : 'Dark mode'}</span>
          </button>
          <button type="button" className="primary-button" onClick={goToProcessing}>
            Launch workspace
          </button>
        </div>
      </header>

      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={handleToastClose}
        />
      )}

      <main className="app-main" role="main">
        {activeSection === 'home' && (
          <>
            <section className="hero-section">
              <div className="hero-content">
                <p className="eyebrow">Hyperspectral intelligence, simplified</p>
                <h1>Discover anomalies hidden across hundreds of spectral bands.</h1>
                <p className="hero-subtitle">
                  Combine an Autoencoder-Transformer backbone with guided visual analytics to isolate
                  anomalies in remote sensing data without losing spatial context.
                </p>
                <div className="hero-actions">
                  <button type="button" className="primary-button" onClick={goToProcessing}>
                    Launch workspace
                  </button>
                  <button type="button" className="ghost-button" onClick={goToHowItWorks}>
                    View pipeline
                  </button>
                </div>
                <div className="metrics-grid" role="list">
                  <div className="metric-card" role="listitem">
                    <h3>Autoencoder + Transformer</h3>
                    <p>Capture spatial and spectral signatures with a hybrid deep-learning backbone.</p>
                  </div>
                  <div className="metric-card" role="listitem">
                    <h3>Insightful visuals</h3>
                    <p>Overlay anomaly maps, confusion matrices, and PCA composites in one dashboard.</p>
                  </div>
                  <div className="metric-card" role="listitem">
                    <h3>Dataset presets</h3>
                    <p>Optimized configurations for Salinas, Indian Pines, and Pavia University scenes.</p>
                  </div>
                </div>
              </div>
              <div className="hero-visual" aria-hidden="true">
                <div className="hero-image-wrapper">
                  <img src={heroVisual} alt="" className="hero-image" />
                  <div className="hero-badge">
                    <span className="badge-dot" />
                    Real-time anomaly overlays
                  </div>
                </div>
              </div>
            </section>

            <section className="info-section">
              <div className="section-header">
                <h2>Why teams choose AnomVisor</h2>
                <p>
                  Upload hyperspectral cubes, iterate quickly, and share results through curated visual reports that
                  highlight what matters.
                </p>
              </div>
              <div className="feature-grid" role="list">
                <article className="feature-card" role="listitem">
                  <h3>Guided workflow</h3>
                  <p>
                    Move smoothly from dataset selection to anomaly classification with contextual cues at every step.
                  </p>
                </article>
                <article className="feature-card" role="listitem">
                  <h3>Explainable insight</h3>
                  <p>
                    Compare anomaly intensity maps, confusion matrices, and classification overlays side-by-side.
                  </p>
                </article>
                <article className="feature-card" role="listitem">
                  <h3>Operational flexibility</h3>
                  <p>
                    Built for agriculture, environmental monitoring, and precision inspection use-cases.
                  </p>
                </article>
              </div>
              <div className="info-card">
                <h3>What is hyperspectral imaging?</h3>
                <p>
                  Hyperspectral Imaging (HSI) captures imagery across hundreds of contiguous spectral bands, revealing
                  subtle signatures beyond traditional RGB data. The technology powers applications ranging from crop
                  health monitoring to material inspection and medical diagnostics.
                </p>
                <p>
                  AnomVisor pairs this rich spectral information with machine learning to flag anomalies. An
                  Autoencoder-Transformer model extracts spectral-spatial features, while an SVM delivers precise
                  classification to separate meaningful signal from noise.
                </p>
              </div>
            </section>
          </>
        )}

        {activeSection === 'howItWorks' && (
          <section className="how-section">
            <div className="section-header">
              <h2>From upload to insight in four steps</h2>
              <p>
                The pipeline streamlines preprocessing, deep feature extraction, and interpretation so you can focus on
                decisions.
              </p>
            </div>
            <PipelineSteps isDarkMode={darkMode} />
            <div className="section-footer">
              <button type="button" className="ghost-button" onClick={goToProcessing}>
                Open workspace
              </button>
            </div>
          </section>
        )}

        {activeSection === 'processing' && (
          <section className="workspace-section">
            <div className="section-header">
              <h2>Workspace</h2>
              <p>Bring your `.mat` spectral cube and ground truth to generate anomaly maps in minutes.</p>
            </div>
            <div className="status-card" role="status" aria-live="polite">
              <div className={`status-chip ${backendHasError ? 'is-error' : 'is-ok'}`}>
                <span className="status-dot" />
                <span>{backendStatus}</span>
              </div>
              <span className="status-timestamp">Last checked: {lastCheckTimestamp}</span>
            </div>
            <div className="workspace-grid">
              <div className="workspace-controls">
                <div className="panel-card">
                  <DatasetSelector
                    selectedDataset={selectedDataset}
                    onDatasetChange={handleDatasetChange}
                  />
                </div>
                <div className="panel-card">
                  <FileUpload
                    selectedDataset={selectedDataset}
                    onUploadSuccess={handleUploadSuccess}
                    onUploadFailure={handleUploadFailure}
                    isLoading={isLoading}
                    setIsLoading={setIsLoading}
                  />
                </div>
              </div>
              <div className="workspace-results">
                {isLoading ? (
                  <div className="loading-state" aria-live="assertive">
                    <div className="loading-spinner" role="presentation" />
                    <p>Processing your request...</p>
                  </div>
                ) : (
                  <>
                    <ModelResults result={result} datasetName={selectedDataset} />
                  </>
                )}
              </div>
            </div>
          </section>
        )}
      </main>

      <footer className="app-footer" role="contentinfo">
        <div className="footer-inner">
          <div className="footer-brand">
            <span className="brand-title">AnomVisor</span>
            <p>Hyperspectral anomaly detection toolkit for modern teams.</p>
          </div>
          <div className="footer-links">
            <button type="button" onClick={goToHome} className={`footer-link ${activeSection === 'home' ? 'is-active' : ''}`}>
              Overview
            </button>
            <button type="button" onClick={goToHowItWorks} className={`footer-link ${activeSection === 'howItWorks' ? 'is-active' : ''}`}>
              Pipeline
            </button>
            <button type="button" onClick={goToProcessing} className={`footer-link ${activeSection === 'processing' ? 'is-active' : ''}`}>
              Workspace
            </button>
          </div>
          <button
            type="button"
            className="ghost-button ghost-button--compact"
            onClick={toggleDarkMode}
            aria-label={`Switch to ${darkMode ? 'light' : 'dark'} mode`}
            title="Toggle dark mode (Ctrl/Cmd + D)"
          >
            {darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
          </button>
        </div>
        <p className="footer-copy">© {new Date().getFullYear()} AnomVisor. All rights reserved.</p>
      </footer>

      {showHelp && (
        <div className="overlay-panel" role="dialog" aria-modal="true" aria-labelledby="help-heading">
          <div className="panel-card">
            <div className="panel-card__header">
              <h2 id="help-heading">Help</h2>
            </div>
            <div className="panel-card__body help-content">
              <section>
                <h3>Getting started</h3>
                <p>Follow these steps to run an analysis:</p>
                <ul>
                  <li>Select a dataset preset that mirrors your scene.</li>
                  <li>Upload the hyperspectral `.mat` cube and its ground truth file.</li>
                  <li>Start the upload and let the backend process the data.</li>
                  <li>Explore anomaly maps, metrics, and classification overlays.</li>
                </ul>
              </section>
              <section>
                <h3>Features</h3>
                <ul>
                  <li>Light and dark workspace modes.</li>
                  <li>Backend health indicators with live status checks.</li>
                  <li>Toast notifications for progress and errors.</li>
                </ul>
              </section>
            </div>
            <div className="panel-card__footer">
              <button type="button" className="primary-button" onClick={toggleHelp}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}

    </div>
  );
}

export default App;

