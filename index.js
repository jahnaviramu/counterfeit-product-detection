import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';
import ErrorBoundary from './components/ErrorBoundary';

// Handle unhandled promise rejections gracefully
window.addEventListener('unhandledrejection', event => {
  console.error('Unhandled promise rejection:', event.reason);
  // Prevent app from crashing, just log it
  event.preventDefault();
});

// Also handle any errors thrown asynchronously
window.addEventListener('error', event => {
  if (event.error) {
    console.error('Global error handler:', event.error);
  }
});

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </React.StrictMode>
);
