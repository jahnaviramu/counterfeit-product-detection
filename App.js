import { useState, useEffect } from 'react';
import AdminPanel from './components/AdminPanel';
import BarcodeScanner from './components/BarcodeScanner';
import ARProductOverlay from './components/ARProductOverlay';
import SellerDashboard from './components/SellerDashboard';
import { BrowserRouter as Router, Routes, Route, Link, useLocation, Navigate, useNavigate } from 'react-router-dom';
import Login from './components/Login';
import ForgotPassword from './components/ForgotPassword';
import ResetPassword from './components/ResetPassword';
import Signup from './components/Signup';
import ProductForm from './components/ProductForm';
import QRGenerator from './components/QRGenerator';
import VerifyProduct from './components/VerifyProduct';
import RoleSelectionPage from './components/RoleSelectionPage';
import BrandAuthForm from './components/BrandAuthForm';
import Dashboard from './components/Dashboard';
import RegisterPage from './components/RegisterPage';
import BuyerDashboard from './components/BuyerDashboard';
import InfluencerDashboard from './components/InfluencerDashboard';

import MultimodalVerify from './components/MultimodalVerify';
import MonitoringDashboard from './components/MonitoringDashboard';
import InfluencerTracking from './components/InfluencerTracking';
import BecomeInfluencer from './components/BecomeInfluencer';
import InfluencerOnboarding from './components/InfluencerOnboarding';

import { registerProduct } from './services/api';

function Navbar({ user, onLogout }) {
  const location = useLocation();
  const [refreshKey, setRefreshKey] = useState(0);
  
  // Detect role changes from localStorage (for when login happens)
  useEffect(() => {
    const checkRole = () => {
      setRefreshKey(k => k + 1);
    };
    
    window.addEventListener('storage', checkRole);
    return () => window.removeEventListener('storage', checkRole);
  }, []);
  
  // Determine role from user state, localStorage, or JWT token fallback
  function detectRole() {
    // Always check fresh from localStorage first (most reliable after login)
    const lsRole = localStorage.getItem('userRole');
    if (lsRole) {
      console.log('Role from localStorage:', lsRole);
      return lsRole;
    }
    
    // Then check user object
    if (user?.role) {
      console.log('Role from user object:', user.role);
      return user.role;
    }
    
    // Try parsing JWT token if available
    const token = localStorage.getItem('authToken') || localStorage.getItem('token');
    if (token) {
      try {
        const payload = JSON.parse(atob(token.split('.')[1]));
        if (payload && payload.role) {
          console.log('Role from JWT:', payload.role);
          return payload.role;
        }
      } catch (e) {
        console.log('JWT parse error:', e);
      }
    }
    
    console.log('No role detected');
    return null;
  }
  const [roleState, setRoleState] = useState(() => detectRole());

  // Poll localStorage every 700ms as a fallback to catch role changes from other login flows
  useEffect(() => {
    let mounted = true;
    const check = () => {
      try {
        const r = detectRole();
        if (mounted && r !== roleState) setRoleState(r);
      } catch (e) {
        // ignore
      }
    };
    const id = setInterval(check, 700);
    // also run once immediately
    check();
    return () => { mounted = false; clearInterval(id); };
  }, [roleState]);

  // Build role-aware navigation
  const navItems = [];
  navItems.push({ name: 'Home', path: '/' });
  // Admin Panel should be accessible to all (per request)
  navItems.push({ name: 'Admin Panel', path: '/admin' });

  const role = roleState;
  if (role === 'seller') {
    navItems.push({ name: 'Product Authentication', path: '/product-auth' });
    navItems.push({ name: 'Merchant Studio', path: '/seller-dashboard' });
    navItems.push({ name: 'Merchant Insights', path: '/seller-behavior' });
    navItems.push({ name: 'Verify & Scan', path: '/scan' });
  } else if (role === 'buyer') {
    navItems.push({ name: 'Verify & Scan', path: '/buyer-verification' });
    navItems.push({ name: 'Buyer Dashboard', path: '/buyer-dashboard' });
  } else if (role === 'influencer') {
    navItems.push({ name: 'Product Authentication', path: '/product-auth' });
    navItems.push({ name: 'Influencer Tracking', path: '/influencer-tracking' });
    navItems.push({ name: 'Influencer Dashboard', path: '/influencer-dashboard' });
    navItems.push({ name: 'Verify & Scan', path: '/scan' });
  } else {
    // not logged-in or unknown role: expose basic public links
    navItems.push({ name: 'Product Authentication', path: '/product-auth' });
    navItems.push({ name: 'Verify & Scan', path: '/scan' });
  }

  // Help always visible
  navItems.push({ name: 'Help & Support', path: '/help' });
  // Show username if available, otherwise use email before @
  const displayName = user?.username || (user?.email ? user.email.split('@')[0] : 'User');
  return (
    <nav className="flex flex-row items-center px-4 py-3 bg-blue-900 shadow-lg sticky top-0 z-50 relative">
      <div className="flex flex-row flex-1">
        {navItems.map(item => (
          <Link
            key={item.name}
            to={item.path}
            className={`px-3 py-2 rounded-md text-white font-semibold transition-all duration-200 ${location.pathname === item.path ? 'bg-blue-700 scale-105 shadow ring-2 ring-white' : 'hover:bg-blue-800 opacity-90'}`}
          >
            {item.name}
          </Link>
        ))}
      </div>
      {user ? (
        <div className="flex items-center ml-4">
          <button className="flex items-center bg-white text-blue-900 font-semibold px-3 py-1 rounded-full shadow hover:bg-blue-100 transition">
            <span className="inline-block w-8 h-8 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold mr-2">
              {displayName.charAt(0).toUpperCase()}
            </span>
            {displayName}
          </button>
          <Link to="/logout" className="text-white underline ml-2">Logout</Link>
          {/* Debug: show detected role and storage keys to help diagnose missing nav items */}
          <div className="ml-3 text-xs text-white/80">
            <div>role: <span className="font-semibold">{role || 'none'}</span></div>
            <div>userRole(ls): <span className="font-semibold">{localStorage.getItem('userRole') || 'none'}</span></div>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3 ml-4">
          <Link to="/signup" className="bg-white text-blue-900 font-semibold px-3 py-1 rounded-full shadow hover:bg-blue-100 transition">Sign Up</Link>
          <Link to="/login" className="bg-white text-blue-900 font-semibold px-3 py-1 rounded-full shadow hover:bg-blue-100 transition">Login</Link>
        </div>
      )}
    </nav>
  );
}

function LoginPage({ onLogin }) {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");
    if (!email || !password) {
      setError("Please enter both email and password.");
      return;
    }
    if (!/^\S+@\S+\.\S+$/.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }
    try {
      const res = await fetch(process.env.REACT_APP_API_URL + '/api/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password })
      });
      const data = await res.json();
      if (res.ok && data.token) {
        // persist token and role, then notify parent App about logged-in user including role
        localStorage.setItem('authToken', data.token);
        if (data.role) localStorage.setItem('userRole', data.role);
        setError('You are now logged in! Redirecting...');
        setTimeout(() => {
          onLogin({ username: username || email.split('@')[0], email, token: data.token, role: data.role });
          navigate('/product-auth');
        }, 1200);
      } else {
        setError(data.error || 'Invalid credentials.');
      }
    } catch (err) {
      // Avoid showing a persistent generic error to end users. Log for debugging only.
      console.warn('Login network error:', err);
      // Do not set a persistent error message here to avoid confusing users when auth server is temporarily unreachable.
      // setError('Unable to reach the authentication server. Please check your connection and try again.');
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-500 via-pink-400 to-yellow-300">
      <div className="w-full max-w-md mx-auto rounded-2xl shadow-2xl bg-white/90 backdrop-blur-md p-10 border-4 border-white/40">
        <h1 className="text-3xl font-bold mb-6 text-blue-700 text-center">Login</h1>
        <form className="flex flex-col gap-4" onSubmit={handleSubmit}>
          <input type="text" placeholder="Username" value={username} onChange={e => setUsername(e.target.value)} className="p-3 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400" />
          <input type="text" placeholder="Email" value={email} onChange={e => setEmail(e.target.value)} className="p-3 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400" />
          <input type="password" placeholder="Password" value={password} onChange={e => setPassword(e.target.value)} className="p-3 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-blue-400" />
          <div className="flex justify-between text-sm mt-1">
            <span></span>
            <button type="button" className="text-blue-700 underline" onClick={() => setError('Forgot password is not implemented yet.')}>Forgot Password?</button>
          </div>
          <button type="submit" className="bg-blue-700 text-white py-3 rounded font-bold hover:bg-blue-800 transition">Login</button>
        </form>
        {error && <div className="mt-2 text-center text-red-600 text-sm">{error}</div>}
        {/* Removed persistent signup prompt to reduce UI noise; registration link available from main nav */}
        <div className="mt-2 text-center text-gray-400 text-xs">For your security, passwords are never stored in the browser.</div>
      </div>
    </div>
  );
}

function RequireLogin({ isLoggedIn, children }) {
  const navigate = useNavigate();
  if (!isLoggedIn) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[300px]">
        <div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center">
          <div className="text-xl font-bold text-blue-700 mb-2">Login Required</div>
          <div className="mb-4 text-gray-600">You must be logged in to access this feature.</div>
          <Link to="/login" className="bg-blue-700 text-white px-4 py-2 rounded font-semibold hover:bg-blue-800 transition">Login</Link>
          {/* Registration link removed from this prompt to avoid duplicate CTAs; use main navigation or help page to register. */}
        </div>
      </div>
    );
  }
  return children;
}

function LogoutPage({ onLogout }) {
  const navigate = useNavigate();
  // Wait 2 seconds before redirecting
  useEffect(() => {
    localStorage.removeItem('authToken');
    onLogout();
    const timer = setTimeout(() => navigate('/login'), 2000);
    return () => clearTimeout(timer);
  }, [onLogout, navigate]);
  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-500 via-pink-400 to-yellow-300">
      <div className="bg-white/90 p-8 rounded-2xl shadow-2xl w-full max-w-md border-4 border-white/40 text-center">
        <div className="text-3xl font-bold text-blue-700 mb-4">Logged out</div>
        <div className="text-gray-700 mb-2">You have been signed out of your account.</div>
        <div className="text-gray-500 text-sm mt-4 text-right">Redirecting to login...</div>
      </div>
    </div>
  );
}

function ScanPage() {
  const [scanResult, setScanResult] = useState(null);
  const [apiResponse, setApiResponse] = useState(null);
  const handleDetected = async (code) => {
    setScanResult(code);
    // Send scanned code to backend for verification
    try {
      const res = await fetch(process.env.REACT_APP_API_URL + '/api/verify-barcode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      });
      const data = await res.json();
      setApiResponse(data);
    } catch (err) {
      setApiResponse({ error: 'Network or server error.' });
    }
  };
  return (
    <div className="p-4">
      <h2 className="text-xl font-bold mb-2">Scan QR/Barcode</h2>
      <BarcodeScanner onDetected={handleDetected} />
      {scanResult && <div className="mt-2">Scanned Code: <span className="font-mono bg-gray-100 px-2 py-1 rounded">{scanResult}</span></div>}
      {/* AR overlay for product visualization */}
      {apiResponse?.product?.registered && (
        <ARProductOverlay productName={apiResponse.product.name || 'Product'} />
      )}
      {apiResponse && (
        <div className="mt-4 p-2 border rounded bg-gray-50">
          {apiResponse.error ? (
            <div className="text-red-600">{apiResponse.error}</div>
          ) : (
            <>
              {apiResponse.product?.registered ? (
                <div>
                  <div className="font-bold text-green-700">Product Registered</div>
                  <div>Name: {apiResponse.product.name}</div>
                  <div>Brand: {apiResponse.product.brand}</div>
                  <div>Blockchain TX: <span className="font-mono">{apiResponse.product.blockchain_tx}</span></div>
                  <div>Contract: <span className="font-mono">{apiResponse.product.contract_address}</span></div>
                  <div>Status: <span className="font-bold text-blue-700">{apiResponse.blockchain?.status}</span></div>
                  {apiResponse.blockchain?.tx_hash && (
                    <div>Transaction Hash: <span className="font-mono">{apiResponse.blockchain.tx_hash}</span></div>
                  )}
                  {/* Dashboard analytics placeholder */}
                  <div className="mt-4 p-2 border rounded bg-blue-50">
                    <div className="font-bold text-blue-700 mb-2">Dashboard Analytics</div>
                    <div>Authenticity Score: <span className="font-mono">{apiResponse.product.authenticity_score || 'N/A'}</span></div>
                    <div>Image Features: <span className="font-mono">{apiResponse.product.image_features || 'N/A'}</span></div>
                    <div>Review Analysis: <span className="font-mono">{apiResponse.product.review_analysis || 'N/A'}</span></div>
                    <div>Seller Behavior: <span className="font-mono">{apiResponse.product.seller_behavior || 'N/A'}</span></div>
                    <div>Influencer Activity: <span className="font-mono">{apiResponse.product.influencer_activity || 'N/A'}</span></div>
                  </div>
                </div>
              ) : (
                <div className="font-bold text-red-700">Product Not Registered</div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}

// Move ScanPage outside App function

function App() {
  const [product, setProduct] = useState(null);
  const [isRegistered, setIsRegistered] = useState(false);
  const [error, setError] = useState(null);
  const [user, setUser] = useState(() => {
    const storedUser = localStorage.getItem('user');
    return storedUser ? JSON.parse(storedUser) : null;
  });
  const [loadingQR, setLoadingQR] = useState(false);

  const handleLogin = (userObj) => {
    setUser(userObj);
    localStorage.setItem('user', JSON.stringify(userObj));
  };
  const handleLogout = () => {
    setUser(null);
    localStorage.removeItem('user');
    localStorage.removeItem('token');
  };

  const handleRegister = async (productData) => {
    setError(null);
    setIsRegistered(false);
    setProduct(null);
    setLoadingQR(true); // Show loading QR immediately
    // Show a temporary QR code (e.g., with 'Registering...' text or a spinner)
    setProduct({ verification_url: 'Registering product...' });
    setIsRegistered(true);
    try {
      const res = await registerProduct(productData);
      const data = await res.json();
      console.log('registerProduct API response:', data); // DEBUG
      if (data.success && data.qr_data) {
        setProduct({ ...data.qr_data });
        setIsRegistered(true);
        setLoadingQR(false);
        console.log('Set product state:', data.qr_data); // DEBUG
      } else {
        setError(data.error || 'Registration failed.');
        setIsRegistered(false);
        setLoadingQR(false);
        console.log('Registration error:', data.error); // DEBUG
      }
    } catch (e) {
      setError('Network or server error.');
      setIsRegistered(false);
      setLoadingQR(false);
      console.log('Network/server error:', e); // DEBUG
    }
  };

  // Helper: get role from token or user object
  function getRole() {
    if (user?.role) return user.role;
    try {
      const token = localStorage.getItem('authToken');
      if (!token) return null;
      const payload = JSON.parse(atob(token.split('.')[1]));
      return payload.role || null;
    } catch {
      return null;
    }
  }

  // Optional: Clear QR code on logout or add a reset button as needed

  return (
    <Router>
      <Navbar user={user} onLogout={handleLogout} />
      {/* Phone / Tunnel link banner (shows when REACT_APP_PHONE_LINK or localStorage.tunnelUrl exists) */}
      {(() => {
        const envLink = process.env.REACT_APP_PHONE_LINK;
        const lsLink = localStorage.getItem('tunnelUrl') || localStorage.getItem('phoneLink');
        const phoneLink = envLink || lsLink;
        if (phoneLink) {
          return (
            <div className="w-full bg-yellow-100 text-yellow-800 py-2 text-center">
              Mobile access: <a className="underline font-semibold" href={phoneLink} target="_blank" rel="noreferrer">{phoneLink}</a>
            </div>
          );
        }
        return null;
      })()}
      
      {/* Phone/Mobile Access Banner */}
      {process.env.REACT_APP_PHONE_URL && (
        <div className="bg-gradient-to-r from-purple-600 to-pink-600 text-white px-4 py-3 text-center shadow-md sticky top-[56px] z-40">
          <div className="text-sm font-semibold">
            📱 Access from your phone: <span className="font-mono font-bold">{process.env.REACT_APP_PHONE_URL}</span>
            <button 
              onClick={() => {
                navigator.clipboard.writeText(process.env.REACT_APP_PHONE_URL);
                alert('Phone link copied to clipboard!');
              }}
              className="ml-2 bg-white/30 hover:bg-white/50 px-2 py-1 rounded text-xs transition"
            >
              Copy
            </button>
          </div>
        </div>
      )}
      
      <Routes>
        <Route path="/scan" element={
          <RequireLogin isLoggedIn={!!user}>
            <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-500 via-pink-400 to-yellow-300 p-4">
              <div className="w-full max-w-4xl mx-auto rounded-2xl shadow-2xl bg-white/90 backdrop-blur-md p-8 border-4 border-white/40">
                <h1 className="text-3xl font-bold mb-2 text-blue-700">Product Verification & Scanning</h1>
                <p className="text-gray-600 mb-6">Scan QR codes or upload product images and descriptions for AI-powered verification.</p>
                <BarcodeScanner onDetected={(code) => alert(`Scanned: ${code}`)} />
              </div>
            </div>
          </RequireLogin>
        } />
        <Route path="/login" element={<LoginPage onLogin={handleLogin} />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route path="/reset-password" element={<ResetPassword />} />
        <Route path="/logout" element={<LogoutPage onLogout={handleLogout} />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/signup" element={<Signup />} />
        <Route path="/" element={
          <div className="min-h-[calc(100vh-56px)] bg-gradient-to-br from-blue-500 via-pink-400 to-yellow-300 flex flex-col items-center justify-center">
            <div className="w-full max-w-6xl mx-auto rounded-2xl shadow-2xl bg-white/80 backdrop-blur-md p-10 border-4 border-white/40">
              <h1 className="text-3xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-fuchsia-700 via-blue-700 to-yellow-500 mb-4 text-center drop-shadow-lg tracking-tight animate-pulse">
                Empowering Authenticity
              </h1>
              <h2 className="text-xl md:text-2xl font-bold text-blue-800 mb-8 text-center tracking-wide">
                AI-Powered Counterfeit Product Detection for Social Media Stores & E-commerce
              </h2>
              {/* Full Project Overview Section with Relevant Images and Content */}
              <div className="mb-10">
                <h2 className="text-2xl font-bold text-blue-700 mb-2">Project Overview</h2>
                <div className="flex flex-col md:flex-row gap-8 items-center mb-6">
                  <img src="https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=400&q=80" alt="Blockchain Ledger" className="w-32 h-32 rounded-xl shadow-lg border-2 border-blue-300 bg-white object-cover" />
                  <p className="mb-4 text-gray-700 text-lg">
                    <span className="font-semibold">Brand Product Authentication</span> is a blockchain-powered platform designed to combat counterfeiting and ensure product authenticity for brands, merchants, and shoppers. The system leverages Ethereum smart contracts to register products, generate unique QR codes, and enable instant verification by end-users. It features a modern, user-friendly interface and is built for extensibility with analytics, AR/AI, and admin tools.
                  </p>
                </div>
                <div className="flex flex-col md:flex-row gap-8 items-center mb-6">
                  <img src="https://images.unsplash.com/photo-1519389950473-47ba0277781c?auto=format&fit=crop&w=400&q=80" alt="Product Tag" className="w-32 h-32 rounded-xl shadow-lg border-2 border-green-300 bg-white object-cover" />
                  <img src="https://images.unsplash.com/photo-1519125323398-675f0ddb6308?auto=format&fit=crop&w=400&q=80" alt="Blockchain Network" className="w-32 h-32 rounded-xl shadow-lg border-2 border-blue-300 bg-white object-cover" />
                  <img src="https://images.unsplash.com/photo-1515378791036-0648a3ef77b2?auto=format&fit=crop&w=400&q=80" alt="QR Code Scan" className="w-32 h-32 rounded-xl shadow-lg border-2 border-yellow-300 bg-white object-cover" />
                </div>
                <h3 className="text-xl font-semibold text-blue-600 mt-6 mb-2">Key Features</h3>
                <ul className="list-disc pl-6 mb-4 text-gray-700">
                  <li>Blockchain-based product registration and verification</li>
                  <li>Role-based flows for sellers and buyers</li>
                  <li>Unique QR code generation for each registered product</li>
                  <li>Modern, persistent navigation and responsive UI</li>
                  <li>Expandable for analytics, admin, AR/AI, and more</li>
                </ul>
                <h3 className="text-xl font-semibold text-blue-600 mt-6 mb-2">Technology Stack</h3>
                <ul className="list-disc pl-6 mb-4 text-gray-700">
                  <li><span className="font-semibold">Smart Contracts:</span> Solidity (Ethereum Sepolia testnet)</li>
                  <li><span className="font-semibold">Backend:</span> Flask (Python), Web3.py, Infura</li>
                  <li><span className="font-semibold">Frontend:</span> React.js, Tailwind CSS</li>
                  <li><span className="font-semibold">Other:</span> QR code generation, RESTful API</li>
                </ul>
                {/* Tech Stack Table */}
                <div className="overflow-x-auto mb-8">
                  <table className="min-w-[350px] max-w-xl w-full mx-auto border border-gray-300 rounded-lg shadow bg-white">
                    <thead className="bg-blue-100">
                      <tr>
                        <th className="px-4 py-2 text-left font-bold text-blue-800">Area</th>
                        <th className="px-4 py-2 text-left font-bold text-blue-800">Tools/Tech</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr><td className="border-t px-4 py-2">Blockchain</td><td className="border-t px-4 py-2">Ethereum + QR Code Integration</td></tr>
                      <tr><td className="border-t px-4 py-2">AR</td><td className="border-t px-4 py-2">WebAR + AR.js</td></tr>
                      <tr><td className="border-t px-4 py-2">Computer Vision</td><td className="border-t px-4 py-2">TensorFlow</td></tr>
                      <tr><td className="border-t px-4 py-2">NLP</td><td className="border-t px-4 py-2">BERT</td></tr>
                      <tr><td className="border-t px-4 py-2">Backend</td><td className="border-t px-4 py-2">Flask</td></tr>
                      <tr><td className="border-t px-4 py-2">Frontend</td><td className="border-t px-4 py-2">React + Tailwind</td></tr>
                      <tr><td className="border-t px-4 py-2">Dashboard</td><td className="border-t px-4 py-2">React + Chart.js</td></tr>
                      <tr><td className="border-t px-4 py-2">Storage</td><td className="border-t px-4 py-2">Firebase, MongoDB</td></tr>
                      <tr><td className="border-t px-4 py-2">Deployment</td><td className="border-t px-4 py-2">Heroku, Vercel</td></tr>
                    </tbody>
                  </table>
                </div>
                <h3 className="text-xl font-semibold text-blue-600 mt-6 mb-2">User Flows</h3>
                <div className="flex flex-col md:flex-row gap-8 items-center mb-4">
                  <img src="https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?auto=format&fit=crop&w=400&q=80" alt="Seller Registering Product" className="w-32 h-32 rounded-xl shadow-lg border-2 border-purple-300 bg-white object-cover" />
                  <img src="https://images.unsplash.com/photo-1519125323398-675f0ddb6308?auto=format&fit=crop&w=400&q=80" alt="Blockchain Verification" className="w-32 h-32 rounded-xl shadow-lg border-2 border-pink-300 bg-white object-cover" />
                  <img src="https://images.unsplash.com/photo-1515378791036-0648a3ef77b2?auto=format&fit=crop&w=400&q=80" alt="Buyer Scanning QR Code" className="w-32 h-32 rounded-xl shadow-lg border-2 border-yellow-300 bg-white object-cover" />
                  <ul className="list-disc pl-6 text-gray-700">
                      <li><span className="font-semibold">Merchant:</span> Registers products on-chain, receives a QR code for each item, and can manage products via the Merchant Studio.</li>
                      <li><span className="font-semibold">Shopper:</span> Scans QR code to verify authenticity and view product details instantly.</li>
                  </ul>
                </div>
                <h3 className="text-xl font-semibold text-blue-600 mt-6 mb-2">Navigation</h3>
                <ul className="list-disc pl-6 mb-4 text-gray-700">
                  <li><span className="font-semibold">Product Authentication:</span> Register new products (Merchant)</li>
                  <li><span className="font-semibold">Merchant Studio:</span> Manage and analyze registered products</li>
                  <li><span className="font-semibold">Verify & Scan:</span> Verify product authenticity (Shopper)</li>
                  <li><span className="font-semibold">Seller Behavior Analysis, Influencer & Marketplace Tracking, Admin Panel:</span> Advanced analytics and management (future expansion)</li>
                  <li><span className="font-semibold">Help & Support:</span> Guidance and troubleshooting</li>
                </ul>
                <h3 className="text-xl font-semibold text-blue-600 mt-6 mb-2">Future Expansion</h3>
                <ul className="list-disc pl-6 mb-4 text-gray-700">
                  <li>AI-powered brand protection and fraud detection</li>
                  <li>Augmented Reality (AR) product visualization</li>
                  <li>Comprehensive analytics and reporting</li>
                  <li>Admin tools for compliance and monitoring</li>
                </ul>
                <h3 className="text-xl font-semibold text-blue-600 mt-6 mb-2">How to Use</h3>
                <ol className="list-decimal pl-6 mb-4 text-gray-700">
                  <li>Navigate to <span className="font-semibold">Product Authentication</span> to register a product (Seller).</li>
                  <li>Scan the generated QR code to verify authenticity (Buyer).</li>
                  <li>Explore dashboards and analytics (coming soon).</li>
                </ol>
                <p className="text-gray-700">
                  For more details, use the navigation bar above or visit Help & Support.
                </p>
              </div>
              {error && <div className="text-red-600 mb-4">{error}</div>}
            </div>
          </div>
        } />
        <Route path="/product-auth" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'seller' ? (
              <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-500 via-pink-400 to-yellow-300">
                <div className="w-full max-w-5xl mx-auto rounded-2xl shadow-2xl bg-white/80 backdrop-blur-md p-10 border-4 border-white/40">
                  <h1 className="text-3xl font-bold mb-6 text-blue-700">Product Authentication (Seller)</h1>
                  <div className="grid md:grid-cols-2 gap-8 items-start">
                    <div>
                      <ProductForm onRegister={handleRegister} />
                    </div>
                    <div>
                      <div className="bg-white/90 p-6 rounded-lg shadow-md border border-gray-200">
                        <h2 className="text-xl font-semibold mb-4 text-blue-700">QR Code</h2>
                        {isRegistered && product ? (
                          <QRGenerator product={product} isRegistered={isRegistered} />
                        ) : (
                          <div className="text-center text-gray-500 py-8">
                            Register a product and the QR code will appear here.
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="min-h-screen flex items-center justify-center">
                <div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center">
                  <div className="text-xl font-bold text-blue-700 mb-2">Access Denied</div>
                  <div className="mb-4 text-gray-600">Only sellers can access this page.</div>
                </div>
              </div>
            )}
          </RequireLogin>
        } />
        <Route path="/buyer-verification" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'buyer' ? (
              <MultimodalVerify />
            ) : (
              <div className="min-h-screen flex items-center justify-center">
                <div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center">
                  <div className="text-xl font-bold text-blue-700 mb-2">Access Denied</div>
                  <div className="mb-4 text-gray-600">Only buyers can access this page.</div>
                </div>
              </div>
            )}
          </RequireLogin>
        } />
        <Route path="/seller-dashboard" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'seller' ? (
              <SellerDashboard />
            ) : (
              <div className="min-h-screen flex items-center justify-center"><div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center"><div className="text-xl font-bold text-blue-700 mb-2">Access Denied</div><div className="mb-4 text-gray-600">Only sellers can access this page.</div></div></div>
            )}
          </RequireLogin>
        } />
        <Route path="/buyer-dashboard" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'buyer' ? (
              <BuyerDashboard />
            ) : (
              <div className="min-h-screen flex items-center justify-center"><div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center"><div className="text-xl font-bold text-blue-700 mb-2">Access Denied</div><div className="mb-4 text-gray-600">Only buyers can access this page.</div></div></div>
            )}
          </RequireLogin>
        } />
        <Route path="/influencer-dashboard" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'influencer' ? (
              <InfluencerDashboard />
            ) : (
              <div className="min-h-screen flex items-center justify-center"><div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center"><div className="text-xl font-bold text-blue-700 mb-2">Access Denied</div><div className="mb-4 text-gray-600">Only influencers can access this page.</div></div></div>
            )}
          </RequireLogin>
        } />
        <Route path="/seller-behavior" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'seller' ? (
              <MonitoringDashboard />
            ) : (
              <div className="min-h-screen flex items-center justify-center"><div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center"><div className="text-xl font-bold text-blue-700 mb-2">Access Denied</div><div className="mb-4 text-gray-600">Only sellers can access this page.</div></div></div>
            )}
          </RequireLogin>
        } />
        <Route path="/influencer-tracking" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'influencer' ? (
              <InfluencerTracking />
            ) : (
              <div className="min-h-screen flex items-center justify-center">
                <div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center">
                  <div className="text-xl font-bold text-blue-700 mb-2">Access Denied</div>
                  <div className="mb-4 text-gray-600">Only influencers can access this page.</div>
                </div>
              </div>
            )}
          </RequireLogin>
        } />
        <Route path="/become-influencer" element={
          <RequireLogin isLoggedIn={!!user}>
            <BecomeInfluencer onRoleChange={(role) => { setUser(u => ({ ...(u||{}), role })); localStorage.setItem('userRole', role); }} />
          </RequireLogin>
        } />
        <Route path="/influencer-onboarding" element={
          <RequireLogin isLoggedIn={!!user}>
            {user?.role === 'influencer' ? (
              <InfluencerOnboarding />
            ) : (
              <div className="min-h-screen flex items-center justify-center">
                <div className="bg-white/90 p-8 rounded-xl shadow border border-blue-200 text-center">
                  <div className="text-xl font-bold text-blue-700 mb-2">Influencer Only</div>
                  <div className="mb-4 text-gray-600">This page is for influencers. <a href="/become-influencer" className="text-blue-700 underline">Become an influencer</a> first.</div>
                </div>
              </div>
            )}
          </RequireLogin>
        } />
  <Route path="/admin" element={<RequireLogin isLoggedIn={!!user}><AdminPanel /></RequireLogin>} />
        <Route path="/help" element={
          <div className="min-h-screen flex flex-col items-center justify-center bg-gradient-to-br from-blue-500 via-pink-400 to-yellow-300">
            <div className="w-full max-w-3xl mx-auto rounded-2xl shadow-2xl bg-white/90 backdrop-blur-md p-10 border-4 border-white/40">
              <h1 className="text-3xl font-bold mb-6 text-blue-700 text-center">Help & Support</h1>
              <p className="mb-4 text-gray-700 text-center">Need assistance? Find answers to common questions, troubleshooting tips, and contact information below.</p>
              <h2 className="text-xl font-semibold text-blue-600 mt-4 mb-2">Frequently Asked Questions</h2>
              <ul className="list-disc pl-6 mb-4 text-gray-700">
                <li><span className="font-semibold">How do I register a product?</span> Go to <span className="text-blue-700">Product Authentication</span> and fill out the registration form as a seller.</li>
                <li><span className="font-semibold">How do I verify a product?</span> Scan the QR code on the product or use the <span className="text-blue-700">Buyer Verification</span> page.</li>
                <li><span className="font-semibold">What if my transaction fails?</span> Ensure your wallet is funded and try again. For persistent issues, contact support.</li>
                <li><span className="font-semibold">Who can access the Admin Panel?</span> Only authorized administrators have access to admin features.</li>
              </ul>
              <h2 className="text-xl font-semibold text-blue-600 mt-4 mb-2">Contact Support</h2>
              <ul className="list-disc pl-6 mb-4 text-gray-700">
                <li>Email: <a href="mailto:support@brandauth.com" className="text-blue-700 underline">support@brandauth.com</a></li>
                <li>Telegram: <a href="https://t.me/brandauthsupport" className="text-blue-700 underline">@brandauthsupport</a></li>
                <li>GitHub Issues: <a href="https://github.com/your-org/brand-auth/issues" className="text-blue-700 underline">brand-auth/issues</a></li>
              </ul>
              <div className="mt-8 text-center text-yellow-700 font-semibold">* For urgent issues, please email or message us directly.</div>
            </div>
          </div>
        } />
        <Route path="/auth-form" element={<BrandAuthForm />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/verify/:product_id" element={<VerifyProduct />} />
        <Route path="*" element={<div className="p-8 text-center">Page not found</div>} />
      </Routes>
    </Router>
  );
}

export default App;
