// src/App.js
import React, { useState } from 'react';
import QRCode from 'qrcode.react';

function App() {
  const [product, setProduct] = useState({
    name: '',
    brand: '',
    category: '',
    manufactureDate: '',
    batchNumber: '',
    description: ''
  });
  const [qrCodeData, setQrCodeData] = useState(null);
  const [isRegistered, setIsRegistered] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setProduct(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await fetch('http://localhost:5000/api/register_product', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(product),
      });
      const data = await response.json();
      if (response.ok && data.success) {
        setIsRegistered(true);
        setQrCodeData(JSON.stringify(data.qr_data));
      } else {
        alert(data.error || 'Registration failed.');
        setIsRegistered(false);
      }
    } catch (err) {
      alert('Network or server error: ' + err.message);
      setIsRegistered(false);
    }
  };

  const generateQRCode = () => {
    // This would normally come from your blockchain service
    const blockchainData = {
      productId: `prod_${Date.now()}`,
      ...product,
      timestamp: new Date().toISOString(),
      contractAddress: '0x123...abc' // Mock Ethereum address
    };
    setQrCodeData(JSON.stringify(blockchainData));
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-3xl font-bold text-gray-800 mb-8">Brand Product Registration</h1>
        
        <div className="flex flex-col md:flex-row gap-8">
          {/* Product Registration Form */}
          <div className="flex-1 bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Register New Product</h2>
            <form onSubmit={handleSubmit}>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Product Name</label>
                  <input
                    type="text"
                    name="name"
                    value={product.name}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md"
                    required
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Brand</label>
                  <input
                    type="text"
                    name="brand"
                    value={product.brand}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md"
                    required
                  />
                </div>
              </div>
              
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Category</label>
                  <select
                    name="category"
                    value={product.category}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md"
                    required
                  >
                    <option value="">Select category</option>
                    <option value="electronics">Electronics</option>
                    <option value="clothing">Clothing</option>
                    <option value="food">Food</option>
                    <option value="cosmetics">Cosmetics</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Manufacture Date</label>
                  <input
                    type="date"
                    name="manufactureDate"
                    value={product.manufactureDate}
                    onChange={handleChange}
                    className="w-full p-2 border border-gray-300 rounded-md"
                    required
                  />
                </div>
              </div>
              
              <div className="mb-4">
                <label className="block text-sm font-medium text-gray-700 mb-1">Batch Number</label>
                <input
                  type="text"
                  name="batchNumber"
                  value={product.batchNumber}
                  onChange={handleChange}
                  className="w-full p-2 border border-gray-300 rounded-md"
                  required
                />
              </div>
              
              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
                <textarea
                  name="description"
                  value={product.description}
                  onChange={handleChange}
                  className="w-full p-2 border border-gray-300 rounded-md"
                  rows="3"
                />
              </div>
              
              <button
                type="submit"
                className="bg-blue-600 text-white py-2 px-6 rounded-md hover:bg-blue-700 transition"
              >
                Register Product
              </button>
            </form>
          </div>
          
          {/* QR Code Generation Box */}
          <div className="flex-1 bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Blockchain QR Code</h2>
            
            {isRegistered ? (
              <div className="flex flex-col items-center">
                {qrCodeData ? (
                  <>
                    <div className="mb-4 p-2 border border-gray-200">
                      <QRCode value={qrCodeData} size={200} />
                    </div>
                    <p className="text-sm text-gray-600 mb-4">
                      This QR contains blockchain authentication data for your product.
                    </p>
                    <button
                      onClick={() => {
                        // Download QR code functionality would go here
                        const canvas = document.getElementById("qr-code");
                        const pngUrl = canvas
                          .toDataURL("image/png")
                          .replace("image/png", "image/octet-stream");
                        let downloadLink = document.createElement("a");
                        downloadLink.href = pngUrl;
                        downloadLink.download = "product_qr.png";
                        document.body.appendChild(downloadLink);
                        downloadLink.click();
                        document.body.removeChild(downloadLink);
                      }}
                      className="bg-green-600 text-white py-2 px-4 rounded-md hover:bg-green-700 transition"
                    >
                      Download QR Code
                    </button>
                  </>
                ) : (
                  <>
                    <div className="bg-gray-100 p-8 rounded-lg mb-4 text-center">
                      <p className="text-gray-500">QR code will be generated after registration</p>
                    </div>
                    <button
                      onClick={generateQRCode}
                      className="bg-blue-600 text-white py-2 px-6 rounded-md hover:bg-blue-700 transition"
                      disabled={!isRegistered}
                    >
                      Generate Blockchain QR Code
                    </button>
                  </>
                )}
              </div>
            ) : (
              <div className="bg-gray-100 p-8 rounded-lg text-center">
                <p className="text-gray-500">Please register a product first to generate QR code</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;