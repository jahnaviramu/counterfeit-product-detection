# Step 1: Install
cd brand-auth-backend
pip install requests beautifulsoup4

# Step 2: Run
python app.py

# Step 3: Test
curl -X POST http://localhost:5000/api/v1/check \
  -H "Content-Type: application/json" \
  -d '{"url": "https://instagram.com/nike"}'