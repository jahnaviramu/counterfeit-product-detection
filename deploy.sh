#!/bin/bash

# Influencer System - Quick Deployment Script
# Run this after pulling the code to set up everything

echo "================================================"
echo "Influencer System - Production Deployment"
echo "================================================"
echo ""

# Check Python
echo "✓ Checking Python version..."
python --version

# Install requirements
echo ""
echo "✓ Installing Python dependencies..."
cd brand-auth-backend
pip install -q requests beautifulsoup4
pip install -q flask flask-cors pymongo pyjwt python-dotenv pillow tensorflow joblib

# Test imports
echo ""
echo "✓ Testing module imports..."
python -c "from influencer_authenticator import InfluencerAuthenticator; print('  ✓ influencer_authenticator imported')"
python -c "from influencer_fraud_detector import InfluencerFraudDetector; print('  ✓ influencer_fraud_detector imported')"
python -c "from influencer_analytics import InfluencerAnalytics; print('  ✓ influencer_analytics imported')"
python -c "from influencer_campaigns_schema import init_influencer_campaigns_schema; print('  ✓ influencer_campaigns_schema imported')"

# Check .env
echo ""
echo "✓ Checking configuration..."
if [ -f ".env" ]; then
    echo "  ✓ .env file found"
    if grep -q "MONGODB_URI" .env; then
        echo "  ✓ MONGODB_URI configured"
    else
        echo "  ⚠ MONGODB_URI not set in .env"
    fi
else
    echo "  ⚠ .env file not found - create it with:"
    echo "    MONGODB_URI=mongodb+srv://user:pass@cluster.mongodb.net/"
    echo "    JWT_SECRET=your_secret_key"
fi

echo ""
echo "================================================"
echo "✅ Deployment Setup Complete"
echo "================================================"
echo ""
echo "Quick Start Commands:"
echo ""
echo "1. Start server:"
echo "   python app.py"
echo ""
echo "2. Test influencer authentication:"
echo "   curl -X POST http://localhost:5000/api/influencer/authenticate \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"instagram_url\": \"https://instagram.com/nike\"}'"
echo ""
echo "3. Create campaign:"
echo "   curl -X POST http://localhost:5000/api/campaigns \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"brand_id\": \"brand_123\", \"title\": \"Campaign\", \"budget\": 10000, \"target_tier\": \"Macro\"}'"
echo ""
echo "4. Search marketplace:"
echo "   curl 'http://localhost:5000/api/marketplace/influencers?tier=Macro'"
echo ""
echo "Documentation:"
echo "  • Full API: See INFLUENCER_SYSTEM_COMPLETE.md"
echo "  • Deployment: See INFLUENCER_SYSTEM_DEPLOYMENT.md"
echo ""
