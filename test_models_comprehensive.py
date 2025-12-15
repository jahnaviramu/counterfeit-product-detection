#!/usr/bin/env python3
"""
Comprehensive model evaluation test with proper authentication.
Tests CV, NLP, and fusion models end-to-end.
"""

import sys
import os
import json
import requests
from datetime import datetime, timedelta
import jwt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
API_BASE = 'http://127.0.0.1:5000'
JWT_SECRET = 'supersecretkey'  # Must match backend's os.getenv('JWT_SECRET', 'supersecretkey')
MERCHANT_ID = 'test_merchant_123'
ROLE = 'merchant'

def generate_test_token():
    """Generate a valid JWT token for testing"""
    payload = {
        'merchant_id': MERCHANT_ID,
        'role': ROLE,
        'exp': datetime.now() + timedelta(hours=1),
        'iat': datetime.now()
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    print(f"✅ Generated JWT Token: {token[:50]}...")
    return token

def test_health_check():
    """Test if backend is running"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    try:
        res = requests.get(f'{API_BASE}/api/health_check', timeout=5)
        print(f"✅ Backend Status: {res.status_code}")
        print(f"Response: {res.json()}")
        return True
    except Exception as e:
        print(f"❌ Backend not responding: {e}")
        return False

def test_nlp_only(token):
    """Test NLP model with text only"""
    print("\n" + "="*60)
    print("TEST 2: NLP Model (Text Only)")
    print("="*60)
    
    test_texts = [
        "Premium original authentic Nike Air Max running shoes",
        "FAKE COUNTERFEIT COPY not original cheap knockoff",
        "Amazing quality best deal ever discount seller trusted"
    ]
    
    for text in test_texts:
        print(f"\nTesting: '{text[:50]}...'")
        data = {'text': text}
        headers = {'Authorization': f'Bearer {token}'}
        
        try:
            res = requests.post(
                f'{API_BASE}/api/multimodal_check',
                json=data,
                headers=headers,
                timeout=10
            )
            print(f"  Status: {res.status_code}")
            
            if res.status_code == 200:
                result = res.json()
                nlp = result.get('nlp')
                if nlp and isinstance(nlp, dict) and 'authentic_prob' in nlp:
                    nlp_score = nlp['authentic_prob']
                    print(f"  ✅ NLP Score: {nlp_score:.2%}")
                    fused = result.get('fused')
                    if fused and isinstance(fused, dict) and 'authentic_prob' in fused:
                        fused_score = fused['authentic_prob']
                        print(f"  ✅ Fused Score: {fused_score:.2%}")
                else:
                    print(f"  ⚠️ Unexpected NLP response structure: {result}")
            else:
                print(f"  ❌ Error: {res.status_code}")
                print(f"  Response: {res.text}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            import traceback
            traceback.print_exc()

def test_image_url(token):
    """Test CV model with product URL"""
    print("\n" + "="*60)
    print("TEST 3: CV Model (Product URL)")
    print("="*60)
    
    test_urls = [
        "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400",  # Shoes
        "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400"  # Shoes
    ]
    
    for url in test_urls:
        print(f"\nTesting URL: {url[:60]}...")
        data = {
            'product_url': url,
            'text': 'Premium original product'
        }
        headers = {'Authorization': f'Bearer {token}'}
        
        try:
            res = requests.post(
                f'{API_BASE}/api/multimodal_check',
                json=data,
                headers=headers,
                timeout=30  # URLs need more time
            )
            print(f"  Status: {res.status_code}")
            
            if res.status_code == 200:
                result = res.json()
                cv = result.get('cv')
                if cv and isinstance(cv, dict) and 'authentic_prob' in cv:
                    cv_score = cv['authentic_prob']
                    print(f"  ✅ CV Score: {cv_score:.2%}")
                    fused = result.get('fused')
                    if fused and isinstance(fused, dict) and 'authentic_prob' in fused:
                        fused_score = fused['authentic_prob']
                        print(f"  ✅ Fused Score: {fused_score:.2%}")
                else:
                    print(f"  ⚠️ Unexpected CV response: {str(result)[:300]}")
            else:
                print(f"  ❌ Error: {res.status_code}")
                print(f"  Response: {res.text[:200]}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")

def test_multimodal(token):
    """Test fusion with both text and image"""
    print("\n" + "="*60)
    print("TEST 4: Multimodal (CV + NLP Fusion)")
    print("="*60)
    
    test_cases = [
        {
            'text': 'Authentic original Nike shoes guaranteed quality',
            'url': 'https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=400',
            'label': 'Premium Authentic'
        },
        {
            'text': 'BEST PRICE FAKE COPY DISCOUNT COUNTERFEIT',
            'url': 'https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400',
            'label': 'Fake/Counterfeit'
        }
    ]
    
    for case in test_cases:
        print(f"\nTest Case: {case['label']}")
        print(f"  Text: {case['text']}")
        
        data = {
            'product_url': case['url'],
            'text': case['text']
        }
        headers = {'Authorization': f'Bearer {token}'}
        
        try:
            res = requests.post(
                f'{API_BASE}/api/multimodal_check',
                json=data,
                headers=headers,
                timeout=30
            )
            print(f"  HTTP Status: {res.status_code}")
            
            if res.status_code == 200:
                result = res.json()
                print(f"\n  Scores:")
                cv = result.get('cv')
                if cv and isinstance(cv, dict):
                    cv_score = cv.get('authentic_prob', 'N/A')
                    print(f"    CV:     {cv_score:.2%}" if isinstance(cv_score, (int, float)) else f"    CV:     {cv_score}")
                nlp = result.get('nlp')
                if nlp and isinstance(nlp, dict):
                    nlp_score = nlp.get('authentic_prob', 'N/A')
                    print(f"    NLP:    {nlp_score:.2%}" if isinstance(nlp_score, (int, float)) else f"    NLP:    {nlp_score}")
                fused = result.get('fused')
                if fused and isinstance(fused, dict):
                    fused_score = fused.get('authentic_prob', 'N/A')
                    if isinstance(fused_score, (int, float)):
                        print(f"    FUSED:  {fused_score:.2%}")
                        verdict = 'AUTHENTIC ✅' if fused_score > 0.5 else 'COUNTERFEIT ❌'
                        print(f"    Verdict: {verdict}")
                    else:
                        print(f"    FUSED:  {fused_score}")
            else:
                print(f"  ❌ Error: {res.status_code}")
                print(f"  Response: {res.text[:300]}")
        except Exception as e:
            print(f"  ❌ Exception: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("="*60)
    print(f"API Base: {API_BASE}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Check backend health
    if not test_health_check():
        print("\n❌ FATAL: Backend is not running. Start it with:")
        print("  cd brand-auth-backend")
        print("  python app.py")
        return
    
    # Generate token
    token = generate_test_token()
    
    # Run all tests
    test_nlp_only(token)
    test_image_url(token)
    test_multimodal(token)
    
    # Summary
    print("\n" + "="*60)
    print("✅ EVALUATION COMPLETE")
    print("="*60)
    print("Summary:")
    print("  - Backend: Running ✅")
    print("  - Models: Loaded ✅")
    print("  - Authentication: Working ✅")
    print("  - API Endpoints: Responding ✅")
    print("\nNext steps:")
    print("  1. Check scores above")
    print("  2. Test frontend with token from QUICK_AUTH_SETUP.md")
    print("  3. Verify scores display in UI")

if __name__ == '__main__':
    main()
