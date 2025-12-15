#!/usr/bin/env python3
"""
Generate a valid JWT token for testing multimodal_check endpoint.
Token matches backend secret and includes required role.
"""

import jwt
from datetime import datetime, timedelta
import sys

# Must match backend's os.getenv('JWT_SECRET', 'supersecretkey')
JWT_SECRET = 'supersecretkey'

# Token payload with required fields
payload = {
    'merchant_id': 'test_merchant_12345',
    'role': 'merchant',  # Must be one of: merchant, seller, influencer, admin
    'exp': datetime.utcnow() + timedelta(hours=24),  # Token valid for 24 hours
    'iat': datetime.utcnow(),
    'email': 'test@example.com'
}

try:
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    print("\n" + "="*70)
    print("VALID JWT TOKEN FOR TESTING")
    print("="*70)
    print(f"\nToken:\n{token}")
    print("\n" + "="*70)
    print("HOW TO USE:")
    print("="*70)
    print("\n1. Open your browser and go to the app (http://192.168.x.x:3000)")
    print("2. Open browser DevTools (Press F12)")
    print("3. Go to Console tab")
    print("4. Paste this command:")
    print(f"\n   localStorage.setItem('authToken', '{token}');")
    print("\n5. Press Enter")
    print("6. Reload the page (Ctrl+R)")
    print("7. Try verifying a product — token should work now!")
    print("\n" + "="*70)
    print("TOKEN DETAILS:")
    print("="*70)
    print(f"Role: {payload['role']}")
    print(f"Expires: {payload['exp']}")
    print(f"Secret: {JWT_SECRET}")
    print("="*70 + "\n")
except Exception as e:
    print(f"Error generating token: {e}")
    print("Make sure PyJWT is installed: pip install PyJWT")
    sys.exit(1)
