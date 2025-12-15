#!/usr/bin/env python3
import requests, jwt, datetime, os, sys

# Config
IMAGE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'val_cv', 'authentic', 'authentic_0069.png')
URL = os.getenv('API_URL', 'http://127.0.0.1:5000') + '/api/multimodal_check'
JWT_SECRET = os.getenv('JWT_SECRET', 'supersecretkey')

# Build token
token = jwt.encode({'email':'tester@example.com','role':'merchant','exp': datetime.datetime.utcnow() + datetime.timedelta(hours=1)}, JWT_SECRET, algorithm='HS256')

print('Using image:', IMAGE_PATH)
print('Posting to', URL)

try:
    with open(IMAGE_PATH, 'rb') as f:
        files = {'image': ('test.png', f, 'image/png')}
        data = {'description': 'bad product', 'archive': '1'}
        headers = {'Authorization': f'Bearer {token}'}
        resp = requests.post(URL, files=files, data=data, headers=headers, timeout=60)
        print('Status:', resp.status_code)
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
except FileNotFoundError:
    print('Image not found at', IMAGE_PATH)
    sys.exit(2)
except requests.exceptions.RequestException as e:
    print('Request failed:', str(e))
    sys.exit(3)
