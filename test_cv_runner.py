"""test_cv_runner.py
Batch-run images through the backend /api/cv_check endpoint and write a CSV report.

Usage:
  python test_cv_runner.py --server http://localhost:5000 --input_dir data/images/debug_test --out results.csv

The script sends multipart POST requests with field 'image' to /api/cv_check and logs the JSON responses.
"""
import os
import argparse
import requests
import csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', default='http://localhost:5000')
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--out', default='cv_test_results.csv')
    args = parser.parse_args()

    files = [os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    rows = []
    for path in files:
        try:
            with open(path, 'rb') as fh:
                resp = requests.post(args.server.rstrip('/') + '/api/cv_check', files={'image': fh})
            j = resp.json()
        except Exception as e:
            j = {'error': str(e)}
        rows.append({'image': path, 'response': j})

    # write CSV with basic columns
    with open(args.out, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['image', 'authentic', 'confidence', 'error'])
        for r in rows:
            resp = r['response']
            if isinstance(resp, dict) and 'error' in resp:
                writer.writerow([r['image'], '', '', resp.get('error')])
            else:
                writer.writerow([r['image'], resp.get('authentic'), resp.get('confidence'), ''])

    print('Wrote results to', args.out)


if __name__ == '__main__':
    main()
