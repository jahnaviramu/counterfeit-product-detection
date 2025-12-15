#!/usr/bin/env python3
"""
Download product images from the frontend archive CSV and save them into
`data/images/authentic` so the CV trainer can use archive images as the
`authentic` class. Skips existing files and limits downloads to a configurable
number per product and overall.
"""
import os
import csv
import ast
import re
import requests
from pathlib import Path

ARCHIVE_PRODUCTS_CSV = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'brand-auth-app', 'archive (1)', 'products.csv')
)

OUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'images', 'authentic'))
os.makedirs(OUT_DIR, exist_ok=True)

MAX_PER_PRODUCT = 1
MAX_TOTAL = None  # set to int to limit total downloads

URL_RE = re.compile(r'https?://[^\]"\']+')

def parse_all_images(field_value):
    """Try to parse the `all_images` field which may be a literal list or
    a string containing URLs."""
    if not field_value or not field_value.strip():
        return []
    # Try literal_eval first
    try:
        val = ast.literal_eval(field_value)
        if isinstance(val, (list, tuple)):
            return [str(x) for x in val if x]
    except Exception:
        pass
    # Fallback: regex find URLs
    return URL_RE.findall(field_value)


def download(url, dest_path, timeout=10):
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in resp.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False


def build_from_archive(max_per_product=MAX_PER_PRODUCT, max_total=MAX_TOTAL):
    print('\n▶ Building CV dataset from archive images')
    print(f'  Archive CSV: {ARCHIVE_PRODUCTS_CSV}')
    print(f'  Output dir: {OUT_DIR}')

    if not os.path.exists(ARCHIVE_PRODUCTS_CSV):
        print('❌ Archive products CSV not found. Aborting.')
        return False

    total_saved = 0
    with open(ARCHIVE_PRODUCTS_CSV, newline='', encoding='utf-8', errors='ignore') as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            all_images_field = row.get('all_images') or row.get('images') or ''
            urls = parse_all_images(all_images_field)
            if not urls:
                continue
            saved_for_product = 0
            for i, url in enumerate(urls):
                if max_per_product is not None and saved_for_product >= max_per_product:
                    break
                if max_total is not None and total_saved >= max_total:
                    break

                # sanitize extension
                ext = os.path.splitext(url.split('?')[0])[1].lower()
                if ext not in ('.jpg', '.jpeg', '.png', '.webp'):
                    ext = '.jpg'

                fname = f"archive_{idx:04d}_{i}{ext}"
                dest = os.path.join(OUT_DIR, fname)
                if os.path.exists(dest):
                    saved_for_product += 1
                    total_saved += 1
                    continue

                ok = download(url, dest)
                if ok:
                    saved_for_product += 1
                    total_saved += 1

            if max_total is not None and total_saved >= max_total:
                break

    print(f"\n✅ Done. Total images saved: {total_saved}")
    if total_saved == 0:
        print('⚠️  No images were downloaded. Check CSV `all_images` column or network access.')
    else:
        print('Next: run `python train_cv_improved.py` to train using these archive images as `authentic`.')

    return True


if __name__ == '__main__':
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument('--per-product', type=int, default=MAX_PER_PRODUCT, help='Max images to download per product')
    p.add_argument('--max-total', type=int, default=MAX_TOTAL, help='Max total images to download')
    args = p.parse_args()
    build_from_archive(max_per_product=args.per_product, max_total=args.max_total)
