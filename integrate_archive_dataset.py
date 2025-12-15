#!/usr/bin/env python3
"""Integrate archive CSV dataset into backend text data safely.

This script will:
- copy `products_archive.csv` -> `products_combined.csv` (append and dedupe by asin)
- merge `reviews_archive.csv` into `reviews_combined.csv` (append and dedupe by reviewID)
- create backups of original target files before overwriting

Run from repository root or this script's directory.
"""
import csv
import os
from shutil import copy2

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
TEXT_DIR = os.path.join(BASE_DIR, 'data', 'text')


def backup(path):
    if os.path.exists(path):
        bak = path + '.bak'
        copy2(path, bak)
        print(f"Backup created: {bak}")


def merge_csv(src_path, dst_path, key_columns):
    """Append rows from src_path into dst_path while deduplicating by key_columns.

    key_columns: list of column names used to build the unique key
    """
    if not os.path.exists(src_path):
        print(f"Source not found: {src_path}")
        return

    # Read existing destination rows
    existing = []
    existing_keys = set()
    if os.path.exists(dst_path):
        with open(dst_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            dst_header = reader.fieldnames
            for r in reader:
                existing.append(r)
                key = tuple((r.get(c, '') or '').strip() for c in key_columns)
                existing_keys.add(key)
    else:
        dst_header = None

    # Read source
    with open(src_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        src_header = reader.fieldnames
        to_add = []
        for r in reader:
            key = tuple((r.get(c, '') or '').strip() for c in key_columns)
            if key not in existing_keys:
                to_add.append(r)
                existing_keys.add(key)

    if not to_add:
        print(f"No new rows to add from {src_path} -> {dst_path}")
        return

    # Determine header to write
    header = dst_header or src_header
    if header is None:
        print("Unable to determine CSV header; aborting")
        return

    # Backup existing destination
    if os.path.exists(dst_path):
        backup(dst_path)

    # Write combined file
    with open(dst_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for r in existing:
            writer.writerow({k: r.get(k, '') for k in header})
        for r in to_add:
            writer.writerow({k: r.get(k, '') for k in header})

    print(f"Merged {len(to_add)} rows from {src_path} into {dst_path}")


def main():
    products_src = os.path.join(TEXT_DIR, 'products_archive.csv')
    reviews_src = os.path.join(TEXT_DIR, 'reviews_archive.csv')

    products_dst = os.path.join(TEXT_DIR, 'products_combined.csv')
    reviews_dst = os.path.join(TEXT_DIR, 'reviews_combined.csv')

    print('Integrating archive dataset...')
    merge_csv(products_src, products_dst, key_columns=['asin'])
    merge_csv(reviews_src, reviews_dst, key_columns=['reviewID'])
    print('Integration complete. Review the combined files before using them for training.')


if __name__ == '__main__':
    main()
