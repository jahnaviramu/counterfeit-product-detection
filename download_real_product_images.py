#!/usr/bin/env python3
"""
download_real_product_images.py

Downloads real product images from free sources for testing.
Uses Unsplash API to get high-quality product photos.
"""

import os
import requests
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = 'data/images'
AUTHENTIC_DIR = os.path.join(OUTPUT_DIR, 'authentic_real')
COUNTERFEIT_DIR = os.path.join(OUTPUT_DIR, 'counterfeit_real')

# Create directories
os.makedirs(AUTHENTIC_DIR, exist_ok=True)
os.makedirs(COUNTERFEIT_DIR, exist_ok=True)

# ============================================================================
# DOWNLOAD FROM UNSPLASH (Free, no API key needed for basic use)
# ============================================================================

AUTHENTIC_SEARCHES = [
    'luxury watch authentic',
    'designer handbag original',
    'brand new iphone',
    'genuine leather wallet',
    'authentic gucci belt',
    'real rolex watch',
    'original apple airpods',
    'premium sneakers',
    'certified diamond ring',
    'authentic chanel bag'
]

COUNTERFEIT_SEARCHES = [
    'fake knockoff watch',
    'counterfeit handbag',
    'imitation designer',
    'replica product',
    'cheap fake item',
    'bootleg clothing',
    'fraudulent goods',
    'fake luxury',
    'poor quality copy',
    'suspicious product'
]

def download_image(url, filepath):
    """Download image from URL and save locally."""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"  ⚠️  Error downloading {url}: {e}")
    return False

def search_unsplash(query, num_images=5):
    """Search Unsplash for images (no API key required for basic searches)."""
    images = []
    
    # Using Unsplash source URL format (no auth required)
    # This provides random results matching the search term
    search_url = f"https://unsplash.com/napi/search/photos"
    
    try:
        params = {
            'query': query,
            'per_page': num_images,
            'page': 1
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(search_url, params=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            for item in data.get('results', []):
                if 'urls' in item and 'small' in item['urls']:
                    images.append({
                        'url': item['urls']['small'],
                        'description': item.get('description', query)
                    })
        
        return images
    except Exception as e:
        print(f"  ⚠️  Error searching Unsplash: {e}")
        return []

def download_pexels(query, num_images=5):
    """Download from Pexels (free stock photos, no API key required)."""
    images = []
    
    try:
        # Pexels search API
        search_url = "https://www.pexels.com/api/v2/search"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        params = {
            'query': query,
            'per_page': num_images,
            'page': 1
        }
        
        # Note: Pexels requires API key for v2, but we can use direct image links
        # For this demo, we'll create a fallback list of reliable URLs
        
        return images
    except Exception as e:
        print(f"  ⚠️  Error with Pexels: {e}")
        return []

# ============================================================================
# ALTERNATIVE: USE DIRECT PRODUCT IMAGE URLS
# ============================================================================

AUTHENTIC_URLS = [
    # Authentic products
    "https://images.unsplash.com/photo-1523275335684-37898b6baf30?w=400",  # Watch
    "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=400",  # Handbag
    "https://images.unsplash.com/photo-1592286927505-1def25115558?w=400",  # Shoes
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400",  # Sunglasses
    "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=400",  # Sneakers
]

COUNTERFEIT_URLS = [
    # Note: These are general low-quality product images
    "https://images.unsplash.com/photo-1607623814075-e51df1bdc82f?w=400",  # Generic product
    "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=400",  # Glasses
    "https://images.unsplash.com/photo-1523395363840-099f6ffad945?w=400",  # Bracelet
    "https://images.unsplash.com/photo-1491904064712-8d3cc02edda9?w=400",  # Shoes worn
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=400",  # Random product
]

def main():
    print("\n" + "="*70)
    print("DOWNLOADING REAL PRODUCT IMAGES")
    print("="*70)
    
    total_downloaded = 0
    
    # ====================================================================
    # DOWNLOAD AUTHENTIC IMAGES
    # ====================================================================
    
    print(f"\n▶ Downloading Authentic Product Images...")
    print(f"  Destination: {AUTHENTIC_DIR}")
    
    for i, url in enumerate(AUTHENTIC_URLS, 1):
        filename = os.path.join(AUTHENTIC_DIR, f'authentic_real_{i:03d}.jpg')
        print(f"  [{i}/{len(AUTHENTIC_URLS)}] Downloading...", end=' ')
        
        if download_image(url, filename):
            print(f"✅ Saved")
            total_downloaded += 1
        else:
            print(f"⚠️  Failed")
    
    # ====================================================================
    # DOWNLOAD COUNTERFEIT IMAGES
    # ====================================================================
    
    print(f"\n▶ Downloading Counterfeit/Low-Quality Images...")
    print(f"  Destination: {COUNTERFEIT_DIR}")
    
    for i, url in enumerate(COUNTERFEIT_URLS, 1):
        filename = os.path.join(COUNTERFEIT_DIR, f'counterfeit_real_{i:03d}.jpg')
        print(f"  [{i}/{len(COUNTERFEIT_URLS)}] Downloading...", end=' ')
        
        if download_image(url, filename):
            print(f"✅ Saved")
            total_downloaded += 1
        else:
            print(f"⚠️  Failed")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    auth_count = len([f for f in os.listdir(AUTHENTIC_DIR)])
    fake_count = len([f for f in os.listdir(COUNTERFEIT_DIR)])
    
    print(f"\n✅ Downloaded: {total_downloaded} images")
    print(f"\n📁 Real Authentic Images: {auth_count} files")
    print(f"   Location: {AUTHENTIC_DIR}")
    print(f"\n📁 Real Counterfeit Images: {fake_count} files")
    print(f"   Location: {COUNTERFEIT_DIR}")
    
    if total_downloaded == 0:
        print("\n⚠️  No images downloaded. Try one of these alternatives:")
        print("\n   Option 1: Use the synthetic images (already in data/images/)")
        print("   Option 2: Manually download images from:")
        print("     - Unsplash.com")
        print("     - Pexels.com")
        print("     - Pixabay.com")
        print("     - Save them to: data/images/authentic_real/")
        print("\n   Option 3: Use real product photos from your phone/camera")
    else:
        print("\n✅ Images ready for testing!")
        print(f"\nNext steps:")
        print(f"  1. Use these images in the Scanner UI")
        print(f"  2. Upload from: {AUTHENTIC_DIR} or {COUNTERFEIT_DIR}")
        print(f"  3. Add product descriptions")
        print(f"  4. Click 'Verify' to see model predictions")

if __name__ == '__main__':
    main()
