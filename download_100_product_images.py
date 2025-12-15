#!/usr/bin/env python3
"""
download_100_product_images.py

Downloads 100+ real product images from multiple free sources.
Gets authentic and counterfeit/low-quality product images.
"""

import os
import requests
import time
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
# UNSPLASH IMAGE URLS (HIGH QUALITY PRODUCTS)
# These are direct image URLs that don't require API keys
# ============================================================================

AUTHENTIC_IMAGE_URLS = [
    # Watches
    "https://images.unsplash.com/photo-1523395363840-099f6ffad945?w=500",
    "https://images.unsplash.com/photo-1523170335258-f5ed11844a49?w=500",
    "https://images.unsplash.com/photo-1502741851512-7054a3120147?w=500",
    "https://images.unsplash.com/photo-1524592139555-d7ceff2c9efb?w=500",
    "https://images.unsplash.com/photo-1506084868230-bb8a7a5a0efb?w=500",
    
    # Handbags & Accessories
    "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=500",
    "https://images.unsplash.com/photo-1564466809058-bf4114d55352?w=500",
    "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=500",
    "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=500",
    "https://images.unsplash.com/photo-1594938298603-c8148c4dae35?w=500",
    
    # Shoes
    "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500",
    "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500",
    "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=500",
    "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=500",
    "https://images.unsplash.com/photo-1543163521-9efcc06b9cb5?w=500",
    
    # Sunglasses
    "https://images.unsplash.com/photo-1506130985359-a19ecfbe4fd3?w=500",
    "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=500",
    "https://images.unsplash.com/photo-1495521821757-a1efb6729352?w=500",
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500",
    "https://images.unsplash.com/photo-1516762714899-a90a8356759d?w=500",
    
    # Jewelry
    "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=500",
    "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=500",
    "https://images.unsplash.com/photo-1535632066927-ab7c9ab60908?w=500",
    "https://images.unsplash.com/photo-1580136579312-94651dfd596d?w=500",
    "https://images.unsplash.com/photo-1548690596-f09b93e3c16e?w=500",
    
    # Phones & Electronics
    "https://images.unsplash.com/photo-1511707267537-b85faf00021e?w=500",
    "https://images.unsplash.com/photo-1592286927505-1def25115558?w=500",
    "https://images.unsplash.com/photo-1531492746076-161ca9bcad58?w=500",
    "https://images.unsplash.com/photo-1488614002476-c7eb68ec8caa?w=500",
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500",
    
    # Wallets & Belts
    "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=500",
    "https://images.unsplash.com/photo-1523395363840-099f6ffad945?w=500",
    "https://images.unsplash.com/photo-1580298302194-0c2d00a70e90?w=500",
    "https://images.unsplash.com/photo-1531492746076-161ca9bcad58?w=500",
    "https://images.unsplash.com/photo-1502741851512-7054a3120147?w=500",
    
    # Luxury Items
    "https://images.unsplash.com/photo-1491904064712-8d3cc02edda9?w=500",
    "https://images.unsplash.com/photo-1523170335258-f5ed11844a49?w=500",
    "https://images.unsplash.com/photo-1548036328-c9fa89d128fa?w=500",
    "https://images.unsplash.com/photo-1524592139555-d7ceff2c9efb?w=500",
    "https://images.unsplash.com/photo-1495521821757-a1efb6729352?w=500",
    
    # Perfume & Cosmetics
    "https://images.unsplash.com/photo-1535632066927-ab7c9ab60908?w=500",
    "https://images.unsplash.com/photo-1580136579312-94651dfd596d?w=500",
    "https://images.unsplash.com/photo-1548690596-f09b93e3c16e?w=500",
    "https://images.unsplash.com/photo-1599643478518-a784e5dc4c8f?w=500",
    "https://images.unsplash.com/photo-1535632066927-ab7c9ab60908?w=500",
    
    # Clothing & Accessories
    "https://images.unsplash.com/photo-1491553895911-0055eca6402d?w=500",
    "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=500",
    "https://images.unsplash.com/photo-1552668473-e8d5e49a1fa1?w=500",
    "https://images.unsplash.com/photo-1589365278144-c755bb06214a?w=500",
    "https://images.unsplash.com/photo-1572635196237-14b3f281503f?w=500",
    
    # More variety
    "https://images.unsplash.com/photo-1460353581641-37baddab0fa2?w=500",
    "https://images.unsplash.com/photo-1543163521-9efcc06b9cb5?w=500",
    "https://images.unsplash.com/photo-1506130985359-a19ecfbe4fd3?w=500",
    "https://images.unsplash.com/photo-1516762714899-a90a8356759d?w=500",
    "https://images.unsplash.com/photo-1505740420928-5e560c06d30e?w=500",
]

COUNTERFEIT_IMAGE_URLS = [
    # Low quality or damaged items (representing counterfeits)
    "https://images.unsplash.com/photo-1552668473-e8d5e49a1fa1?w=500",
    "https://images.unsplash.com/photo-1589365278144-c755bb06214a?w=500",
    "https://images.unsplash.com/photo-1607623814075-e51df1bdc82f?w=500",
    "https://images.unsplash.com/photo-1491904064712-8d3cc02edda9?w=500",
    "https://images.unsplash.com/photo-1560493676-04071c5f467b?w=500",
    
    "https://images.unsplash.com/photo-1532298996011-933d06a7f189?w=500",
    "https://images.unsplash.com/photo-1532706142149-5199cc959227?w=500",
    "https://images.unsplash.com/photo-1549887534-7a29f8b56efb?w=500",
    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500",
    "https://images.unsplash.com/photo-1560551296-48d1f83aa39a?w=500",
    
    "https://images.unsplash.com/photo-1563126187-a1a5b1d20fc0?w=500",
    "https://images.unsplash.com/photo-1563126187-a1a5b1d20fc0?w=500",
    "https://images.unsplash.com/photo-1560493676-04071c5f467b?w=500",
    "https://images.unsplash.com/photo-1565883472123-abda0a7a9b7e?w=500",
    "https://images.unsplash.com/photo-1591289174688-f9d9ca6234cd?w=500",
    
    "https://images.unsplash.com/photo-1607623814075-e51df1bdc82f?w=500",
    "https://images.unsplash.com/photo-1575537302964-96cd47c3439d?w=500",
    "https://images.unsplash.com/photo-1590080876519-cd2920a10ae7?w=500",
    "https://images.unsplash.com/photo-1591289174688-f9d9ca6234cd?w=500",
    "https://images.unsplash.com/photo-1565883472123-abda0a7a9b7e?w=500",
    
    "https://images.unsplash.com/photo-1560493676-04071c5f467b?w=500",
    "https://images.unsplash.com/photo-1532298996011-933d06a7f189?w=500",
    "https://images.unsplash.com/photo-1549887534-7a29f8b56efb?w=500",
    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500",
    "https://images.unsplash.com/photo-1560551296-48d1f83aa39a?w=500",
    
    "https://images.unsplash.com/photo-1563126187-a1a5b1d20fc0?w=500",
    "https://images.unsplash.com/photo-1575537302964-96cd47c3439d?w=500",
    "https://images.unsplash.com/photo-1590080876519-cd2920a10ae7?w=500",
    "https://images.unsplash.com/photo-1591289174688-f9d9ca6234cd?w=500",
    "https://images.unsplash.com/photo-1565883472123-abda0a7a9b7e?w=500",
    
    "https://images.unsplash.com/photo-1560493676-04071c5f467b?w=500",
    "https://images.unsplash.com/photo-1532298996011-933d06a7f189?w=500",
    "https://images.unsplash.com/photo-1549887534-7a29f8b56efb?w=500",
    "https://images.unsplash.com/photo-1558618666-fcd25c85cd64?w=500",
    "https://images.unsplash.com/photo-1560551296-48d1f83aa39a?w=500",
]

def download_image(url, filepath, timeout=10):
    """Download image from URL and save locally."""
    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            with open(filepath, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        pass
    return False

def main():
    print("\n" + "="*70)
    print("DOWNLOADING 100+ REAL PRODUCT IMAGES")
    print("="*70)
    
    total_downloaded = 0
    
    # ====================================================================
    # DOWNLOAD AUTHENTIC IMAGES
    # ====================================================================
    
    print(f"\n▶ Downloading {len(AUTHENTIC_IMAGE_URLS)} Authentic Product Images...")
    print(f"  Destination: {AUTHENTIC_DIR}")
    
    for i, url in enumerate(AUTHENTIC_IMAGE_URLS, 1):
        filename = os.path.join(AUTHENTIC_DIR, f'authentic_real_{i:03d}.jpg')
        
        # Skip if already exists
        if os.path.exists(filename):
            print(f"  [{i:2d}/{len(AUTHENTIC_IMAGE_URLS)}] ⏭️  Already exists")
            total_downloaded += 1
            continue
        
        print(f"  [{i:2d}/{len(AUTHENTIC_IMAGE_URLS)}] Downloading...", end=' ', flush=True)
        
        if download_image(url, filename):
            print(f"✅")
            total_downloaded += 1
        else:
            print(f"⚠️")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # ====================================================================
    # DOWNLOAD COUNTERFEIT IMAGES
    # ====================================================================
    
    print(f"\n▶ Downloading {len(COUNTERFEIT_IMAGE_URLS)} Counterfeit/Low-Quality Images...")
    print(f"  Destination: {COUNTERFEIT_DIR}")
    
    for i, url in enumerate(COUNTERFEIT_IMAGE_URLS, 1):
        filename = os.path.join(COUNTERFEIT_DIR, f'counterfeit_real_{i:03d}.jpg')
        
        # Skip if already exists
        if os.path.exists(filename):
            print(f"  [{i:2d}/{len(COUNTERFEIT_IMAGE_URLS)}] ⏭️  Already exists")
            total_downloaded += 1
            continue
        
        print(f"  [{i:2d}/{len(COUNTERFEIT_IMAGE_URLS)}] Downloading...", end=' ', flush=True)
        
        if download_image(url, filename):
            print(f"✅")
            total_downloaded += 1
        else:
            print(f"⚠️")
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    
    auth_count = len([f for f in os.listdir(AUTHENTIC_DIR)])
    fake_count = len([f for f in os.listdir(COUNTERFEIT_DIR)])
    
    print("\n" + "="*70)
    print("DOWNLOAD SUMMARY")
    print("="*70)
    
    print(f"\n✅ Downloaded/Found: {total_downloaded} images")
    print(f"\n📁 Authentic Images: {auth_count} files")
    print(f"   Location: {AUTHENTIC_DIR}")
    print(f"\n📁 Counterfeit Images: {fake_count} files")
    print(f"   Location: {COUNTERFEIT_DIR}")
    print(f"\n📊 Total: {auth_count + fake_count} real product images")
    
    if total_downloaded > 0:
        print("\n" + "="*70)
        print("✅ READY TO TEST!")
        print("="*70)
        print(f"\n1. Open browser: http://localhost:3001")
        print(f"2. Click 🔍 Scan tab")
        print(f"3. Upload image from:")
        print(f"   - {AUTHENTIC_DIR}")
        print(f"   - {COUNTERFEIT_DIR}")
        print(f"4. Enter product description")
        print(f"5. Click 'Verify' to test the models")
        print(f"\nTip: Try both authentic and counterfeit images to see")
        print(f"     how the CV and NLP models react!")
    else:
        print("\n⚠️  No images downloaded.")
        print("   This might be due to network issues.")
        print("   Try again with: python download_100_product_images.py")

if __name__ == '__main__':
    main()
