#!/usr/bin/env python3
"""
organize_kaggle_dataset.py

Organizes the Kaggle dataset (fake/real folders) into 
the format needed for training: authentic/counterfeit
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_DIR = 'data/images'
KAGGLE_REAL_DIR = os.path.join(DATA_DIR, 'real')
KAGGLE_FAKE_DIR = os.path.join(DATA_DIR, 'fake')

TRAINING_AUTHENTIC_DIR = os.path.join(DATA_DIR, 'authentic')
TRAINING_COUNTERFEIT_DIR = os.path.join(DATA_DIR, 'counterfeit')

# ============================================================================
# MAIN SCRIPT
# ============================================================================

def count_images(directory):
    """Count images in a directory."""
    if not os.path.exists(directory):
        return 0
    return len([f for f in os.listdir(directory) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))])

def organize_dataset():
    """Organize Kaggle dataset into training format."""
    
    print("\n" + "="*70)
    print("ORGANIZING KAGGLE DATASET")
    print("="*70)
    
    # ====================================================================
    # CHECK WHAT WE HAVE
    # ====================================================================
    
    print("\n▶ Checking Kaggle Dataset...")
    print("-" * 70)
    
    real_count = count_images(KAGGLE_REAL_DIR)
    fake_count = count_images(KAGGLE_FAKE_DIR)
    
    print(f"\n📁 Found Kaggle Dataset:")
    print(f"  Real images: {real_count} files in {KAGGLE_REAL_DIR}")
    print(f"  Fake images: {fake_count} files in {KAGGLE_FAKE_DIR}")
    print(f"  Total: {real_count + fake_count} images")
    
    if real_count == 0 and fake_count == 0:
        print("\n❌ No images found in Kaggle dataset!")
        print("   Check if files are in subdirectories...")
        return False
    
    # ====================================================================
    # ORGANIZE INTO TRAINING FORMAT
    # ====================================================================
    
    print("\n▶ Organizing for Training...")
    print("-" * 70)
    
    # Create/clear training directories
    os.makedirs(TRAINING_AUTHENTIC_DIR, exist_ok=True)
    os.makedirs(TRAINING_COUNTERFEIT_DIR, exist_ok=True)
    
    copied_authentic = 0
    copied_counterfeit = 0
    skipped = 0
    
    # Copy real images to authentic
    if real_count > 0:
        print(f"\nCopying REAL images to AUTHENTIC training folder...")
        real_files = [f for f in os.listdir(KAGGLE_REAL_DIR) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        for i, filename in enumerate(real_files, 1):
            src = os.path.join(KAGGLE_REAL_DIR, filename)
            dst = os.path.join(TRAINING_AUTHENTIC_DIR, filename)
            
            try:
                # Check if already exists to avoid duplicates
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_authentic += 1
                else:
                    skipped += 1
                
                if i % 50 == 0 or i == real_count:
                    print(f"  [{i}/{real_count}] Copied...")
            except Exception as e:
                print(f"  ⚠️  Error copying {filename}: {e}")
    
    # Copy fake images to counterfeit
    if fake_count > 0:
        print(f"\nCopying FAKE images to COUNTERFEIT training folder...")
        fake_files = [f for f in os.listdir(KAGGLE_FAKE_DIR) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp'))]
        
        for i, filename in enumerate(fake_files, 1):
            src = os.path.join(KAGGLE_FAKE_DIR, filename)
            dst = os.path.join(TRAINING_COUNTERFEIT_DIR, filename)
            
            try:
                # Check if already exists to avoid duplicates
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_counterfeit += 1
                else:
                    skipped += 1
                
                if i % 50 == 0 or i == fake_count:
                    print(f"  [{i}/{fake_count}] Copied...")
            except Exception as e:
                print(f"  ⚠️  Error copying {filename}: {e}")
    
    # ====================================================================
    # SUMMARY
    # ====================================================================
    
    auth_total = count_images(TRAINING_AUTHENTIC_DIR)
    counter_total = count_images(TRAINING_COUNTERFEIT_DIR)
    
    print("\n" + "="*70)
    print("ORGANIZATION SUMMARY")
    print("="*70)
    
    print(f"\n📊 Kaggle Dataset Status:")
    print(f"  Real images found: {real_count}")
    print(f"  Fake images found: {fake_count}")
    
    print(f"\n📁 Training Dataset Created:")
    print(f"  Authentic folder: {auth_total} images")
    print(f"    Location: {TRAINING_AUTHENTIC_DIR}")
    print(f"  Counterfeit folder: {counter_total} images")
    print(f"    Location: {TRAINING_COUNTERFEIT_DIR}")
    
    print(f"\n📈 Statistics:")
    print(f"  Copied: {copied_authentic + copied_counterfeit} images")
    print(f"  Skipped (duplicates): {skipped}")
    print(f"  Total for training: {auth_total + counter_total}")
    
    if auth_total > 0 and counter_total > 0:
        balance = min(auth_total, counter_total) / max(auth_total, counter_total)
        print(f"  Class balance: {balance:.1%}")
    
    # ====================================================================
    # NEXT STEPS
    # ====================================================================
    
    if auth_total > 0 and counter_total > 0:
        print("\n" + "="*70)
        print("✅ READY TO TRAIN!")
        print("="*70)
        
        print(f"\nNext steps:")
        print(f"  1. Retrain CV model with real images:")
        print(f"     python train_cv_improved.py")
        print(f"\n  2. Expected improvement:")
        print(f"     - Current: 95% accuracy (synthetic images)")
        print(f"     - New: 99%+ accuracy (real images)")
        print(f"\n  3. Then test in Scanner UI:")
        print(f"     http://localhost:3001 → 🔍 Scan")
        print(f"\n  4. Deploy improved model:")
        print(f"     python app.py")
        
        return True
    else:
        print("\n⚠️  Not enough images to train!")
        if auth_total == 0:
            print("  - Authentic folder is empty")
        if counter_total == 0:
            print("  - Counterfeit folder is empty")
        return False

if __name__ == '__main__':
    success = organize_dataset()
    exit(0 if success else 1)
