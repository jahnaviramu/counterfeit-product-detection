#!/usr/bin/env python3
"""
prepare_datasets.py

Prepares synthetic datasets for CV and NLP model training.
Creates authentic and counterfeit product images and text data.
"""

import os
import csv
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import json
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

DATASET_DIR = 'data'
IMAGES_DIR = os.path.join(DATASET_DIR, 'images')
TEXT_DIR = os.path.join(DATASET_DIR, 'text')

# Image dataset config
NUM_AUTHENTIC_IMAGES = 200
NUM_COUNTERFEIT_IMAGES = 200
IMAGE_SIZE = (224, 224)

# Text dataset config
NUM_AUTHENTIC_TEXTS = 300
NUM_COUNTERFEIT_TEXTS = 300

# ============================================================================
# SAMPLE DATA
# ============================================================================

AUTHENTIC_DESCRIPTIONS = [
    "Premium authentic designer handbag with genuine leather.",
    "Original brand name watch with Swiss movement.",
    "Genuine product with official certification and warranty.",
    "Authentic luxury item with hologram and serial number verification.",
    "Real branded perfume with original bottle and cap design.",
    "Legitimate product from authorized dealer.",
    "Certified authentic with quality assurance stamp.",
    "Original merchandise with authentic packaging and labeling.",
    "Genuine high-end product with premium materials.",
    "Real brand item with proper documentation.",
    "Authentic designer shoes with genuine materials and craftsmanship.",
    "Official product with manufacturer seal and authenticity code.",
    "Legitimate branded item with warranty card included.",
    "Genuine luxury goods with certification of authenticity.",
    "Real product verified by brand representative.",
]

COUNTERFEIT_DESCRIPTIONS = [
    "Super cheap designer bag, looks like real thing.",
    "Best quality fake watches at unbeatable price.",
    "Almost identical to original, nobody can tell the difference.",
    "Replica luxury items, perfect copy of expensive brands.",
    "Knockoff perfume that smells the same as original.",
    "Imitation branded product for sale, very affordable.",
    "Copy of famous brand, same quality, half the price.",
    "Counterfeit shoes that look exactly like designer ones.",
    "Fake item that looks real, great deal.",
    "Unauthorized reproduction of brand name goods.",
    "Imitation product, similar appearance to original.",
    "Replicated item, not official but looks good.",
    "Duplicate design, unofficial but quality made.",
    "Knockoff version of expensive product.",
    "Fake brand merchandise, very convincing copy.",
]

AUTHENTIC_KEYWORDS = [
    "authentic", "genuine", "original", "certified", "official",
    "authorized", "warranty", "premium", "quality", "legitimate",
    "verified", "real", "authentic brand", "original item", "genuine product"
]

COUNTERFEIT_KEYWORDS = [
    "replica", "fake", "copy", "knockoff", "imitation",
    "counterfeit", "cheap", "affordable fake", "unbeatable price",
    "looks like", "almost identical", "cannot tell difference",
    "unauthorized", "unauthorized reproduction", "duplicate", "not official"
]

# ============================================================================
# IMAGE GENERATION
# ============================================================================

def generate_authentic_image():
    """Generate a synthetic authentic product image."""
    img = Image.new('RGB', IMAGE_SIZE, color=(random.randint(200, 255), 
                                                 random.randint(200, 255), 
                                                 random.randint(200, 255)))
    draw = ImageDraw.Draw(img)
    
    # Draw product shape (rectangle/circle)
    if random.random() > 0.5:
        # Rectangle product
        x1, y1 = random.randint(20, 50), random.randint(20, 50)
        x2, y2 = random.randint(174, 204), random.randint(174, 204)
        draw.rectangle([x1, y1, x2, y2], fill=(random.randint(50, 150),
                                                 random.randint(50, 150),
                                                 random.randint(50, 150)))
    else:
        # Circle product
        center_x, center_y = IMAGE_SIZE[0]//2, IMAGE_SIZE[1]//2
        radius = random.randint(40, 80)
        draw.ellipse([center_x-radius, center_y-radius, 
                      center_x+radius, center_y+radius],
                     fill=(random.randint(50, 150),
                           random.randint(50, 150),
                           random.randint(50, 150)))
    
    # Add quality pattern (genuine products have better quality)
    for _ in range(5):
        x, y = random.randint(30, 194), random.randint(30, 194)
        draw.rectangle([x, y, x+10, y+10], outline=(255, 215, 0), width=2)
    
    # Add hologram-like element
    for i in range(10):
        y = 20 + i * 20
        draw.line([(50, y), (150, y)], fill=(200, 200, 255), width=1)
    
    return img

def generate_counterfeit_image():
    """Generate a synthetic counterfeit product image."""
    img = Image.new('RGB', IMAGE_SIZE, color=(random.randint(100, 200),
                                                 random.randint(100, 200),
                                                 random.randint(100, 200)))
    draw = ImageDraw.Draw(img)
    
    # Draw product shape (poorly made)
    if random.random() > 0.5:
        # Poor rectangle
        x1, y1 = random.randint(10, 40), random.randint(10, 40)
        x2, y2 = random.randint(184, 214), random.randint(184, 214)
        draw.rectangle([x1, y1, x2, y2], fill=(random.randint(100, 200),
                                                 random.randint(100, 200),
                                                 random.randint(100, 200)),
                       outline=(50, 50, 50), width=1)
    else:
        # Irregular circle
        center_x, center_y = IMAGE_SIZE[0]//2 + random.randint(-10, 10), IMAGE_SIZE[1]//2 + random.randint(-10, 10)
        radius = random.randint(50, 70)
        draw.ellipse([center_x-radius, center_y-radius,
                      center_x+radius, center_y+radius],
                     fill=(random.randint(100, 200),
                           random.randint(100, 200),
                           random.randint(100, 200)),
                     outline=(50, 50, 50), width=2)
    
    # Add blurry/poor quality artifacts
    for _ in range(15):
        x, y = random.randint(20, 204), random.randint(20, 204)
        draw.point((x, y), fill=(50, 50, 50))
    
    # Add noise pattern (sign of poor quality)
    for _ in range(30):
        x, y = random.randint(0, 224), random.randint(0, 224)
        draw.point((x, y), fill=(random.randint(0, 100), 
                                 random.randint(0, 100),
                                 random.randint(0, 100)))
    
    return img

# ============================================================================
# TEXT GENERATION
# ============================================================================

def generate_authentic_text():
    """Generate authentic product review text."""
    template = random.choice([
        "This is a {} original product. Verified authentic with {}.",
        "Genuine {} item with {}. Highly recommended.",
        "Certified {} product. Quality is {}.",
        "{} brand item, {} materials. Very satisfied.",
        "Official {} from {}. Excellent condition."
    ])
    
    quality = random.choice(["excellent", "premium", "superior", "certified"])
    feature = random.choice(["hologram verification", "serial number", "warranty card", "original packaging"])
    
    description = random.choice(AUTHENTIC_DESCRIPTIONS)
    
    return template.format(quality, feature) + " " + description

def generate_counterfeit_text():
    """Generate counterfeit/suspicious product review text."""
    template = random.choice([
        "This {} is {} but looks {} the real thing.",
        "Found this {} {} online, {} the original.",
        "Best {} deal! Looks {} but much cheaper.",
        "This is a {} product. {} but quality is good.",
        "{} that is {} {}. Great price!"
    ])
    
    adj1 = random.choice(["replica", "copy", "knockoff", "imitation"])
    adj2 = random.choice(["cheap", "affordable", "discounted", "inexpensive"])
    phrase = random.choice(["similar to", "looks like", "almost identical to", "very similar to"])
    
    description = random.choice(COUNTERFEIT_DESCRIPTIONS)
    
    return description

# ============================================================================
# DATASET CREATION
# ============================================================================

def create_image_dataset():
    """Create synthetic image dataset."""
    print("\n" + "="*60)
    print("CREATING IMAGE DATASET")
    print("="*60)
    
    # Create directories
    auth_dir = os.path.join(IMAGES_DIR, 'authentic')
    fake_dir = os.path.join(IMAGES_DIR, 'counterfeit')
    
    os.makedirs(auth_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Generate authentic images
    print(f"\nGenerating {NUM_AUTHENTIC_IMAGES} authentic images...")
    for i in range(NUM_AUTHENTIC_IMAGES):
        img = generate_authentic_image()
        img.save(os.path.join(auth_dir, f'authentic_{i:04d}.png'))
        if (i + 1) % 50 == 0:
            print(f"  ✓ Created {i + 1}/{NUM_AUTHENTIC_IMAGES}")
    
    # Generate counterfeit images
    print(f"\nGenerating {NUM_COUNTERFEIT_IMAGES} counterfeit images...")
    for i in range(NUM_COUNTERFEIT_IMAGES):
        img = generate_counterfeit_image()
        img.save(os.path.join(fake_dir, f'counterfeit_{i:04d}.png'))
        if (i + 1) % 50 == 0:
            print(f"  ✓ Created {i + 1}/{NUM_COUNTERFEIT_IMAGES}")
    
    print(f"\n✅ Image dataset created!")
    print(f"   Authentic: {auth_dir}")
    print(f"   Counterfeit: {fake_dir}")

def create_text_dataset():
    """Create text dataset CSV for NLP training."""
    print("\n" + "="*60)
    print("CREATING TEXT DATASET")
    print("="*60)
    
    os.makedirs(TEXT_DIR, exist_ok=True)
    
    csv_path = os.path.join(TEXT_DIR, 'reviews.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])  # Header
        
        # Authentic reviews (label=1)
        print(f"\nGenerating {NUM_AUTHENTIC_TEXTS} authentic reviews...")
        for i in range(NUM_AUTHENTIC_TEXTS):
            text = generate_authentic_text()
            writer.writerow([text, 1])  # 1 = authentic
            if (i + 1) % 100 == 0:
                print(f"  ✓ Created {i + 1}/{NUM_AUTHENTIC_TEXTS}")
        
        # Counterfeit reviews (label=0)
        print(f"\nGenerating {NUM_COUNTERFEIT_TEXTS} counterfeit reviews...")
        for i in range(NUM_COUNTERFEIT_TEXTS):
            text = generate_counterfeit_text()
            writer.writerow([text, 0])  # 0 = counterfeit/suspicious
            if (i + 1) % 100 == 0:
                print(f"  ✓ Created {i + 1}/{NUM_COUNTERFEIT_TEXTS}")
    
    print(f"\n✅ Text dataset created!")
    print(f"   Path: {csv_path}")
    print(f"   Total samples: {NUM_AUTHENTIC_TEXTS + NUM_COUNTERFEIT_TEXTS}")
    print(f"   Authentic: {NUM_AUTHENTIC_TEXTS}, Counterfeit: {NUM_COUNTERFEIT_TEXTS}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*60)
    print("DATA PREPARATION TOOL")
    print("="*60)
    print("\nThis script will generate synthetic datasets for:")
    print("  1. CV Model Training (Product Images)")
    print("  2. NLP Model Training (Product Reviews)")
    print("\nConfiguration:")
    print(f"  - Authentic Images: {NUM_AUTHENTIC_IMAGES}")
    print(f"  - Counterfeit Images: {NUM_COUNTERFEIT_IMAGES}")
    print(f"  - Authentic Reviews: {NUM_AUTHENTIC_TEXTS}")
    print(f"  - Counterfeit Reviews: {NUM_COUNTERFEIT_TEXTS}")
    
    try:
        create_image_dataset()
        create_text_dataset()
        
        print("\n" + "="*60)
        print("✅ DATASET PREPARATION COMPLETE!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Train CV model:  python train_cv_model.py")
        print("  2. Train NLP model: python train_sbert_finetune.py")
        print("  3. Test endpoint:   curl -X POST http://localhost:5000/api/multimodal_check")
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n❌ Error during dataset preparation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
