"""
Database Schema Setup for Instagram Seller Authenticity System

Creates MongoDB collections with proper indexes for:
- checks: Authentication check history
- profiles: Cached Instagram profile data
- brands: Trusted brand references
- images_reference: Reference images for similarity matching
"""

import logging
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

def init_authenticity_schema(db):
    """
    Initialize authentication check collections.
    
    Args:
        db: MongoDB database instance (e.g., db from pymongo)
    """
    
    # === CHECKS COLLECTION ===
    # Stores all authenticity checks performed by users
    if 'checks' not in db.list_collection_names():
        db.create_collection('checks')
        logger.info("Created 'checks' collection")
    
    checks = db['checks']
    
    # Index for fast lookup by user/timestamp
    checks.create_index([('user_id', ASCENDING), ('created_at', DESCENDING)])
    checks.create_index([('handle', ASCENDING)])
    checks.create_index([('verdict', ASCENDING)])
    checks.create_index([('score', DESCENDING)])
    checks.create_index([('created_at', DESCENDING)])
    
    # === PROFILES COLLECTION ===
    # Caches Instagram profile metadata to avoid re-fetching
    if 'profiles' not in db.list_collection_names():
        db.create_collection('profiles')
        logger.info("Created 'profiles' collection")
    
    profiles = db['profiles']
    
    # Unique index on handle
    profiles.create_index([('handle', ASCENDING)], unique=True)
    profiles.create_index([('follower_count', DESCENDING)])
    profiles.create_index([('verified', ASCENDING)])
    profiles.create_index([('last_fetched', DESCENDING)])
    
    # === BRANDS COLLECTION ===
    # Trusted brands with reference data for verification
    if 'brands' not in db.list_collection_names():
        db.create_collection('brands')
        logger.info("Created 'brands' collection")
    
    brands = db['brands']
    
    brands.create_index([('name', ASCENDING)], unique=True)
    brands.create_index([('domain', ASCENDING)])
    brands.create_index([('verified_by_admin', ASCENDING)])
    brands.create_index([('official_handles', ASCENDING)])
    
    # === IMAGES_REFERENCE COLLECTION ===
    # Reference images for perceptual hashing and similarity
    if 'images_reference' not in db.list_collection_names():
        db.create_collection('images_reference')
        logger.info("Created 'images_reference' collection")
    
    images_ref = db['images_reference']
    
    images_ref.create_index([('brand_id', ASCENDING)])
    images_ref.create_index([('image_hash', ASCENDING)])
    images_ref.create_index([('created_at', DESCENDING)])
    
    logger.info("All collections initialized with indexes")
    
    # === SAMPLE DATA ===
    # Insert sample trusted brands
    _seed_sample_brands(brands)
    _seed_sample_profiles(profiles)


def _seed_sample_brands(brands_collection):
    """Insert sample trusted brands."""
    sample_brands = [
        {
            'name': 'Nike',
            'domain': 'nike.com',
            'verified_by_admin': True,
            'official_handles': ['nike', 'nikestore', 'nikesportswear'],
            'notes': 'Major sportswear brand',
            'created_at': datetime.utcnow().isoformat(),
        },
        {
            'name': 'Apple',
            'domain': 'apple.com',
            'verified_by_admin': True,
            'official_handles': ['apple', 'applepodcasts', 'applemusicuk'],
            'notes': 'Technology company',
            'created_at': datetime.utcnow().isoformat(),
        },
        {
            'name': 'Gucci',
            'domain': 'gucci.com',
            'verified_by_admin': True,
            'official_handles': ['gucci'],
            'notes': 'Luxury fashion brand',
            'created_at': datetime.utcnow().isoformat(),
        },
    ]
    
    for brand in sample_brands:
        try:
            brands_collection.update_one(
                {'name': brand['name']},
                {'$set': brand},
                upsert=True
            )
            logger.info(f"Seeded brand: {brand['name']}")
        except Exception as e:
            logger.warning(f"Could not seed brand {brand['name']}: {e}")


def _seed_sample_profiles(profiles_collection):
    """Insert sample profiles for testing."""
    sample_profiles = [
        {
            'handle': 'official_nike',
            'verified': True,
            'follower_count': 5200000,
            'following_count': 150,
            'post_count': 8500,
            'bio': 'Just Do It. ™ For sport inquiries: sponsorships@nike.com',
            'website': 'https://www.nike.com',
            'has_shop': True,
            'last_fetched': datetime.utcnow().isoformat(),
        },
        {
            'handle': 'fake_nike_deals',
            'verified': False,
            'follower_count': 2300,
            'following_count': 4500,
            'post_count': 45,
            'bio': 'NIKE DISCOUNT STORE 70% OFF! DM for orders WhatsApp +1234567890',
            'website': None,
            'has_shop': False,
            'last_fetched': datetime.utcnow().isoformat(),
        },
    ]
    
    for profile in sample_profiles:
        try:
            profiles_collection.update_one(
                {'handle': profile['handle']},
                {'$set': profile},
                upsert=True
            )
            logger.info(f"Seeded profile: {profile['handle']}")
        except Exception as e:
            logger.warning(f"Could not seed profile {profile['handle']}: {e}")


# Example document structures for reference:

"""
=== CHECKS COLLECTION ===
{
  "_id": ObjectId(...),
  "user_id": "user123",
  "input_url": "https://instagram.com/some_seller",
  "handle": "some_seller",
  "score": 28,
  "verdict": "Likely Fake",
  "reasons": [
    "Profile age: 2 months",
    "No Instagram Shop found",
    "Caption asks to DM/WhatsApp for orders",
    "Username contains many underscores/numbers"
  ],
  "metadata": {
    "followers": 1743,
    "posts": 14,
    "website": null,
    "og_image": "https://..."
  },
  "image_similarity": 0.32,
  "created_at": "2025-12-09T12:00:00Z"
}

=== PROFILES COLLECTION ===
{
  "_id": ObjectId(...),
  "handle": "alice_influencer",
  "profile_url": "https://instagram.com/alice_influencer",
  "verified": true,
  "follower_count": 125000,
  "following_count": 450,
  "post_count": 2100,
  "bio": "Fashion & lifestyle | DM for collabs",
  "website": "https://aliceblog.com",
  "has_shop": true,
  "og_image": "https://...",
  "first_post_date": "2022-03-15T10:30:00Z",
  "last_fetched": "2025-12-09T12:00:00Z"
}

=== BRANDS COLLECTION ===
{
  "_id": ObjectId(...),
  "name": "Nike",
  "domain": "nike.com",
  "verified_by_admin": true,
  "official_handles": ["nike", "nikestore"],
  "reference_images": [ObjectId(...), ObjectId(...)],
  "notes": "Major sportswear brand, use only official_handles",
  "created_at": "2025-01-01T00:00:00Z"
}

=== IMAGES_REFERENCE COLLECTION ===
{
  "_id": ObjectId(...),
  "brand_id": ObjectId("..."),
  "image_hash": "d1a2b3c4...",  # perceptual hash (pHash)
  "image_url": "s3://bucket/nike_shoe_001.jpg",
  "description": "Nike Air Max authentic product shot",
  "created_at": "2025-01-01T00:00:00Z"
}
"""
