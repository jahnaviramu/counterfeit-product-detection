"""
Extended Database Schema for Influencer Campaigns, Analytics, and Marketplace
Adds collections for campaigns, proposals, fraud logs, and performance analytics.
"""

import logging
from datetime import datetime
from pymongo import MongoClient, ASCENDING, DESCENDING

logger = logging.getLogger(__name__)


def init_influencer_campaigns_schema(db):
    """
    Initialize collections for campaign management, fraud detection, and analytics.
    
    Args:
        db: MongoDB database instance
    """
    
    # === CAMPAIGNS COLLECTION ===
    # Campaigns created by brands to hire influencers
    if 'campaigns' not in db.list_collection_names():
        db.create_collection('campaigns')
        logger.info("Created 'campaigns' collection")
    
    campaigns = db['campaigns']
    campaigns.create_index([('brand_id', ASCENDING)])
    campaigns.create_index([('status', ASCENDING)])
    campaigns.create_index([('created_at', DESCENDING)])
    campaigns.create_index([('budget', DESCENDING)])
    campaigns.create_index([('target_tier', ASCENDING)])
    
    # === CAMPAIGN_PROPOSALS COLLECTION ===
    # Proposals sent to influencers for campaigns
    if 'campaign_proposals' not in db.list_collection_names():
        db.create_collection('campaign_proposals')
        logger.info("Created 'campaign_proposals' collection")
    
    proposals = db['campaign_proposals']
    proposals.create_index([('campaign_id', ASCENDING)])
    proposals.create_index([('influencer_id', ASCENDING)])
    proposals.create_index([('status', ASCENDING)])
    proposals.create_index([('created_at', DESCENDING)])
    proposals.create_index([('expires_at', ASCENDING)])
    
    # === CAMPAIGN_ANALYTICS COLLECTION ===
    # Tracks performance of each campaign
    if 'campaign_analytics' not in db.list_collection_names():
        db.create_collection('campaign_analytics')
        logger.info("Created 'campaign_analytics' collection")
    
    analytics = db['campaign_analytics']
    analytics.create_index([('campaign_id', ASCENDING)])
    analytics.create_index([('influencer_id', ASCENDING)])
    analytics.create_index([('updated_at', DESCENDING)])
    
    # === INFLUENCER_AUTHENTICITY COLLECTION ===
    # Stores authentication results for influencers (separate from sellers)
    if 'influencer_authenticity' not in db.list_collection_names():
        db.create_collection('influencer_authenticity')
        logger.info("Created 'influencer_authenticity' collection")
    
    inf_auth = db['influencer_authenticity']
    inf_auth.create_index([('handle', ASCENDING)])
    inf_auth.create_index([('score', DESCENDING)])
    inf_auth.create_index([('verified', ASCENDING)])
    inf_auth.create_index([('checked_at', DESCENDING)])
    
    # === INFLUENCER_FRAUD_LOGS COLLECTION ===
    # Stores fraud detection results and alerts
    if 'influencer_fraud_logs' not in db.list_collection_names():
        db.create_collection('influencer_fraud_logs')
        logger.info("Created 'influencer_fraud_logs' collection")
    
    fraud_logs = db['influencer_fraud_logs']
    fraud_logs.create_index([('influencer_id', ASCENDING)])
    fraud_logs.create_index([('risk_level', ASCENDING)])
    fraud_logs.create_index([('detected_at', DESCENDING)])
    fraud_logs.create_index([('fraud_score', DESCENDING)])
    
    # === PERFORMANCE_ANALYTICS COLLECTION ===
    # Stores calculated performance metrics for analytics dashboard
    if 'performance_analytics' not in db.list_collection_names():
        db.create_collection('performance_analytics')
        logger.info("Created 'performance_analytics' collection")
    
    perf_analytics = db['performance_analytics']
    perf_analytics.create_index([('influencer_id', ASCENDING)])
    perf_analytics.create_index([('calculated_at', DESCENDING)])
    perf_analytics.create_index([('health_score', DESCENDING)])
    perf_analytics.create_index([('tier', ASCENDING)])
    
    # === INFLUENCER_MARKETPLACE COLLECTION ===
    # Indexed influencers for marketplace search/filtering
    if 'influencer_marketplace' not in db.list_collection_names():
        db.create_collection('influencer_marketplace')
        logger.info("Created 'influencer_marketplace' collection")
    
    marketplace = db['influencer_marketplace']
    marketplace.create_index([('handle', ASCENDING)])
    marketplace.create_index([('tier', ASCENDING)])
    marketplace.create_index([('followers', DESCENDING)])
    marketplace.create_index([('niche', ASCENDING)])
    marketplace.create_index([('engagement_rate', DESCENDING)])
    marketplace.create_index([('conversion_rate', DESCENDING)])
    marketplace.create_index([('authenticity_score', DESCENDING)])
    marketplace.create_index([('available_for_hire', ASCENDING)])
    
    # === A/B_TESTS COLLECTION ===
    # A/B testing framework for campaign optimization
    if 'ab_tests' not in db.list_collection_names():
        db.create_collection('ab_tests')
        logger.info("Created 'ab_tests' collection")
    
    ab_tests = db['ab_tests']
    ab_tests.create_index([('campaign_id', ASCENDING)])
    ab_tests.create_index([('status', ASCENDING)])
    ab_tests.create_index([('created_at', DESCENDING)])
    
    logger.info("All influencer campaign collections initialized")
    

def init_payment_schema(db):
    """
    Initialize payment and commission tracking collections.
    
    Args:
        db: MongoDB database instance
    """
    
    # === PAYOUTS COLLECTION ===
    # Tracks payouts to influencers
    if 'payouts' not in db.list_collection_names():
        db.create_collection('payouts')
        logger.info("Created 'payouts' collection")
    
    payouts = db['payouts']
    payouts.create_index([('influencer_id', ASCENDING)])
    payouts.create_index([('status', ASCENDING)])
    payouts.create_index([('payout_date', DESCENDING)])
    payouts.create_index([('created_at', DESCENDING)])
    
    # === COMMISSIONS COLLECTION ===
    # Tracks commission calculations and earnings
    if 'commissions' not in db.list_collection_names():
        db.create_collection('commissions')
        logger.info("Created 'commissions' collection")
    
    commissions = db['commissions']
    commissions.create_index([('influencer_id', ASCENDING)])
    commissions.create_index([('campaign_id', ASCENDING)])
    commissions.create_index([('calculated_at', DESCENDING)])
    
    logger.info("Payment collections initialized")


# Example document structures:

CAMPAIGN_EXAMPLE = {
    "_id": "ObjectId(...)",
    "brand_id": "brand_123",
    "brand_name": "Nike",
    "title": "Summer Collection Launch",
    "description": "Promote new summer collection with product photography",
    "budget": 15000.00,
    "currency": "USD",
    "target_tier": "Macro",  # Nano, Micro, Macro, Mega
    "target_niche": ["fashion", "sports"],
    "target_followers_min": 50000,
    "target_followers_max": 500000,
    "target_engagement_min": 2.0,  # percentage
    "deliverables": {
        "posts": 3,
        "stories": 10,
        "reels": 2,
        "mentions": 5
    },
    "timeline": {
        "start_date": "2025-12-15",
        "end_date": "2025-12-31",
        "campaign_duration_days": 16
    },
    "commission_structure": {
        "base_payment": 5000.00,
        "per_click": 0.50,
        "per_conversion": 5.00,
        "performance_bonus": 2000.00  # If targets met
    },
    "status": "active",  # draft, active, paused, completed
    "proposals_count": 23,
    "accepted_proposals": 5,
    "created_at": "2025-12-09T10:00:00Z",
    "updated_at": "2025-12-09T12:00:00Z"
}

PROPOSAL_EXAMPLE = {
    "_id": "ObjectId(...)",
    "campaign_id": "campaign_123",
    "influencer_id": "influencer_456",
    "influencer_handle": "fashion_guru",
    "influencer_followers": 125000,
    "influencer_engagement_rate": 3.2,
    "influencer_niche": "fashion",
    "status": "pending",  # pending, accepted, rejected, completed
    "proposal_message": "I'm very interested in this campaign! I have a highly engaged audience interested in fashion.",
    "requested_fee": 8000.00,
    "counter_offer": None,
    "terms": {
        "posting_schedule": "Within 7 days",
        "content_approval": "Influencer chooses content",
        "exclusivity": "Not exclusive",
        "usage_rights": "Limited to 3 months"
    },
    "metrics_guarantee": {
        "minimum_impressions": 50000,
        "estimated_reach": 85000,
        "minimum_engagement": 2700
    },
    "sent_at": "2025-12-09T08:00:00Z",
    "expires_at": "2025-12-12T08:00:00Z",
    "response_at": None
}

FRAUD_LOG_EXAMPLE = {
    "_id": "ObjectId(...)",
    "influencer_id": "influencer_456",
    "influencer_handle": "suspicious_account",
    "fraud_score": 0.78,
    "risk_level": "High",
    "flags": [
        "Conversion anomaly detected",
        "Unnatural click consistency pattern",
        "High probability of bot followers"
    ],
    "affected_campaigns": ["campaign_123", "campaign_124"],
    "recommended_action": "Review and pause campaigns with this influencer",
    "detected_at": "2025-12-09T14:30:00Z",
    "reviewed": False,
    "reviewed_by": None,
    "action_taken": None
}

ANALYTICS_EXAMPLE = {
    "_id": "ObjectId(...)",
    "influencer_id": "influencer_456",
    "influencer_handle": "fashion_guru",
    "analysis_period_days": 30,
    "aov": {
        "value": 67.50,
        "sample_size": 120,
        "status": "calculated"
    },
    "ltv": {
        "historical_value": 8100.00,
        "estimated_future_value": 16200.00,
        "total_ltv": 24300.00
    },
    "roi": {
        "roi_percent": 285.0,
        "revenue_generated": 8100.00,
        "estimated_cost": 2000.00,
        "profit": 6100.00
    },
    "engagement": {
        "conversion_rate": 4.2,
        "engagement_quality": "excellent",
        "total_interactions": 8500
    },
    "health_score": 82,
    "health_status": "Excellent",
    "tier": "Micro",
    "performance_vs_tier": "exceeds_tier",
    "calculated_at": "2025-12-09T15:00:00Z"
}

MARKETPLACE_EXAMPLE = {
    "_id": "ObjectId(...)",
    "influencer_id": "influencer_456",
    "handle": "fashion_guru",
    "name": "Sarah Fashion",
    "bio": "Fashion blogger & content creator",
    "profile_url": "https://instagram.com/fashion_guru",
    "profile_image": "https://...",
    "followers": 125000,
    "engagement_rate": 3.2,
    "conversion_rate": 4.2,
    "niche": ["fashion", "lifestyle", "beauty"],
    "tier": "Micro",
    "authenticity_score": 88,
    "verified": True,
    "has_shop": False,
    "average_post_rate": 4.5,  # posts per week
    "audience_demographics": {
        "age_18_24": 35,
        "age_25_34": 40,
        "age_35_44": 15,
        "female_percent": 92
    },
    "available_for_hire": True,
    "rate_per_post": 8000,
    "rate_per_story": 2000,
    "rate_per_reel": 5000,
    "response_rate": 0.95,
    "average_response_time_hours": 2,
    "past_campaigns": 15,
    "completion_rate": 1.0,
    "listed_at": "2025-12-09T10:00:00Z",
    "updated_at": "2025-12-09T15:00:00Z"
}

AB_TEST_EXAMPLE = {
    "_id": "ObjectId(...)",
    "campaign_id": "campaign_123",
    "test_name": "Hashtag Strategy A/B Test",
    "test_type": "hashtag",  # hashtag, posting_time, content_type, call_to_action
    "variant_a": {
        "name": "Branded Hashtags Only",
        "hashtags": ["#nike", "#summercollection", "#fashion"],
        "influencers": ["influencer_1", "influencer_2"],
        "results": {
            "impressions": 125000,
            "engagement": 4200,
            "clicks": 850,
            "conversions": 45
        }
    },
    "variant_b": {
        "name": "Mixed Hashtags",
        "hashtags": ["#nike", "#summercollection", "#fashion", "#ootd", "#instagood"],
        "influencers": ["influencer_3", "influencer_4"],
        "results": {
            "impressions": 138000,
            "engagement": 5100,
            "clicks": 980,
            "conversions": 52
        }
    },
    "status": "completed",  # draft, running, completed
    "winner": "variant_b",
    "statistical_significance": 0.92,
    "started_at": "2025-12-01T00:00:00Z",
    "ended_at": "2025-12-09T00:00:00Z"
}

PAYOUT_EXAMPLE = {
    "_id": "ObjectId(...)",
    "influencer_id": "influencer_456",
    "campaign_ids": ["campaign_123", "campaign_124"],
    "payout_period": "2025-12",
    "total_amount": 15000.00,
    "currency": "USD",
    "breakdown": {
        "base_payments": 12000.00,
        "performance_bonus": 2000.00,
        "platform_fee": -1500.00,
        "tax_withholding": -3000.00
    },
    "status": "pending",  # pending, processed, paid, failed
    "payment_method": "stripe_connect",
    "stripe_transfer_id": None,
    "created_at": "2025-12-09T10:00:00Z",
    "processed_at": None,
    "paid_at": None
}
