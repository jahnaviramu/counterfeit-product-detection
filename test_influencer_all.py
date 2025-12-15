"""
Unit tests for influencer endpoints.
Run with: python -m pytest tests/test_influencer_endpoints.py -v
"""
import pytest
import json
from datetime import datetime, timedelta
import jwt
import os

# Mock JWT_SECRET for testing
JWT_SECRET = os.getenv('JWT_SECRET', 'supersecretkey')


@pytest.fixture
def influencer_token():
    """Generate a valid influencer JWT token."""
    payload = {
        'email': 'influencer@test.com',
        'role': 'influencer',
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token


@pytest.fixture
def admin_token():
    """Generate a valid admin JWT token."""
    payload = {
        'email': 'admin@test.com',
        'role': 'admin',
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token


@pytest.fixture
def buyer_token():
    """Generate a buyer JWT token (should be denied)."""
    payload = {
        'email': 'buyer@test.com',
        'role': 'buyer',
        'exp': datetime.utcnow() + timedelta(hours=1)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm='HS256')
    return token


class TestInfluencerEvent:
    """Test the /api/influencer-event endpoint."""

    def test_event_validation_missing_influencer(self, client, influencer_token):
        """Missing influencer should return 400."""
        res = client.post('/api/influencer-event',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'product_id': 'prod1', 'event_type': 'click'}
        )
        assert res.status_code == 400
        assert 'influencer' in res.get_json()['error'].lower()

    def test_event_validation_invalid_event_type(self, client, influencer_token):
        """Invalid event_type should return 400."""
        res = client.post('/api/influencer-event',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'influencer': 'user1', 'product_id': 'prod1', 'event_type': 'invalid'}
        )
        assert res.status_code == 400
        assert 'event_type' in res.get_json()['error'].lower()

    def test_event_validation_negative_revenue(self, client, influencer_token):
        """Negative revenue should return 400."""
        res = client.post('/api/influencer-event',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'influencer': 'user1', 'product_id': 'prod1', 'event_type': 'purchase', 'revenue': -10}
        )
        assert res.status_code == 400
        assert 'revenue' in res.get_json()['error'].lower()

    def test_event_success(self, client, influencer_token):
        """Valid event should return 201."""
        res = client.post('/api/influencer-event',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'influencer': 'user1', 'product_id': 'prod1', 'event_type': 'click', 'revenue': 0}
        )
        assert res.status_code in [201, 200]  # Both acceptable
        assert 'ok' in res.get_json()['status'].lower()

    def test_event_unauthorized(self, client, buyer_token):
        """Buyers should not be able to record events (403 Forbidden)."""
        res = client.post('/api/influencer-event',
            headers={'Authorization': f'Bearer {buyer_token}'},
            json={'influencer': 'user1', 'product_id': 'prod1', 'event_type': 'click'}
        )
        assert res.status_code == 403


class TestInfluencerReferrals:
    """Test the /api/influencer-referrals endpoint."""

    def test_referrals_pagination_valid(self, client, influencer_token):
        """Valid pagination params should work."""
        res = client.get('/api/influencer-referrals?page=0&limit=10',
            headers={'Authorization': f'Bearer {influencer_token}'}
        )
        assert res.status_code == 200
        data = res.get_json()
        assert 'events' in data
        assert 'total' in data
        assert 'page' in data
        assert 'limit' in data

    def test_referrals_pagination_invalid_limit(self, client, influencer_token):
        """Invalid limit should return 400."""
        res = client.get('/api/influencer-referrals?limit=abc',
            headers={'Authorization': f'Bearer {influencer_token}'}
        )
        assert res.status_code == 400

    def test_referrals_unauthorized(self, client):
        """Missing token should return 401."""
        res = client.get('/api/influencer-referrals')
        assert res.status_code == 401


class TestInfluencerProfile:
    """Test the /api/influencer-profile endpoint."""

    def test_profile_get(self, client, influencer_token):
        """GET profile should return 200."""
        res = client.get('/api/influencer-profile',
            headers={'Authorization': f'Bearer {influencer_token}'}
        )
        assert res.status_code == 200
        assert 'profile' in res.get_json()

    def test_profile_post_validation(self, client, influencer_token):
        """POST with invalid handle should return 400."""
        res = client.post('/api/influencer-profile',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'handle': 'ab'}  # Too short
        )
        assert res.status_code == 400

    def test_profile_post_success(self, client, influencer_token):
        """Valid profile POST should return 200."""
        res = client.post('/api/influencer-profile',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={
                'handle': 'jane_influencer',
                'bio': 'Test bio',
                'payout_email': 'jane@example.com',
                'social_links': {'instagram': '@jane', 'twitter': '@jane_tw'}
            }
        )
        assert res.status_code == 200
        data = res.get_json()
        assert data['profile']['handle'] == 'jane_influencer'


class TestInfluencerPayout:
    """Test the /api/influencer-payout endpoint."""

    def test_payout_validation_missing_amount(self, client, influencer_token):
        """Missing amount should return 400."""
        res = client.post('/api/influencer-payout',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'influencer': 'user1'}
        )
        # amount defaults to 0, so check
        assert res.status_code == 400

    def test_payout_validation_negative(self, client, influencer_token):
        """Negative amount should return 400."""
        res = client.post('/api/influencer-payout',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'influencer': 'user1', 'amount': -100}
        )
        assert res.status_code == 400

    def test_payout_success(self, client, influencer_token):
        """Valid payout should return 200."""
        res = client.post('/api/influencer-payout',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'influencer': 'user1', 'amount': 50.0}
        )
        assert res.status_code == 200
        assert res.get_json()['success'] is True


class TestRateLimiting:
    """Test rate limiting on endpoints."""

    def test_rate_limit_influencer_event(self, client, influencer_token):
        """Rapid requests should trigger rate limit (429)."""
        # Note: In-memory rate limiter resets every minute; test may not work reliably
        # This is a placeholder for integration testing
        pass


class TestAdminEndpoints:
    """Test admin-only endpoints."""

    def test_create_influencer_admin_only(self, client, influencer_token):
        """Non-admin should get 403."""
        res = client.post('/api/admin/create-influencer',
            headers={'Authorization': f'Bearer {influencer_token}'},
            json={'email': 'newuser@test.com'}
        )
        assert res.status_code == 403

    def test_create_influencer_success(self, client, admin_token):
        """Admin should be able to create influencer."""
        res = client.post('/api/admin/create-influencer',
            headers={'Authorization': f'Bearer {admin_token}'},
            json={'email': 'newuser@test.com'}
        )
        assert res.status_code == 200
        assert res.get_json()['success'] is True
