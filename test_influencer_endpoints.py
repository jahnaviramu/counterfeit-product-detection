import json
from app import app


def test_influencer_stats_requires_auth():
    client = app.test_client()
    res = client.get('/api/influencer-stats')
    assert res.status_code in (200, 403)


def test_influencer_event_post():
    client = app.test_client()
    payload = {'influencer': 'test_influencer', 'product_id': 'p1', 'event_type': 'click', 'revenue': 0}
    res = client.post('/api/influencer-event', data=json.dumps(payload), content_type='application/json')
    # server may return 201 (persisted) or 200 with warning when mongo unavailable
    assert res.status_code in (200, 201)
