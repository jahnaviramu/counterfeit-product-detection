# Instagram Seller Authenticity Check API

## Overview

Complete RESTful API for checking Instagram seller profiles/posts and determining authenticity.

**Base URL**: `http://192.168.1.34:5000` (or localhost:5000)

---

## Endpoints

### 1. POST /api/v1/check

Check Instagram profile/post authenticity and get a trust score.

**Request:**
```json
{
  "url": "https://instagram.com/brandname_store",
  "user_image": null,
  "check_type": "auto"
}
```

**Response (200 OK):**
```json
{
  "handle": "brandname_store",
  "score": 28,
  "verdict": "Likely Fake",
  "reasons": [
    "✗ Payment method flag: suggests orders via DM/WhatsApp (high fraud risk)",
    "⚠ New account: 65 days old",
    "✗ Product image does not match known references (similarity: 0.15)"
  ],
  "metadata": {
    "followers": 1743,
    "following": 450,
    "posts": 14,
    "verified": false,
    "website": null,
    "bio": "FAKE DEALS 70% OFF! DM for orders WhatsApp: +1234567890",
    "og_image": "https://..."
  }
}
```

**Error (400):**
```json
{
  "error": "url required"
}
```

**Error (503):**
```json
{
  "error": "Authenticity check service unavailable"
}
```

---

### 2. GET /api/v1/checks/{check_id}

Retrieve a previously saved authenticity check by ID.

**Parameters:**
- `check_id` (path): MongoDB ObjectId of the check

**Response (200 OK):**
```json
{
  "_id": "507f1f77bcf86cd799439011",
  "input_url": "https://instagram.com/brandname_store",
  "handle": "brandname_store",
  "score": 28,
  "verdict": "Likely Fake",
  "reasons": [...],
  "metadata": {...},
  "user_email": "user@example.com",
  "created_at": "2025-12-09T12:00:00"
}
```

**Error (404):**
```json
{
  "error": "Check not found"
}
```

---

### 3. GET /api/v1/brand/{domain}

Lookup a known brand by domain.

**Parameters:**
- `domain` (path): Brand domain (e.g., `nike.com`)

**Response (200 OK) - Brand found:**
```json
{
  "brand": {
    "_id": "507f1f77bcf86cd799439011",
    "name": "Nike",
    "domain": "nike.com",
    "verified_by_admin": true,
    "official_handles": ["nike", "nikestore", "nikesportswear"],
    "notes": "Major sportswear brand"
  }
}
```

**Response (200 OK) - Brand not found:**
```json
{
  "brand": null
}
```

---

### 4. GET /api/v1/brands

List all trusted brands (paginated).

**Query Parameters:**
- `limit` (integer): Results per page (default: 50, max: 200)
- `page` (integer): Page number (default: 0)

**Response (200 OK):**
```json
{
  "brands": [
    {
      "_id": "507f1f77bcf86cd799439011",
      "name": "Nike",
      "domain": "nike.com",
      "verified_by_admin": true,
      "official_handles": ["nike", "nikestore"]
    },
    {
      "_id": "507f1f77bcf86cd799439012",
      "name": "Apple",
      "domain": "apple.com",
      "verified_by_admin": true,
      "official_handles": ["apple"]
    }
  ],
  "total": 42,
  "page": 0,
  "limit": 50
}
```

---

### 5. POST /api/v1/brand

Create or update a trusted brand (**admin only**).

**Authentication:** Requires JWT token with `role: "admin"`

**Request:**
```json
{
  "name": "Nike",
  "domain": "nike.com",
  "official_handles": ["nike", "nikestore", "nikesportswear"],
  "verified_by_admin": true,
  "notes": "Major sportswear brand. Use only these official handles."
}
```

**Response (201 Created):**
```json
{
  "status": "ok",
  "upserted": false,
  "modified": true
}
```

**Error (403):**
```json
{
  "error": "Admin access required"
}
```

---

## Scoring Algorithm

The authenticity score (0-100) is calculated using the following weighted signals:

### Positive Signals (Add Points)
| Signal | Weight | Criteria |
|--------|--------|----------|
| Verified Badge | +30 | Instagram verified (blue checkmark) |
| Shop Tab | +25 | Has Instagram Shop tab or shopping features |
| High Followers & Engagement | +15 | >10k followers, 20+ posts, healthy engagement ratio |
| Account Age | +8 | Account >1 year old |
| Clean Username | +5 | Professional username pattern (no spam indicators) |
| Website Trust | +10 | Domain is established (>1 year old) |
| Image Similarity | +18 | Product image matches known brand reference (>85% match) |

### Negative Signals (Subtract Points)
| Signal | Weight | Criteria |
|--------|--------|----------|
| DM/WhatsApp Payment | -30 | "Order via DM/WhatsApp" in caption or bio |
| High Discount | -20 | Claims >50% discount |
| New Account | -15 | Account <3 months old |
| Bot Engagement | -10 | High followers but very few posts |
| Suspicious Username | -8 | Pattern suggests fake account (underscores, numbers, spam suffixes) |

### Verdicts
- **0-40**: "Likely Fake" (High risk)
- **41-70**: "Suspicious" (Manual verification recommended)
- **71-100**: "Likely Genuine" (Low risk)

---

## Examples

### Example 1: Check Fake Nike Store

```bash
curl -X POST http://localhost:5000/api/v1/check \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://instagram.com/fake_nike_deals"
  }'
```

**Response:**
```json
{
  "handle": "fake_nike_deals",
  "score": 18,
  "verdict": "Likely Fake",
  "reasons": [
    "✗ Payment method flag: suggests orders via DM/WhatsApp (high fraud risk)",
    "⚠ New account: 65 days old",
    "⚠ Suspicious username pattern: 'fake_nike_deals'"
  ],
  "metadata": {
    "followers": 2300,
    "following": 4500,
    "posts": 45,
    "verified": false,
    "website": null,
    "bio": "NIKE DISCOUNT STORE 70% OFF! DM for orders WhatsApp +1234567890"
  }
}
```

---

### Example 2: Check Official Nike Profile

```bash
curl -X POST http://localhost:5000/api/v1/check \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://instagram.com/nike"
  }'
```

**Response:**
```json
{
  "handle": "nike",
  "score": 95,
  "verdict": "Likely Genuine",
  "reasons": [
    "✓ Instagram verified (blue checkmark)",
    "✓ Instagram Shop detected",
    "✓ Healthy follower/engagement metrics (5200000 followers)",
    "✓ Account age: 13 year(s)",
    "✓ Professional username pattern",
    "✓ Website domain established (nike.com)"
  ],
  "metadata": {
    "followers": 5200000,
    "following": 150,
    "posts": 8500,
    "verified": true,
    "website": "https://www.nike.com",
    "bio": "Just Do It. ™ For sport inquiries: sponsorships@nike.com"
  }
}
```

---

### Example 3: Retrieve Brand Info

```bash
curl -X GET http://localhost:5000/api/v1/brand/nike.com
```

**Response:**
```json
{
  "brand": {
    "_id": "507f...",
    "name": "Nike",
    "domain": "nike.com",
    "verified_by_admin": true,
    "official_handles": ["nike", "nikestore", "nikesportswear"],
    "notes": "Major sportswear brand"
  }
}
```

---

### Example 4: Admin Creates Trusted Brand

```bash
curl -X POST http://localhost:5000/api/v1/brand \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_ADMIN_JWT_TOKEN" \
  -d '{
    "name": "Gucci",
    "domain": "gucci.com",
    "official_handles": ["gucci"],
    "verified_by_admin": true,
    "notes": "Luxury fashion brand. Monitor for counterfeits closely."
  }'
```

**Response:**
```json
{
  "status": "ok",
  "upserted": true,
  "modified": false
}
```

---

## Rate Limiting

- **Global**: 100 requests per 60 seconds per IP
- **Recommended**: Implement client-side caching (12-24 hours)

---

## Error Codes

| Code | Message | Reason |
|------|---------|--------|
| 200 | OK | Check successful |
| 201 | Created | Brand created/updated |
| 400 | Bad Request | Missing required fields or invalid URL |
| 403 | Forbidden | Admin access required |
| 404 | Not Found | Check or brand not found |
| 500 | Internal Server Error | Backend error |
| 503 | Service Unavailable | Instagram fetcher modules not loaded |

---

## Database Collections

### checks
Stores all authenticity checks.

```json
{
  "_id": ObjectId,
  "input_url": "https://instagram.com/...",
  "handle": "handle",
  "score": 50,
  "verdict": "Suspicious",
  "reasons": ["...", "..."],
  "metadata": { ... },
  "user_email": "optional",
  "created_at": "2025-12-09T12:00:00Z"
}
```

### profiles
Cached Instagram profile metadata.

```json
{
  "_id": ObjectId,
  "handle": "unique_handle",
  "verified": true,
  "follower_count": 125000,
  "following_count": 450,
  "post_count": 2100,
  "bio": "...",
  "website": "https://...",
  "last_fetched": "2025-12-09T12:00:00Z"
}
```

### brands
Trusted brands reference.

```json
{
  "_id": ObjectId,
  "name": "Nike",
  "domain": "nike.com",
  "verified_by_admin": true,
  "official_handles": ["nike", "nikestore"],
  "notes": "...",
  "updated_at": "2025-12-09T12:00:00Z"
}
```

---

## Notes & Future Enhancements

1. **Image Similarity**: Currently returns placeholder score; implement pHash or CLIP for real image matching.
2. **Domain Age Checking**: Currently returns `None`; integrate WHOIS API for real domain age checks.
3. **Real-time Updates**: Cache Instagram profile data for 12-24 hours to avoid re-fetching.
4. **Machine Learning**: Train classifier on known fake accounts to improve scoring accuracy.
5. **Graph API Integration**: For business accounts, use official Instagram Graph API instead of scraping.

---

## Testing

**Quick test with Python:**

```python
import requests
import json

resp = requests.post('http://localhost:5000/api/v1/check', json={
    'url': 'https://instagram.com/nike'
})

print(json.dumps(resp.json(), indent=2))
```

---

Generated: 2025-12-09
