Influencer Tracking - Setup & Notes

Overview

This document describes the influencer tracking features implemented in the backend and frontend.

Endpoints

- POST `/api/influencer-event`
  - Payload: `{ influencer, product_id, event_type, revenue?, metadata?, timestamp? }`
  - Records an influencer event. If MongoDB is unavailable, the endpoint returns 200 with a warning.

- GET `/api/influencer-stats`
  - Requires an Authorization Bearer JWT with `role` equal to `influencer` or `admin`.
  - Optional query params: `start`, `end` (ISO timestamps), `influencer`.
  - Returns impressions, clicks, verifications, purchases, conversionRate, estRevenue, top influencers, and a timeline.

- GET `/api/influencer-referrals`
  - Pagination: `limit` and `page`.
  - Requires influencer/admin role.

- GET `/api/influencer-top`
  - Requires influencer/admin role.
  - Returns top influencers aggregated by purchases.

- GET `/api/influencer-report?type=top|events`
  - Returns a CSV. Requires influencer/admin role.

- GET/POST `/api/influencer-settings`
  - Read and upsert influencer settings (e.g., commission_pct). Requires influencer/admin role.

Frontend

- `src/components/InfluencerTracking.jsx` - UI scaffold with metrics, top influencers, timeline, CSV export, summary copy, a dev "Simulate Event" form, Manage Commission modal, and polling (10s) to fetch recent events.
- `src/components/QRGenerator.jsx` - added referral link copy input which appends `?ref=<handle>` to `verification_url`.
- `src/components/VerifyProduct.jsx` - records `click` when a referral param is present and records `verification` event on successful verification.

Security

- The server enforces JWT-based role checks for sensitive influencer endpoints:
  - Only JWT tokens whose decoded payload contains `role` equal to `influencer` or `admin` are allowed to fetch influencer data or reports.

Notes & Next Steps

- The current implementation uses an `influencer_events` collection to persist events. The analytics are computed on-demand via MongoDB queries and simple aggregations. Consider pre-aggregating or using time-series collections for large event volumes.
- Add more formal tests and CI integration for the new endpoints.
- Add UI pages to manage social profiles and link them to influencer handles.

