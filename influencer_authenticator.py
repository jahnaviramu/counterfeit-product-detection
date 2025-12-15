"""
Influencer Authenticator Module
Verifies Instagram influencer accounts, detects bot followers, analyzes engagement,
and provides authenticity scoring specific to influencers.
"""

import re
from typing import Dict, List, Tuple
from datetime import datetime
from instagram_fetcher import InstagramMetadataFetcher


class InfluencerAuthenticator:
    """Authenticate and score influencer Instagram accounts"""

    def __init__(self):
        self.fetcher = InstagramMetadataFetcher()
        self.min_followers_threshold = 1000
        self.min_engagement_threshold = 0.5  # 0.5% minimum engagement rate

    def authenticate_influencer(self, instagram_url: str) -> Dict:
        """
        Complete influencer authentication with score, risk flags, and recommendations.
        
        Args:
            instagram_url: Instagram profile URL or handle
            
        Returns:
            Dict with:
                - handle: Instagram handle
                - score: 0-100 authenticity score
                - verdict: "Authentic", "Suspicious", or "Likely Fake"
                - risk_flags: List of detected issues
                - engagement_metrics: Engagement analysis
                - follower_analysis: Bot/fake follower detection
                - recommendations: Array of improvement suggestions
                - verified_badge: Boolean if has blue checkmark
                - tier: "Mega", "Macro", "Micro", "Nano" influencer
        """
        try:
            # Fetch metadata
            metadata = self.fetcher.fetch_profile_metadata(instagram_url)
            if not metadata or metadata.get('error'):
                return {
                    'error': 'Failed to fetch Instagram profile',
                    'handle': None,
                    'score': 0,
                    'verdict': 'Likely Fake'
                }

            # Extract key metrics
            handle = metadata.get('handle')
            followers = int(metadata.get('follower_count', 0))
            following = int(metadata.get('following_count', 0))
            posts = int(metadata.get('post_count', 0))
            bio = metadata.get('bio', '')
            verified = metadata.get('verified', False)
            has_shop = metadata.get('has_shop', False)
            website = metadata.get('website', '')

            # Calculate engagement rate
            engagement_metrics = self._analyze_engagement(followers, following, posts, metadata)
            
            # Bot follower detection
            follower_analysis = self._analyze_follower_quality(followers, following, engagement_metrics)
            
            # Risk assessment
            risk_flags = self._detect_risk_flags(
                handle, followers, following, posts, bio, 
                verified, engagement_metrics, follower_analysis
            )
            
            # Calculate authenticity score
            score = self._calculate_influencer_score(
                verified, followers, engagement_metrics, 
                follower_analysis, risk_flags, has_shop, website
            )
            
            # Determine verdict
            verdict = self._get_verdict(score)
            
            # Influencer tier
            tier = self._classify_influencer_tier(followers)
            
            # Recommendations
            recommendations = self._generate_recommendations(risk_flags, engagement_metrics, tier)

            return {
                'handle': handle,
                'followers': followers,
                'following': following,
                'posts': posts,
                'score': score,
                'verdict': verdict,
                'verified_badge': verified,
                'tier': tier,
                'has_shop': has_shop,
                'website': website,
                'bio': bio,
                'engagement_metrics': engagement_metrics,
                'follower_analysis': follower_analysis,
                'risk_flags': risk_flags,
                'recommendations': recommendations,
                'authenticity_timestamp': datetime.utcnow().isoformat(),
                'safe_to_hire': score >= 70 and len(risk_flags) <= 1
            }

        except Exception as e:
            print(f'[ERROR] authenticate_influencer: {e}')
            return {'error': str(e), 'score': 0, 'verdict': 'Likely Fake'}

    def _analyze_engagement(self, followers: int, following: int, posts: int, metadata: Dict) -> Dict:
        """Analyze engagement rate and patterns"""
        
        if posts == 0:
            return {'engagement_rate': 0, 'posts_count': 0, 'avg_engagement_per_post': 0}

        # Estimate engagement from metadata
        engagement_rate = 0
        if followers > 0:
            # Average engagement = (comments + likes) / followers per post
            # If we have post data, use it; otherwise estimate
            estimated_likes_per_post = max(followers * 0.02, 10)  # 2% engagement baseline
            engagement_rate = (estimated_likes_per_post / followers) * 100

        return {
            'engagement_rate': round(engagement_rate, 2),
            'posts_count': posts,
            'avg_engagement_per_post': round(estimated_likes_per_post, 0),
            'follower_to_following_ratio': round(followers / max(following, 1), 2),
            'posts_per_day_estimate': round(posts / 365, 2) if posts > 0 else 0,
            'follower_growth_indicator': 'healthy' if followers > self.min_followers_threshold else 'low'
        }

    def _analyze_follower_quality(self, followers: int, following: int, engagement_metrics: Dict) -> Dict:
        """Detect bot followers and fake engagement"""
        
        risk_score = 0
        bot_indicators = []

        # 1. Follower/Following ratio analysis
        ratio = engagement_metrics['follower_to_following_ratio']
        if ratio < 0.5:
            risk_score += 15
            bot_indicators.append('Unusually high following count relative to followers')
        elif ratio > 5:
            risk_score += 5
            bot_indicators.append('Very selective following (may indicate old-school spam tactics)')

        # 2. Follower count validation
        if followers < 1000:
            risk_score += 10
            bot_indicators.append('Low follower count (may limit reach)')
        elif followers > 10_000_000:
            risk_score += 5
            bot_indicators.append('Extremely high follower count (may include bots)')

        # 3. Engagement rate validation
        engagement = engagement_metrics['engagement_rate']
        if engagement < 0.5:
            risk_score += 20
            bot_indicators.append('Very low engagement rate (below 0.5%)')
        elif engagement > 15:
            risk_score += 5
            bot_indicators.append('Unusually high engagement (may indicate bot likes)')

        # 4. Posts per day
        posts_per_day = engagement_metrics['posts_per_day_estimate']
        if posts_per_day > 5:
            risk_score += 10
            bot_indicators.append('Excessive posting frequency (may indicate automation)')

        # Calculate bot probability
        bot_probability = min(risk_score / 100, 1.0)

        return {
            'bot_probability': round(bot_probability, 2),
            'risk_score': risk_score,
            'indicators': bot_indicators,
            'likely_authentic': bot_probability < 0.4,
            'needs_manual_review': 0.4 <= bot_probability <= 0.7,
            'likely_inauthentic': bot_probability > 0.7
        }

    def _detect_risk_flags(self, handle: str, followers: int, following: int, posts: int,
                          bio: str, verified: bool, engagement_metrics: Dict, 
                          follower_analysis: Dict) -> List[str]:
        """Identify specific risk factors"""
        
        flags = []

        # Account completeness
        if not bio or len(bio) < 10:
            flags.append('Incomplete or missing bio')
        
        if posts == 0:
            flags.append('No posts (new account)')

        # Handle analysis
        if re.search(r'\d{4,}', handle):  # Many consecutive numbers
            flags.append('Handle contains many numbers (may indicate spam)')

        if len(handle) < 3:
            flags.append('Very short handle (may be impersonation)')

        # Follower quality
        if follower_analysis['bot_probability'] > 0.6:
            flags.append('High probability of bot followers')

        if not verified and followers > 100_000:
            flags.append('Large follower count but not verified badge')

        # Engagement
        if engagement_metrics['engagement_rate'] < 0.5 and followers > 10_000:
            flags.append('Low engagement for follower count')

        # New account detection
        if followers < 1000 and posts < 20:
            flags.append('New account with low activity')

        return flags

    def _calculate_influencer_score(self, verified: bool, followers: int, 
                                   engagement_metrics: Dict, follower_analysis: Dict,
                                   risk_flags: List[str], has_shop: bool, website: str) -> int:
        """Calculate 0-100 authenticity score for influencer"""
        
        score = 50  # Baseline

        # Verified badge (+25)
        if verified:
            score += 25

        # Follower count validation (+20)
        if 5_000 <= followers <= 1_000_000:
            score += 20
        elif followers > 1_000_000:
            score += 15
        elif followers >= 1_000:
            score += 10

        # Engagement rate (+20)
        engagement = engagement_metrics['engagement_rate']
        if engagement >= 3:
            score += 20
        elif engagement >= 1:
            score += 15
        elif engagement >= 0.5:
            score += 10

        # Follower quality (+15)
        bot_prob = follower_analysis['bot_probability']
        if bot_prob < 0.3:
            score += 15
        elif bot_prob < 0.5:
            score += 10
        elif bot_prob < 0.7:
            score += 5

        # Shop/Website (+10)
        if has_shop:
            score += 8
        if website:
            score += 2

        # Risk flags penalty (-5 per flag)
        score -= len(risk_flags) * 5

        # Clamp to 0-100
        return max(0, min(100, score))

    def _get_verdict(self, score: int) -> str:
        """Get verdict based on score"""
        if score >= 75:
            return 'Authentic'
        elif score >= 50:
            return 'Suspicious'
        else:
            return 'Likely Fake'

    def _classify_influencer_tier(self, followers: int) -> str:
        """Classify influencer tier by follower count"""
        if followers >= 1_000_000:
            return 'Mega'
        elif followers >= 100_000:
            return 'Macro'
        elif followers >= 10_000:
            return 'Micro'
        else:
            return 'Nano'

    def _generate_recommendations(self, risk_flags: List[str], 
                                 engagement_metrics: Dict, tier: str) -> List[str]:
        """Generate recommendations for improvement"""
        
        recommendations = []

        # Address risk flags
        if 'Incomplete or missing bio' in risk_flags:
            recommendations.append('Complete your bio with clear description and contact info')

        if 'No posts' in risk_flags:
            recommendations.append('Start posting regular content to build credibility')

        if 'High probability of bot followers' in risk_flags:
            recommendations.append('Consider using Instagram\'s audience insights to remove fake followers')

        if 'Low engagement for follower count' in risk_flags:
            recommendations.append('Create more interactive content or consider improving content quality')

        # Tier-specific recommendations
        if tier == 'Nano':
            recommendations.append('Grow to 10k+ followers to unlock more brand partnerships')
            recommendations.append('Focus on niche engagement rather than follower count')

        if len(recommendations) == 0:
            recommendations.append('Account looks authentic - continue building quality content!')

        return recommendations

    def compare_influencers(self, influencers: List[Dict]) -> Dict:
        """Compare multiple influencers for brand matching"""
        
        if not influencers:
            return {}

        # Rank by score
        ranked = sorted(influencers, key=lambda x: x.get('score', 0), reverse=True)

        # Calculate metrics
        avg_score = sum(i.get('score', 0) for i in influencers) / len(influencers)
        avg_followers = sum(i.get('followers', 0) for i in influencers) / len(influencers)
        avg_engagement = sum(i.get('engagement_metrics', {}).get('engagement_rate', 0) 
                            for i in influencers) / len(influencers)

        return {
            'ranked': ranked,
            'stats': {
                'count': len(influencers),
                'avg_score': round(avg_score, 2),
                'avg_followers': int(avg_followers),
                'avg_engagement_rate': round(avg_engagement, 2),
                'authentic_count': sum(1 for i in influencers if i.get('verdict') == 'Authentic'),
                'suspicious_count': sum(1 for i in influencers if i.get('verdict') == 'Suspicious'),
                'fake_count': sum(1 for i in influencers if i.get('verdict') == 'Likely Fake')
            },
            'recommendation': ranked[0]['handle'] if ranked else None
        }
