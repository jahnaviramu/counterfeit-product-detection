"""
Instagram Influencer Verification Module
Verifies Instagram accounts, detects fake followers, checks verification badges, and validates influencer authenticity
Author: System
Version: 1.0.0
"""

import re
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InstagramInfluencerVerifier:
    """
    Advanced Instagram influencer verification system
    
    Features:
    - Verification badge detection (✓ check)
    - Follower count validation
    - Engagement rate analysis
    - Bot follower detection
    - Account authenticity scoring
    - Risk flag generation
    - Tier classification
    """
    
    def __init__(self):
        """Initialize verifier with scoring weights and thresholds"""
        self.verification_badge_weight = 0.15
        self.follower_quality_weight = 0.25
        self.engagement_weight = 0.20
        self.growth_pattern_weight = 0.15
        self.account_age_weight = 0.10
        self.bio_quality_weight = 0.15
        
        # Risk flags thresholds
        self.bot_follower_threshold = 0.30  # 30% bot followers = suspicious
        self.engagement_drop_threshold = 0.20  # 20% engagement drop = warning
        self.growth_spike_threshold = 2.5  # 250% follower growth in month = suspicious
        self.unnatural_comment_ratio = 0.05  # Comments < 5% of likes = suspicious
        
        # Tier definitions
        self.tiers = {
            'nano': {'min': 0, 'max': 10000, 'label': 'Nano Influencer'},
            'micro': {'min': 10000, 'max': 100000, 'label': 'Micro Influencer'},
            'macro': {'min': 100000, 'max': 1000000, 'label': 'Macro Influencer'},
            'mega': {'min': 1000000, 'max': float('inf'), 'label': 'Mega Influencer'}
        }
    
    def verify_instagram_account(self, instagram_url: str, profile_data: Optional[Dict] = None) -> Dict:
        """
        Main verification method for Instagram accounts
        
        Args:
            instagram_url: Instagram profile URL (e.g., https://instagram.com/username)
            profile_data: Optional pre-fetched profile data
        
        Returns:
            Dict with verification results:
            {
                'handle': str,
                'verification_badge': bool,
                'followers': int,
                'following': int,
                'posts': int,
                'bio_length': int,
                'website_present': bool,
                'authenticity_score': float (0-100),
                'verdict': str ('genuine', 'suspicious', 'likely_fake'),
                'safe_to_hire': bool,
                'follower_quality_score': float,
                'engagement_rate': float,
                'engagement_authenticity': float,
                'account_age_days': int,
                'account_age_status': str,
                'risk_flags': List[str],
                'tier': str,
                'recommendations': List[str],
                'checked_at': str
            }
        """
        
        try:
            # Extract handle from URL
            handle = self._extract_handle(instagram_url)
            if not handle:
                return self._error_response(f"Invalid Instagram URL: {handle}")
            
            # Use provided data or fetch new data
            if not profile_data:
                profile_data = self._fetch_instagram_profile(handle)
                if not profile_data:
                    return self._error_response(f"Could not fetch profile data for {handle}")
            
            # Analyze each component
            verification_result = self._check_verification_badge(profile_data)
            follower_quality = self._analyze_follower_quality(profile_data)
            engagement_analysis = self._analyze_engagement(profile_data)
            account_age_analysis = self._analyze_account_age(profile_data)
            bio_quality = self._analyze_bio_quality(profile_data)
            growth_pattern = self._analyze_growth_pattern(profile_data)
            
            # Detect risk flags
            risk_flags = self._detect_risk_flags(
                verification_result,
                follower_quality,
                engagement_analysis,
                account_age_analysis,
                growth_pattern,
                profile_data
            )
            
            # Calculate authenticity score
            authenticity_score = self._calculate_authenticity_score(
                verification_result['badge_present'],
                follower_quality['quality_score'],
                engagement_analysis['engagement_authenticity'],
                account_age_analysis['is_mature'],
                bio_quality['quality_score'],
                growth_pattern['is_natural']
            )
            
            # Determine verdict
            verdict = self._determine_verdict(authenticity_score, risk_flags)
            safe_to_hire = verdict == 'genuine'
            
            # Classify tier
            tier = self._classify_tier(profile_data.get('followers', 0))
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                verification_result,
                follower_quality,
                engagement_analysis,
                account_age_analysis,
                risk_flags,
                profile_data
            )
            
            return {
                'handle': handle,
                'verification_badge': verification_result['badge_present'],
                'verification_source': verification_result['source'],
                'followers': profile_data.get('followers', 0),
                'following': profile_data.get('following', 0),
                'posts': profile_data.get('posts', 0),
                'bio_length': len(profile_data.get('bio', '')),
                'website_present': bool(profile_data.get('website')),
                'authenticity_score': round(authenticity_score, 2),
                'verdict': verdict,
                'safe_to_hire': safe_to_hire,
                'follower_quality_score': round(follower_quality['quality_score'], 2),
                'engagement_rate': round(engagement_analysis['engagement_rate'], 4),
                'engagement_authenticity': round(engagement_analysis['engagement_authenticity'], 2),
                'account_age_days': account_age_analysis['days_old'],
                'account_age_status': account_age_analysis['status'],
                'follower_following_ratio': round(
                    profile_data.get('followers', 1) / max(profile_data.get('following', 1), 1),
                    2
                ),
                'posts_per_month': round(
                    (profile_data.get('posts', 0) * 30) / max(account_age_analysis['days_old'], 1),
                    2
                ),
                'risk_flags': risk_flags,
                'tier': tier,
                'recommendations': recommendations,
                'checked_at': datetime.utcnow().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error verifying Instagram account: {str(e)}")
            return self._error_response(str(e))
    
    def compare_influencers(self, influencer_list: List[Dict]) -> List[Dict]:
        """
        Compare multiple influencers and rank by reliability
        
        Args:
            influencer_list: List of Instagram handles or profile data
        
        Returns:
            Ranked list of influencers with comparison scores
        """
        results = []
        
        for influencer in influencer_list:
            handle = influencer if isinstance(influencer, str) else influencer.get('handle')
            profile_data = None if isinstance(influencer, str) else influencer
            
            result = self.verify_instagram_account(handle, profile_data)
            if 'error' not in result:
                results.append(result)
        
        # Sort by authenticity score descending
        results.sort(key=lambda x: x.get('authenticity_score', 0), reverse=True)
        
        return results
    
    # ==================== HELPER METHODS ====================
    
    def _extract_handle(self, url: str) -> Optional[str]:
        """Extract Instagram handle from URL"""
        patterns = [
            r'instagram\.com/([a-zA-Z0-9._-]+)/?',
            r'^([a-zA-Z0-9._-]+)$',  # Direct handle
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url.lower())
            if match:
                return match.group(1).rstrip('/')
        
        return None
    
    def _fetch_instagram_profile(self, handle: str) -> Optional[Dict]:
        """
        Fetch Instagram profile data
        Uses Open Graph meta tags and public profile information
        """
        try:
            # Try Instagram direct fetch with Open Graph
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            url = f"https://www.instagram.com/{handle}/?__a=1"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                user = data.get('graphql', {}).get('user', {})
                
                return {
                    'followers': user.get('edge_followed_by', {}).get('total_count', 0),
                    'following': user.get('edge_follow', {}).get('total_count', 0),
                    'posts': user.get('edge_owner_to_timeline_media', {}).get('total_count', 0),
                    'is_verified': user.get('is_verified', False),
                    'is_business_account': user.get('is_business_account', False),
                    'is_professional_account': user.get('is_professional_account', False),
                    'bio': user.get('biography', ''),
                    'website': user.get('external_url', ''),
                    'profile_pic_url': user.get('profile_pic_url_hd', ''),
                    'created_at': None,  # Not available from public API
                    'recent_posts': user.get('edge_owner_to_timeline_media', {}).get('edges', [])[:12]
                }
        
        except Exception as e:
            logger.warning(f"Could not fetch Instagram profile for {handle}: {str(e)}")
            return None
    
    def _check_verification_badge(self, profile_data: Dict) -> Dict:
        """Check for Instagram verification badge (✓)"""
        badge_present = profile_data.get('is_verified', False)
        
        return {
            'badge_present': badge_present,
            'source': 'instagram_official' if badge_present else 'not_verified',
            'confidence': 1.0 if badge_present else 0.0
        }
    
    def _analyze_follower_quality(self, profile_data: Dict) -> Dict:
        """
        Analyze follower quality and detect fake followers
        
        Metrics:
        - Follower/Following ratio (too high = fake followers)
        - Account history consistency
        - Bio quality
        """
        followers = profile_data.get('followers', 0)
        following = profile_data.get('following', 0)
        posts = profile_data.get('posts', 0)
        
        # Calculate follower/following ratio
        ratio = followers / max(following, 1)
        
        quality_score = 100.0
        issues = []
        
        # Check follower/following ratio
        if ratio > 10 and followers > 100000:
            # Mega influencers can have high ratios naturally
            quality_score -= 5
        elif ratio > 5:
            quality_score -= 15
            issues.append("Unusually high follower/following ratio")
        elif ratio < 0.1:
            quality_score -= 10
            issues.append("Very low follower/following ratio")
        
        # Check if account has posts (necessary for influencer)
        if posts == 0:
            quality_score -= 30
            issues.append("No posts on account")
        elif followers > 0 and posts < 5:
            quality_score -= 20
            issues.append("Very few posts for follower count")
        
        # Check for bot-like follower growth
        posts_per_follower = posts / max(followers, 1)
        if posts_per_follower > 0.01:  # Less than 1 follower per post on average
            quality_score -= 10
            issues.append("Suspicious posts-to-followers ratio")
        
        return {
            'quality_score': max(0, quality_score),
            'ratio': round(ratio, 2),
            'issues': issues,
            'bot_probability': 1 - (quality_score / 100)
        }
    
    def _analyze_engagement(self, profile_data: Dict) -> Dict:
        """
        Analyze engagement patterns
        
        Metrics:
        - Average likes per post
        - Comment rate
        - Engagement rate
        """
        posts = profile_data.get('recent_posts', [])
        followers = profile_data.get('followers', 1)
        
        if not posts:
            return {
                'engagement_rate': 0.0,
                'engagement_authenticity': 50.0,
                'avg_likes': 0,
                'avg_comments': 0,
                'comment_like_ratio': 0.0
            }
        
        total_likes = 0
        total_comments = 0
        like_counts = []
        comment_counts = []
        
        for post in posts[:12]:  # Last 12 posts
            node = post.get('node', post)
            likes = node.get('edge_liked_by', {}).get('total_count', 0)
            comments = node.get('edge_media_to_comment', {}).get('total_count', 0)
            
            total_likes += likes
            total_comments += comments
            like_counts.append(likes)
            comment_counts.append(comments)
        
        avg_likes = total_likes / len(posts) if posts else 0
        avg_comments = total_comments / len(posts) if posts else 0
        
        # Calculate engagement rate (likes + comments) / followers
        engagement_rate = (total_likes + total_comments) / (followers * len(posts)) if followers > 0 else 0
        
        # Analyze engagement authenticity
        authenticity_score = 100.0
        
        # Check comment-to-like ratio (should be 5-15%)
        comment_like_ratio = total_comments / max(total_likes, 1)
        if comment_like_ratio < 0.02:
            authenticity_score -= 15  # Too few comments
        elif comment_like_ratio > 0.20:
            authenticity_score -= 5  # Slightly unusual but OK
        
        # Check for unnatural consistency
        if like_counts:
            avg_like = sum(like_counts) / len(like_counts)
            variance = sum((x - avg_like) ** 2 for x in like_counts) / len(like_counts)
            std_dev = variance ** 0.5
            
            # Very low variance = unnatural
            if avg_like > 100 and std_dev / avg_like < 0.1:
                authenticity_score -= 20
            # High variance = more natural
            elif std_dev / avg_like > 1.0:
                authenticity_score -= 5
        
        # Realistic engagement rates by follower count
        if followers < 10000:
            expected_rate = 0.05  # 5% for nano
        elif followers < 100000:
            expected_rate = 0.03  # 3% for micro
        else:
            expected_rate = 0.01  # 1% for macro/mega
        
        if engagement_rate > expected_rate * 2:
            authenticity_score -= 10  # Suspiciously high
        
        return {
            'engagement_rate': min(0.15, engagement_rate),  # Cap at 15%
            'engagement_authenticity': max(0, authenticity_score),
            'avg_likes': round(avg_likes, 0),
            'avg_comments': round(avg_comments, 0),
            'comment_like_ratio': round(comment_like_ratio, 4),
            'total_engagement': total_likes + total_comments
        }
    
    def _analyze_account_age(self, profile_data: Dict) -> Dict:
        """Analyze account age (newer accounts = higher risk)"""
        # Instagram created_at not available in public API
        # Use heuristic: accounts with low posts and high followers = new
        
        posts = profile_data.get('posts', 0)
        followers = profile_data.get('followers', 0)
        
        # Estimate account maturity
        is_mature = posts > 20 or followers == 0  # At least 20 posts or brand new
        
        # Since we can't get exact age, estimate based on post count
        # Assume average 2-3 posts per month for active accounts
        estimated_months = posts / 2.5 if posts > 0 else 1
        estimated_days = estimated_months * 30
        
        if estimated_days < 30:
            status = 'very_new'
            score_impact = -30
        elif estimated_days < 90:
            status = 'new'
            score_impact = -15
        elif estimated_days < 365:
            status = 'young'
            score_impact = -5
        else:
            status = 'established'
            score_impact = 0
        
        return {
            'days_old': int(estimated_days),
            'is_mature': is_mature,
            'status': status,
            'score_impact': score_impact
        }
    
    def _analyze_bio_quality(self, profile_data: Dict) -> Dict:
        """Analyze bio quality (fake accounts often have empty bios)"""
        bio = profile_data.get('bio', '')
        website = bool(profile_data.get('website'))
        
        quality_score = 60.0
        
        # Check bio length
        if len(bio) == 0:
            quality_score -= 20  # No bio = suspicious
        elif len(bio) < 10:
            quality_score -= 10  # Very short bio
        elif len(bio) > 150:
            quality_score += 5  # Detailed bio
        
        # Check for website
        if website:
            quality_score += 10
        
        # Check for business account indicators
        is_business = profile_data.get('is_business_account', False)
        is_professional = profile_data.get('is_professional_account', False)
        
        if is_business or is_professional:
            quality_score += 15
        
        return {
            'quality_score': min(100, quality_score),
            'bio_present': len(bio) > 0,
            'website_present': website,
            'is_business_account': is_business
        }
    
    def _analyze_growth_pattern(self, profile_data: Dict) -> Dict:
        """Analyze growth pattern (sudden spikes = bot followers)"""
        # With public API, we can't get historical data
        # But we can check current metrics for anomalies
        
        followers = profile_data.get('followers', 0)
        posts = profile_data.get('posts', 0)
        
        # Check for natural growth pattern
        # Ratio should be somewhat consistent
        
        is_natural = True
        anomalies = []
        
        # Very high followers with few posts = suspicious
        if followers > 100000 and posts < 10:
            is_natural = False
            anomalies.append("High followers with very few posts")
        
        # Very high follower count but low engagement = suspicious
        # (checked separately in engagement analysis)
        
        return {
            'is_natural': is_natural,
            'anomalies': anomalies,
            'score_impact': 0 if is_natural else -20
        }
    
    def _detect_risk_flags(self, verification_result: Dict, follower_quality: Dict,
                          engagement_analysis: Dict, account_age_analysis: Dict,
                          growth_pattern: Dict, profile_data: Dict) -> List[str]:
        """Detect all risk flags"""
        flags = []
        
        # Verification badge missing
        if not verification_result['badge_present']:
            flags.append("No Instagram verification badge")
        
        # Follower quality issues
        if follower_quality['quality_score'] < 60:
            flags.append("Low follower quality score")
        
        if follower_quality['bot_probability'] > 0.30:
            flags.append("High probability of bot followers (>30%)")
        
        for issue in follower_quality.get('issues', []):
            flags.append(f"Follower quality: {issue}")
        
        # Engagement issues
        if engagement_analysis['engagement_authenticity'] < 50:
            flags.append("Engagement patterns appear artificial")
        
        if engagement_analysis['comment_like_ratio'] < 0.02:
            flags.append("Suspiciously few comments (engagement might be fake)")
        
        # Account age issues
        if account_age_analysis['status'] == 'very_new':
            flags.append("Account is very new (<30 days)")
        elif account_age_analysis['status'] == 'new':
            flags.append("Account is new (<90 days)")
        
        # Growth pattern issues
        for anomaly in growth_pattern.get('anomalies', []):
            flags.append(f"Growth anomaly: {anomaly}")
        
        # Bio quality issues
        if profile_data.get('posts', 0) == 0:
            flags.append("No posts on account")
        
        return flags
    
    def _calculate_authenticity_score(self, badge: bool, follower_quality: float,
                                     engagement_auth: float, mature_account: bool,
                                     bio_quality: float, natural_growth: bool) -> float:
        """
        Calculate overall authenticity score (0-100)
        
        Weights:
        - Verification badge: 15%
        - Follower quality: 25%
        - Engagement authenticity: 20%
        - Account maturity: 10%
        - Bio quality: 15%
        - Natural growth: 15%
        """
        score = 0.0
        
        # Verification badge (0-15)
        if badge:
            score += 15.0
        
        # Follower quality (0-25)
        score += (follower_quality / 100) * 25
        
        # Engagement authenticity (0-20)
        score += (engagement_auth / 100) * 20
        
        # Account maturity (0-10)
        if mature_account:
            score += 10.0
        
        # Bio quality (0-15)
        score += (bio_quality / 100) * 15
        
        # Natural growth (0-15)
        if natural_growth:
            score += 15.0
        
        return min(100.0, max(0.0, score))
    
    def _determine_verdict(self, score: float, risk_flags: List[str]) -> str:
        """Determine verdict based on score and risk flags"""
        if score >= 75:
            return 'genuine'
        elif score >= 50:
            if len(risk_flags) > 5:
                return 'likely_fake'
            return 'suspicious'
        else:
            return 'likely_fake'
    
    def _classify_tier(self, followers: int) -> str:
        """Classify influencer tier based on follower count"""
        for tier_name, tier_def in self.tiers.items():
            if tier_def['min'] <= followers < tier_def['max']:
                return tier_name
        return 'nano'
    
    def _generate_recommendations(self, verification_result: Dict, follower_quality: Dict,
                                 engagement_analysis: Dict, account_age_analysis: Dict,
                                 risk_flags: List[str], profile_data: Dict) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Verification badge recommendation
        if not verification_result['badge_present']:
            recommendations.append(
                "Encourage influencer to apply for Instagram verification badge "
                "(requires 10k+ followers, active account, clear bio)"
            )
        
        # Engagement recommendation
        if engagement_analysis['comment_like_ratio'] < 0.03:
            recommendations.append(
                "Review engagement quality - encourage more conversation with followers "
                "(ask questions, respond to comments)"
            )
        
        # Bio recommendation
        if profile_data.get('bio', '') == '':
            recommendations.append("Add a professional bio with relevant keywords and website")
        
        if not profile_data.get('website'):
            recommendations.append("Add website link to profile (links to store/portfolio)")
        
        # Account maturity
        if account_age_analysis['status'] == 'very_new':
            recommendations.append("Wait 30+ days for account to establish credibility before major campaigns")
        elif account_age_analysis['status'] == 'new':
            recommendations.append("Consider smaller test campaigns until account matures (90+ days)")
        
        # Follower quality
        if follower_quality['ratio'] > 5:
            recommendations.append(
                "High follower/following ratio - audit followers for authenticity. "
                "Consider unfollowing inactive accounts to improve ratio."
            )
        
        # General recommendations
        if len(risk_flags) == 0:
            recommendations.append("✓ Account appears authentic and ready for partnerships")
        
        return recommendations[:5]  # Max 5 recommendations
    
    def _error_response(self, error: str) -> Dict:
        """Generate error response"""
        return {
            'error': error,
            'success': False,
            'checked_at': datetime.utcnow().isoformat()
        }
