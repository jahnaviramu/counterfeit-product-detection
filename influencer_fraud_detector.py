"""
Influencer Fraud Detection Engine
Detects anomalies, bot engagement, fake followers, and suspicious activity patterns.
Provides risk scoring and fraud alerts.
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import statistics


class InfluencerFraudDetector:
    """Detect fraudulent influencer activity and suspicious patterns"""

    def __init__(self):
        self.anomaly_threshold = 2.0  # Standard deviations
        self.suspicious_activity_threshold = 0.6

    def analyze_influencer_activity(self, influencer_data: Dict, 
                                   historical_events: List[Dict]) -> Dict:
        """
        Analyze influencer activity for fraud indicators.
        
        Args:
            influencer_data: Influencer profile data
            historical_events: List of past events (clicks, purchases, etc.)
            
        Returns:
            Dict with fraud risk assessment, flags, and recommendations
        """
        try:
            if not historical_events:
                return self._empty_fraud_report(influencer_data.get('handle', 'unknown'))

            # Extract metrics from events
            events_by_date = self._group_events_by_date(historical_events)
            
            # Run detection algorithms
            conversion_anomalies = self._detect_conversion_anomalies(historical_events)
            click_patterns = self._analyze_click_patterns(historical_events)
            temporal_patterns = self._analyze_temporal_patterns(events_by_date)
            engagement_anomalies = self._detect_engagement_anomalies(historical_events, influencer_data)

            # Calculate fraud risk score
            fraud_score = self._calculate_fraud_score(
                conversion_anomalies, click_patterns, 
                temporal_patterns, engagement_anomalies
            )

            # Generate fraud flags
            fraud_flags = self._generate_fraud_flags(
                conversion_anomalies, click_patterns, 
                temporal_patterns, engagement_anomalies
            )

            # Risk level
            risk_level = self._get_risk_level(fraud_score)

            # Recommendations
            recommendations = self._get_fraud_recommendations(fraud_flags)

            return {
                'handle': influencer_data.get('handle'),
                'fraud_score': round(fraud_score, 2),  # 0-100
                'risk_level': risk_level,  # Low, Medium, High, Critical
                'fraud_flags': fraud_flags,
                'is_suspicious': fraud_score > self.suspicious_activity_threshold * 100,
                'events_analyzed': len(historical_events),
                'conversion_anomalies': conversion_anomalies,
                'click_patterns': click_patterns,
                'temporal_patterns': temporal_patterns,
                'engagement_anomalies': engagement_anomalies,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.utcnow().isoformat()
            }

        except Exception as e:
            print(f'[ERROR] analyze_influencer_activity: {e}')
            return {'error': str(e), 'fraud_score': 0, 'risk_level': 'Unknown'}

    def _group_events_by_date(self, events: List[Dict]) -> Dict[str, List[Dict]]:
        """Group events by date"""
        grouped = {}
        for event in events:
            timestamp = event.get('timestamp') or event.get('time', '')
            if not timestamp:
                continue
            
            date_str = timestamp[:10]  # YYYY-MM-DD
            if date_str not in grouped:
                grouped[date_str] = []
            grouped[date_str].append(event)

        return grouped

    def _detect_conversion_anomalies(self, events: List[Dict]) -> Dict:
        """Detect suspicious conversion rate changes"""
        
        # Separate event types
        clicks = [e for e in events if e.get('event_type') == 'click']
        verifications = [e for e in events if e.get('event_type') == 'verification']
        purchases = [e for e in events if e.get('event_type') == 'purchase']

        if not clicks:
            return {'click_to_purchase_ratio': 0, 'anomaly_detected': False}

        # Calculate conversion rate
        click_to_purchase = len(purchases) / max(len(clicks), 1)
        click_to_verification = len(verifications) / max(len(clicks), 1)

        # Expected rates for legitimate influencers: 2-5% conversion
        is_too_high = click_to_purchase > 0.10
        is_too_low = len(clicks) > 100 and click_to_purchase < 0.001
        is_suspicious = is_too_high or is_too_low

        return {
            'click_to_purchase_ratio': round(click_to_purchase, 4),
            'click_to_verification_ratio': round(click_to_verification, 4),
            'total_clicks': len(clicks),
            'total_purchases': len(purchases),
            'anomaly_detected': is_suspicious,
            'anomaly_type': 'too_high_conversion' if is_too_high else 'too_low_conversion' if is_too_low else None,
            'severity': 'critical' if is_too_high else 'low' if is_suspicious else 'none'
        }

    def _analyze_click_patterns(self, events: List[Dict]) -> Dict:
        """Analyze click behavior for bot indicators"""
        
        clicks = [e for e in events if e.get('event_type') == 'click']
        
        if not clicks:
            return {'total_clicks': 0, 'bot_indicator': False}

        # Extract click times
        click_times = [e.get('timestamp') or e.get('time', '') for e in clicks]
        
        # Count clicks per hour (if timestamp available)
        clicks_per_hour = {}
        for ct in click_times:
            if ct and len(ct) >= 13:  # YYYY-MM-DDTHH:MM format
                hour_key = ct[:13]
                clicks_per_hour[hour_key] = clicks_per_hour.get(hour_key, 0) + 1

        # Detect unnatural click patterns
        if clicks_per_hour:
            click_counts = list(clicks_per_hour.values())
            mean_clicks = statistics.mean(click_counts)
            stdev = statistics.stdev(click_counts) if len(click_counts) > 1 else 0
            
            # Bot indicator: extremely consistent clicks per hour
            consistency = stdev / max(mean_clicks, 1)
            bot_indicator = consistency < 0.3 and len(click_counts) > 10
        else:
            bot_indicator = False
            mean_clicks = 0

        return {
            'total_clicks': len(clicks),
            'clicks_per_hour': round(mean_clicks, 2) if clicks_per_hour else 0,
            'consistency_score': round((1 - consistency) if clicks_per_hour else 0, 2),
            'bot_indicator': bot_indicator,
            'bot_probability': 0.9 if bot_indicator else 0.1,
            'pattern': 'suspicious_consistency' if bot_indicator else 'normal'
        }

    def _analyze_temporal_patterns(self, events_by_date: Dict[str, List[Dict]]) -> Dict:
        """Detect suspicious temporal patterns"""
        
        if not events_by_date:
            return {'dates_with_activity': 0, 'suspicious_pattern': False}

        dates = sorted(events_by_date.keys())
        
        if len(dates) < 2:
            return {
                'dates_with_activity': len(dates),
                'suspicious_pattern': False,
                'pattern_type': 'insufficient_data'
            }

        # Check for weekend/weekday patterns
        weekend_activity = sum(len(events_by_date[d]) for d in dates 
                              if datetime.fromisoformat(d).weekday() >= 5)
        weekday_activity = sum(len(events_by_date[d]) for d in dates 
                              if datetime.fromisoformat(d).weekday() < 5)

        total_activity = weekend_activity + weekday_activity
        
        # Natural influencers have more activity during peak hours
        weekend_ratio = weekend_activity / max(total_activity, 1)
        
        # Suspicious if activity only on weekends (suggests part-time fake activity)
        suspicious = weekend_ratio > 0.8 or weekend_ratio < 0.1

        # Check date gaps (dormant periods)
        date_gaps = []
        for i in range(len(dates) - 1):
            gap = (datetime.fromisoformat(dates[i+1]) - 
                   datetime.fromisoformat(dates[i])).days
            if gap > 30:
                date_gaps.append(gap)

        return {
            'dates_with_activity': len(dates),
            'first_activity': dates[0],
            'last_activity': dates[-1],
            'weekend_ratio': round(weekend_ratio, 2),
            'weekday_activity': weekday_activity,
            'weekend_activity': weekend_activity,
            'long_dormant_periods': len(date_gaps),
            'suspicious_pattern': suspicious,
            'pattern_type': 'weekend_heavy' if weekend_ratio > 0.8 else 'weekday_only' if weekend_ratio < 0.1 else 'normal'
        }

    def _detect_engagement_anomalies(self, events: List[Dict], influencer_data: Dict) -> Dict:
        """Detect anomalies in engagement metrics"""
        
        # Check revenue correlation with engagement
        high_revenue_events = [e for e in events if e.get('revenue', 0) > 100]
        low_revenue_events = [e for e in events if e.get('revenue', 0) <= 100]

        avg_high_revenue = (sum(e.get('revenue', 0) for e in high_revenue_events) / 
                           len(high_revenue_events)) if high_revenue_events else 0
        avg_low_revenue = (sum(e.get('revenue', 0) for e in low_revenue_events) / 
                          len(low_revenue_events)) if low_revenue_events else 0

        # Check for suspiciously consistent engagement (bot indicator)
        revenues = [e.get('revenue', 0) for e in events if e.get('revenue', 0) > 0]
        if revenues:
            revenue_mean = statistics.mean(revenues)
            revenue_stdev = statistics.stdev(revenues) if len(revenues) > 1 else 0
            revenue_consistency = revenue_stdev / max(revenue_mean, 1)
            
            suspicious_consistency = revenue_consistency < 0.2 and len(revenues) > 20
        else:
            suspicious_consistency = False
            revenue_consistency = 0

        return {
            'total_events': len(events),
            'high_revenue_events': len(high_revenue_events),
            'avg_revenue_per_event': round(avg_high_revenue, 2),
            'revenue_consistency': round(revenue_consistency, 2),
            'suspicious_revenue_pattern': suspicious_consistency,
            'estimated_aov': round(avg_high_revenue, 2),  # Average Order Value
            'pattern': 'suspicious_consistency' if suspicious_consistency else 'normal'
        }

    def _calculate_fraud_score(self, conversion_anomalies: Dict, click_patterns: Dict,
                              temporal_patterns: Dict, engagement_anomalies: Dict) -> float:
        """Calculate 0-100 fraud risk score"""
        
        score = 0

        # Conversion anomaly (up to 30 points)
        if conversion_anomalies.get('anomaly_detected'):
            score += 30 if conversion_anomalies['severity'] == 'critical' else 15

        # Click pattern (up to 25 points)
        if click_patterns.get('bot_indicator'):
            score += 25

        # Temporal pattern (up to 25 points)
        if temporal_patterns.get('suspicious_pattern'):
            score += 20

        # Engagement anomaly (up to 20 points)
        if engagement_anomalies.get('suspicious_revenue_pattern'):
            score += 20

        return min(score, 100)

    def _generate_fraud_flags(self, conversion_anomalies: Dict, click_patterns: Dict,
                             temporal_patterns: Dict, engagement_anomalies: Dict) -> List[str]:
        """Generate specific fraud flags"""
        
        flags = []

        if conversion_anomalies.get('anomaly_detected'):
            flags.append(f"Conversion anomaly: {conversion_anomalies['anomaly_type']}")

        if click_patterns.get('bot_indicator'):
            flags.append('Unnatural click consistency pattern detected')

        if temporal_patterns.get('suspicious_pattern'):
            flags.append(f"Suspicious temporal pattern: {temporal_patterns['pattern_type']}")

        if temporal_patterns.get('long_dormant_periods', 0) > 2:
            flags.append('Multiple long dormant periods in activity')

        if engagement_anomalies.get('suspicious_revenue_pattern'):
            flags.append('Suspiciously consistent revenue engagement')

        if not flags:
            flags.append('No fraudulent indicators detected')

        return flags

    def _get_risk_level(self, fraud_score: float) -> str:
        """Get risk level from score"""
        if fraud_score >= 75:
            return 'Critical'
        elif fraud_score >= 50:
            return 'High'
        elif fraud_score >= 25:
            return 'Medium'
        else:
            return 'Low'

    def _get_fraud_recommendations(self, fraud_flags: List[str]) -> List[str]:
        """Generate recommendations based on fraud flags"""
        
        recommendations = []

        if 'Conversion anomaly' in str(fraud_flags):
            recommendations.append('Review this influencer\'s conversion data manually')

        if 'Unnatural click consistency' in str(fraud_flags):
            recommendations.append('Consider suspending campaigns with this influencer pending review')

        if 'Suspicious temporal pattern' in str(fraud_flags):
            recommendations.append('Monitor activity more closely for patterns')

        if 'Suspiciously consistent revenue' in str(fraud_flags):
            recommendations.append('Verify payouts are legitimate before processing')

        if not recommendations:
            recommendations.append('This influencer appears legitimate based on fraud analysis')

        return recommendations

    def _empty_fraud_report(self, handle: str) -> Dict:
        """Generate empty report for influencers with no activity"""
        return {
            'handle': handle,
            'fraud_score': 0,
            'risk_level': 'Unknown',
            'fraud_flags': ['Insufficient activity data for analysis'],
            'is_suspicious': False,
            'events_analyzed': 0,
            'recommendation': 'Monitor after influencer generates more activity'
        }

    def bulk_fraud_check(self, influencers_with_events: List[Tuple[Dict, List[Dict]]]) -> List[Dict]:
        """Check multiple influencers for fraud at once"""
        
        reports = []
        for influencer_data, events in influencers_with_events:
            report = self.analyze_influencer_activity(influencer_data, events)
            reports.append(report)

        # Summary stats
        high_risk = [r for r in reports if r.get('risk_level') == 'High']
        critical_risk = [r for r in reports if r.get('risk_level') == 'Critical']

        return {
            'individual_reports': reports,
            'summary': {
                'total_analyzed': len(reports),
                'high_risk_count': len(high_risk),
                'critical_risk_count': len(critical_risk),
                'avg_fraud_score': round(sum(r.get('fraud_score', 0) for r in reports) / 
                                         len(reports), 2) if reports else 0,
                'recommendation': f'Review {len(critical_risk)} critical and {len(high_risk)} high-risk influencers'
            }
        }
