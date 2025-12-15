"""
Influencer Performance Analytics Engine
Calculates AOV, LTV, ROI, A/B testing metrics, and custom performance KPIs.
"""

from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import statistics


class InfluencerAnalytics:
    """Calculate comprehensive performance analytics for influencers and campaigns"""

    def __init__(self):
        self.date_format = '%Y-%m-%d'

    def calculate_comprehensive_metrics(self, influencer_data: Dict, 
                                       events: List[Dict],
                                       time_period_days: int = 30) -> Dict:
        """
        Calculate all key performance metrics.
        
        Args:
            influencer_data: Influencer profile info
            events: List of historical events
            time_period_days: Analysis period (default 30 days)
            
        Returns:
            Dict with AOV, LTV, ROI, engagement metrics, and trends
        """
        
        try:
            handle = influencer_data.get('handle', 'unknown')
            
            # Filter events by time period
            cutoff_date = datetime.utcnow() - timedelta(days=time_period_days)
            recent_events = self._filter_events_by_date(events, cutoff_date)

            if not recent_events:
                return self._empty_analytics_report(handle)

            # Calculate core metrics
            aov = self._calculate_aov(recent_events)
            ltv = self._calculate_ltv(influencer_data, recent_events)
            roi = self._calculate_roi(recent_events, influencer_data)
            engagement = self._calculate_engagement(recent_events, influencer_data)
            retention = self._calculate_retention(recent_events)
            growth = self._calculate_growth(recent_events, time_period_days)

            # Conversion funnel
            funnel = self._calculate_conversion_funnel(recent_events)

            # Performance trends
            trends = self._calculate_trends(recent_events)

            # Benchmarking
            tier = self._get_influencer_tier(influencer_data.get('followers', 0))
            benchmarks = self._get_tier_benchmarks(tier)

            return {
                'handle': handle,
                'analysis_period_days': time_period_days,
                'events_analyzed': len(recent_events),
                'analysis_timestamp': datetime.utcnow().isoformat(),
                
                # Core metrics
                'aov': aov,  # Average Order Value
                'ltv': ltv,  # Lifetime Value
                'roi': roi,  # Return on Investment
                'engagement': engagement,
                'retention': retention,
                'growth': growth,
                'funnel': funnel,
                'trends': trends,
                
                # Benchmarking
                'tier': tier,
                'benchmarks': benchmarks,
                'performance_vs_tier': self._compare_to_benchmark(aov, roi, engagement, benchmarks),
                
                # Health check
                'health_score': self._calculate_health_score(aov, roi, retention, engagement),
                'health_status': self._get_health_status(self._calculate_health_score(aov, roi, retention, engagement))
            }

        except Exception as e:
            print(f'[ERROR] calculate_comprehensive_metrics: {e}')
            return {'error': str(e)}

    def _filter_events_by_date(self, events: List[Dict], cutoff_date: datetime) -> List[Dict]:
        """Filter events to recent period"""
        filtered = []
        for event in events:
            timestamp_str = event.get('timestamp') or event.get('time', '')
            if not timestamp_str:
                continue
            try:
                event_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                if event_date >= cutoff_date:
                    filtered.append(event)
            except:
                pass
        return filtered

    def _calculate_aov(self, events: List[Dict]) -> Dict:
        """Calculate Average Order Value"""
        
        purchase_events = [e for e in events if e.get('event_type') == 'purchase']
        
        if not purchase_events:
            return {
                'value': 0,
                'currency': 'USD',
                'sample_size': 0,
                'status': 'insufficient_data'
            }

        revenues = [e.get('revenue', 0) for e in purchase_events if e.get('revenue', 0) > 0]
        
        if not revenues:
            return {
                'value': 0,
                'currency': 'USD',
                'sample_size': len(purchase_events),
                'status': 'no_revenue_data'
            }

        aov = statistics.mean(revenues)
        median = statistics.median(revenues)
        stdev = statistics.stdev(revenues) if len(revenues) > 1 else 0

        return {
            'value': round(aov, 2),
            'median': round(median, 2),
            'min': round(min(revenues), 2),
            'max': round(max(revenues), 2),
            'std_dev': round(stdev, 2),
            'currency': 'USD',
            'sample_size': len(revenues),
            'status': 'calculated'
        }

    def _calculate_ltv(self, influencer_data: Dict, events: List[Dict]) -> Dict:
        """Calculate Lifetime Value (total revenue generated)"""
        
        total_revenue = sum(e.get('revenue', 0) for e in events if e.get('revenue', 0) > 0)
        
        purchase_count = len([e for e in events if e.get('event_type') == 'purchase'])
        
        # Estimate future LTV based on average monthly revenue
        months_active = 6  # Conservative estimate
        estimated_future_value = (total_revenue / max(months_active, 1)) * 12

        return {
            'historical_value': round(total_revenue, 2),
            'estimated_future_value': round(estimated_future_value, 2),
            'total_ltv': round(total_revenue + estimated_future_value, 2),
            'purchase_count': purchase_count,
            'value_per_purchase': round(total_revenue / max(purchase_count, 1), 2),
            'currency': 'USD'
        }

    def _calculate_roi(self, events: List[Dict], influencer_data: Dict) -> Dict:
        """Calculate Return on Investment"""
        
        # Estimate investment: brand probably pays influencer 10-30% of revenue
        total_revenue = sum(e.get('revenue', 0) for e in events)
        estimated_cost = total_revenue * 0.15  # 15% average influencer cut
        
        # Profit
        profit = total_revenue - estimated_cost
        
        # ROI = (Profit / Cost) * 100
        roi_percent = (profit / max(estimated_cost, 1)) * 100 if estimated_cost > 0 else 0

        # Click cost (assume $0.50 per click)
        clicks = len([e for e in events if e.get('event_type') == 'click'])
        click_cost = clicks * 0.50
        
        # CPC (Cost Per Click)
        cpc = click_cost / max(clicks, 1) if clicks > 0 else 0

        return {
            'roi_percent': round(roi_percent, 2),
            'estimated_cost': round(estimated_cost, 2),
            'revenue_generated': round(total_revenue, 2),
            'profit': round(profit, 2),
            'cost_per_click': round(cpc, 2),
            'profit_margin_percent': round((profit / max(total_revenue, 1)) * 100, 2),
            'status': 'positive' if roi_percent > 0 else 'negative' if roi_percent < 0 else 'break_even'
        }

    def _calculate_engagement(self, events: List[Dict], influencer_data: Dict) -> Dict:
        """Calculate engagement metrics"""
        
        followers = influencer_data.get('followers', 1000)
        
        # Event counts
        clicks = len([e for e in events if e.get('event_type') == 'click'])
        verifications = len([e for e in events if e.get('event_type') == 'verification'])
        purchases = len([e for e in events if e.get('event_type') == 'purchase'])
        impressions = len([e for e in events if e.get('event_type') == 'impression'])

        total_interactions = clicks + verifications + purchases

        # Engagement rates
        impression_rate = (impressions / max(followers, 1)) * 100
        click_rate = (clicks / max(impressions, 1)) * 100
        verification_rate = (verifications / max(clicks, 1)) * 100
        conversion_rate = (purchases / max(clicks, 1)) * 100

        return {
            'total_interactions': total_interactions,
            'impressions': impressions,
            'clicks': clicks,
            'verifications': verifications,
            'conversions': purchases,
            'impression_rate': round(impression_rate, 2),
            'click_through_rate': round(click_rate, 2),
            'verification_rate': round(verification_rate, 2),
            'conversion_rate': round(conversion_rate, 2),
            'engagement_quality': 'excellent' if conversion_rate > 5 else 'good' if conversion_rate > 2 else 'average' if conversion_rate > 0.5 else 'low'
        }

    def _calculate_retention(self, events: List[Dict]) -> Dict:
        """Calculate customer retention metrics"""
        
        # Group purchases by customer (inferred from revenue patterns)
        purchases = [e for e in events if e.get('event_type') == 'purchase']
        
        if not purchases:
            return {
                'repeat_purchase_rate': 0,
                'customer_retention': 'insufficient_data',
                'churn_rate': 0
            }

        # Estimate unique customers
        unique_customers = max(1, len(purchases) * 0.7)  # Rough estimate
        repeat_customers = max(0, len(purchases) - unique_customers)

        retention_rate = (repeat_customers / max(unique_customers, 1)) * 100

        return {
            'total_purchases': len(purchases),
            'estimated_unique_customers': int(unique_customers),
            'repeat_purchase_rate': round(retention_rate, 2),
            'churn_rate': round(100 - retention_rate, 2),
            'customer_lifetime_purchases': round(len(purchases) / max(unique_customers, 1), 2)
        }

    def _calculate_growth(self, events: List[Dict], time_period_days: int) -> Dict:
        """Calculate growth trends over time period"""
        
        # Group by week
        events_by_week = {}
        for event in events:
            timestamp_str = event.get('timestamp') or event.get('time', '')
            if not timestamp_str:
                continue
            try:
                event_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                week_key = event_date.isocalendar()[1]  # Week number
                if week_key not in events_by_week:
                    events_by_week[week_key] = 0
                events_by_week[week_key] += 1
            except:
                pass

        if len(events_by_week) < 2:
            return {
                'weeks_of_data': len(events_by_week),
                'growth_rate': 0,
                'trend': 'insufficient_data'
            }

        weeks = sorted(events_by_week.keys())
        first_week_events = events_by_week[weeks[0]]
        last_week_events = events_by_week[weeks[-1]]

        growth_rate = ((last_week_events - first_week_events) / max(first_week_events, 1)) * 100

        return {
            'weeks_of_data': len(weeks),
            'first_week_events': first_week_events,
            'last_week_events': last_week_events,
            'growth_rate': round(growth_rate, 2),
            'trend': 'growing' if growth_rate > 10 else 'stable' if growth_rate > -10 else 'declining'
        }

    def _calculate_conversion_funnel(self, events: List[Dict]) -> Dict:
        """Calculate conversion funnel: Impressions → Clicks → Verifications → Purchases"""
        
        impressions = len([e for e in events if e.get('event_type') == 'impression'])
        clicks = len([e for e in events if e.get('event_type') == 'click'])
        verifications = len([e for e in events if e.get('event_type') == 'verification'])
        purchases = len([e for e in events if e.get('event_type') == 'purchase'])

        # Calculate drop-off rates
        impression_to_click = (clicks / max(impressions, 1)) * 100 if impressions > 0 else 0
        click_to_verification = (verifications / max(clicks, 1)) * 100 if clicks > 0 else 0
        verification_to_purchase = (purchases / max(verifications, 1)) * 100 if verifications > 0 else 0
        overall_conversion = (purchases / max(impressions, 1)) * 100 if impressions > 0 else 0

        return {
            'stage_1_impressions': impressions,
            'stage_2_clicks': clicks,
            'stage_3_verifications': verifications,
            'stage_4_purchases': purchases,
            'impression_to_click_rate': round(impression_to_click, 2),
            'click_to_verification_rate': round(click_to_verification, 2),
            'verification_to_purchase_rate': round(verification_to_purchase, 2),
            'overall_conversion_rate': round(overall_conversion, 2)
        }

    def _calculate_trends(self, events: List[Dict]) -> Dict:
        """Calculate trend analysis"""
        
        recent_3_days = [e for e in events if self._days_ago(e) <= 3]
        recent_7_days = [e for e in events if self._days_ago(e) <= 7]
        recent_30_days = events

        conversion_3d = len([e for e in recent_3_days if e.get('event_type') == 'purchase'])
        conversion_7d = len([e for e in recent_7_days if e.get('event_type') == 'purchase'])
        conversion_30d = len([e for e in recent_30_days if e.get('event_type') == 'purchase'])

        return {
            'purchases_last_3_days': conversion_3d,
            'purchases_last_7_days': conversion_7d,
            'purchases_last_30_days': conversion_30d,
            'momentum': 'accelerating' if conversion_3d > conversion_7d / 2 else 'steady' if conversion_7d > conversion_30d / 4 else 'declining'
        }

    def _days_ago(self, event: Dict) -> int:
        """Calculate days since event"""
        timestamp_str = event.get('timestamp') or event.get('time', '')
        if not timestamp_str:
            return 999

        try:
            event_date = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return (datetime.utcnow() - event_date).days
        except:
            return 999

    def _get_influencer_tier(self, followers: int) -> str:
        """Get influencer tier"""
        if followers >= 1_000_000:
            return 'Mega'
        elif followers >= 100_000:
            return 'Macro'
        elif followers >= 10_000:
            return 'Micro'
        else:
            return 'Nano'

    def _get_tier_benchmarks(self, tier: str) -> Dict:
        """Get industry benchmarks by tier"""
        
        benchmarks = {
            'Mega': {
                'avg_aov': 75.00,
                'avg_roi': 150,
                'avg_conversion_rate': 1.5,
                'expected_growth_rate': 5.0
            },
            'Macro': {
                'avg_aov': 60.00,
                'avg_roi': 200,
                'avg_conversion_rate': 2.5,
                'expected_growth_rate': 10.0
            },
            'Micro': {
                'avg_aov': 45.00,
                'avg_roi': 250,
                'avg_conversion_rate': 4.0,
                'expected_growth_rate': 20.0
            },
            'Nano': {
                'avg_aov': 30.00,
                'avg_roi': 300,
                'avg_conversion_rate': 5.0,
                'expected_growth_rate': 30.0
            }
        }
        
        return benchmarks.get(tier, benchmarks['Nano'])

    def _compare_to_benchmark(self, aov: Dict, roi: Dict, engagement: Dict, benchmarks: Dict) -> Dict:
        """Compare metrics to tier benchmarks"""
        
        aov_value = aov.get('value', 0)
        roi_percent = roi.get('roi_percent', 0)
        conversion_rate = engagement.get('conversion_rate', 0)

        return {
            'aov_vs_benchmark': round((aov_value / benchmarks['avg_aov'] - 1) * 100, 2) if benchmarks['avg_aov'] > 0 else 0,
            'roi_vs_benchmark': round((roi_percent / benchmarks['avg_roi'] - 1) * 100, 2) if benchmarks['avg_roi'] > 0 else 0,
            'conversion_vs_benchmark': round((conversion_rate / benchmarks['avg_conversion_rate'] - 1) * 100, 2) if benchmarks['avg_conversion_rate'] > 0 else 0,
            'overall_performance': 'exceeds_tier' if all([
                aov_value > benchmarks['avg_aov'],
                roi_percent > benchmarks['avg_roi'],
                conversion_rate > benchmarks['avg_conversion_rate']
            ]) else 'meets_tier' if all([
                aov_value >= benchmarks['avg_aov'] * 0.8,
                roi_percent >= benchmarks['avg_roi'] * 0.8,
                conversion_rate >= benchmarks['avg_conversion_rate'] * 0.8
            ]) else 'below_tier'
        }

    def _calculate_health_score(self, aov: Dict, roi: Dict, retention: Dict, engagement: Dict) -> int:
        """Calculate overall health score 0-100"""
        
        score = 50

        # AOV contribution
        if aov.get('value', 0) > 50:
            score += 20
        elif aov.get('value', 0) > 30:
            score += 10

        # ROI contribution
        if roi.get('roi_percent', 0) > 100:
            score += 20
        elif roi.get('roi_percent', 0) > 50:
            score += 10

        # Retention contribution
        if retention.get('repeat_purchase_rate', 0) > 20:
            score += 10
        elif retention.get('repeat_purchase_rate', 0) > 10:
            score += 5

        # Engagement contribution
        if engagement.get('conversion_rate', 0) > 3:
            score += 10
        elif engagement.get('conversion_rate', 0) > 1:
            score += 5

        return min(score, 100)

    def _get_health_status(self, score: int) -> str:
        """Get health status from score"""
        if score >= 80:
            return 'Excellent'
        elif score >= 60:
            return 'Good'
        elif score >= 40:
            return 'Fair'
        else:
            return 'Poor'

    def _empty_analytics_report(self, handle: str) -> Dict:
        """Generate empty report for insufficient data"""
        return {
            'handle': handle,
            'events_analyzed': 0,
            'status': 'insufficient_data',
            'message': 'Not enough event data to calculate analytics'
        }

    def compare_influencers(self, analytics_reports: List[Dict]) -> Dict:
        """Compare multiple influencers based on analytics"""
        
        if not analytics_reports:
            return {}

        # Rank by health score
        ranked = sorted(analytics_reports, 
                       key=lambda x: x.get('health_score', 0), 
                       reverse=True)

        return {
            'ranked_by_health': ranked,
            'summary': {
                'total_influencers': len(analytics_reports),
                'avg_health_score': round(sum(r.get('health_score', 0) for r in analytics_reports) / len(analytics_reports), 2),
                'avg_aov': round(sum(r.get('aov', {}).get('value', 0) for r in analytics_reports) / len(analytics_reports), 2),
                'avg_roi': round(sum(r.get('roi', {}).get('roi_percent', 0) for r in analytics_reports) / len(analytics_reports), 2),
                'total_revenue_generated': round(sum(r.get('roi', {}).get('revenue_generated', 0) for r in analytics_reports), 2)
            }
        }
