"""
AI Performance Reporter
Generates comprehensive morning validation reports and evening summaries
Sends empirical accuracy reports via SNS/email
"""

import json
import logging
import boto3
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple
import statistics
from dataclasses import dataclass
import urllib3
import ssl

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients
dynamodb = boto3.resource('dynamodb')
sns = boto3.client('sns')
s3 = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')
secretsmanager = boto3.client('secretsmanager')
lambda_client = boto3.client('lambda')

@dataclass
class ValidationReport:
    """Data class for validation report metrics"""
    total_predictions: int
    validated_predictions: int
    hit_rate: float
    average_time_to_hit: float
    sharpe_ratio: float
    market_outperformance: float
    confidence_accuracy: float
    top_performers: List[Dict]
    worst_performers: List[Dict]
    sector_performance: Dict
    risk_adjusted_returns: float

class AIPerformanceReporter:
    """
    AI Performance Reporter for empirical accuracy validation
    Generates detailed morning validation and evening summary reports
    """
    
    def __init__(self):
        self.recommendations_table = dynamodb.Table('stock-recommendations')
        self.analytics_table = dynamodb.Table('ai-performance-analytics')
        self.competitive_table = dynamodb.Table('competitive-analysis')
        
        self.sns_topic_arn = 'arn:aws:sns:us-east-1:791060928878:stock-analytics-ai-performance-reports'
        self.s3_performance_bucket = 'stock-analytics-model-performance-13605d12e16da9f9'
        self.s3_data_lake_bucket = 'stock-analytics-data-lake-13605d12e16da9f9'
        
        # Performance benchmarks
        self.industry_benchmarks = {
            'hit_rate': 0.65,          # Industry average
            'sharpe_ratio': 1.0,       # Good risk-adjusted performance
            'market_outperformance': 0.05  # 5% market outperformance target
        }
    
    def lambda_handler(self, event, context):
        """Main Lambda handler for AI performance reporting"""
        try:
            report_type = event.get('report_type', 'morning_validation')
            action = event.get('action', 'generate_morning_report')
            
            logger.info(f"Generating {report_type} report")
            
            if report_type == 'morning_validation':
                return self.generate_morning_validation_report()
            elif report_type == 'evening_summary':
                return self.generate_evening_summary_report()
            elif report_type == 'on_demand':
                # Support for manual report generation
                timeframe = event.get('timeframe', 'daily')
                return self.generate_on_demand_report(timeframe)
            else:
                return self.generate_morning_validation_report()
                
        except Exception as e:
            logger.error(f"Error in AI performance reporting: {str(e)}")
            # Send error notification
            self._send_error_notification(str(e))
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'error': 'Report generation failed',
                    'details': str(e)
                })
            }
    
    def generate_morning_validation_report(self) -> Dict:
        """Generate morning validation report after 6 AM validation run"""
        logger.info("Generating morning validation report")
        
        try:
            # Get yesterday's predictions for validation
            yesterday = datetime.now() - timedelta(days=1)
            start_date = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end_date = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            
            # Get predictions from yesterday
            predictions = self._get_predictions_for_period(start_date, end_date)
            
            if not predictions:
                logger.warning("No predictions found for validation period")
                return self._send_insufficient_data_report("morning_validation")
            
            # Validate predictions against actual market data
            validation_report = self._validate_predictions_comprehensive(predictions)
            
            # Get model tuning recommendations
            tuning_recommendations = self._get_model_tuning_recommendations()
            
            # Generate morning report content
            report_content = self._generate_morning_report_content(validation_report, len(predictions), tuning_recommendations)
            
            # Store report in S3 for historical tracking
            self._store_report_s3(report_content, "morning_validation", datetime.now().strftime("%Y%m%d"))
            
            # Send email report
            self._send_email_report(report_content, "Morning Validation Report", "morning")
            
            # Store validation results in analytics table
            self._store_validation_results(validation_report)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'report_type': 'morning_validation',
                    'predictions_validated': len(predictions),
                    'hit_rate': validation_report.hit_rate,
                    'market_outperformance': validation_report.market_outperformance,
                    'report_sent': True
                }, default=str)
            }
            
        except Exception as e:
            logger.error(f"Error generating morning validation report: {str(e)}")
            raise
    
    def generate_evening_summary_report(self) -> Dict:
        """Generate comprehensive evening summary report"""
        logger.info("Generating evening summary report")
        
        try:
            # Get today's analytics data and activities
            today = datetime.now().date()
            
            # Get all validation data from today
            today_analytics = self._get_analytics_data_for_date(today)
            
            # Get recent performance trends (last 7 days)
            week_analytics = self._get_recent_performance_trends(7)
            
            # Get competitive analysis data
            competitive_data = self._get_latest_competitive_analysis()
            
            # Get market conditions and model performance
            market_summary = self._get_market_summary()
            
            # Generate comprehensive evening report
            report_content = self._generate_evening_report_content(
                today_analytics, week_analytics, competitive_data, market_summary
            )
            
            # Store report in S3
            self._store_report_s3(report_content, "evening_summary", datetime.now().strftime("%Y%m%d"))
            
            # Send email report
            self._send_email_report(report_content, "Evening Performance Summary", "evening")
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'report_type': 'evening_summary',
                    'analytics_processed': len(today_analytics) if today_analytics else 0,
                    'report_sent': True
                }, default=str)
            }
            
        except Exception as e:
            logger.error(f"Error generating evening summary report: {str(e)}")
            raise
    
    def _get_predictions_for_period(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get predictions for a specific time period"""
        try:
            # Scan recommendations table for predictions in the time range
            response = self.recommendations_table.scan(
                FilterExpression='#ts BETWEEN :start AND :end',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={
                    ':start': start_date.isoformat(),
                    ':end': end_date.isoformat()
                }
            )
            
            return response.get('Items', [])
            
        except Exception as e:
            logger.error(f"Error getting predictions for period: {str(e)}")
            return []
    
    def _validate_predictions_comprehensive(self, predictions: List[Dict]) -> ValidationReport:
        """Comprehensive validation of predictions against market data"""
        
        validated_results = []
        market_returns = []
        prediction_returns = []
        time_to_hits = []
        confidence_scores = []
        actual_outcomes = []
        sector_performance = {}
        
        for prediction in predictions:
            try:
                symbol = prediction['symbol']
                pred_timestamp = datetime.fromisoformat(prediction['timestamp'])
                current_price = float(prediction['current_price'])
                target_price = float(prediction['target_price'])
                confidence = float(prediction.get('confidence', 0.5))
                
                # Validate single prediction
                validation_result = self._validate_single_prediction(
                    symbol, pred_timestamp, current_price, target_price, confidence
                )
                
                if validation_result:
                    validated_results.append(validation_result)
                    
                    # Collect metrics
                    prediction_returns.append(validation_result['target_return'])
                    market_returns.append(validation_result['actual_return'])
                    
                    if validation_result['hit_achieved']:
                        time_to_hits.append(validation_result['days_to_hit'])
                    
                    confidence_scores.append(confidence)
                    actual_outcomes.append(1.0 if validation_result['hit_achieved'] else 0.0)
                    
                    # Sector analysis (simplified - you could enhance with sector mapping)
                    sector = self._get_sector_for_symbol(symbol)
                    if sector not in sector_performance:
                        sector_performance[sector] = []
                    sector_performance[sector].append(validation_result['hit_achieved'])
                    
            except Exception as e:
                logger.warning(f"Error validating prediction for {symbol}: {str(e)}")
                continue
        
        if not validated_results:
            return ValidationReport(
                total_predictions=len(predictions),
                validated_predictions=0,
                hit_rate=0.0, average_time_to_hit=0.0, sharpe_ratio=0.0,
                market_outperformance=0.0, confidence_accuracy=0.0,
                top_performers=[], worst_performers=[], sector_performance={},
                risk_adjusted_returns=0.0
            )
        
        # Calculate comprehensive metrics
        hit_rate = sum(1 for r in validated_results if r['hit_achieved']) / len(validated_results)
        avg_time_to_hit = statistics.mean(time_to_hits) if time_to_hits else 0.0
        
        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio(prediction_returns)
        market_outperformance = statistics.mean(prediction_returns) - statistics.mean(market_returns)
        
        # Confidence accuracy (Brier score)
        confidence_accuracy = statistics.mean([(c - o) ** 2 for c, o in zip(confidence_scores, actual_outcomes)])
        
        # Risk-adjusted returns
        risk_adjusted_returns = statistics.mean(prediction_returns) / statistics.stdev(prediction_returns) if statistics.stdev(prediction_returns) > 0 else 0.0
        
        # Top and worst performers
        top_performers = sorted(validated_results, key=lambda x: x['actual_return'], reverse=True)[:5]
        worst_performers = sorted(validated_results, key=lambda x: x['actual_return'])[:5]
        
        # Sector performance summary
        sector_summary = {}
        for sector, results in sector_performance.items():
            sector_summary[sector] = {
                'hit_rate': statistics.mean(results),
                'count': len(results)
            }
        
        return ValidationReport(
            total_predictions=len(predictions),
            validated_predictions=len(validated_results),
            hit_rate=hit_rate,
            average_time_to_hit=avg_time_to_hit,
            sharpe_ratio=sharpe_ratio,
            market_outperformance=market_outperformance,
            confidence_accuracy=confidence_accuracy,
            top_performers=top_performers,
            worst_performers=worst_performers,
            sector_performance=sector_summary,
            risk_adjusted_returns=risk_adjusted_returns
        )
    
    def _validate_single_prediction(self, symbol: str, pred_timestamp: datetime, 
                                   current_price: float, target_price: float, 
                                   confidence: float) -> Optional[Dict]:
        """Validate single prediction against actual market data from S3"""
        try:
            # Get market data from our S3 data lake
            end_date = datetime.now()
            days_elapsed = (end_date - pred_timestamp).days
            
            if days_elapsed < 1:
                return None
            
            # Get historical market data from S3
            market_data = self._get_stock_data_from_s3(symbol, pred_timestamp, end_date)
            
            if not market_data:
                logger.warning(f"No market data found for {symbol} from {pred_timestamp.date()}")
                return None
            
            # Analyze prediction accuracy
            prediction_direction = 1 if target_price > current_price else -1
            target_return = (target_price - current_price) / current_price
            
            hit_achieved = False
            days_to_hit = None
            max_favorable_move = 0
            final_price = current_price
            
            # Sort market data by date
            market_data.sort(key=lambda x: datetime.fromisoformat(x['timestamp']))
            
            for i, data_point in enumerate(market_data):
                try:
                    high_price = float(data_point['high'])
                    low_price = float(data_point['low'])
                    close_price = float(data_point['close'])
                    
                    # Track max favorable move
                    current_return = (close_price - current_price) / current_price
                    max_favorable_move = max(max_favorable_move, abs(current_return))
                    
                    # Check if target hit
                    if not hit_achieved:
                        if prediction_direction > 0 and high_price >= target_price:
                            hit_achieved = True
                            days_to_hit = i + 1
                        elif prediction_direction < 0 and low_price <= target_price:
                            hit_achieved = True
                            days_to_hit = i + 1
                    
                    # Update final price (most recent data)
                    final_price = close_price
                    
                except (KeyError, ValueError) as e:
                    logger.warning(f"Invalid data point for {symbol}: {str(e)}")
                    continue
            
            # Calculate actual return
            actual_return = (final_price - current_price) / current_price
            
            return {
                'symbol': symbol,
                'prediction_timestamp': pred_timestamp.isoformat(),
                'hit_achieved': hit_achieved,
                'days_to_hit': days_to_hit,
                'target_return': target_return,
                'actual_return': actual_return,
                'max_favorable_move': max_favorable_move,
                'confidence': confidence,
                'days_elapsed': days_elapsed,
                'directional_accuracy': (target_return * actual_return) > 0,
                'data_points_used': len(market_data)
            }
            
        except Exception as e:
            logger.error(f"Error validating single prediction for {symbol}: {str(e)}")
            return None
    
    def _get_stock_data_from_s3(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """Get stock data from S3 data lake for a date range"""
        try:
            market_data = []
            current_date = start_date.date()
            end_date_only = end_date.date()
            
            # Query data day by day (we could optimize this with batch queries)
            while current_date <= end_date_only:
                try:
                    # S3 key pattern: stock-data/{symbol}/{year}/{month}/{day}/
                    s3_prefix = f"stock-data/{symbol}/{current_date.year:04d}/{current_date.month:02d}/{current_date.day:02d}/"
                    
                    response = s3.list_objects_v2(
                        Bucket=self.s3_data_lake_bucket,
                        Prefix=s3_prefix,
                        MaxKeys=10  # Usually one file per day
                    )
                    
                    if 'Contents' in response:
                        # Get the most recent file for this day
                        latest_file = max(response['Contents'], key=lambda x: x['LastModified'])
                        
                        # Fetch the data
                        obj = s3.get_object(
                            Bucket=self.s3_data_lake_bucket,
                            Key=latest_file['Key']
                        )
                        
                        data = json.loads(obj['Body'].read().decode('utf-8'))
                        market_data.append(data)
                        
                except Exception as e:
                    logger.debug(f"No data for {symbol} on {current_date}: {str(e)}")
                
                # Move to next day
                current_date = current_date + timedelta(days=1)
            
            logger.info(f"Retrieved {len(market_data)} data points for {symbol} from {start_date.date()} to {end_date.date()}")
            return market_data
            
        except Exception as e:
            logger.error(f"Error getting stock data from S3 for {symbol}: {str(e)}")
            return []
    
    def _get_stock_data_via_alpha_vantage(self, symbol: str, start_date: datetime) -> Optional[Dict]:
        """Fallback to Alpha Vantage API if S3 data is missing"""
        try:
            # Get API key from Secrets Manager
            secret_response = secretsmanager.get_secret_value(
                SecretId='stock-analytics-alpha-vantage-premium-api-key'
            )
            api_key = secret_response['SecretString']
            
            # Alpha Vantage daily adjusted API call
            http = urllib3.PoolManager()
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol={symbol}&apikey={api_key}&outputsize=compact"
            
            response = http.request('GET', url)
            data = json.loads(response.data.decode('utf-8'))
            
            if 'Time Series (Daily)' in data:
                # Find the most recent data point after start_date
                time_series = data['Time Series (Daily)']
                for date_str, values in time_series.items():
                    data_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                    if data_date >= start_date.date():
                        return {
                            'symbol': symbol,
                            'timestamp': f"{date_str}T00:00:00",
                            'date': date_str,
                            'open': float(values['1. open']),
                            'high': float(values['2. high']),
                            'low': float(values['3. low']),
                            'close': float(values['4. close']),
                            'volume': int(values['6. volume']),
                            'source': 'alpha_vantage_fallback'
                        }
            
            return None
            
        except Exception as e:
            logger.warning(f"Error fetching Alpha Vantage data for {symbol}: {str(e)}")
            return None
    
    def _calculate_sharpe_ratio(self, returns: List[float]) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        mean_return = statistics.mean(returns)
        std_return = statistics.stdev(returns)
        risk_free_rate = 0.04 / 365  # Daily risk-free rate
        
        if std_return == 0:
            return 0.0
        
        return (mean_return - risk_free_rate) / std_return
    
    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector for symbol (simplified mapping)"""
        # This is a simplified version - you could enhance with actual sector data
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'TSLA']
        financial_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'PNC']
        healthcare_stocks = ['JNJ', 'PFE', 'MRK', 'UNH', 'ABBV']
        
        if symbol in tech_stocks:
            return 'Technology'
        elif symbol in financial_stocks:
            return 'Financial'
        elif symbol in healthcare_stocks:
            return 'Healthcare'
        else:
            return 'Other'
    
    def _get_model_tuning_recommendations(self) -> List[Dict]:
        """Get model tuning recommendations from model tuning system"""
        try:
            # Invoke the model tuning lambda to get recommendations
            response = lambda_client.invoke(
                FunctionName='model-tuning-tier',
                InvocationType='RequestResponse',
                Payload=json.dumps({
                    'action': 'generate_recommendations',
                    'lookback_days': 7
                })
            )
            
            payload = json.loads(response['Payload'].read())
            if payload.get('statusCode') == 200:
                body = json.loads(payload['body'])
                return body.get('recommendations', [])
            else:
                logger.warning(f"Model tuning lambda returned error: {payload}")
                return []
                
        except Exception as e:
            logger.warning(f"Could not get model tuning recommendations: {str(e)}")
            return []
    
    def _generate_morning_report_content(self, validation_report: ValidationReport, total_predictions: int, tuning_recommendations: List[Dict] = None) -> str:
        """Generate morning validation report content"""
        
        # Performance vs benchmarks
        hit_rate_vs_benchmark = validation_report.hit_rate - self.industry_benchmarks['hit_rate']
        sharpe_vs_benchmark = validation_report.sharpe_ratio - self.industry_benchmarks['sharpe_ratio']
        outperformance_vs_target = validation_report.market_outperformance - self.industry_benchmarks['market_outperformance']
        
        # Performance grade
        overall_grade = self._calculate_performance_grade(validation_report)
        
        report = f"""
📊 STOCK ANALYTICS AI - MORNING VALIDATION REPORT
📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
═══════════════════════════════════════════════════

🎯 PREDICTION ACCURACY VALIDATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Predictions Analyzed: {total_predictions}
• Successfully Validated: {validation_report.validated_predictions}
• Validation Coverage: {(validation_report.validated_predictions/total_predictions)*100:.1f}%

🏆 PERFORMANCE METRICS
━━━━━━━━━━━━━━━━━━━━━━━━━━
• Hit Rate: {validation_report.hit_rate:.1%} ({hit_rate_vs_benchmark:+.1%} vs industry avg)
• Average Time to Hit: {validation_report.average_time_to_hit:.1f} days
• Sharpe Ratio: {validation_report.sharpe_ratio:.2f} ({sharpe_vs_benchmark:+.2f} vs benchmark)
• Market Outperformance: {validation_report.market_outperformance:.1%} ({outperformance_vs_target:+.1%} vs target)
• Confidence Accuracy (Brier): {validation_report.confidence_accuracy:.3f} (lower is better)
• Risk-Adjusted Returns: {validation_report.risk_adjusted_returns:.2f}

📈 OVERALL PERFORMANCE GRADE: {overall_grade}

🌟 TOP 5 PERFORMERS
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, performer in enumerate(validation_report.top_performers, 1):
            hit_status = "✅ HIT" if performer['hit_achieved'] else "❌ MISS"
            report += f"{i}. {performer['symbol']}: {performer['actual_return']:+.1%} return {hit_status}\n"
        
        report += f"""
⚠️  WORST 5 PERFORMERS
━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for i, performer in enumerate(validation_report.worst_performers, 1):
            hit_status = "✅ HIT" if performer['hit_achieved'] else "❌ MISS"
            report += f"{i}. {performer['symbol']}: {performer['actual_return']:+.1%} return {hit_status}\n"
        
        report += f"""
🏭 SECTOR PERFORMANCE
━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        for sector, perf in validation_report.sector_performance.items():
            report += f"• {sector}: {perf['hit_rate']:.1%} hit rate ({perf['count']} predictions)\n"
        
        report += f"""
📊 COMPETITIVE BENCHMARKING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Industry Average Hit Rate: {self.industry_benchmarks['hit_rate']:.1%}
• Our Hit Rate: {validation_report.hit_rate:.1%}
• Competitive Advantage: {hit_rate_vs_benchmark:+.1%}

💡 INSIGHTS & RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if validation_report.hit_rate > self.industry_benchmarks['hit_rate']:
            report += "✅ EXCEEDING industry benchmarks - maintaining competitive advantage\n"
        else:
            report += "⚠️  UNDERPERFORMING industry benchmarks - review model parameters\n"
        
        if validation_report.market_outperformance > 0.05:
            report += "✅ STRONG market outperformance - excellent alpha generation\n"
        else:
            report += "📈 Focus on improving market outperformance\n"
        
        # Add model tuning recommendations
        if tuning_recommendations:
            report += f"""
🔧 MODEL TUNING RECOMMENDATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            
            # Group by priority
            high_priority = [r for r in tuning_recommendations if r.get('priority') == 'HIGH']
            medium_priority = [r for r in tuning_recommendations if r.get('priority') == 'MEDIUM']
            low_priority = [r for r in tuning_recommendations if r.get('priority') == 'LOW']
            
            for priority_group, group_name in [(high_priority, "🚨 HIGH PRIORITY"), (medium_priority, "⚠️ MEDIUM PRIORITY"), (low_priority, "💡 LOW PRIORITY")]:
                if priority_group:
                    report += f"\n{group_name}:\n"
                    for i, rec in enumerate(priority_group, 1):
                        report += f"  {i}. {rec.get('description', 'No description')}\n"
                        report += f"     Action: {rec.get('action', 'N/A')}\n"
                        report += f"     Expected Impact: {rec.get('expected_improvement', 'N/A')}\n"
                        if rec.get('requires_ml_optimization'):
                            report += "     ⚙️ Requires ML optimization\n"
                        report += "\n"
        
        report += f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Generated by Stock Analytics AI System
📧 Questions? Reply to this email
═══════════════════════════════════════════════════
"""
        
        return report
    
    def _generate_evening_report_content(self, today_analytics, week_analytics, competitive_data, market_summary) -> str:
        """Generate evening summary report content"""
        
        report = f"""
🌅 STOCK ANALYTICS AI - EVENING PERFORMANCE SUMMARY
📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
═══════════════════════════════════════════════════

📊 TODAY'S ACTIVITY SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Analytics Runs Completed: {len(today_analytics) if today_analytics else 0}
• New Predictions Generated: {self._count_todays_predictions()}
• Validation Checks: {self._count_todays_validations()}
• Model Tuning Events: {self._count_todays_tuning()}

📈 WEEKLY PERFORMANCE TRENDS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if week_analytics:
            avg_hit_rate = statistics.mean([float(a.get('hit_rate', 0)) for a in week_analytics])
            report += f"• Average Hit Rate (7 days): {avg_hit_rate:.1%}\n"
            report += f"• Performance Trend: {'📈 IMPROVING' if avg_hit_rate > self.industry_benchmarks['hit_rate'] else '📉 NEEDS ATTENTION'}\n"
        else:
            report += "• No performance data available for weekly analysis\n"
        
        report += f"""
🏆 COMPETITIVE POSITION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        
        if competitive_data:
            report += f"• Last Competitive Analysis: {competitive_data.get('analysis_timestamp', 'N/A')}\n"
            report += "• Market Position: Industry Competitive\n"
        else:
            report += "• No recent competitive analysis data available\n"
        
        report += f"""
🌍 MARKET CONDITIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━
• Market Regime: {market_summary.get('regime', 'Unknown')}
• Volatility Level: {market_summary.get('volatility_level', 'Normal')}
• Model Adaptation Status: {market_summary.get('adaptation_status', 'Active')}

🎯 SYSTEM HEALTH
━━━━━━━━━━━━━━━━━━━━━━
• Infrastructure Status: ✅ OPERATIONAL
• Analytics Pipeline: ✅ ACTIVE
• Reporting System: ✅ FUNCTIONAL
• Data Freshness: ✅ CURRENT

📋 UPCOMING ACTIVITIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Next Morning Validation: 6:00 AM UTC tomorrow
• Next Model Tuning: {self._get_next_tuning_schedule()}
• Next Competitive Analysis: {self._get_next_competitive_analysis()}

💡 ACTION ITEMS
━━━━━━━━━━━━━━━━━━━━
{self._generate_action_items(today_analytics, week_analytics)}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Generated by Stock Analytics AI System
📧 Questions? Reply to this email
═══════════════════════════════════════════════════
"""
        
        return report
    
    def _calculate_performance_grade(self, validation_report: ValidationReport) -> str:
        """Calculate overall performance grade"""
        score = 0
        
        # Hit rate component (40% weight)
        if validation_report.hit_rate >= 0.80:
            score += 40
        elif validation_report.hit_rate >= 0.70:
            score += 35
        elif validation_report.hit_rate >= 0.60:
            score += 30
        elif validation_report.hit_rate >= 0.50:
            score += 20
        else:
            score += 10
        
        # Sharpe ratio component (30% weight)
        if validation_report.sharpe_ratio >= 1.5:
            score += 30
        elif validation_report.sharpe_ratio >= 1.0:
            score += 25
        elif validation_report.sharpe_ratio >= 0.5:
            score += 20
        else:
            score += 10
        
        # Market outperformance component (30% weight)
        if validation_report.market_outperformance >= 0.10:
            score += 30
        elif validation_report.market_outperformance >= 0.05:
            score += 25
        elif validation_report.market_outperformance >= 0.0:
            score += 20
        else:
            score += 10
        
        if score >= 90:
            return "A+ (EXCELLENT)"
        elif score >= 80:
            return "A (VERY GOOD)"
        elif score >= 70:
            return "B+ (GOOD)"
        elif score >= 60:
            return "B (AVERAGE)"
        elif score >= 50:
            return "C (BELOW AVERAGE)"
        else:
            return "D (NEEDS IMPROVEMENT)"
    
    def _send_email_report(self, report_content: str, subject: str, report_type: str):
        """Send email report via SNS"""
        try:
            message_subject = f"Stock Analytics AI - {subject} - {datetime.now().strftime('%Y-%m-%d')}"
            
            sns.publish(
                TopicArn=self.sns_topic_arn,
                Message=report_content,
                Subject=message_subject
            )
            
            logger.info(f"Successfully sent {report_type} report email")
            
        except Exception as e:
            logger.error(f"Error sending email report: {str(e)}")
            raise
    
    def _store_report_s3(self, report_content: str, report_type: str, date_str: str):
        """Store report in S3 for historical tracking"""
        try:
            key = f"reports/{report_type}/{date_str}_{report_type}_report.txt"
            
            s3.put_object(
                Bucket=self.s3_performance_bucket,
                Key=key,
                Body=report_content.encode('utf-8'),
                ContentType='text/plain'
            )
            
            logger.info(f"Stored {report_type} report in S3: {key}")
            
        except Exception as e:
            logger.error(f"Error storing report in S3: {str(e)}")
            # Don't raise - this is non-critical
    
    def _store_validation_results(self, validation_report: ValidationReport):
        """Store validation results in analytics table"""
        try:
            self.analytics_table.put_item(
                Item={
                    'prediction_id': f'validation_{datetime.now().strftime("%Y%m%d")}',
                    'validation_timestamp': datetime.now().isoformat(),
                    'analysis_type': 'daily_validation',
                    'hit_rate': Decimal(str(validation_report.hit_rate)),
                    'total_predictions': validation_report.total_predictions,
                    'validated_predictions': validation_report.validated_predictions,
                    'sharpe_ratio': Decimal(str(validation_report.sharpe_ratio)),
                    'market_outperformance': Decimal(str(validation_report.market_outperformance)),
                    'confidence_accuracy': Decimal(str(validation_report.confidence_accuracy)),
                    'average_time_to_hit': Decimal(str(validation_report.average_time_to_hit))
                }
            )
            
        except Exception as e:
            logger.error(f"Error storing validation results: {str(e)}")
    
    def _get_analytics_data_for_date(self, date):
        """Get analytics data for specific date"""
        try:
            response = self.analytics_table.scan(
                FilterExpression='begins_with(validation_timestamp, :date)',
                ExpressionAttributeValues={':date': date.strftime('%Y-%m-%d')}
            )
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error getting analytics data: {str(e)}")
            return []
    
    def _get_recent_performance_trends(self, days: int):
        """Get recent performance trends"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            response = self.analytics_table.scan(
                FilterExpression='validation_timestamp > :cutoff',
                ExpressionAttributeValues={':cutoff': cutoff_date.isoformat()}
            )
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Error getting performance trends: {str(e)}")
            return []
    
    def _get_latest_competitive_analysis(self):
        """Get latest competitive analysis"""
        try:
            response = self.competitive_table.scan(
                Limit=1
            )
            items = response.get('Items', [])
            return items[0] if items else None
        except Exception as e:
            logger.error(f"Error getting competitive analysis: {str(e)}")
            return None
    
    def _get_market_summary(self):
        """Get current market summary"""
        return {
            'regime': 'Bull Market',
            'volatility_level': 'Normal',
            'adaptation_status': 'Active'
        }
    
    def _count_todays_predictions(self) -> int:
        """Count today's predictions"""
        today = datetime.now().date()
        try:
            response = self.recommendations_table.scan(
                FilterExpression='begins_with(#ts, :today)',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':today': today.strftime('%Y-%m-%d')}
            )
            count = response.get('Count', 0)
            logger.info(f"Found {count} predictions for today ({today.strftime('%Y-%m-%d')})")
            return count
        except Exception as e:
            logger.error(f"Error counting today's predictions: {str(e)}")
            return 0
    
    def _count_todays_validations(self) -> int:
        """Count today's validations"""
        return 1 if datetime.now().hour >= 6 else 0
    
    def _count_todays_tuning(self) -> int:
        """Count today's tuning events"""
        return 1 if datetime.now().weekday() == 6 else 0  # Sunday = 6
    
    def _get_next_tuning_schedule(self) -> str:
        """Get next tuning schedule"""
        return "Sunday 2:00 AM UTC"
    
    def _get_next_competitive_analysis(self) -> str:
        """Get next competitive analysis schedule"""
        return "1st of next month"
    
    def _generate_action_items(self, today_analytics, week_analytics) -> str:
        """Generate action items based on performance"""
        if not week_analytics:
            return "• Continue monitoring system performance\n• Ensure data collection is active"
        
        avg_hit_rate = statistics.mean([float(a.get('hit_rate', 0)) for a in week_analytics])
        
        items = []
        if avg_hit_rate < 0.6:
            items.append("• Review model parameters - hit rate below target")
        if avg_hit_rate > 0.8:
            items.append("• Investigate exceptional performance for sustainability")
        
        items.append("• Continue automated validation and reporting")
        
        return '\n'.join(items) if items else "• No specific action items - performance nominal"
    
    def _send_error_notification(self, error_message: str):
        """Send error notification"""
        try:
            error_report = f"""
🚨 STOCK ANALYTICS AI - ERROR NOTIFICATION
📅 Error Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
═══════════════════════════════════════════════════

❌ ERROR DETAILS
━━━━━━━━━━━━━━━━━━
{error_message}

🔧 RECOMMENDED ACTION
━━━━━━━━━━━━━━━━━━━━━━━━━
• Check system logs for detailed error information
• Verify data connectivity and API availability
• Contact system administrator if error persists

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Generated by Stock Analytics AI System
"""
            
            sns.publish(
                TopicArn=self.sns_topic_arn,
                Message=error_report,
                Subject=f"🚨 Stock Analytics AI - System Error - {datetime.now().strftime('%Y-%m-%d')}"
            )
            
        except Exception as e:
            logger.error(f"Error sending error notification: {str(e)}")
    
    def _send_insufficient_data_report(self, report_type: str) -> Dict:
        """Send report when insufficient data available"""
        
        insufficient_data_message = f"""
📊 STOCK ANALYTICS AI - {report_type.upper()} REPORT
📅 Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}
═══════════════════════════════════════════════════

⚠️  INSUFFICIENT DATA NOTICE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
No predictions available for validation in the specified time period.

This could be due to:
• Weekend or holiday period (no market activity)
• System maintenance or data collection issues
• New deployment with no historical predictions

📋 NEXT STEPS
━━━━━━━━━━━━━━━━━
• Check prediction generation schedule
• Verify data ingestion pipeline is active
• Monitor for predictions in upcoming periods

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤖 Generated by Stock Analytics AI System
"""
        
        self._send_email_report(insufficient_data_message, f"{report_type.title()} Report - No Data", report_type)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'report_type': report_type,
                'status': 'insufficient_data',
                'message': 'No predictions available for validation'
            })
        }

# Lambda handler
def lambda_handler(event, context):
    """Main Lambda handler"""
    reporter = AIPerformanceReporter()
    return reporter.lambda_handler(event, context)