#!/usr/bin/env python3
"""
Market Validation Engine
Validates model predictions against real S&P 500 returns and competitive benchmarks
"""

import yfinance as yf
import numpy as np
import pandas as pd
import json
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')

class MarketValidationEngine:
    """
    Validates model performance against market benchmarks and competitive standards
    """
    
    def __init__(self):
        self.recommendations_table = dynamodb.Table('stock-recommendations')
        self.analytics_table = dynamodb.Table('ai-performance-analytics')
        self.validation_bucket = 'stock-analytics-model-performance-13605d12e16da9f9'
        
        # Market benchmarks
        self.benchmarks = {
            'SPY': 'S&P 500',
            'QQQ': 'NASDAQ-100',
            'IWM': 'Russell 2000',
            'VTI': 'Total Stock Market',
            'VOO': 'S&P 500 (Vanguard)'
        }
        
        # Industry competitive standards
        self.competitive_benchmarks = {
            'hit_rate_minimum': 0.55,      # Minimum to be competitive
            'hit_rate_good': 0.65,         # Good performance
            'hit_rate_excellent': 0.75,    # Excellent performance
            'sharpe_ratio_minimum': 0.8,   # Risk-adjusted return minimum
            'information_ratio_good': 1.2, # Good active management
            'max_drawdown_acceptable': 0.15 # Maximum acceptable drawdown
        }
    
    def validate_against_market(self, validation_period_days: int = 90) -> Dict:
        """
        Comprehensive validation against market benchmarks
        """
        logger.info(f"Starting market validation for {validation_period_days} days")
        
        try:
            # Get model predictions from recent period
            model_predictions = self._get_recent_predictions(validation_period_days)
            
            if not model_predictions:
                return {'error': 'No recent predictions found for validation'}
            
            # Get benchmark data for same period
            benchmark_data = self._get_benchmark_data(validation_period_days)
            
            # Calculate model performance
            model_performance = self._calculate_model_performance(model_predictions)
            
            # Calculate benchmark performance  
            benchmark_performance = self._calculate_benchmark_performance(benchmark_data)
            
            # Compare model vs benchmarks
            comparison_results = self._compare_model_vs_benchmarks(
                model_performance, benchmark_performance
            )
            
            # Validate against competitive standards
            competitive_assessment = self._assess_competitive_standing(model_performance)
            
            # Generate improvement recommendations
            improvement_plan = self._generate_improvement_plan(
                comparison_results, competitive_assessment
            )
            
            # Store validation results
            validation_results = {
                'validation_timestamp': datetime.now().isoformat(),
                'validation_period_days': validation_period_days,
                'model_performance': model_performance,
                'benchmark_performance': benchmark_performance,
                'comparison_results': comparison_results,
                'competitive_assessment': competitive_assessment,
                'improvement_plan': improvement_plan,
                'predictions_analyzed': len(model_predictions)
            }
            
            self._store_validation_results(validation_results)
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error in market validation: {e}")
            return {'error': str(e)}
    
    def _get_recent_predictions(self, days: int) -> List[Dict]:
        """Get recent model predictions with outcomes"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            response = self.recommendations_table.scan(
                FilterExpression='#ts >= :cutoff',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':cutoff': cutoff_date.isoformat()}
            )
            
            predictions = []
            for item in response.get('Items', []):
                # Convert Decimal to float
                prediction = {}
                for k, v in item.items():
                    if isinstance(v, Decimal):
                        prediction[k] = float(v)
                    else:
                        prediction[k] = v
                
                predictions.append(prediction)
            
            logger.info(f"Retrieved {len(predictions)} recent predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error getting recent predictions: {e}")
            return []
    
    def _get_benchmark_data(self, days: int) -> Dict:
        """Get benchmark index data for comparison period"""
        try:
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            end_date = datetime.now().strftime('%Y-%m-%d')
            
            benchmark_data = {}
            
            for symbol, name in self.benchmarks.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date)
                    
                    if not hist.empty:
                        hist['Returns'] = hist['Close'].pct_change()
                        benchmark_data[symbol] = {
                            'name': name,
                            'data': hist,
                            'total_return': (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1),
                            'daily_returns': hist['Returns'].dropna().tolist()
                        }
                        
                except Exception as e:
                    logger.warning(f"Error fetching {symbol}: {e}")
                    continue
            
            logger.info(f"Retrieved benchmark data for {len(benchmark_data)} indexes")
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Error getting benchmark data: {e}")
            return {}
    
    def _calculate_model_performance(self, predictions: List[Dict]) -> Dict:
        """Calculate comprehensive model performance metrics"""
        try:
            if not predictions:
                return {'error': 'No predictions to analyze'}
            
            # Get actual outcomes for predictions
            predictions_with_outcomes = []
            
            for pred in predictions:
                outcome = self._get_actual_outcome(pred)
                if outcome is not None:
                    pred_with_outcome = pred.copy()
                    pred_with_outcome['actual_outcome'] = outcome
                    pred_with_outcome['hit_target'] = outcome > 0.5
                    predictions_with_outcomes.append(pred_with_outcome)
            
            if not predictions_with_outcomes:
                return {'error': 'No outcomes available'}
            
            # Calculate hit rate
            total_predictions = len(predictions_with_outcomes)
            successful_hits = sum(1 for p in predictions_with_outcomes if p['hit_target'])
            hit_rate = successful_hits / total_predictions
            
            # Calculate returns if following recommendations
            model_returns = []
            for pred in predictions_with_outcomes:
                # Simulate taking position based on recommendation
                rec_type = pred.get('recommendation_type', 'HOLD')
                target_return = self._calculate_prediction_return(pred)
                
                if rec_type in ['BUY', 'STRONG_BUY']:
                    model_returns.append(target_return)
                elif rec_type in ['SELL', 'STRONG_SELL']:
                    model_returns.append(-target_return)  # Short position
                # HOLD positions contribute 0 return
            
            if model_returns:
                total_model_return = sum(model_returns)
                avg_return_per_prediction = total_model_return / len(model_returns)
                
                # Annualize assuming predictions every 15 days
                annualized_return = avg_return_per_prediction * (365 / 15)
                
                # Calculate Sharpe ratio
                return_std = np.std(model_returns) if len(model_returns) > 1 else 0.1
                sharpe_ratio = (avg_return_per_prediction * np.sqrt(365/15)) / (return_std * np.sqrt(365/15))
            else:
                total_model_return = 0
                annualized_return = 0
                sharpe_ratio = 0
            
            # Confidence analysis
            high_conf_preds = [p for p in predictions_with_outcomes if p.get('confidence', 0) > 0.8]
            high_conf_hit_rate = (
                sum(1 for p in high_conf_preds if p['hit_target']) / len(high_conf_preds)
                if high_conf_preds else 0
            )
            
            # Time analysis
            hit_times = []
            for pred in predictions_with_outcomes:
                if pred['hit_target']:
                    # Calculate how long it took to hit target
                    hit_time = self._calculate_time_to_hit(pred)
                    if hit_time:
                        hit_times.append(hit_time)
            
            avg_time_to_hit = np.mean(hit_times) if hit_times else 0
            
            return {
                'hit_rate': hit_rate,
                'total_predictions': total_predictions,
                'successful_predictions': successful_hits,
                'high_confidence_hit_rate': high_conf_hit_rate,
                'total_return': total_model_return,
                'annualized_return': annualized_return,
                'sharpe_ratio': sharpe_ratio,
                'average_time_to_hit_days': avg_time_to_hit,
                'return_per_prediction': avg_return_per_prediction,
                'predictions_with_outcomes': len(predictions_with_outcomes)
            }
            
        except Exception as e:
            logger.error(f"Error calculating model performance: {e}")
            return {'error': str(e)}
    
    def _get_actual_outcome(self, prediction: Dict) -> Optional[float]:
        """Get actual market outcome for a prediction"""
        try:
            symbol = prediction['symbol']
            pred_timestamp = datetime.fromisoformat(prediction['timestamp'])
            current_price = float(prediction['current_price'])
            target_price = float(prediction.get('target_price', current_price))
            
            # Get actual price data
            end_date = min(datetime.now(), pred_timestamp + timedelta(days=30))
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=pred_timestamp.date(), end=end_date.date())
            
            if hist.empty:
                return None
            
            # Check if target was hit
            target_return = (target_price - current_price) / current_price
            direction = 1 if target_return > 0 else -1
            
            hit_achieved = False
            actual_return = 0
            
            for date, row in hist.iterrows():
                if direction > 0 and row['High'] >= target_price:
                    hit_achieved = True
                    actual_return = (target_price - current_price) / current_price
                    break
                elif direction < 0 and row['Low'] <= target_price:
                    hit_achieved = True
                    actual_return = (current_price - target_price) / current_price
                    break
            
            if not hit_achieved:
                # Calculate final return
                final_price = hist['Close'].iloc[-1]
                actual_return = (final_price - current_price) / current_price
                if direction < 0:
                    actual_return = -actual_return
            
            return 1.0 if hit_achieved else 0.0
            
        except Exception as e:
            logger.warning(f"Error getting actual outcome: {e}")
            return None
    
    def _calculate_prediction_return(self, prediction: Dict) -> float:
        """Calculate actual return achieved by prediction"""
        try:
            current_price = float(prediction['current_price'])
            symbol = prediction['symbol']
            pred_timestamp = datetime.fromisoformat(prediction['timestamp'])
            
            # Get actual price movement over holding period (15 days default)
            end_date = min(datetime.now(), pred_timestamp + timedelta(days=15))
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(start=pred_timestamp.date(), end=end_date.date())
            
            if hist.empty or len(hist) < 2:
                return 0.0
            
            final_price = hist['Close'].iloc[-1]
            actual_return = (final_price - current_price) / current_price
            
            return float(actual_return)
            
        except Exception as e:
            logger.warning(f"Error calculating prediction return: {e}")
            return 0.0
    
    def _calculate_time_to_hit(self, prediction: Dict) -> Optional[float]:
        """Calculate time taken to hit target"""
        try:
            symbol = prediction['symbol']
            pred_timestamp = datetime.fromisoformat(prediction['timestamp'])
            current_price = float(prediction['current_price'])
            target_price = float(prediction.get('target_price', current_price))
            
            ticker = yf.Ticker(symbol)
            end_date = min(datetime.now(), pred_timestamp + timedelta(days=30))
            hist = ticker.history(start=pred_timestamp.date(), end=end_date.date())
            
            if hist.empty:
                return None
            
            target_return = (target_price - current_price) / current_price
            direction = 1 if target_return > 0 else -1
            
            for i, (date, row) in enumerate(hist.iterrows()):
                if direction > 0 and row['High'] >= target_price:
                    return float(i + 1)  # Days to hit
                elif direction < 0 and row['Low'] <= target_price:
                    return float(i + 1)
            
            return None  # Target not hit
            
        except Exception as e:
            logger.warning(f"Error calculating time to hit: {e}")
            return None
    
    def _calculate_benchmark_performance(self, benchmark_data: Dict) -> Dict:
        """Calculate benchmark performance metrics"""
        try:
            benchmark_performance = {}
            
            for symbol, data in benchmark_data.items():
                hist = data['data']
                returns = hist['Returns'].dropna()
                
                if len(returns) < 2:
                    continue
                
                # Performance metrics
                total_return = data['total_return']
                daily_returns = returns.values
                
                # Risk metrics
                volatility = returns.std() * np.sqrt(252)  # Annualized
                sharpe_ratio = (returns.mean() * 252) / volatility if volatility > 0 else 0
                
                # Drawdown analysis
                cumulative = (1 + returns).cumprod()
                rolling_max = cumulative.expanding().max()
                drawdown = (cumulative / rolling_max - 1)
                max_drawdown = drawdown.min()
                
                # Win rate (positive return days)
                win_rate = (returns > 0).sum() / len(returns)
                
                benchmark_performance[symbol] = {
                    'name': data['name'],
                    'total_return': float(total_return),
                    'annualized_return': float(total_return * 365 / len(hist)),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown),
                    'win_rate': float(win_rate),
                    'trading_days': len(hist)
                }
            
            # Calculate market average
            if benchmark_performance:
                avg_return = np.mean([b['total_return'] for b in benchmark_performance.values()])
                avg_sharpe = np.mean([b['sharpe_ratio'] for b in benchmark_performance.values()])
                avg_volatility = np.mean([b['volatility'] for b in benchmark_performance.values()])
                
                benchmark_performance['market_average'] = {
                    'total_return': float(avg_return),
                    'annualized_return': float(avg_return * 365 / validation_period_days),
                    'sharpe_ratio': float(avg_sharpe),
                    'volatility': float(avg_volatility)
                }
            
            return benchmark_performance
            
        except Exception as e:
            logger.error(f"Error calculating benchmark performance: {e}")
            return {}
    
    def _compare_model_vs_benchmarks(self, model_perf: Dict, benchmark_perf: Dict) -> Dict:
        """Compare model performance against market benchmarks"""
        try:
            if 'market_average' not in benchmark_perf:
                return {'error': 'No benchmark data for comparison'}
            
            market_avg = benchmark_perf['market_average']
            
            # Convert hit rate to expected return
            hit_rate = model_perf.get('hit_rate', 0.0)
            model_return = model_perf.get('annualized_return', 0.0)
            
            # Market comparison
            market_return = market_avg['total_return']
            outperformance = model_return - market_return
            
            # Risk-adjusted comparison
            model_sharpe = model_perf.get('sharpe_ratio', 0.0)
            market_sharpe = market_avg['sharpe_ratio']
            
            # Detailed benchmark comparisons
            individual_comparisons = {}
            for symbol, bench_data in benchmark_perf.items():
                if symbol == 'market_average':
                    continue
                
                individual_comparisons[symbol] = {
                    'beats_benchmark': model_return > bench_data['total_return'],
                    'outperformance': model_return - bench_data['total_return'],
                    'risk_adjusted_outperformance': model_sharpe - bench_data['sharpe_ratio'],
                    'benchmark_name': bench_data['name']
                }
            
            return {
                'beats_market_average': outperformance > 0,
                'outperformance': float(outperformance),
                'outperformance_percentage': float(outperformance * 100),
                'risk_adjusted_outperformance': float(model_sharpe - market_sharpe),
                'model_metrics': {
                    'hit_rate': model_perf.get('hit_rate', 0.0),
                    'annualized_return': model_return,
                    'sharpe_ratio': model_sharpe
                },
                'market_metrics': {
                    'average_return': float(market_return),
                    'average_sharpe': float(market_sharpe)
                },
                'individual_benchmark_comparisons': individual_comparisons,
                'benchmarks_beaten': sum(1 for comp in individual_comparisons.values() if comp['beats_benchmark']),
                'total_benchmarks': len(individual_comparisons)
            }
            
        except Exception as e:
            logger.error(f"Error comparing model vs benchmarks: {e}")
            return {'error': str(e)}
    
    def _assess_competitive_standing(self, model_performance: Dict) -> Dict:
        """Assess model performance against competitive industry standards"""
        try:
            hit_rate = model_performance.get('hit_rate', 0.0)
            sharpe_ratio = model_performance.get('sharpe_ratio', 0.0)
            
            # Hit rate assessment
            if hit_rate >= self.competitive_benchmarks['hit_rate_excellent']:
                hit_rate_tier = 'excellent'
                hit_rate_percentile = 95
            elif hit_rate >= self.competitive_benchmarks['hit_rate_good']:
                hit_rate_tier = 'good'
                hit_rate_percentile = 80
            elif hit_rate >= self.competitive_benchmarks['hit_rate_minimum']:
                hit_rate_tier = 'competitive'
                hit_rate_percentile = 60
            else:
                hit_rate_tier = 'underperforming'
                hit_rate_percentile = 30
            
            # Risk-adjusted performance assessment
            if sharpe_ratio >= 1.5:
                risk_adjusted_tier = 'excellent'
            elif sharpe_ratio >= 1.0:
                risk_adjusted_tier = 'good'
            elif sharpe_ratio >= self.competitive_benchmarks['sharpe_ratio_minimum']:
                risk_adjusted_tier = 'acceptable'
            else:
                risk_adjusted_tier = 'poor'
            
            # Overall competitive score (0-100)
            competitive_score = (
                (hit_rate / 0.75) * 60 +  # Hit rate worth 60 points (75% = perfect)
                (sharpe_ratio / 2.0) * 40   # Sharpe worth 40 points (2.0 = perfect)
            )
            competitive_score = min(100, max(0, competitive_score))
            
            # Industry standing
            if competitive_score >= 85:
                industry_standing = 'market_leader'
            elif competitive_score >= 70:
                industry_standing = 'above_average'
            elif competitive_score >= 55:
                industry_standing = 'average'
            else:
                industry_standing = 'below_average'
            
            return {
                'competitive_score': float(competitive_score),
                'industry_standing': industry_standing,
                'hit_rate_assessment': {
                    'tier': hit_rate_tier,
                    'percentile': hit_rate_percentile,
                    'actual': hit_rate,
                    'target': self.competitive_benchmarks['hit_rate_good']
                },
                'risk_adjusted_assessment': {
                    'tier': risk_adjusted_tier,
                    'sharpe_ratio': sharpe_ratio,
                    'target': self.competitive_benchmarks['sharpe_ratio_minimum']
                },
                'meets_minimum_standards': (
                    hit_rate >= self.competitive_benchmarks['hit_rate_minimum'] and
                    sharpe_ratio >= self.competitive_benchmarks['sharpe_ratio_minimum']
                ),
                'competitive_advantages': self._identify_competitive_advantages(model_performance),
                'competitive_gaps': self._identify_competitive_gaps(model_performance)
            }
            
        except Exception as e:
            logger.error(f"Error assessing competitive standing: {e}")
            return {'error': str(e)}
    
    def _identify_competitive_advantages(self, model_performance: Dict) -> List[str]:
        """Identify model's competitive advantages"""
        advantages = []
        
        hit_rate = model_performance.get('hit_rate', 0.0)
        high_conf_hit_rate = model_performance.get('high_confidence_hit_rate', 0.0)
        sharpe_ratio = model_performance.get('sharpe_ratio', 0.0)
        avg_time_to_hit = model_performance.get('average_time_to_hit_days', 0)
        
        if hit_rate >= 0.70:
            advantages.append('high_accuracy_predictions')
        
        if high_conf_hit_rate >= 0.80:
            advantages.append('excellent_confidence_calibration')
        
        if sharpe_ratio >= 1.3:
            advantages.append('superior_risk_adjusted_returns')
        
        if 0 < avg_time_to_hit <= 10:
            advantages.append('fast_target_achievement')
        
        if model_performance.get('return_per_prediction', 0) > 0.03:
            advantages.append('strong_return_generation')
        
        return advantages
    
    def _identify_competitive_gaps(self, model_performance: Dict) -> List[Dict]:
        """Identify performance gaps vs competitive standards"""
        gaps = []
        
        hit_rate = model_performance.get('hit_rate', 0.0)
        sharpe_ratio = model_performance.get('sharpe_ratio', 0.0)
        
        # Hit rate gap
        hit_rate_target = self.competitive_benchmarks['hit_rate_good']
        if hit_rate < hit_rate_target:
            gaps.append({
                'metric': 'hit_rate',
                'current': hit_rate,
                'target': hit_rate_target,
                'gap': hit_rate_target - hit_rate,
                'improvement_needed': f"{(hit_rate_target - hit_rate) * 100:.1f}%",
                'priority': 'high' if hit_rate < 0.55 else 'medium'
            })
        
        # Sharpe ratio gap
        sharpe_target = self.competitive_benchmarks['information_ratio_good']
        if sharpe_ratio < sharpe_target:
            gaps.append({
                'metric': 'sharpe_ratio',
                'current': sharpe_ratio,
                'target': sharpe_target,
                'gap': sharpe_target - sharpe_ratio,
                'improvement_needed': f"{sharpe_target - sharpe_ratio:.2f} points",
                'priority': 'medium'
            })
        
        return gaps
    
    def _generate_improvement_plan(self, comparison_results: Dict, 
                                 competitive_assessment: Dict) -> Dict:
        """Generate specific improvement plan based on validation results"""
        try:
            plan = {
                'priority_actions': [],
                'improvement_targets': {},
                'timeline': 'next_30_days',
                'success_criteria': {}
            }
            
            # Analyze current standing
            beats_market = comparison_results.get('beats_market_average', False)
            competitive_score = competitive_assessment.get('competitive_score', 0)
            gaps = competitive_assessment.get('competitive_gaps', [])
            
            if not beats_market:
                plan['priority_actions'].append({
                    'action': 'enhance_feature_engineering',
                    'description': 'Add fundamental and alternative data features',
                    'expected_impact': 'Improve hit rate by 8-12%',
                    'priority': 'critical',
                    'timeline_days': 14
                })
                
                plan['priority_actions'].append({
                    'action': 'implement_ensemble_models',
                    'description': 'Deploy XGBoost, LightGBM, and neural network ensemble',
                    'expected_impact': 'Improve accuracy by 5-8%',
                    'priority': 'high',
                    'timeline_days': 21
                })
            
            if competitive_score < 70:
                plan['priority_actions'].append({
                    'action': 'optimize_prediction_targets',
                    'description': 'Adjust price targets and timeframes based on backtesting',
                    'expected_impact': 'Improve hit rate by 3-5%',
                    'priority': 'high',
                    'timeline_days': 7
                })
            
            # Set specific targets
            current_hit_rate = competitive_assessment.get('hit_rate_assessment', {}).get('actual', 0)
            target_hit_rate = max(0.65, current_hit_rate + 0.10)  # Aim for 10% improvement or 65% minimum
            
            plan['improvement_targets'] = {
                'hit_rate_target': target_hit_rate,
                'sharpe_ratio_target': 1.2,
                'market_outperformance_target': 0.05,  # 5% annual outperformance
                'confidence_calibration_target': 0.75   # High confidence predictions should hit 75%+
            }
            
            # Success criteria
            plan['success_criteria'] = {
                'beats_spy_consistently': True,
                'hit_rate_above_65': True,
                'positive_sharpe_ratio': True,
                'outperforms_market_by_3_percent': True
            }
            
            return plan
            
        except Exception as e:
            logger.error(f"Error generating improvement plan: {e}")
            return {'error': str(e)}
    
    def _store_validation_results(self, results: Dict):
        """Store validation results for tracking"""
        try:
            # Store in analytics table
            validation_item = {
                'analysis_id': f"market_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'analysis_type': 'market_validation',
                'analysis_timestamp': results['validation_timestamp'],
                'validation_period_days': results['validation_period_days'],
                'model_hit_rate': Decimal(str(results['model_performance'].get('hit_rate', 0))),
                'model_sharpe_ratio': Decimal(str(results['model_performance'].get('sharpe_ratio', 0))),
                'beats_market': results['comparison_results'].get('beats_market_average', False),
                'market_outperformance': Decimal(str(results['comparison_results'].get('outperformance', 0))),
                'competitive_score': Decimal(str(results['competitive_assessment'].get('competitive_score', 0))),
                'industry_standing': results['competitive_assessment'].get('industry_standing', 'unknown'),
                'predictions_analyzed': results['predictions_analyzed']
            }
            
            self.analytics_table.put_item(Item=validation_item)
            
            # Store detailed results in S3
            s3_key = f"validations/{datetime.now().strftime('%Y/%m/%d')}/market_validation_{datetime.now().strftime('%H%M%S')}.json"
            
            s3.put_object(
                Bucket=self.validation_bucket,
                Key=s3_key,
                Body=json.dumps(results, default=str, indent=2),
                ContentType='application/json'
            )
            
            # Publish CloudWatch metrics
            self._publish_validation_metrics(results)
            
            logger.info(f"Validation results stored: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error storing validation results: {e}")
    
    def _publish_validation_metrics(self, results: Dict):
        """Publish key metrics to CloudWatch"""
        try:
            metrics = [
                ('ModelHitRate', results['model_performance'].get('hit_rate', 0) * 100, 'Percent'),
                ('ModelSharpeRatio', results['model_performance'].get('sharpe_ratio', 0), 'None'),
                ('MarketOutperformance', results['comparison_results'].get('outperformance', 0) * 100, 'Percent'),
                ('CompetitiveScore', results['competitive_assessment'].get('competitive_score', 0), 'None'),
                ('BeatsMarket', 1 if results['comparison_results'].get('beats_market_average', False) else 0, 'Count')
            ]
            
            for metric_name, value, unit in metrics:
                cloudwatch.put_metric_data(
                    Namespace='StockAnalytics/ModelValidation',
                    MetricData=[{
                        'MetricName': metric_name,
                        'Value': float(value),
                        'Unit': unit,
                        'Timestamp': datetime.now()
                    }]
                )
            
            logger.info("Published validation metrics to CloudWatch")
            
        except Exception as e:
            logger.error(f"Error publishing validation metrics: {e}")

def lambda_handler(event, context):
    """Lambda handler for market validation engine"""
    try:
        action = event.get('action', 'validate_performance')
        engine = MarketValidationEngine()
        
        if action == 'validate_performance':
            validation_period = event.get('validation_period_days', 90)
            results = engine.validate_against_market(validation_period)
            
            return {
                'statusCode': 200,
                'body': json.dumps(results, default=str)
            }
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Unknown action: {action}'})
            }
            
    except Exception as e:
        logger.error(f"Error in market validation: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }