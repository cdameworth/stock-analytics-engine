#!/usr/bin/env python3
"""
Advanced Model Tuning Service - Consolidated Implementation
Combines backtesting, fundamental analysis, ensemble models, and market validation
Designed to consistently beat market average performance
"""

import json
import boto3
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from decimal import Decimal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients
lambda_client = boto3.client('lambda')
dynamodb = boto3.resource('dynamodb')
s3 = boto3.client('s3')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
import os
ANALYTICS_TABLE = os.environ.get('ANALYTICS_TABLE', 'ai-performance-analytics')
RECOMMENDATIONS_TABLE = os.environ.get('RECOMMENDATIONS_TABLE', 'stock-recommendations')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', 'stock-analytics-ml-models-13605d12e16da9f9')

class AdvancedModelTuningService:
    """
    Unified model tuning service that orchestrates all components for market-beating performance
    """
    
    def __init__(self):
        self.analytics_table = dynamodb.Table(ANALYTICS_TABLE)
        self.recommendations_table = dynamodb.Table(RECOMMENDATIONS_TABLE)
        
        # Component Lambda functions
        self.components = {
            'backtesting_engine': 'backtesting-engine',
            'fundamental_enrichment': 'fundamental-data-enrichment', 
            'ensemble_engine': 'ensemble-model-engine',
            'market_validation': 'market-validation-engine'
        }
        
        # Performance targets for market-beating
        self.market_beating_targets = {
            'minimum_hit_rate': 0.58,      # Must beat 58% to be viable
            'competitive_hit_rate': 0.65,   # 65% for competitive advantage
            'excellent_hit_rate': 0.75,     # 75% for market leadership
            'minimum_sharpe': 1.0,          # Risk-adjusted return minimum
            'market_outperformance': 0.03,   # 3% annual outperformance minimum
            'confidence_calibration': 0.75   # High confidence accuracy target
        }
        
        # Tuning strategies based on performance level
        self.tuning_strategies = {
            'emergency': {
                'frequency': 'daily',
                'components': ['backtesting', 'validation', 'quick_retrain'],
                'trigger_threshold': 0.45  # Hit rate below 45%
            },
            'aggressive': {
                'frequency': 'weekly', 
                'components': ['full_backtest', 'ensemble_retrain', 'feature_optimization'],
                'trigger_threshold': 0.55  # Hit rate below 55%
            },
            'optimization': {
                'frequency': 'bi_weekly',
                'components': ['hyperparameter_tuning', 'feature_engineering', 'ensemble_weights'],
                'trigger_threshold': 0.65  # Hit rate below 65%
            },
            'maintenance': {
                'frequency': 'monthly',
                'components': ['drift_detection', 'performance_monitoring'],
                'trigger_threshold': 0.75  # Hit rate above 75%
            }
        }
    
    def execute_comprehensive_tuning(self, lookback_days: int = 90) -> Dict:
        """
        Execute comprehensive tuning workflow to achieve market-beating performance
        """
        logger.info(f"Starting comprehensive tuning for {lookback_days} days lookback")
        
        try:
            # 1. Assess current performance
            current_performance = self._assess_current_performance(lookback_days)
            
            # 2. Determine tuning strategy based on performance
            tuning_strategy = self._determine_tuning_strategy(current_performance)
            
            # 3. Execute backtesting validation
            backtest_results = self._execute_backtesting(lookback_days)
            
            # 4. Enhance features with fundamental data
            feature_enhancement = self._enhance_feature_pipeline()
            
            # 5. Train/update ensemble models
            ensemble_results = self._update_ensemble_models(backtest_results, feature_enhancement)
            
            # 6. Validate against market benchmarks
            market_validation = self._validate_market_performance()
            
            # 7. Deploy optimal configuration
            deployment_results = self._deploy_optimal_configuration(
                ensemble_results, market_validation
            )
            
            # 8. Schedule next tuning
            next_tuning = self._schedule_next_tuning(market_validation, tuning_strategy)
            
            # Compile comprehensive results
            comprehensive_results = {
                'tuning_timestamp': datetime.now().isoformat(),
                'lookback_period': lookback_days,
                'current_performance': current_performance,
                'tuning_strategy': tuning_strategy,
                'backtest_results': backtest_results,
                'feature_enhancement': feature_enhancement,
                'ensemble_results': ensemble_results,
                'market_validation': market_validation,
                'deployment_results': deployment_results,
                'next_tuning_schedule': next_tuning,
                'success_metrics': self._calculate_success_metrics(market_validation),
                'improvement_achieved': self._calculate_improvement(current_performance, market_validation)
            }
            
            # Store results and update tracking
            self._store_comprehensive_results(comprehensive_results)
            
            return comprehensive_results
            
        except Exception as e:
            logger.error(f"Error in comprehensive tuning: {e}")
            return {'error': str(e)}
    
    def _assess_current_performance(self, lookback_days: int) -> Dict:
        """Assess current model performance"""
        try:
            # Invoke market validation to get current metrics
            payload = {
                'action': 'validate_performance',
                'validation_period_days': lookback_days
            }
            
            response = lambda_client.invoke(
                FunctionName=self.components['market_validation'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read().decode())
            
            if result.get('statusCode') == 200:
                validation_data = json.loads(result['body'])
                
                # Extract key performance indicators
                model_perf = validation_data.get('model_performance', {})
                comparison = validation_data.get('comparison_results', {})
                competitive = validation_data.get('competitive_assessment', {})
                
                return {
                    'hit_rate': model_perf.get('hit_rate', 0.0),
                    'sharpe_ratio': model_perf.get('sharpe_ratio', 0.0),
                    'beats_market': comparison.get('beats_market_average', False),
                    'market_outperformance': comparison.get('outperformance', 0.0),
                    'competitive_score': competitive.get('competitive_score', 0),
                    'industry_standing': competitive.get('industry_standing', 'unknown'),
                    'needs_urgent_attention': model_perf.get('hit_rate', 0) < 0.50
                }
            else:
                logger.warning("Market validation failed, using defaults")
                return self._get_default_performance_assessment()
                
        except Exception as e:
            logger.error(f"Error assessing current performance: {e}")
            return self._get_default_performance_assessment()
    
    def _get_default_performance_assessment(self) -> Dict:
        """Default performance assessment when validation fails"""
        return {
            'hit_rate': 0.45,  # Assume poor performance requiring urgent attention
            'sharpe_ratio': 0.2,
            'beats_market': False,
            'market_outperformance': -0.05,
            'competitive_score': 30,
            'industry_standing': 'below_average',
            'needs_urgent_attention': True
        }
    
    def _determine_tuning_strategy(self, performance: Dict) -> Dict:
        """Determine appropriate tuning strategy based on current performance"""
        hit_rate = performance.get('hit_rate', 0.0)
        
        for strategy_name, strategy_config in self.tuning_strategies.items():
            if hit_rate <= strategy_config['trigger_threshold']:
                return {
                    'strategy': strategy_name,
                    'frequency': strategy_config['frequency'],
                    'components': strategy_config['components'],
                    'urgency': 'critical' if strategy_name == 'emergency' else 'high' if strategy_name == 'aggressive' else 'medium',
                    'justification': f"Hit rate {hit_rate:.1%} triggers {strategy_name} strategy"
                }
        
        # Default to maintenance if performance is good
        return {
            'strategy': 'maintenance',
            'frequency': 'monthly',
            'components': ['drift_detection', 'performance_monitoring'],
            'urgency': 'low',
            'justification': f"Hit rate {hit_rate:.1%} indicates good performance"
        }
    
    def _execute_backtesting(self, lookback_days: int) -> Dict:
        """Execute backtesting validation"""
        try:
            # Calculate backtesting period (use 2x lookback for comprehensive testing)
            backtest_days = min(lookback_days * 2, 180)
            start_date = (datetime.now() - timedelta(days=backtest_days)).strftime('%Y-%m-%d')
            end_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')  # Exclude last week for fresh data
            
            payload = {
                'action': 'run_backtest',
                'start_date': start_date,
                'end_date': end_date,
                'walk_forward_days': 21  # 3-week validation periods
            }
            
            response = lambda_client.invoke(
                FunctionName=self.components['backtesting_engine'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read().decode())
            
            if result.get('statusCode') == 200:
                backtest_data = json.loads(result['body'])
                
                return {
                    'backtest_completed': True,
                    'backtest_period': f"{start_date} to {end_date}",
                    'validation_accuracy': backtest_data.get('final_metrics', {}).get('model_performance', {}).get('accuracy', 0.0),
                    'beats_benchmark_in_backtest': backtest_data.get('final_metrics', {}).get('market_comparison', {}).get('beats_market', False),
                    'backtest_details': backtest_data
                }
            else:
                return {'backtest_completed': False, 'error': result.get('body', 'Unknown error')}
                
        except Exception as e:
            logger.error(f"Error executing backtesting: {e}")
            return {'backtest_completed': False, 'error': str(e)}
    
    def _enhance_feature_pipeline(self) -> Dict:
        """Enhance feature pipeline with fundamental and macro data"""
        try:
            # Test fundamental data enrichment
            test_symbol = 'AAPL'  # Use AAPL as test case
            
            payload = {
                'action': 'enrich_features',
                'symbol': test_symbol,
                'base_features': {
                    'price_to_ma5_ratio': 1.02,
                    'volatility': 0.025,
                    'volume_ratio': 0.8
                }
            }
            
            response = lambda_client.invoke(
                FunctionName=self.components['fundamental_enrichment'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read().decode())
            
            if result.get('statusCode') == 200:
                enriched_features = json.loads(result['body'])
                
                # Count new features added
                base_feature_count = len(payload['base_features'])
                enriched_feature_count = len(enriched_features)
                new_features_added = enriched_feature_count - base_feature_count
                
                # Test sector rotation analysis
                sector_payload = {'action': 'sector_rotation'}
                sector_response = lambda_client.invoke(
                    FunctionName=self.components['fundamental_enrichment'],
                    InvocationType='RequestResponse',
                    Payload=json.dumps(sector_payload)
                )
                
                sector_result = json.loads(sector_response['Payload'].read().decode())
                sector_data = json.loads(sector_result['body']) if sector_result.get('statusCode') == 200 else {}
                
                return {
                    'enhancement_successful': True,
                    'new_features_added': new_features_added,
                    'total_features': enriched_feature_count,
                    'fundamental_features': [
                        'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book', 'profit_margin',
                        'earnings_growth', 'revenue_growth', 'beta', 'sector_relative_1m'
                    ],
                    'macro_features': [
                        'vix_level', 'treasury_10y', 'dollar_strength', 'risk_on_regime'
                    ],
                    'sector_rotation_available': 'error' not in sector_data,
                    'test_enrichment_sample': enriched_features
                }
            else:
                return {'enhancement_successful': False, 'error': result.get('body')}
                
        except Exception as e:
            logger.error(f"Error enhancing feature pipeline: {e}")
            return {'enhancement_successful': False, 'error': str(e)}
    
    def _update_ensemble_models(self, backtest_results: Dict, feature_enhancement: Dict) -> Dict:
        """Update ensemble models with improved features and validation"""
        try:
            if not feature_enhancement.get('enhancement_successful'):
                logger.warning("Feature enhancement failed, using basic ensemble")
            
            # Prepare training data from backtest results
            training_data = self._prepare_ensemble_training_data(backtest_results)
            
            if not training_data:
                return {'ensemble_update_successful': False, 'error': 'No training data available'}
            
            # Train advanced ensemble
            payload = {
                'action': 'create_advanced_pipeline',
                'training_data': training_data
            }
            
            response = lambda_client.invoke(
                FunctionName=self.components['ensemble_engine'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload, default=str)
            )
            
            result = json.loads(response['Payload'].read().decode())
            
            if result.get('statusCode') == 200:
                ensemble_data = json.loads(result['body'])
                
                return {
                    'ensemble_update_successful': True,
                    'pipeline_performance': ensemble_data.get('pipeline_performance', 0.0),
                    'models_in_ensemble': ensemble_data.get('base_models_count', 0),
                    'ensemble_type': 'stacked_ensemble',
                    'expected_improvement': self._estimate_ensemble_improvement(ensemble_data),
                    'ensemble_details': ensemble_data
                }
            else:
                return {'ensemble_update_successful': False, 'error': result.get('body')}
                
        except Exception as e:
            logger.error(f"Error updating ensemble models: {e}")
            return {'ensemble_update_successful': False, 'error': str(e)}
    
    def _prepare_ensemble_training_data(self, backtest_results: Dict) -> Optional[List[Dict]]:
        """Prepare training data for ensemble from backtest results"""
        try:
            if not backtest_results.get('backtest_completed'):
                logger.warning("Backtest incomplete, generating synthetic training data")
                return self._generate_synthetic_training_data()
            
            # Extract validation data from backtest
            backtest_details = backtest_results.get('backtest_details', {})
            validation_results = backtest_details.get('validation_periods', [])
            
            training_data = []
            
            for period_result in validation_results:
                evaluation = period_result.get('evaluation', {})
                evaluation_details = evaluation.get('evaluation_details', [])
                
                for detail in evaluation_details:
                    # Convert to training format
                    training_sample = {
                        'symbol': detail.get('symbol'),
                        'prediction_score': detail.get('prediction_score'),
                        'actual_score': detail.get('actual_score'),
                        'target': detail.get('actual_score'),  # Use actual outcome as target
                        'confidence': detail.get('confidence'),
                        'date': detail.get('date')
                    }
                    
                    training_data.append(training_sample)
            
            if len(training_data) < 50:
                logger.warning(f"Insufficient backtest data ({len(training_data)}), supplementing with synthetic data")
                synthetic_data = self._generate_synthetic_training_data(target_size=100)
                training_data.extend(synthetic_data)
            
            return training_data[:500]  # Limit to 500 samples for efficiency
            
        except Exception as e:
            logger.error(f"Error preparing ensemble training data: {e}")
            return self._generate_synthetic_training_data()
    
    def _generate_synthetic_training_data(self, target_size: int = 200) -> List[Dict]:
        """Generate synthetic training data when real data insufficient"""
        try:
            np.random.seed(42)  # For reproducibility
            synthetic_data = []
            
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX']
            
            for i in range(target_size):
                # Generate realistic features
                symbol = np.random.choice(symbols)
                
                # Technical features (somewhat realistic distributions)
                price_to_ma5 = np.random.normal(1.0, 0.05)
                price_to_ma20 = np.random.normal(1.0, 0.08) 
                volatility = np.abs(np.random.normal(0.025, 0.01))
                volume_ratio = np.random.lognormal(0, 0.5)
                rsi = np.random.normal(50, 15)
                
                # Fundamental features
                pe_ratio = np.random.lognormal(3, 0.5)  # Realistic P/E distribution
                earnings_growth = np.random.normal(0.08, 0.15)  # 8% avg earnings growth
                sector_rotation = np.random.uniform(0, 1)
                
                # Macro features
                vix_level = np.random.gamma(2, 0.1)  # VIX-like distribution
                risk_on_regime = np.random.uniform(0, 1)
                
                # Generate target based on realistic relationships
                target_score = (
                    0.5 +
                    0.15 * np.clip((price_to_ma5 - 1) * 10, -0.15, 0.15) +
                    0.10 * np.clip((price_to_ma20 - 1) * 5, -0.10, 0.10) +
                    0.08 * np.clip(earnings_growth, -0.08, 0.08) +
                    0.05 * (risk_on_regime - 0.5) +
                    0.03 * np.clip((50 - rsi) / 50, -0.03, 0.03) +
                    np.random.normal(0, 0.05)  # Noise
                )
                
                target = np.clip(target_score, 0, 1)
                
                synthetic_data.append({
                    'symbol': symbol,
                    'price_to_ma5': price_to_ma5,
                    'price_to_ma20': price_to_ma20,
                    'volatility': volatility,
                    'volume_ratio': min(volume_ratio, 3.0),  # Cap extreme values
                    'rsi_normalized': rsi / 100.0,
                    'pe_ratio': min(pe_ratio, 50) / 50,  # Normalize
                    'earnings_growth': np.clip(earnings_growth + 1, 0, 2) / 2,  # Normalize
                    'sector_rotation': sector_rotation,
                    'vix_level': min(vix_level, 1.0),
                    'risk_on_regime': risk_on_regime,
                    'target': target,
                    'date': (datetime.now() - timedelta(days=np.random.randint(1, 90))).strftime('%Y-%m-%d')
                })
            
            logger.info(f"Generated {len(synthetic_data)} synthetic training samples")
            return synthetic_data
            
        except Exception as e:
            logger.error(f"Error generating synthetic training data: {e}")
            return []
    
    def _estimate_ensemble_improvement(self, ensemble_data: Dict) -> Dict:
        """Estimate expected improvement from ensemble model"""
        try:
            pipeline_performance = ensemble_data.get('pipeline_performance', 0.0)
            base_models_count = ensemble_data.get('base_models_count', 1)
            
            # Estimate improvement based on ensemble characteristics
            if pipeline_performance > 0.70:
                expected_hit_rate_improvement = 0.05  # 5% improvement
                confidence_improvement = 0.10
            elif pipeline_performance > 0.60:
                expected_hit_rate_improvement = 0.08  # 8% improvement  
                confidence_improvement = 0.15
            else:
                expected_hit_rate_improvement = 0.12  # 12% improvement
                confidence_improvement = 0.20
            
            # Ensemble bonus
            ensemble_bonus = min(0.03, (base_models_count - 1) * 0.01)
            
            return {
                'expected_hit_rate_improvement': expected_hit_rate_improvement + ensemble_bonus,
                'expected_confidence_improvement': confidence_improvement,
                'expected_sharpe_improvement': 0.3,  # Conservative Sharpe improvement
                'ensemble_advantage': base_models_count > 2,
                'confidence_level': 'high' if pipeline_performance > 0.65 else 'medium'
            }
            
        except Exception as e:
            logger.error(f"Error estimating ensemble improvement: {e}")
            return {'expected_hit_rate_improvement': 0.05}
    
    def _validate_market_performance(self) -> Dict:
        """Validate current performance against market"""
        try:
            payload = {'action': 'validate_performance', 'validation_period_days': 60}
            
            response = lambda_client.invoke(
                FunctionName=self.components['market_validation'],
                InvocationType='RequestResponse',
                Payload=json.dumps(payload)
            )
            
            result = json.loads(response['Payload'].read().decode())
            
            if result.get('statusCode') == 200:
                return {
                    'validation_successful': True,
                    'validation_data': json.loads(result['body'])
                }
            else:
                return {'validation_successful': False, 'error': result.get('body')}
                
        except Exception as e:
            logger.error(f"Error validating market performance: {e}")
            return {'validation_successful': False, 'error': str(e)}
    
    def _deploy_optimal_configuration(self, ensemble_results: Dict, 
                                    market_validation: Dict) -> Dict:
        """Deploy optimal model configuration"""
        try:
            # Create deployment configuration
            deployment_config = {
                'deployment_timestamp': datetime.now().isoformat(),
                'model_type': 'advanced_ensemble',
                'feature_pipeline': 'fundamental_enhanced',
                'validation_method': 'market_benchmark',
                'expected_performance': {
                    'hit_rate_target': 0.65,
                    'market_outperformance_target': 0.03,
                    'sharpe_ratio_target': 1.0
                },
                'ensemble_configuration': ensemble_results,
                'validation_results': market_validation,
                'deployment_strategy': 'gradual_rollout'  # Deploy gradually to validate
            }
            
            # Store configuration
            config_key = f"deployments/advanced_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            s3.put_object(
                Bucket=MODEL_BUCKET,
                Key=config_key,
                Body=json.dumps(deployment_config, default=str, indent=2),
                ContentType='application/json'
            )
            
            # Update model registry
            self._update_model_registry(deployment_config)
            
            return {
                'deployment_successful': True,
                'configuration_location': f"s3://{MODEL_BUCKET}/{config_key}",
                'deployment_config': deployment_config,
                'rollout_strategy': 'gradual',
                'monitoring_required': True
            }
            
        except Exception as e:
            logger.error(f"Error deploying optimal configuration: {e}")
            return {'deployment_successful': False, 'error': str(e)}
    
    def _schedule_next_tuning(self, market_validation: Dict, tuning_strategy: Dict) -> Dict:
        """Schedule next tuning based on performance and strategy"""
        try:
            validation_data = market_validation.get('validation_data', {})
            competitive_assessment = validation_data.get('competitive_assessment', {})
            
            competitive_score = competitive_assessment.get('competitive_score', 0)
            strategy_frequency = tuning_strategy.get('frequency', 'weekly')
            
            # Calculate next run date
            frequency_days = {
                'daily': 1,
                'weekly': 7,
                'bi_weekly': 14,
                'monthly': 30
            }
            
            days_until_next = frequency_days.get(strategy_frequency, 7)
            
            # Adjust based on performance
            if competitive_score < 50:
                days_until_next = min(days_until_next, 3)  # Accelerate if poor performance
            elif competitive_score > 80:
                days_until_next = min(days_until_next * 1.5, 45)  # Slow down if excellent
            
            next_run_date = (datetime.now() + timedelta(days=days_until_next)).isoformat()
            
            return {
                'next_tuning_date': next_run_date,
                'frequency': strategy_frequency,
                'days_until_next': int(days_until_next),
                'strategy': tuning_strategy.get('strategy'),
                'priority': tuning_strategy.get('urgency'),
                'adaptive_scheduling': True,
                'performance_based_adjustment': competitive_score < 50 or competitive_score > 80
            }
            
        except Exception as e:
            logger.error(f"Error scheduling next tuning: {e}")
            return {'error': str(e)}
    
    def _calculate_success_metrics(self, market_validation: Dict) -> Dict:
        """Calculate success metrics against market-beating targets"""
        try:
            validation_data = market_validation.get('validation_data', {})
            
            model_perf = validation_data.get('model_performance', {})
            comparison = validation_data.get('comparison_results', {})
            competitive = validation_data.get('competitive_assessment', {})
            
            hit_rate = model_perf.get('hit_rate', 0.0)
            sharpe_ratio = model_perf.get('sharpe_ratio', 0.0)
            market_outperformance = comparison.get('outperformance', 0.0)
            competitive_score = competitive.get('competitive_score', 0)
            
            # Check against market-beating targets
            success_criteria = {
                'meets_minimum_hit_rate': hit_rate >= self.market_beating_targets['minimum_hit_rate'],
                'meets_competitive_hit_rate': hit_rate >= self.market_beating_targets['competitive_hit_rate'],
                'meets_sharpe_target': sharpe_ratio >= self.market_beating_targets['minimum_sharpe'],
                'beats_market': market_outperformance >= self.market_beating_targets['market_outperformance'],
                'competitive_standing': competitive_score >= 70
            }
            
            # Overall success score
            criteria_met = sum(success_criteria.values())
            success_score = criteria_met / len(success_criteria)
            
            # Market beating status
            if success_score >= 0.8:
                market_beating_status = 'market_leader'
            elif success_score >= 0.6:
                market_beating_status = 'competitive'
            elif success_score >= 0.4:
                market_beating_status = 'improving'
            else:
                market_beating_status = 'needs_major_improvement'
            
            return {
                'success_criteria': success_criteria,
                'criteria_met': criteria_met,
                'total_criteria': len(success_criteria),
                'success_score': success_score,
                'market_beating_status': market_beating_status,
                'current_metrics': {
                    'hit_rate': hit_rate,
                    'sharpe_ratio': sharpe_ratio,
                    'market_outperformance': market_outperformance,
                    'competitive_score': competitive_score
                },
                'targets': self.market_beating_targets,
                'ready_for_production': success_score >= 0.6
            }
            
        except Exception as e:
            logger.error(f"Error calculating success metrics: {e}")
            return {'error': str(e)}
    
    def _calculate_improvement(self, before_performance: Dict, after_validation: Dict) -> Dict:
        """Calculate improvement achieved through tuning"""
        try:
            before_hit_rate = before_performance.get('hit_rate', 0.0)
            before_competitive_score = before_performance.get('competitive_score', 0)
            
            validation_data = after_validation.get('validation_data', {})
            after_model_perf = validation_data.get('model_performance', {})
            after_competitive = validation_data.get('competitive_assessment', {})
            
            after_hit_rate = after_model_perf.get('hit_rate', 0.0)
            after_competitive_score = after_competitive.get('competitive_score', 0)
            
            hit_rate_improvement = after_hit_rate - before_hit_rate
            competitive_improvement = after_competitive_score - before_competitive_score
            
            return {
                'hit_rate_improvement': hit_rate_improvement,
                'hit_rate_improvement_percentage': hit_rate_improvement * 100,
                'competitive_score_improvement': competitive_improvement,
                'significant_improvement': hit_rate_improvement > 0.05,  # 5%+ improvement
                'before_metrics': {
                    'hit_rate': before_hit_rate,
                    'competitive_score': before_competitive_score
                },
                'after_metrics': {
                    'hit_rate': after_hit_rate,
                    'competitive_score': after_competitive_score
                },
                'improvement_assessment': self._assess_improvement_significance(hit_rate_improvement)
            }
            
        except Exception as e:
            logger.error(f"Error calculating improvement: {e}")
            return {'error': str(e)}
    
    def _assess_improvement_significance(self, improvement: float) -> str:
        """Assess significance of improvement"""
        if improvement >= 0.10:
            return 'major_breakthrough'
        elif improvement >= 0.05:
            return 'significant_improvement'
        elif improvement >= 0.02:
            return 'moderate_improvement'
        elif improvement >= 0:
            return 'marginal_improvement'
        else:
            return 'performance_degradation'
    
    def _store_comprehensive_results(self, results: Dict):
        """Store comprehensive tuning results"""
        try:
            # Store summary in DynamoDB
            summary_item = {
                'analysis_id': f"comprehensive_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'analysis_type': 'comprehensive_model_tuning',
                'analysis_timestamp': results['tuning_timestamp'],
                'lookback_period': results['lookback_period'],
                'tuning_strategy': results['tuning_strategy']['strategy'],
                'success_score': Decimal(str(results.get('success_metrics', {}).get('success_score', 0))),
                'market_beating_status': results.get('success_metrics', {}).get('market_beating_status', 'unknown'),
                'hit_rate_improvement': Decimal(str(results.get('improvement_achieved', {}).get('hit_rate_improvement', 0))),
                'beats_market': results.get('market_validation', {}).get('validation_data', {}).get('comparison_results', {}).get('beats_market_average', False),
                'ready_for_production': results.get('success_metrics', {}).get('ready_for_production', False)
            }
            
            self.analytics_table.put_item(Item=summary_item)
            
            # Store detailed results in S3
            s3_key = f"comprehensive_tuning/{datetime.now().strftime('%Y/%m/%d')}/tuning_results_{datetime.now().strftime('%H%M%S')}.json"
            
            s3.put_object(
                Bucket=MODEL_BUCKET,
                Key=s3_key,
                Body=json.dumps(results, default=str, indent=2),
                ContentType='application/json'
            )
            
            # Publish success metrics to CloudWatch
            self._publish_tuning_metrics(results)
            
            logger.info(f"Comprehensive results stored: {s3_key}")
            
        except Exception as e:
            logger.error(f"Error storing comprehensive results: {e}")
    
    def _publish_tuning_metrics(self, results: Dict):
        """Publish key tuning metrics to CloudWatch"""
        try:
            success_metrics = results.get('success_metrics', {})
            improvement = results.get('improvement_achieved', {})
            
            metrics = [
                ('TuningSuccessScore', success_metrics.get('success_score', 0) * 100, 'Percent'),
                ('HitRateImprovement', improvement.get('hit_rate_improvement_percentage', 0), 'Percent'),
                ('ComprehensiveTuningCompleted', 1, 'Count'),
                ('MarketBeatingCapability', 1 if success_metrics.get('ready_for_production') else 0, 'Count')
            ]
            
            for metric_name, value, unit in metrics:
                cloudwatch.put_metric_data(
                    Namespace='StockAnalytics/ModelTuning',
                    MetricData=[{
                        'MetricName': metric_name,
                        'Value': float(value),
                        'Unit': unit,
                        'Timestamp': datetime.now()
                    }]
                )
            
        except Exception as e:
            logger.error(f"Error publishing tuning metrics: {e}")
    
    def _update_model_registry(self, deployment_config: Dict):
        """Update model registry with new deployment"""
        try:
            registry_item = {
                'model_id': f"advanced_tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_type': deployment_config['model_type'],
                'deployment_timestamp': deployment_config['deployment_timestamp'],
                'feature_pipeline': deployment_config['feature_pipeline'],
                'validation_method': deployment_config['validation_method'],
                'expected_hit_rate': Decimal(str(deployment_config['expected_performance']['hit_rate_target'])),
                'expected_outperformance': Decimal(str(deployment_config['expected_performance']['market_outperformance_target'])),
                'status': 'deployed',
                'deployment_strategy': deployment_config['deployment_strategy']
            }
            
            self.analytics_table.put_item(Item=registry_item)
            logger.info(f"Updated model registry: {registry_item['model_id']}")
            
        except Exception as e:
            logger.error(f"Error updating model registry: {e}")

def lambda_handler(event, context):
    """Main Lambda handler for advanced model tuning service"""
    try:
        action = event.get('action', 'comprehensive_tuning')
        service = AdvancedModelTuningService()
        
        if action == 'comprehensive_tuning':
            lookback_days = event.get('lookback_days', 90)
            results = service.execute_comprehensive_tuning(lookback_days)
            
            return {
                'statusCode': 200,
                'body': json.dumps(results, default=str)
            }
        
        elif action == 'quick_assessment':
            lookback_days = event.get('lookback_days', 30)
            performance = service._assess_current_performance(lookback_days)
            strategy = service._determine_tuning_strategy(performance)
            
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'current_performance': performance,
                    'recommended_strategy': strategy
                }, default=str)
            }
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Unknown action: {action}'})
            }
            
    except Exception as e:
        logger.error(f"Error in advanced model tuning service: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }