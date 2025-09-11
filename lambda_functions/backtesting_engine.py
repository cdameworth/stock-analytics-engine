#!/usr/bin/env python3
"""
Comprehensive Backtesting Engine for Stock Prediction Models
Implements walk-forward validation with real market data and benchmark comparisons
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.ensemble import RandomForestRegressor
import boto3
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

class BacktestingEngine:
    """
    Walk-forward backtesting engine that validates model performance against market benchmarks
    """
    
    def __init__(self):
        self.benchmark_symbols = ['SPY', 'QQQ', 'IWM']  # Market benchmarks
        self.test_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'NVDA', 'META', 'NFLX', 'AMD', 'CRM']
        self.analytics_table = dynamodb.Table('ai-performance-analytics')
        
    def run_comprehensive_backtest(self, start_date: str, end_date: str, 
                                 walk_forward_days: int = 30) -> Dict:
        """
        Run walk-forward backtesting against market benchmarks
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Get market data
        market_data = self._fetch_market_data(start_date, end_date)
        if not market_data:
            return {'error': 'Failed to fetch market data'}
        
        # Run walk-forward validation
        backtest_results = self._walk_forward_validation(
            market_data, start_date, end_date, walk_forward_days
        )
        
        # Calculate benchmark performance
        benchmark_performance = self._calculate_benchmark_performance(market_data)
        
        # Compare against benchmarks
        performance_comparison = self._compare_against_benchmarks(
            backtest_results, benchmark_performance
        )
        
        # Generate comprehensive metrics
        final_metrics = self._calculate_comprehensive_metrics(
            backtest_results, benchmark_performance
        )
        
        return {
            'backtest_results': backtest_results,
            'benchmark_performance': benchmark_performance,
            'performance_comparison': performance_comparison,
            'final_metrics': final_metrics,
            'backtest_period': f"{start_date} to {end_date}",
            'validation_method': 'walk_forward',
            'walk_forward_days': walk_forward_days
        }
    
    def _fetch_market_data(self, start_date: str, end_date: str) -> Dict:
        """Fetch historical market data for backtesting"""
        try:
            all_symbols = self.benchmark_symbols + self.test_symbols
            market_data = {}
            
            for symbol in all_symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(start=start_date, end=end_date, interval='1d')
                    
                    if not hist.empty:
                        # Calculate additional features
                        hist['Returns'] = hist['Close'].pct_change()
                        hist['MA_5'] = hist['Close'].rolling(window=5).mean()
                        hist['MA_20'] = hist['Close'].rolling(window=20).mean()
                        hist['MA_50'] = hist['Close'].rolling(window=50).mean()
                        hist['Volatility'] = hist['Returns'].rolling(window=20).std() * np.sqrt(252)
                        hist['RSI'] = self._calculate_rsi(hist['Close'].values)
                        
                        market_data[symbol] = hist
                    
                except Exception as e:
                    logger.warning(f"Failed to fetch data for {symbol}: {e}")
                    continue
            
            logger.info(f"Fetched data for {len(market_data)} symbols")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = np.convolve(gains, np.ones(period)/period, mode='valid')
        avg_losses = np.convolve(losses, np.ones(period)/period, mode='valid')
        
        rs = avg_gains / (avg_losses + 1e-10)  # Avoid division by zero
        rsi = 100 - (100 / (1 + rs))
        
        # Pad with neutral RSI for initial period
        return np.concatenate([np.full(period, 50.0), rsi])
    
    def _walk_forward_validation(self, market_data: Dict, start_date: str, 
                                end_date: str, walk_forward_days: int) -> Dict:
        """
        Implement walk-forward validation methodology
        """
        logger.info("Starting walk-forward validation")
        
        # Convert dates
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        validation_results = []
        current_date = start_dt
        
        # Initial training period (60 days minimum)
        training_period = 60
        
        while current_date + timedelta(days=walk_forward_days) <= end_dt:
            # Define training and test windows
            train_start = current_date - timedelta(days=training_period)
            train_end = current_date
            test_start = current_date
            test_end = current_date + timedelta(days=walk_forward_days)
            
            logger.info(f"Validating period: {test_start.date()} to {test_end.date()}")
            
            # Train model on training window
            model, features = self._train_model_for_period(
                market_data, train_start, train_end
            )
            
            if model is None:
                current_date += timedelta(days=walk_forward_days)
                continue
            
            # Generate predictions for test window
            predictions = self._generate_predictions_for_period(
                model, features, market_data, test_start, test_end
            )
            
            # Evaluate predictions against actual outcomes
            evaluation = self._evaluate_predictions(
                predictions, market_data, test_start, test_end
            )
            
            validation_results.append({
                'period': f"{test_start.date()} to {test_end.date()}",
                'train_period': f"{train_start.date()} to {train_end.date()}",
                'predictions': len(predictions),
                'evaluation': evaluation,
                'model_features': len(features) if features else 0
            })
            
            # Move to next period
            current_date += timedelta(days=walk_forward_days)
            
            # Extend training period progressively (max 252 days = 1 year)
            training_period = min(training_period + 10, 252)
        
        # Aggregate results
        aggregated_results = self._aggregate_validation_results(validation_results)
        
        return {
            'validation_periods': validation_results,
            'aggregated_performance': aggregated_results,
            'total_periods': len(validation_results)
        }
    
    def _train_model_for_period(self, market_data: Dict, start_date: pd.Timestamp, 
                               end_date: pd.Timestamp) -> Tuple[Optional[object], Optional[List]]:
        """Train model for specific period"""
        try:
            # Prepare training data
            training_features, training_targets = self._prepare_training_data(
                market_data, start_date, end_date
            )
            
            if len(training_features) < 20:  # Minimum samples required
                return None, None
            
            # Train RandomForest (current model)
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                random_state=42
            )
            
            model.fit(training_features, training_targets)
            
            # Get feature names
            feature_names = self._get_feature_names()
            
            return model, feature_names
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, None
    
    def _prepare_training_data(self, market_data: Dict, start_date: pd.Timestamp, 
                              end_date: pd.Timestamp) -> Tuple[List, List]:
        """Prepare training features and targets for given period"""
        features = []
        targets = []
        
        for symbol in self.test_symbols:
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            period_data = df[(df.index >= start_date) & (df.index <= end_date)]
            
            if len(period_data) < 20:
                continue
            
            # Generate features and targets for each trading day
            for i in range(20, len(period_data)):  # Need 20 days for indicators
                current_row = period_data.iloc[i]
                
                # Feature extraction
                feature_vector = self._extract_features(period_data, i)
                if feature_vector is None:
                    continue
                
                # Target calculation (future return over next 5-15 days)
                target = self._calculate_target(period_data, i)
                if target is None:
                    continue
                
                features.append(feature_vector)
                targets.append(target)
        
        return features, targets
    
    def _extract_features(self, df: pd.DataFrame, idx: int) -> Optional[List]:
        """Extract features for specific data point"""
        try:
            row = df.iloc[idx]
            
            # Price ratios
            price_to_ma5 = row['Close'] / row['MA_5'] if pd.notna(row['MA_5']) else 1.0
            price_to_ma20 = row['Close'] / row['MA_20'] if pd.notna(row['MA_20']) else 1.0
            ma5_to_ma20 = row['MA_5'] / row['MA_20'] if pd.notna(row['MA_5']) and pd.notna(row['MA_20']) else 1.0
            
            # Volatility features
            volatility = row['Volatility'] if pd.notna(row['Volatility']) else 0.02
            
            # Volume features
            avg_volume = df['Volume'].iloc[max(0, idx-20):idx].mean()
            volume_ratio = row['Volume'] / avg_volume if avg_volume > 0 else 1.0
            
            # Technical indicators
            rsi = row['RSI'] if pd.notna(row['RSI']) else 50.0
            
            # Momentum features
            return_5d = df['Returns'].iloc[max(0, idx-5):idx].sum() if idx >= 5 else 0.0
            return_20d = df['Returns'].iloc[max(0, idx-20):idx].sum() if idx >= 20 else 0.0
            
            return [
                price_to_ma5,
                price_to_ma20, 
                ma5_to_ma20,
                volatility,
                volume_ratio,
                rsi / 100.0,  # Normalize RSI
                return_5d,
                return_20d
            ]
            
        except Exception as e:
            logger.warning(f"Error extracting features at index {idx}: {e}")
            return None
    
    def _calculate_target(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """Calculate target variable (future return classification)"""
        try:
            current_price = df.iloc[idx]['Close']
            
            # Look ahead 5-15 days for target achievement
            future_window = df.iloc[idx+1:min(idx+16, len(df))]
            if len(future_window) < 5:
                return None
            
            # Target: Did price move >2% in favorable direction within window?
            max_price = future_window['High'].max()
            min_price = future_window['Low'].min()
            
            upside_return = (max_price - current_price) / current_price
            downside_return = (min_price - current_price) / current_price
            
            # Binary classification: 1 = profitable opportunity, 0 = not profitable
            if upside_return >= 0.02:  # >2% upside achieved
                return 1.0
            elif downside_return <= -0.02:  # >2% downside - short opportunity
                return 1.0
            else:
                return 0.0  # No significant move
            
        except Exception as e:
            logger.warning(f"Error calculating target at index {idx}: {e}")
            return None
    
    def _get_feature_names(self) -> List[str]:
        """Get feature column names"""
        return [
            'price_to_ma5',
            'price_to_ma20',
            'ma5_to_ma20',
            'volatility',
            'volume_ratio',
            'rsi_normalized',
            'return_5d',
            'return_20d'
        ]
    
    def _generate_predictions_for_period(self, model: object, features: List, 
                                       market_data: Dict, start_date: pd.Timestamp,
                                       end_date: pd.Timestamp) -> List[Dict]:
        """Generate predictions for test period"""
        predictions = []
        
        for symbol in self.test_symbols:
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            period_data = df[(df.index >= start_date) & (df.index <= end_date)]
            
            for i in range(min(20, len(period_data)), len(period_data)):
                try:
                    feature_vector = self._extract_features(period_data, i)
                    if feature_vector is None:
                        continue
                    
                    # Generate prediction
                    pred_score = model.predict([feature_vector])[0]
                    
                    # Calculate confidence based on feature consistency
                    confidence = self._calculate_prediction_confidence(feature_vector, model)
                    
                    prediction = {
                        'symbol': symbol,
                        'date': period_data.index[i].strftime('%Y-%m-%d'),
                        'price': float(period_data.iloc[i]['Close']),
                        'prediction_score': float(pred_score),
                        'confidence': float(confidence),
                        'features': feature_vector
                    }
                    
                    predictions.append(prediction)
                    
                except Exception as e:
                    logger.warning(f"Error generating prediction for {symbol}: {e}")
                    continue
        
        return predictions
    
    def _calculate_prediction_confidence(self, features: List, model: object) -> float:
        """Calculate prediction confidence based on model uncertainty"""
        try:
            # Use ensemble variance as confidence measure
            if hasattr(model, 'estimators_'):
                # For RandomForest, use variance across trees
                tree_predictions = [tree.predict([features])[0] for tree in model.estimators_]
                prediction_std = np.std(tree_predictions)
                
                # Convert std to confidence (lower std = higher confidence)
                confidence = max(0.5, min(0.95, 1.0 - prediction_std * 2))
                return confidence
            else:
                return 0.75  # Default confidence
                
        except Exception as e:
            logger.warning(f"Error calculating confidence: {e}")
            return 0.75
    
    def _evaluate_predictions(self, predictions: List[Dict], market_data: Dict,
                            start_date: pd.Timestamp, end_date: pd.Timestamp) -> Dict:
        """Evaluate predictions against actual market outcomes"""
        evaluation_results = []
        
        for pred in predictions:
            try:
                symbol = pred['symbol']
                pred_date = pd.to_datetime(pred['date'])
                pred_price = pred['price']
                pred_score = pred['prediction_score']
                
                # Get actual outcome
                actual_outcome = self._get_actual_outcome(
                    market_data[symbol], pred_date, pred_price
                )
                
                if actual_outcome is not None:
                    evaluation_results.append({
                        'symbol': symbol,
                        'date': pred['date'],
                        'predicted': pred_score > 0.5,  # Binary prediction
                        'actual': actual_outcome > 0.5,
                        'prediction_score': pred_score,
                        'actual_score': actual_outcome,
                        'confidence': pred['confidence'],
                        'correct': (pred_score > 0.5) == (actual_outcome > 0.5)
                    })
                    
            except Exception as e:
                logger.warning(f"Error evaluating prediction: {e}")
                continue
        
        if not evaluation_results:
            return {'error': 'No valid evaluations'}
        
        # Calculate metrics
        total_predictions = len(evaluation_results)
        correct_predictions = sum(1 for r in evaluation_results if r['correct'])
        accuracy = correct_predictions / total_predictions
        
        # Confidence-based accuracy
        high_conf_results = [r for r in evaluation_results if r['confidence'] > 0.8]
        high_conf_accuracy = (
            sum(1 for r in high_conf_results if r['correct']) / len(high_conf_results)
            if high_conf_results else 0
        )
        
        # Calculate returns if taking positions based on predictions
        total_return = sum(
            (r['actual_score'] - 0.5) if r['predicted'] else 0
            for r in evaluation_results
        )
        
        return {
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'accuracy': accuracy,
            'high_confidence_accuracy': high_conf_accuracy,
            'total_return': total_return,
            'average_return_per_prediction': total_return / total_predictions if total_predictions > 0 else 0,
            'evaluation_details': evaluation_results
        }
    
    def _get_actual_outcome(self, df: pd.DataFrame, pred_date: pd.Timestamp, 
                           pred_price: float) -> Optional[float]:
        """Get actual market outcome for prediction"""
        try:
            pred_idx = df.index.get_loc(pred_date, method='nearest')
            
            # Look ahead 5-15 days
            future_data = df.iloc[pred_idx+1:min(pred_idx+16, len(df))]
            if len(future_data) < 5:
                return None
            
            # Calculate actual outcome (same logic as training target)
            max_price = future_data['High'].max()
            min_price = future_data['Low'].min()
            
            upside_return = (max_price - pred_price) / pred_price
            downside_return = (min_price - pred_price) / pred_price
            
            if upside_return >= 0.02:
                return 1.0
            elif downside_return <= -0.02:
                return 1.0  # Short opportunity
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Error getting actual outcome: {e}")
            return None
    
    def _calculate_benchmark_performance(self, market_data: Dict) -> Dict:
        """Calculate benchmark index performance"""
        benchmark_returns = {}
        
        for symbol in self.benchmark_symbols:
            if symbol not in market_data:
                continue
            
            df = market_data[symbol]
            
            # Calculate total return
            start_price = df['Close'].iloc[0]
            end_price = df['Close'].iloc[-1]
            total_return = (end_price - start_price) / start_price
            
            # Calculate Sharpe ratio
            returns = df['Returns'].dropna()
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            
            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative / rolling_max - 1)
            max_drawdown = drawdown.min()
            
            benchmark_returns[symbol] = {
                'total_return': float(total_return),
                'annualized_return': float(total_return * 252 / len(df)),
                'sharpe_ratio': float(sharpe_ratio) if not np.isnan(sharpe_ratio) else 0.0,
                'max_drawdown': float(max_drawdown),
                'volatility': float(returns.std() * np.sqrt(252))
            }
        
        # Calculate average benchmark performance
        if benchmark_returns:
            avg_return = np.mean([b['total_return'] for b in benchmark_returns.values()])
            avg_sharpe = np.mean([b['sharpe_ratio'] for b in benchmark_returns.values()])
            
            benchmark_returns['market_average'] = {
                'total_return': float(avg_return),
                'sharpe_ratio': float(avg_sharpe)
            }
        
        return benchmark_returns
    
    def _compare_against_benchmarks(self, backtest_results: Dict, 
                                   benchmark_performance: Dict) -> Dict:
        """Compare model performance against market benchmarks"""
        if 'aggregated_performance' not in backtest_results:
            return {'error': 'No backtest results to compare'}
        
        model_performance = backtest_results['aggregated_performance']
        market_avg = benchmark_performance.get('market_average', {})
        
        # Calculate key comparisons
        model_accuracy = model_performance.get('overall_accuracy', 0.0)
        model_return_per_pred = model_performance.get('average_return_per_prediction', 0.0)
        
        market_return = market_avg.get('total_return', 0.0)
        market_sharpe = market_avg.get('sharpe_ratio', 0.0)
        
        # Convert model performance to comparable metrics
        # Assume we make predictions every 30 days and hold positions
        periods_per_year = 365 / 30  # ~12 periods per year
        annualized_model_return = model_return_per_pred * periods_per_year
        
        return {
            'model_vs_market': {
                'model_annualized_return': float(annualized_model_return),
                'market_annualized_return': float(market_return),
                'outperformance': float(annualized_model_return - market_return),
                'beats_market': annualized_model_return > market_return
            },
            'model_accuracy': float(model_accuracy),
            'accuracy_target': 0.60,  # 60% minimum for market beating
            'meets_accuracy_target': model_accuracy >= 0.60,
            'market_competitiveness': {
                'competitive_score': float(min(model_accuracy / 0.78, 1.0)),  # Target 78%
                'industry_percentile': self._estimate_industry_percentile(model_accuracy),
                'needs_improvement': model_accuracy < 0.65
            }
        }
    
    def _estimate_industry_percentile(self, accuracy: float) -> float:
        """Estimate industry percentile based on accuracy"""
        # Based on academic research on stock prediction accuracy
        if accuracy >= 0.75:
            return 95.0  # Top 5%
        elif accuracy >= 0.65:
            return 80.0  # Top 20%
        elif accuracy >= 0.55:
            return 60.0  # Above average
        elif accuracy >= 0.50:
            return 40.0  # Below average
        else:
            return 20.0  # Bottom 20%
    
    def _aggregate_validation_results(self, validation_results: List[Dict]) -> Dict:
        """Aggregate results across all validation periods"""
        if not validation_results:
            return {}
        
        # Collect all evaluations
        all_evaluations = []
        for result in validation_results:
            if 'evaluation' in result and 'evaluation_details' in result['evaluation']:
                all_evaluations.extend(result['evaluation']['evaluation_details'])
        
        if not all_evaluations:
            return {'error': 'No evaluations to aggregate'}
        
        # Calculate overall metrics
        total_predictions = len(all_evaluations)
        correct_predictions = sum(1 for e in all_evaluations if e['correct'])
        overall_accuracy = correct_predictions / total_predictions
        
        # Period-by-period accuracy
        period_accuracies = [
            r['evaluation']['accuracy'] for r in validation_results 
            if 'evaluation' in r and 'accuracy' in r['evaluation']
        ]
        
        # Confidence analysis
        high_conf_evals = [e for e in all_evaluations if e['confidence'] > 0.8]
        high_conf_accuracy = (
            sum(1 for e in high_conf_evals if e['correct']) / len(high_conf_evals)
            if high_conf_evals else 0
        )
        
        # Return analysis
        total_return = sum(e.get('actual_score', 0) - 0.5 for e in all_evaluations if e['predicted'])
        avg_return_per_prediction = total_return / total_predictions if total_predictions > 0 else 0
        
        return {
            'overall_accuracy': overall_accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'high_confidence_accuracy': high_conf_accuracy,
            'high_confidence_count': len(high_conf_evals),
            'average_return_per_prediction': avg_return_per_prediction,
            'period_accuracies': period_accuracies,
            'accuracy_std': float(np.std(period_accuracies)) if period_accuracies else 0,
            'validation_periods': len(validation_results)
        }
    
    def _calculate_comprehensive_metrics(self, backtest_results: Dict, 
                                       benchmark_performance: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        
        model_perf = backtest_results.get('aggregated_performance', {})
        comparison = self._compare_against_benchmarks(backtest_results, benchmark_performance)
        
        # Risk-adjusted metrics
        accuracy = model_perf.get('overall_accuracy', 0.0)
        return_per_pred = model_perf.get('average_return_per_prediction', 0.0)
        accuracy_std = model_perf.get('accuracy_std', 0.0)
        
        # Calculate information ratio (risk-adjusted outperformance)
        market_return = benchmark_performance.get('market_average', {}).get('total_return', 0.0)
        model_return = comparison.get('model_vs_market', {}).get('model_annualized_return', 0.0)
        
        outperformance = model_return - market_return
        information_ratio = outperformance / accuracy_std if accuracy_std > 0 else 0
        
        return {
            'model_performance': {
                'accuracy': accuracy,
                'confidence_calibration': model_perf.get('high_confidence_accuracy', 0.0),
                'return_per_prediction': return_per_pred,
                'consistency': 1.0 - accuracy_std  # Higher consistency = lower std
            },
            'market_comparison': {
                'beats_market': comparison.get('model_vs_market', {}).get('beats_market', False),
                'outperformance': outperformance,
                'information_ratio': information_ratio
            },
            'competitive_assessment': {
                'industry_percentile': comparison.get('market_competitiveness', {}).get('industry_percentile', 50),
                'competitive_advantage': accuracy > 0.65 and outperformance > 0.05,
                'improvement_needed': accuracy < 0.60 or outperformance < 0.02
            },
            'recommendation': self._generate_improvement_recommendation(accuracy, outperformance)
        }
    
    def _generate_improvement_recommendation(self, accuracy: float, outperformance: float) -> Dict:
        """Generate specific recommendations for improvement"""
        
        if accuracy >= 0.70 and outperformance >= 0.08:
            return {
                'status': 'market_beating',
                'action': 'optimize_further',
                'priority': 'low',
                'message': 'Model successfully beats market. Focus on risk management and scaling.'
            }
        elif accuracy >= 0.60 and outperformance >= 0.03:
            return {
                'status': 'competitive',
                'action': 'fine_tune',
                'priority': 'medium', 
                'message': 'Model is competitive. Enhance feature engineering and ensemble methods.'
            }
        elif accuracy >= 0.52:
            return {
                'status': 'underperforming',
                'action': 'major_overhaul',
                'priority': 'high',
                'message': 'Model underperforms market. Requires fundamental features and advanced ML.'
            }
        else:
            return {
                'status': 'failing',
                'action': 'complete_redesign', 
                'priority': 'critical',
                'message': 'Model performs poorly. Complete redesign with new data sources required.'
            }

def lambda_handler(event, context):
    """Lambda handler for backtesting engine"""
    try:
        engine = BacktestingEngine()
        
        action = event.get('action', 'run_backtest')
        
        if action == 'run_backtest':
            start_date = event.get('start_date', (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d'))
            end_date = event.get('end_date', datetime.now().strftime('%Y-%m-%d'))
            walk_forward_days = event.get('walk_forward_days', 30)
            
            results = engine.run_comprehensive_backtest(start_date, end_date, walk_forward_days)
            
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
        logger.error(f"Error in backtesting lambda: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }