#!/usr/bin/env python3
"""
Fundamental Data Enrichment Service
Adds P/E ratios, earnings data, sector rotation analysis, and macroeconomic indicators
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import boto3
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from decimal import Decimal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')

class FundamentalDataEnricher:
    """
    Enriches technical analysis with fundamental and macroeconomic data
    """
    
    def __init__(self):
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV', 
            'Financial': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Consumer Staples': 'XLP',
            'Industrial': 'XLI',
            'Materials': 'XLB',
            'Real Estate': 'XLRE',
            'Utilities': 'XLU',
            'Communication': 'XLC'
        }
        
        self.macro_indicators = {
            'VIX': '^VIX',      # Volatility Index
            'TNX': '^TNX',      # 10-Year Treasury
            'DXY': 'DX-Y.NYB',  # Dollar Index
            'GLD': 'GLD',       # Gold ETF
            'TLT': 'TLT'        # 20+ Year Treasury Bond ETF
        }
        
    def enrich_stock_features(self, symbol: str, base_features: Dict) -> Dict:
        """
        Enrich base technical features with fundamental data
        """
        logger.info(f"Enriching features for {symbol}")
        
        try:
            # Get stock info and financials
            ticker = yf.Ticker(symbol)
            
            # Fundamental features
            fundamental_features = self._extract_fundamental_features(ticker)
            
            # Sector features
            sector_features = self._extract_sector_features(ticker)
            
            # Macroeconomic features
            macro_features = self._extract_macro_features()
            
            # Earnings calendar features
            earnings_features = self._extract_earnings_features(ticker)
            
            # Combine all features
            enriched_features = {
                **base_features,
                **fundamental_features,
                **sector_features, 
                **macro_features,
                **earnings_features,
                'enrichment_timestamp': datetime.now().isoformat(),
                'feature_version': 'v2.0_fundamental'
            }
            
            logger.info(f"Enriched {symbol} with {len(enriched_features) - len(base_features)} additional features")
            return enriched_features
            
        except Exception as e:
            logger.error(f"Error enriching features for {symbol}: {e}")
            return base_features
    
    def _extract_fundamental_features(self, ticker) -> Dict:
        """Extract fundamental financial metrics"""
        try:
            info = ticker.info
            
            # Key fundamental ratios
            pe_ratio = info.get('trailingPE', None)
            forward_pe = info.get('forwardPE', None)
            peg_ratio = info.get('pegRatio', None)
            price_to_book = info.get('priceToBook', None)
            price_to_sales = info.get('priceToSalesTrailing12Months', None)
            
            # Profitability metrics
            profit_margin = info.get('profitMargins', None)
            operating_margin = info.get('operatingMargins', None)
            roe = info.get('returnOnEquity', None)
            roa = info.get('returnOnAssets', None)
            
            # Growth metrics
            earnings_growth = info.get('earningsGrowth', None)
            revenue_growth = info.get('revenueGrowth', None)
            
            # Financial health
            debt_to_equity = info.get('debtToEquity', None)
            current_ratio = info.get('currentRatio', None)
            quick_ratio = info.get('quickRatio', None)
            
            # Market metrics
            market_cap = info.get('marketCap', None)
            enterprise_value = info.get('enterpriseValue', None)
            beta = info.get('beta', None)
            
            # Dividend metrics
            dividend_yield = info.get('dividendYield', None)
            payout_ratio = info.get('payoutRatio', None)
            
            # Normalize and handle missing values
            features = {
                # Valuation (normalized to reasonable ranges)
                'pe_ratio': self._normalize_ratio(pe_ratio, 5, 50),
                'forward_pe': self._normalize_ratio(forward_pe, 5, 50),
                'peg_ratio': self._normalize_ratio(peg_ratio, 0.5, 3.0),
                'price_to_book': self._normalize_ratio(price_to_book, 0.5, 10),
                'price_to_sales': self._normalize_ratio(price_to_sales, 0.5, 20),
                
                # Profitability (as percentages normalized to 0-1)
                'profit_margin': self._normalize_percentage(profit_margin),
                'operating_margin': self._normalize_percentage(operating_margin),
                'roe': self._normalize_percentage(roe),
                'roa': self._normalize_percentage(roa),
                
                # Growth (normalized growth rates)
                'earnings_growth': self._normalize_growth_rate(earnings_growth),
                'revenue_growth': self._normalize_growth_rate(revenue_growth),
                
                # Financial health
                'debt_to_equity': self._normalize_ratio(debt_to_equity, 0, 5),
                'current_ratio': self._normalize_ratio(current_ratio, 0.5, 5),
                'quick_ratio': self._normalize_ratio(quick_ratio, 0.5, 3),
                
                # Market position
                'market_cap_log': self._normalize_market_cap(market_cap),
                'enterprise_value_log': self._normalize_market_cap(enterprise_value),
                'beta': self._normalize_ratio(beta, 0, 3),
                
                # Income
                'dividend_yield': self._normalize_percentage(dividend_yield),
                'payout_ratio': self._normalize_percentage(payout_ratio)
            }
            
            return features
            
        except Exception as e:
            logger.warning(f"Error extracting fundamental features: {e}")
            return self._get_default_fundamental_features()
    
    def _normalize_ratio(self, value: Optional[float], min_val: float, max_val: float) -> float:
        """Normalize ratio to 0-1 range with outlier handling"""
        if value is None or not isinstance(value, (int, float)) or np.isnan(value):
            return 0.5  # Neutral for missing data
        
        # Clip outliers
        clipped = max(min_val, min(max_val, value))
        
        # Normalize to 0-1
        return (clipped - min_val) / (max_val - min_val)
    
    def _normalize_percentage(self, value: Optional[float]) -> float:
        """Normalize percentage values"""
        if value is None or not isinstance(value, (int, float)) or np.isnan(value):
            return 0.5
        
        # Convert to percentage if needed and normalize
        if abs(value) <= 1:  # Already in decimal form
            return max(0, min(1, (value + 0.5)))  # Shift and normalize
        else:  # In percentage form
            return max(0, min(1, (value / 100 + 0.5)))
    
    def _normalize_growth_rate(self, value: Optional[float]) -> float:
        """Normalize growth rates (can be negative)"""
        if value is None or not isinstance(value, (int, float)) or np.isnan(value):
            return 0.5
        
        # Clip extreme values and normalize around 0
        clipped = max(-1.0, min(3.0, value))  # -100% to 300% growth
        return (clipped + 1.0) / 4.0  # Normalize to 0-1
    
    def _normalize_market_cap(self, value: Optional[float]) -> float:
        """Normalize market cap using log scale"""
        if value is None or not isinstance(value, (int, float)) or value <= 0:
            return 0.5
        
        # Log normalize (typical range: $100M to $3T)
        log_value = np.log10(value)
        normalized = (log_value - 8) / 4  # 10^8 to 10^12 range
        return max(0, min(1, normalized))
    
    def _get_default_fundamental_features(self) -> Dict:
        """Return default values when fundamental data unavailable"""
        return {
            'pe_ratio': 0.5, 'forward_pe': 0.5, 'peg_ratio': 0.5,
            'price_to_book': 0.5, 'price_to_sales': 0.5,
            'profit_margin': 0.5, 'operating_margin': 0.5,
            'roe': 0.5, 'roa': 0.5,
            'earnings_growth': 0.5, 'revenue_growth': 0.5,
            'debt_to_equity': 0.5, 'current_ratio': 0.5, 'quick_ratio': 0.5,
            'market_cap_log': 0.5, 'enterprise_value_log': 0.5, 'beta': 0.5,
            'dividend_yield': 0.5, 'payout_ratio': 0.5
        }
    
    def _extract_sector_features(self, ticker) -> Dict:
        """Extract sector rotation and relative performance features"""
        try:
            info = ticker.info
            sector = info.get('sector', 'Unknown')
            
            # Map to sector ETF
            sector_etf = self.sector_etfs.get(sector, 'SPY')
            
            # Get sector performance
            sector_ticker = yf.Ticker(sector_etf)
            sector_hist = sector_ticker.history(period='3mo')
            spy_ticker = yf.Ticker('SPY')
            spy_hist = spy_ticker.history(period='3mo')
            
            if sector_hist.empty or spy_hist.empty:
                return self._get_default_sector_features()
            
            # Calculate sector relative performance
            sector_return_1m = (sector_hist['Close'].iloc[-1] / sector_hist['Close'].iloc[-20] - 1)
            spy_return_1m = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[-20] - 1)
            sector_relative_1m = sector_return_1m - spy_return_1m
            
            # 3-month relative performance
            sector_return_3m = (sector_hist['Close'].iloc[-1] / sector_hist['Close'].iloc[0] - 1)
            spy_return_3m = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1)
            sector_relative_3m = sector_return_3m - spy_return_3m
            
            # Sector momentum
            sector_momentum = (sector_hist['Close'].rolling(5).mean().iloc[-1] / 
                             sector_hist['Close'].rolling(20).mean().iloc[-1] - 1)
            
            return {
                'sector': self._encode_sector(sector),
                'sector_relative_1m': self._normalize_return(sector_relative_1m),
                'sector_relative_3m': self._normalize_return(sector_relative_3m),
                'sector_momentum': self._normalize_return(sector_momentum),
                'sector_strength': 1.0 if sector_relative_1m > 0.02 else 0.0
            }
            
        except Exception as e:
            logger.warning(f"Error extracting sector features: {e}")
            return self._get_default_sector_features()
    
    def _encode_sector(self, sector: str) -> float:
        """One-hot encode sectors into single value for now (can expand later)"""
        sector_rankings = {
            'Technology': 0.9,
            'Healthcare': 0.8,
            'Financial': 0.7,
            'Consumer Discretionary': 0.6,
            'Communication': 0.6,
            'Industrial': 0.5,
            'Consumer Staples': 0.4,
            'Materials': 0.4,
            'Energy': 0.3,
            'Utilities': 0.2,
            'Real Estate': 0.2
        }
        
        return sector_rankings.get(sector, 0.5)
    
    def _normalize_return(self, return_value: float) -> float:
        """Normalize return values to 0-1 range"""
        if return_value is None or np.isnan(return_value):
            return 0.5
        
        # Clip to reasonable range (-50% to +100%) and normalize
        clipped = max(-0.5, min(1.0, return_value))
        return (clipped + 0.5) / 1.5
    
    def _get_default_sector_features(self) -> Dict:
        """Default sector features when data unavailable"""
        return {
            'sector': 0.5,
            'sector_relative_1m': 0.5,
            'sector_relative_3m': 0.5, 
            'sector_momentum': 0.5,
            'sector_strength': 0.0
        }
    
    def _extract_macro_features(self) -> Dict:
        """Extract macroeconomic environment features"""
        try:
            macro_features = {}
            
            for indicator, symbol in self.macro_indicators.items():
                try:
                    ticker = yf.Ticker(symbol)
                    hist = ticker.history(period='1mo')
                    
                    if not hist.empty:
                        current_value = hist['Close'].iloc[-1]
                        month_ago_value = hist['Close'].iloc[0]
                        
                        # Calculate change
                        change = (current_value - month_ago_value) / month_ago_value
                        
                        # Normalize based on indicator type
                        if indicator == 'VIX':
                            # VIX: 10-50 range, higher = more fearful
                            macro_features['vix_level'] = min(1.0, max(0.0, (current_value - 10) / 40))
                            macro_features['vix_change'] = self._normalize_return(change)
                        elif indicator == 'TNX':
                            # 10Y Treasury: 0-10% range  
                            macro_features['treasury_10y'] = min(1.0, max(0.0, current_value / 10))
                            macro_features['rates_direction'] = 1.0 if change > 0 else 0.0
                        elif indicator == 'DXY':
                            # Dollar strength
                            macro_features['dollar_strength'] = self._normalize_return(change)
                        elif indicator == 'GLD':
                            # Gold as risk-off indicator
                            macro_features['gold_momentum'] = self._normalize_return(change)
                        elif indicator == 'TLT':
                            # Bond momentum
                            macro_features['bond_momentum'] = self._normalize_return(change)
                            
                except Exception as e:
                    logger.warning(f"Error fetching {indicator}: {e}")
                    continue
            
            # Add default values for missing indicators
            defaults = {
                'vix_level': 0.4, 'vix_change': 0.5,
                'treasury_10y': 0.4, 'rates_direction': 0.5,
                'dollar_strength': 0.5, 'gold_momentum': 0.5,
                'bond_momentum': 0.5
            }
            
            for key, default_value in defaults.items():
                if key not in macro_features:
                    macro_features[key] = default_value
            
            # Calculate macro regime
            macro_features['risk_on_regime'] = self._calculate_risk_regime(macro_features)
            
            return macro_features
            
        except Exception as e:
            logger.error(f"Error extracting macro features: {e}")
            return self._get_default_macro_features()
    
    def _calculate_risk_regime(self, macro_features: Dict) -> float:
        """Calculate risk-on vs risk-off regime"""
        try:
            risk_on_score = 0.5
            
            # Low VIX = risk on
            if macro_features.get('vix_level', 0.5) < 0.3:
                risk_on_score += 0.2
            elif macro_features.get('vix_level', 0.5) > 0.7:
                risk_on_score -= 0.2
            
            # Rising rates can be risk-on (growth) or risk-off (tightening)
            rates_level = macro_features.get('treasury_10y', 0.4)
            if 0.2 <= rates_level <= 0.6:  # Goldilocks zone
                risk_on_score += 0.1
            
            # Strong dollar can hurt stocks
            if macro_features.get('dollar_strength', 0.5) > 0.7:
                risk_on_score -= 0.1
            
            # Gold strength indicates risk-off
            if macro_features.get('gold_momentum', 0.5) > 0.6:
                risk_on_score -= 0.1
            
            return max(0, min(1, risk_on_score))
            
        except Exception as e:
            logger.warning(f"Error calculating risk regime: {e}")
            return 0.5
    
    def _get_default_macro_features(self) -> Dict:
        """Default macro features when data unavailable"""
        return {
            'vix_level': 0.4, 'vix_change': 0.5,
            'treasury_10y': 0.4, 'rates_direction': 0.5,
            'dollar_strength': 0.5, 'gold_momentum': 0.5,
            'bond_momentum': 0.5, 'risk_on_regime': 0.5
        }
    
    def _extract_earnings_features(self, ticker) -> Dict:
        """Extract earnings calendar and expectations features"""
        try:
            # Get earnings data
            earnings_calendar = ticker.calendar
            earnings_history = ticker.earnings_history
            
            earnings_features = {}
            
            # Days to next earnings
            if earnings_calendar is not None and not earnings_calendar.empty:
                next_earnings = earnings_calendar.index[0]
                days_to_earnings = (next_earnings - datetime.now()).days
                earnings_features['days_to_earnings'] = min(1.0, max(0.0, (90 - days_to_earnings) / 90))
                earnings_features['earnings_this_month'] = 1.0 if days_to_earnings <= 30 else 0.0
            else:
                earnings_features['days_to_earnings'] = 0.5
                earnings_features['earnings_this_month'] = 0.0
            
            # Earnings surprise history
            if earnings_history is not None and not earnings_history.empty and len(earnings_history) > 0:
                recent_surprises = []
                for _, row in earnings_history.head(4).iterrows():  # Last 4 quarters
                    eps_estimate = row.get('epsestimate', 0)
                    eps_actual = row.get('epsactual', 0)
                    
                    if eps_estimate and eps_estimate != 0:
                        surprise = (eps_actual - eps_estimate) / abs(eps_estimate)
                        recent_surprises.append(surprise)
                
                if recent_surprises:
                    avg_surprise = np.mean(recent_surprises)
                    earnings_features['earnings_surprise_avg'] = self._normalize_return(avg_surprise)
                    earnings_features['positive_surprise_trend'] = 1.0 if avg_surprise > 0.05 else 0.0
                else:
                    earnings_features['earnings_surprise_avg'] = 0.5
                    earnings_features['positive_surprise_trend'] = 0.0
            else:
                earnings_features['earnings_surprise_avg'] = 0.5
                earnings_features['positive_surprise_trend'] = 0.0
            
            return earnings_features
            
        except Exception as e:
            logger.warning(f"Error extracting earnings features: {e}")
            return {
                'days_to_earnings': 0.5,
                'earnings_this_month': 0.0,
                'earnings_surprise_avg': 0.5,
                'positive_surprise_trend': 0.0
            }
    
    def calculate_sector_rotation_signal(self) -> Dict:
        """Calculate sector rotation signals based on relative performance"""
        try:
            # Get sector ETF performance
            sector_performance = {}
            
            for sector, etf_symbol in self.sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf_symbol)
                    hist = ticker.history(period='3mo')
                    
                    if not hist.empty:
                        # 1-month and 3-month returns
                        return_1m = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1)
                        return_3m = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1)
                        
                        # Momentum
                        momentum = (hist['Close'].rolling(5).mean().iloc[-1] / 
                                  hist['Close'].rolling(20).mean().iloc[-1] - 1)
                        
                        sector_performance[sector] = {
                            'return_1m': return_1m,
                            'return_3m': return_3m,
                            'momentum': momentum,
                            'etf_symbol': etf_symbol
                        }
                        
                except Exception as e:
                    logger.warning(f"Error fetching sector data for {sector}: {e}")
                    continue
            
            if not sector_performance:
                return {'error': 'No sector data available'}
            
            # Rank sectors by performance
            sectors_by_1m = sorted(sector_performance.items(), 
                                 key=lambda x: x[1]['return_1m'], reverse=True)
            sectors_by_3m = sorted(sector_performance.items(),
                                 key=lambda x: x[1]['return_3m'], reverse=True) 
            sectors_by_momentum = sorted(sector_performance.items(),
                                       key=lambda x: x[1]['momentum'], reverse=True)
            
            # Calculate rotation signals
            rotation_signals = {
                'leading_sectors_1m': [s[0] for s in sectors_by_1m[:3]],
                'lagging_sectors_1m': [s[0] for s in sectors_by_1m[-3:]],
                'leading_sectors_3m': [s[0] for s in sectors_by_3m[:3]],
                'momentum_leaders': [s[0] for s in sectors_by_momentum[:3]],
                'sector_dispersion': self._calculate_sector_dispersion(sector_performance),
                'rotation_strength': self._calculate_rotation_strength(sectors_by_1m)
            }
            
            return rotation_signals
            
        except Exception as e:
            logger.error(f"Error calculating sector rotation: {e}")
            return {'error': str(e)}
    
    def _calculate_sector_dispersion(self, sector_performance: Dict) -> float:
        """Calculate how spread out sector performance is"""
        returns_1m = [data['return_1m'] for data in sector_performance.values()]
        
        if len(returns_1m) < 2:
            return 0.5
        
        std_dev = np.std(returns_1m)
        
        # Normalize: high dispersion = strong rotation
        return min(1.0, max(0.0, std_dev / 0.1))  # 10% std = max dispersion
    
    def _calculate_rotation_strength(self, sorted_sectors: List) -> float:
        """Calculate strength of sector rotation"""
        if len(sorted_sectors) < 6:
            return 0.5
        
        # Compare top 3 vs bottom 3 sectors
        top_3_avg = np.mean([s[1]['return_1m'] for s in sorted_sectors[:3]])
        bottom_3_avg = np.mean([s[1]['return_1m'] for s in sorted_sectors[-3:]])
        
        rotation_spread = top_3_avg - bottom_3_avg
        
        # Strong rotation = large spread between winners and losers
        return min(1.0, max(0.0, rotation_spread / 0.2))  # 20% spread = max rotation

def lambda_handler(event, context):
    """Lambda handler for fundamental data enrichment"""
    try:
        action = event.get('action', 'enrich_features')
        enricher = FundamentalDataEnricher()
        
        if action == 'enrich_features':
            symbol = event.get('symbol')
            base_features = event.get('base_features', {})
            
            if not symbol:
                return {
                    'statusCode': 400,
                    'body': json.dumps({'error': 'Symbol required'})
                }
            
            enriched = enricher.enrich_stock_features(symbol, base_features)
            
            return {
                'statusCode': 200,
                'body': json.dumps(enriched, default=str)
            }
            
        elif action == 'sector_rotation':
            rotation_signals = enricher.calculate_sector_rotation_signal()
            
            return {
                'statusCode': 200,
                'body': json.dumps(rotation_signals, default=str)
            }
        
        else:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': f'Unknown action: {action}'})
            }
            
    except Exception as e:
        logger.error(f"Error in fundamental enrichment: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }