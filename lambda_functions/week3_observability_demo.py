"""
Week 3 Advanced Observability Demo
Demonstrates the new observability intelligence features and SigNoz integration
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Demo imports
try:
    from shared.observability_intelligence import (
        get_performance_monitor, get_sampling_optimizer, get_trading_intelligence
    )
    from shared.signoz_integration import (
        SigNozDashboardBuilder, SigNozQueryBuilder, generate_signoz_dashboard_config
    )
    from shared.business_tracing import get_financial_tracer
    from shared.market_utils import get_market_session, classify_symbol
    WEEK3_FEATURES_AVAILABLE = True
except ImportError as e:
    print(f"Week 3 features not available: {e}")
    WEEK3_FEATURES_AVAILABLE = False

logger = logging.getLogger(__name__)

def lambda_handler(event, context):
    """
    Demo handler showcasing Week 3 advanced observability features
    """
    if not WEEK3_FEATURES_AVAILABLE:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Week 3 observability features not available',
                'message': 'Missing required modules'
            })
        }

    demo_type = event.get('demo_type', 'full_demo')

    results = {}

    try:
        if demo_type in ['full_demo', 'performance_monitoring']:
            results['performance_monitoring'] = demo_performance_monitoring()

        if demo_type in ['full_demo', 'trading_intelligence']:
            results['trading_intelligence'] = demo_trading_intelligence()

        if demo_type in ['full_demo', 'dynamic_sampling']:
            results['dynamic_sampling'] = demo_dynamic_sampling()

        if demo_type in ['full_demo', 'signoz_dashboards']:
            results['signoz_dashboards'] = demo_signoz_integration()

        return {
            'statusCode': 200,
            'body': json.dumps(results, default=str, indent=2)
        }

    except Exception as e:
        logger.error(f"Error in Week 3 demo: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': 'Demo failed',
                'message': str(e)
            })
        }

def demo_performance_monitoring() -> Dict[str, Any]:
    """
    Demonstrate advanced performance monitoring capabilities
    """
    print("🔍 Demonstrating Performance Monitoring...")

    # Initialize performance monitor
    performance_monitor = get_performance_monitor("demo_service")

    # Create sample ML accuracy tracking
    test_symbols = ['AAPL', 'GOOGL', 'TSLA', 'AMD', 'NVDA']
    demo_results = []

    for symbol in test_symbols:
        # Simulate prediction vs actual price comparison
        predicted_price = 150.0
        actual_price = 148.5  # Simulate slight difference
        confidence_score = 0.85

        # Track accuracy
        accuracy_span = performance_monitor.track_ml_accuracy(
            symbol=symbol,
            predicted_price=predicted_price,
            actual_price=actual_price,
            confidence_score=confidence_score,
            prediction_timestamp=datetime.utcnow() - timedelta(hours=2),
            actual_timestamp=datetime.utcnow()
        )
        accuracy_span.end()

        demo_results.append({
            'symbol': symbol,
            'predicted_price': predicted_price,
            'actual_price': actual_price,
            'accuracy_pct': ((1 - abs(predicted_price - actual_price) / actual_price) * 100)
        })

    # Generate performance summary
    performance_summary = performance_monitor.get_performance_summary(lookback_hours=24)

    return {
        'demo_predictions': demo_results,
        'performance_summary': performance_summary,
        'market_session': get_market_session().value,
        'timestamp': datetime.utcnow().isoformat()
    }

def demo_trading_intelligence() -> Dict[str, Any]:
    """
    Demonstrate trading intelligence and market opportunity analysis
    """
    print("📊 Demonstrating Trading Intelligence...")

    trading_intelligence = get_trading_intelligence()

    # Analyze opportunities for major symbols
    major_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    opportunities = []

    for symbol in major_symbols:
        # Simulate trace data (in real scenario, this would come from actual traces)
        mock_traces = [
            {
                'finance.symbol': symbol,
                'finance.confidence_score': 0.87,
                'ml.accuracy.price_error_pct': 3.2,
                'finance.target_price': 175.50,
                'timestamp': datetime.utcnow().isoformat()
            }
        ]

        opportunity = trading_intelligence.analyze_market_opportunity(symbol, mock_traces)
        opportunities.append(opportunity)

    # Sort by opportunity score
    opportunities.sort(key=lambda x: x['opportunity_score'], reverse=True)

    return {
        'market_opportunities': opportunities,
        'top_opportunity': opportunities[0] if opportunities else None,
        'current_market_session': get_market_session().value,
        'analysis_timestamp': datetime.utcnow().isoformat()
    }

def demo_dynamic_sampling() -> Dict[str, Any]:
    """
    Demonstrate dynamic sampling optimization
    """
    print("⚙️ Demonstrating Dynamic Sampling...")

    sampling_optimizer = get_sampling_optimizer()

    # Simulate performance summary for optimization
    mock_performance = {
        'accuracy_mean': 78.5,
        'total_predictions': 150,
        'lookback_hours': 24
    }

    # Get optimized sampling rates
    optimized_rates = sampling_optimizer.optimize_sampling_rates(
        performance_summary=mock_performance,
        cost_budget_daily=50.0
    )

    # Show symbol tier classifications
    symbol_classifications = {}
    test_symbols = ['AAPL', 'AMD', 'UNKNOWN_STOCK', 'SPY', 'SMALL_CAP_STOCK']

    for symbol in test_symbols:
        tier = classify_symbol(symbol)
        symbol_classifications[symbol] = {
            'tier': tier.value,
            'sampling_rate': optimized_rates.get(tier.value, 0.25)
        }

    return {
        'performance_input': mock_performance,
        'optimized_sampling_rates': optimized_rates,
        'symbol_classifications': symbol_classifications,
        'optimization_timestamp': datetime.utcnow().isoformat()
    }

def demo_signoz_integration() -> Dict[str, Any]:
    """
    Demonstrate SigNoz dashboard and query integration
    """
    print("📈 Demonstrating SigNoz Integration...")

    # Build dashboard configurations
    dashboard_builder = SigNozDashboardBuilder()

    # Generate sample queries
    query_builder = SigNozQueryBuilder()

    sample_queries = {
        'symbol_performance_AAPL': query_builder.get_symbol_performance_query('AAPL', 24),
        'market_hours_analysis': query_builder.get_market_session_analysis_query('market_hours'),
        'low_confidence_symbols': query_builder.get_low_confidence_symbols_query(0.6)
    }

    # Create sample dashboard
    ml_dashboard = dashboard_builder.create_ml_performance_dashboard()
    trading_dashboard = dashboard_builder.create_trading_signals_dashboard()

    return {
        'sample_queries': sample_queries,
        'ml_performance_dashboard': {
            'title': ml_dashboard['title'],
            'widget_count': len(ml_dashboard['widgets']),
            'tags': ml_dashboard['tags']
        },
        'trading_signals_dashboard': {
            'title': trading_dashboard['title'],
            'widget_count': len(trading_dashboard['widgets']),
            'tags': trading_dashboard['tags']
        },
        'dashboard_config_available': True,
        'export_timestamp': datetime.utcnow().isoformat()
    }

def export_dashboard_configs():
    """
    Export complete dashboard configurations for SigNoz import
    """
    if not WEEK3_FEATURES_AVAILABLE:
        print("Week 3 features not available for dashboard export")
        return None

    config_json = generate_signoz_dashboard_config()

    # Save to file for manual import
    with open('/tmp/signoz_dashboards.json', 'w') as f:
        f.write(config_json)

    print("📁 Dashboard configurations exported to /tmp/signoz_dashboards.json")
    print("Import these into SigNoz for complete observability setup")

    return config_json

if __name__ == "__main__":
    # Run demo if executed directly
    print("🚀 Starting Week 3 Advanced Observability Demo")
    print("=" * 60)

    demo_event = {'demo_type': 'full_demo'}
    demo_context = {}

    result = lambda_handler(demo_event, demo_context)

    if result['statusCode'] == 200:
        print("✅ Demo completed successfully!")
        print("\nKey Features Demonstrated:")
        print("• Performance monitoring with ML accuracy tracking")
        print("• Trading intelligence and market opportunity analysis")
        print("• Dynamic sampling rate optimization")
        print("• SigNoz dashboard integration")

        # Export dashboard configs
        export_dashboard_configs()

    else:
        print("❌ Demo failed:")
        print(result['body'])

    print("\n" + "=" * 60)
    print("Week 3 Advanced Observability Demo Complete")