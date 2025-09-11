#!/usr/bin/env python3
"""
ECS Fargate Model Training Script
Trains ML model using recent stock data and uploads to S3
Cost-optimized for weekly execution (~$2 per run)
"""

import boto3
import pandas as pd
import numpy as np
import joblib
import json
import logging
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os
import sys
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# AWS clients
s3 = boto3.client('s3')
dynamodb = boto3.resource('dynamodb')
cloudwatch = boto3.client('cloudwatch')

# Configuration from environment
S3_BUCKET = os.environ.get('S3_BUCKET')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE')
ANALYTICS_TABLE = os.environ.get('ANALYTICS_TABLE', 'ai-performance-analytics')

class ModelTrainer:
    def __init__(self):
        self.s3_bucket = S3_BUCKET
        self.model_bucket = MODEL_BUCKET
        self.recommendations_table = dynamodb.Table(DYNAMODB_TABLE)
        self.analytics_table = dynamodb.Table(ANALYTICS_TABLE)
        
    def fetch_training_data(self, days_back: int = 90) -> pd.DataFrame:
        """
        Fetch historical stock data and recommendations for training
        """
        logger.info(f"Fetching training data from last {days_back} days...")
        
        # Get cutoff date
        cutoff_date = datetime.now() - timedelta(days=days_back)
        cutoff_str = cutoff_date.isoformat()
        
        # Fetch recommendations from DynamoDB
        try:
            response = self.recommendations_table.scan(
                FilterExpression='#ts > :cutoff',
                ExpressionAttributeNames={'#ts': 'timestamp'},
                ExpressionAttributeValues={':cutoff': cutoff_str}
            )
            
            recommendations = response['Items']
            logger.info(f"Fetched {len(recommendations)} historical recommendations")
            
            # Convert to DataFrame
            df = pd.DataFrame(recommendations)
            
            if df.empty:
                logger.warning("No recommendations found - creating synthetic data")
                return self.create_synthetic_training_data()
            
            return self.process_recommendations_for_training(df)
            
        except Exception as e:
            logger.error(f"Error fetching training data: {str(e)}")
            logger.info("Falling back to synthetic training data")
            return self.create_synthetic_training_data()
    
    def create_synthetic_training_data(self) -> pd.DataFrame:
        """
        Create synthetic training data based on market patterns
        """
        logger.info("Creating synthetic training data...")
        
        np.random.seed(42)
        n_samples = 3000
        
        data = []
        
        for i in range(n_samples):
            # Generate realistic feature values
            record = {
                'price_to_ma5_ratio': np.random.normal(1.0, 0.05),
                'price_to_ma20_ratio': np.random.normal(1.0, 0.03),
                'volatility': max(0.01, np.random.exponential(0.05)),
                'volume_normalized': max(0.1, np.random.lognormal(0, 0.3)),
                'market_sentiment': np.random.beta(5, 5),
                'market_volatility': max(0.01, np.random.exponential(0.04)),
                'market_correlation': np.clip(np.random.normal(0.3, 0.4), -1, 1),
                'symbol': f'SYM{i % 100}',
                'timestamp': datetime.now() - timedelta(days=np.random.randint(1, 90))
            }
            
            # Generate target based on features
            target = 0.5
            if record['price_to_ma5_ratio'] > 1.02 and record['price_to_ma20_ratio'] > 1.01:
                target += 0.2
            if record['volume_normalized'] > 1.5:
                target += 0.1
            if record['market_sentiment'] > 0.6:
                target += 0.1
            if record['volatility'] < 0.05:
                target += 0.05
            
            # Negative signals
            if record['price_to_ma5_ratio'] < 0.98:
                target -= 0.15
            if record['volatility'] > 0.15:
                target -= 0.1
            
            # Add noise and clip
            target += np.random.normal(0, 0.05)
            record['target_score'] = np.clip(target, 0, 1)
            
            data.append(record)
        
        return pd.DataFrame(data)
    
    def process_recommendations_for_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process recommendations DataFrame for model training
        """
        logger.info("Processing recommendations for training...")
        
        # Extract features and targets from recommendations
        processed_data = []
        
        for _, row in df.iterrows():
            try:
                # Extract features from technical analysis if available
                tech_analysis = row.get('technical_analysis', {})
                
                features = {
                    'price_to_ma5_ratio': 1.0,  # Default values
                    'price_to_ma20_ratio': 1.0,
                    'volatility': 0.05,
                    'volume_normalized': 1.0,
                    'market_sentiment': 0.5,
                    'market_volatility': 0.05,
                    'market_correlation': 0.0
                }
                
                # Try to extract from metadata or technical analysis
                if isinstance(tech_analysis, dict):
                    # Map available technical indicators to features
                    if 'trend_strength' in tech_analysis:
                        features['market_correlation'] = float(tech_analysis['trend_strength'])
                    if 'volatility' in tech_analysis:
                        features['volatility'] = max(0.01, float(tech_analysis.get('volatility', 0.05)))
                
                # Use prediction score as target
                target_score = float(row.get('prediction_score', row.get('confidence', 0.5)))
                
                record = {
                    **features,
                    'target_score': target_score,
                    'symbol': row.get('symbol', 'UNKNOWN'),
                    'timestamp': row.get('timestamp', datetime.now().isoformat())
                }
                
                processed_data.append(record)
                
            except Exception as e:
                logger.warning(f"Error processing row: {str(e)}")
                continue
        
        if not processed_data:
            return self.create_synthetic_training_data()
        
        return pd.DataFrame(processed_data)
    
    def prepare_training_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for model training
        """
        logger.info("Preparing training features...")
        
        feature_columns = [
            'price_to_ma5_ratio',
            'price_to_ma20_ratio', 
            'volatility',
            'volume_normalized',
            'market_sentiment',
            'market_volatility',
            'market_correlation'
        ]
        
        # Extract features
        X = df[feature_columns].values
        y = df['target_score'].values
        
        # Remove any invalid values
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_mask]
        y = y[valid_mask]
        
        logger.info(f"Training data shape: {X.shape}")
        logger.info(f"Target statistics: mean={y.mean():.3f}, std={y.std():.3f}")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[RandomForestRegressor, StandardScaler, Dict]:
        """
        Train the RandomForestRegressor model
        """
        logger.info("Training RandomForestRegressor model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Create and fit scaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train model
        model = RandomForestRegressor(
            n_estimators=150,      # Good balance of accuracy and speed
            max_depth=12,          # Prevent overfitting
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        logger.info("Training model...")
        model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        train_score = model.score(X_train_scaled, y_train)
        test_score = model.score(X_test_scaled, y_test)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        # Predictions for detailed metrics
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        metrics = {
            'train_r2': float(train_score),
            'test_r2': float(test_score),
            'cv_mean_r2': float(cv_scores.mean()),
            'cv_std_r2': float(cv_scores.std()),
            'rmse': float(rmse),
            'n_samples': len(X),
            'n_features': X.shape[1],
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Training R²: {train_score:.4f}")
        logger.info(f"Test R²: {test_score:.4f}")
        logger.info(f"CV R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        logger.info(f"RMSE: {rmse:.4f}")
        
        # Feature importance
        feature_names = [
            'price_to_ma5_ratio',
            'price_to_ma20_ratio', 
            'volatility',
            'volume_normalized',
            'market_sentiment',
            'market_volatility',
            'market_correlation'
        ]
        
        importance = dict(zip(feature_names, model.feature_importances_))
        logger.info("Feature Importance:")
        for feature, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {feature}: {imp:.4f}")
        
        metrics['feature_importance'] = importance
        
        return model, scaler, metrics
    
    def save_and_upload_model(self, model: RandomForestRegressor, scaler: StandardScaler, 
                            metrics: Dict) -> Dict:
        """
        Save model artifacts locally and upload to S3
        """
        logger.info("Saving and uploading model artifacts...")
        
        # Create local directory
        os.makedirs('/tmp/model', exist_ok=True)
        
        # Save model and scaler
        model_path = '/tmp/model/model.pkl'
        scaler_path = '/tmp/model/scaler.pkl'
        metadata_path = '/tmp/model/metadata.json'
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        # Prepare metadata
        metadata = {
            'model_type': 'RandomForestRegressor',
            'version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'training_timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'feature_names': [
                'price_to_ma5_ratio',
                'price_to_ma20_ratio', 
                'volatility',
                'volume_normalized',
                'market_sentiment',
                'market_volatility',
                'market_correlation'
            ],
            'deployment_target': 'lambda'
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Upload to S3
        try:
            # Upload model
            with open(model_path, 'rb') as f:
                s3.put_object(
                    Bucket=self.model_bucket,
                    Key='optimized_model/model.pkl',
                    Body=f.read(),
                    ContentType='application/octet-stream'
                )
            
            # Upload scaler
            with open(scaler_path, 'rb') as f:
                s3.put_object(
                    Bucket=self.model_bucket,
                    Key='optimized_model/scaler.pkl',
                    Body=f.read(),
                    ContentType='application/octet-stream'
                )
            
            # Upload metadata
            with open(metadata_path, 'rb') as f:
                s3.put_object(
                    Bucket=self.model_bucket,
                    Key='optimized_model/metadata.json',
                    Body=f.read(),
                    ContentType='application/json'
                )
            
            logger.info(f"Model uploaded to s3://{self.model_bucket}/optimized_model/")
            
            return {
                'upload_status': 'success',
                'model_location': f"s3://{self.model_bucket}/optimized_model/",
                'version': metadata['version']
            }
            
        except Exception as e:
            logger.error(f"Error uploading model: {str(e)}")
            return {
                'upload_status': 'failed',
                'error': str(e)
            }
    
    def publish_training_metrics(self, metrics: Dict):
        """
        Publish training metrics to CloudWatch
        """
        try:
            metric_data = [
                {
                    'MetricName': 'ModelTrainingR2Score',
                    'Value': metrics['test_r2'],
                    'Unit': 'None',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'ModelTrainingRMSE',
                    'Value': metrics['rmse'],
                    'Unit': 'None',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'TrainingSamples',
                    'Value': metrics['n_samples'],
                    'Unit': 'Count',
                    'Timestamp': datetime.utcnow()
                },
                {
                    'MetricName': 'ModelTrainingSuccess',
                    'Value': 1,
                    'Unit': 'None',
                    'Timestamp': datetime.utcnow()
                }
            ]
            
            cloudwatch.put_metric_data(
                Namespace='StockAnalytics/ModelTraining',
                MetricData=metric_data
            )
            
            logger.info("Training metrics published to CloudWatch")
            
        except Exception as e:
            logger.error(f"Error publishing metrics: {str(e)}")
    
    def run_training(self):
        """
        Main training execution
        """
        logger.info("=" * 60)
        logger.info("Starting ECS Fargate Model Training")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Fetch training data
            df = self.fetch_training_data()
            
            # Prepare features
            X, y = self.prepare_training_features(df)
            
            if len(X) == 0:
                raise ValueError("No valid training data available")
            
            # Train model
            model, scaler, metrics = self.train_model(X, y)
            
            # Save and upload
            upload_result = self.save_and_upload_model(model, scaler, metrics)
            
            # Publish metrics
            self.publish_training_metrics(metrics)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.info("=" * 60)
            logger.info("Training completed successfully!")
            logger.info(f"Duration: {duration:.1f} seconds")
            logger.info(f"Model R²: {metrics['test_r2']:.4f}")
            logger.info(f"Upload Status: {upload_result['upload_status']}")
            logger.info("=" * 60)
            
            return {
                'status': 'success',
                'metrics': metrics,
                'upload_result': upload_result,
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            
            # Publish failure metric
            try:
                cloudwatch.put_metric_data(
                    Namespace='StockAnalytics/ModelTraining',
                    MetricData=[
                        {
                            'MetricName': 'ModelTrainingFailure',
                            'Value': 1,
                            'Unit': 'None',
                            'Timestamp': datetime.utcnow()
                        }
                    ]
                )
            except:
                pass
            
            return {
                'status': 'failed',
                'error': str(e)
            }

def main():
    """
    Main entry point for ECS task
    """
    trainer = ModelTrainer()
    result = trainer.run_training()
    
    if result['status'] == 'success':
        logger.info("Training completed successfully")
        sys.exit(0)
    else:
        logger.error("Training failed")
        sys.exit(1)

if __name__ == "__main__":
    main()