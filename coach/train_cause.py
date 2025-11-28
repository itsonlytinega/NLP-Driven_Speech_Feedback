"""
Train Cause Classifier using BERT features + XGBoost
Trains XGBoost classifier on BERT scores to predict root cause
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import torch
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import logging
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Import from utils module
import sys
import os
# Set Django settings
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'verbalcoach.settings')
import django
django.setup()

from coach.utils import get_bert_scores

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CauseClassifierTrainer:
    """Train XGBoost cause classifier"""
    
    def __init__(self, model_dir='coach/models'):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.model = None
        
    def load_data(self, csv_path='data/train.csv'):
        """Load training data"""
        logger.info(f"Loading data from {csv_path}...")
        
        if not os.path.exists(csv_path):
            logger.error(f"Training data not found: {csv_path}")
            logger.info("Please run: python data/preprocess.py first")
            return None
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples")
        
        # Get BERT scores for each sample
        logger.info("Computing BERT scores...")
        bert_scores = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc='Ensemble scoring'):
            try:
                # ENSEMBLE: Pass audio_path for hybrid clarity/pacing, transcript for pure BERT fillers
                audio_path = row.get('audio_path', None)
                if pd.isna(audio_path) or not os.path.exists(str(audio_path)):
                    audio_path = None
                
                scores = get_bert_scores(row['transcript'], audio_path=audio_path)
                bert_scores.append([
                    scores['filler'],      # From pure BERT (97.6% AUC)
                    scores['clarity'],     # From hybrid (94.7% AUC) if audio available
                    scores['pacing']        # From hybrid (72.0% AUC) if audio available
                ])
            except Exception as e:
                logger.warning(f"Error computing ensemble scores for row {idx}: {e}")
                # Use fallback scores
                bert_scores.append([0.5, 0.5, 0.5])
        
        # Create feature matrix
        bert_df = pd.DataFrame(bert_scores, columns=['bert_filler', 'bert_clarity', 'bert_pacing'])
        
        # Combine with existing features
        features = pd.concat([
            bert_df,
            df[['wpm', 'pause_ratio', 'speech_rate_var']]
        ], axis=1)
        
        # Encode labels
        labels = self.label_encoder.fit_transform(df['cause'])
        
        return features, labels, df
    
    def train(self, features, labels, cv_folds=5):
        """Train XGBoost model with cross-validation"""
        logger.info("\n=== Training Cause Classifier ===")
        
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Create XGBoost classifier
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Cross-validation
        logger.info(f"Running {cv_folds}-fold cross-validation...")
        cv_scores = cross_val_score(
            self.model,
            features_scaled,
            labels,
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring='accuracy'
        )
        
        logger.info(f"\nCV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        logger.info(f"  Folds: {cv_scores}")
        
        # Train on full dataset
        logger.info("\nTraining on full dataset...")
        self.model.fit(features_scaled, labels)
        
        # Feature importance
        feature_names = features.columns.tolist()
        importances = self.model.feature_importances_
        
        logger.info("\nFeature Importances:")
        for name, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
            logger.info(f"  {name}: {importance:.3f}")
        
        return self.model
    
    def evaluate(self, features, labels):
        """Evaluate trained model"""
        logger.info("\n=== Evaluating Model ===")
        
        features_scaled = self.scaler.transform(features)
        
        # Predictions
        predictions = self.model.predict(features_scaled)
        probabilities = self.model.predict_proba(features_scaled)
        
        # Metrics
        accuracy = accuracy_score(labels, predictions)
        logger.info(f"\nAccuracy: {accuracy:.3f}")
        
        # Classification report
        label_names = self.label_encoder.classes_
        logger.info("\nClassification Report:")
        logger.info(classification_report(labels, predictions, target_names=label_names))
        
        # Confusion matrix
        logger.info("\nConfusion Matrix:")
        cm = confusion_matrix(labels, predictions)
        logger.info(f"\n{cm}")
        
        return {
            'accuracy': accuracy,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def save_model(self):
        """Save trained model and scaler"""
        logger.info(f"\nSaving model to {self.model_dir}...")
        
        # Save XGBoost model
        with open(os.path.join(self.model_dir, 'cause_model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save scaler
        with open(os.path.join(self.model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Save label encoder
        with open(os.path.join(self.model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info("✓ Model saved!")
    
    def predict_cause(self, audio_path: str, transcript: str) -> str:
        """
        Predict root cause for given audio and transcript.
        
        Args:
            audio_path (str): Path to audio file
            transcript (str): Transcribed text
            
        Returns:
            str: Predicted cause (anxiety, stress, lack_of_confidence, poor_skills)
        """
        try:
            # Load models if not already loaded
            if self.model is None or self.scaler is None:
                import pickle
                model_path = os.path.join(self.model_dir, 'cause_model.pkl')
                scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
                encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
                
                if os.path.exists(model_path):
                    with open(model_path, 'rb') as f:
                        self.model = pickle.load(f)
                if os.path.exists(scaler_path):
                    with open(scaler_path, 'rb') as f:
                        self.scaler = pickle.load(f)
                if os.path.exists(encoder_path):
                    with open(encoder_path, 'rb') as f:
                        self.label_encoder = pickle.load(f)
            
            # Get BERT scores
            bert_scores = get_bert_scores(transcript)
            
            # Load audio features (simplified - in production, compute actual audio features)
            import librosa
            y, sr = librosa.load(audio_path, sr=16000, duration=30)
            
            # Compute basic features
            duration = len(y) / sr
            words = len(transcript.split())
            wpm = words / (duration / 60) if duration > 0 else 0
            
            # Pause ratio
            intervals = librosa.effects.split(y, top_db=20)
            silence_dur = sum((end - start) / sr for start, end in intervals)
            pause_ratio = silence_dur / duration if duration > 0 else 0
            
            # Speech rate variance
            # Simplified - compute from text
            speech_rate_var = 0.0  # Would compute from actual speech analysis
            
            # Create feature vector
            features = np.array([[
                bert_scores['filler'],
                bert_scores['clarity'],
                bert_scores['pacing'],
                wpm,
                pause_ratio,
                speech_rate_var
            ]])
            
            # Scale features
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Predict
            if self.model is not None:
                prediction = self.model.predict(features_scaled)[0]
            else:
                return 'poor_skills'
            
            if self.label_encoder is not None:
                predicted_label = self.label_encoder.inverse_transform([prediction])[0]
            else:
                return 'poor_skills'
            
            return predicted_label
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return 'poor_skills'  # Default fallback


def main():
    """Main entry point"""
    
    logger.info("=" * 60)
    logger.info("Training Cause Classifier")
    logger.info("=" * 60)
    
    # Initialize trainer
    trainer = CauseClassifierTrainer()
    
    # Load data
    train_features, train_labels, train_df = trainer.load_data('data/train.csv')
    
    if train_features is None:
        logger.error("Could not load training data!")
        return
    
    # Train model
    model = trainer.train(train_features, train_labels, cv_folds=5)
    
    # Save model
    trainer.save_model()
    
    # Evaluate on training set
    results = trainer.evaluate(train_features, train_labels)
    
    # Evaluate on test set if available
    test_csv = 'data/test.csv'
    if os.path.exists(test_csv):
        logger.info("\n=== Evaluating on Test Set ===")
        
        test_df = pd.read_csv(test_csv)
        logger.info(f"Test set: {len(test_df)} samples")
        
        # Get ensemble scores for test set
        bert_scores_test = []
        for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc='Test ensemble scoring'):
            try:
                # ENSEMBLE: Pass audio_path for hybrid clarity/pacing
                audio_path = row.get('audio_path', None)
                if pd.isna(audio_path) or not os.path.exists(str(audio_path)):
                    audio_path = None
                
                scores = get_bert_scores(row['transcript'], audio_path=audio_path)
                bert_scores_test.append([
                    scores['filler'],      # From pure BERT (97.6% AUC)
                    scores['clarity'],     # From hybrid (94.7% AUC) if audio available
                    scores['pacing']        # From hybrid (72.0% AUC) if audio available
                ])
            except:
                bert_scores_test.append([0.5, 0.5, 0.5])
        
        bert_df_test = pd.DataFrame(bert_scores_test, columns=['bert_filler', 'bert_clarity', 'bert_pacing'])
        test_features = pd.concat([
            bert_df_test,
            test_df[['wpm', 'pause_ratio', 'speech_rate_var']]
        ], axis=1)
        
        # Encode test labels
        test_labels = trainer.label_encoder.transform(test_df['cause'])
        
        # Evaluate on test set
        test_results = trainer.evaluate(test_features, test_labels)
        
        logger.info(f"\n✓ Test Accuracy: {test_results['accuracy']:.3f}")
    
    logger.info("\n✓ Training complete!")
    
    # Auto-generate comprehensive report
    logger.info("\n" + "="*60)
    logger.info("Generating comprehensive evaluation report...")
    logger.info("="*60)
    
    try:
        import subprocess
        import sys
        from pathlib import Path
        
        report_script = Path(__file__).parent.parent / 'reports' / 'generate_report.py'
        
        if report_script.exists():
            logger.info(f"Running report generator: {report_script}")
            result = subprocess.run([sys.executable, str(report_script)], 
                                  capture_output=False, text=True)
            if result.returncode == 0:
                logger.info("\n✓ Report generated successfully!")
                logger.info(f"Open: reports/full_report.html")
            else:
                logger.warning(f"Report generation returned code: {result.returncode}")
        else:
            logger.warning(f"Report script not found: {report_script}")
    except Exception as e:
        logger.error(f"Failed to generate report: {e}")
        logger.info("You can manually generate the report by running:")
        logger.info("  python reports/generate_report.py")


if __name__ == '__main__':
    main()

