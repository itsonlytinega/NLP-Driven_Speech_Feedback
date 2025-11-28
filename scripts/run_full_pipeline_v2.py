"""
Automated Full Pipeline Runner V2
Runs preprocessing, training, and evaluation automatically
All output logged to files
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import logging

# Setup logging
TIMESTAMP = datetime.now().strftime("%Y%m%d")
V2_SUFFIX = f"v2_{TIMESTAMP}"

log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)
pipeline_log = log_dir / f'pipeline_{V2_SUFFIX}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(pipeline_log),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("AUTOMATED FULL PIPELINE RUNNER V2")
logger.info(f"Timestamp: {TIMESTAMP}")
logger.info("="*80)


def run_command(cmd, description, log_file=None):
    """Run a command and log output"""
    logger.info(f"\n{'='*80}")
    logger.info(f"STEP: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info("="*80)
    
    try:
        if log_file:
            with open(log_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False
                )
            # Read and log summary
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                logger.info(f"Output saved to {log_file}")
                logger.info(f"Last 20 lines:")
                for line in lines[-20:]:
                    logger.info(line.strip())
        else:
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            logger.info(result.stdout)
            if result.stderr:
                logger.warning(result.stderr)
        
        if result.returncode == 0:
            logger.info(f"[✓] {description} completed successfully")
            return True
        else:
            logger.error(f"[✗] {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        logger.error(f"[✗] Error running {description}: {e}")
        return False


def main():
    """Run full pipeline"""
    logger.info("Starting automated pipeline...")
    
    # Step 1: Preprocessing
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("="*80)
    
    preprocess_log = log_dir / f'preprocessing_{V2_SUFFIX}.log'
    success = run_command(
        [sys.executable, 'data/preprocess_v2.py'],
        'Data Preprocessing',
        log_file=str(preprocess_log)
    )
    
    if not success:
        logger.error("Preprocessing failed! Stopping pipeline.")
        return
    
    # Step 2: Training
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("="*80)
    
    training_log = log_dir / f'training_{V2_SUFFIX}.log'
    success = run_command(
        [sys.executable, 'coach/train_bert_v2.py'],
        'Model Training',
        log_file=str(training_log)
    )
    
    if not success:
        logger.error("Training failed! Stopping pipeline.")
        return
    
    # Step 3: Evaluation
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: MODEL EVALUATION")
    logger.info("="*80)
    
    eval_log = log_dir / f'evaluation_{V2_SUFFIX}.log'
    success = run_command(
        [sys.executable, 'coach/evaluate_v2.py'],
        'Model Evaluation',
        log_file=str(eval_log)
    )
    
    if not success:
        logger.error("Evaluation failed!")
        return
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info("\nGenerated files:")
    logger.info(f"  Preprocessing log: {preprocess_log}")
    logger.info(f"  Training log: {training_log}")
    logger.info(f"  Evaluation log: {eval_log}")
    logger.info(f"  Model: coach/models/hybrid_bert_wav2vec_{V2_SUFFIX}/")
    logger.info(f"  Reports: reports/*_{V2_SUFFIX}.*")
    logger.info("\n[OK] All steps completed successfully!")


if __name__ == '__main__':
    main()




