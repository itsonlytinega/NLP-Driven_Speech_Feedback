"""
Complete pipeline runner for VerbalCoach NLP system.
Runs preprocessing, training, and report generation sequentially.
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def print_status(text, status="info"):
    """Print status message"""
    prefixes = {
        "info": "[INFO]",
        "success": "[✓ SUCCESS]",
        "warning": "[⚠ WARNING]",
        "error": "[✗ ERROR]"
    }
    prefix = prefixes.get(status, "[INFO]")
    print(f"{prefix} {text}")

def check_prerequisites():
    """Check if required dependencies are installed"""
    print_header("Checking Prerequisites")
    
    required_packages = [
        'torch', 'transformers', 'librosa', 'pandas', 
        'numpy', 'sklearn', 'matplotlib', 'seaborn',
        'xgboost', 'textstat', 'soundfile', 'jinja2'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print_status(f"{package} is installed", "success")
        except ImportError:
            print_status(f"{package} is NOT installed", "error")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print_status("All prerequisites met!", "success")
    return True

def run_preprocessing():
    """Run data preprocessing"""
    print_header("STEP 1: Preprocessing Data (2-4 hours)")
    
    preprocess_script = Path('data/preprocess.py')
    if not preprocess_script.exists():
        print_status("preprocess.py not found!", "error")
        return False
    
    print_status("Starting preprocessing... This will take 2-4 hours.", "warning")
    print_status("You can close this window - it will run in background.")
    print_status("Or press Ctrl+C to cancel and run manually later.\n")
    
    try:
        # Check if data already exists
        if Path('data/train.csv').exists() and Path('data/bert_train.pt').exists():
            response = input("Preprocessed data already exists. Skip preprocessing? (y/n): ")
            if response.lower() == 'y':
                print_status("Skipping preprocessing", "warning")
                return True
        
        # Run preprocessing
        result = subprocess.run(
            [sys.executable, str(preprocess_script)],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print_status("Preprocessing completed!", "success")
            return True
        else:
            print_status(f"Preprocessing failed with code: {result.returncode}", "error")
            return False
            
    except KeyboardInterrupt:
        print_status("\nPreprocessing interrupted by user", "warning")
        return False
    except Exception as e:
        print_status(f"Preprocessing error: {e}", "error")
        return False

def run_bert_training():
    """Run BERT model training"""
    print_header("STEP 2: Training BERT Model (30-60 mins)")
    
    bert_script = Path('coach/train_bert.py')
    if not bert_script.exists():
        print_status("train_bert.py not found!", "error")
        return False
    
    print_status("Starting BERT training... This will take 30-60 minutes.")
    
    try:
        result = subprocess.run(
            [sys.executable, str(bert_script)],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print_status("BERT training completed!", "success")
            return True
        else:
            print_status(f"BERT training failed with code: {result.returncode}", "error")
            return False
            
    except KeyboardInterrupt:
        print_status("\nBERT training interrupted by user", "warning")
        return False
    except Exception as e:
        print_status(f"BERT training error: {e}", "error")
        return False

def run_cause_training():
    """Run cause classifier training"""
    print_header("STEP 3: Training Cause Classifier (5-10 mins)")
    
    cause_script = Path('coach/train_cause.py')
    if not cause_script.exists():
        print_status("train_cause.py not found!", "error")
        return False
    
    print_status("Starting cause classifier training...")
    
    try:
        result = subprocess.run(
            [sys.executable, str(cause_script)],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print_status("Cause classifier training completed!", "success")
            print_status("Report auto-generated!", "success")
            return True
        else:
            print_status(f"Cause training failed with code: {result.returncode}", "error")
            return False
            
    except KeyboardInterrupt:
        print_status("\nCause training interrupted by user", "warning")
        return False
    except Exception as e:
        print_status(f"Cause training error: {e}", "error")
        return False

def open_report():
    """Open the generated report"""
    print_header("STEP 4: Opening Report")
    
    report_script = Path('open_report.py')
    if report_script.exists():
        print_status("Opening report in Chrome...")
        subprocess.run([sys.executable, str(report_script)])
    else:
        # Fallback: generate and open report
        report_gen = Path('reports/generate_report.py')
        if report_gen.exists():
            print_status("Generating report...")
            subprocess.run([sys.executable, str(report_gen)])
            print_status("Report generated at: reports/full_report.html")
        else:
            print_status("Could not find report generator", "error")

def main():
    """Main pipeline runner"""
    print_header("VerbalCoach NLP Pipeline Runner")
    print_status("Complete pipeline for 2,000+ sample dataset")
    print_status("This will take approximately 3-5 hours total.")
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠ Please install missing dependencies first.")
        print("Run: pip install -r requirements.txt\n")
        sys.exit(1)
    
    # Confirm start
    print_header("Ready to Start")
    print("This pipeline will:")
    print("1. Preprocess 2,000+ samples (2-4 hours)")
    print("2. Train BERT model (30-60 mins)")
    print("3. Train cause classifier (5-10 mins)")
    print("4. Generate and open report\n")
    
    response = input("Start pipeline now? (y/n): ")
    if response.lower() != 'y':
        print_status("Pipeline cancelled by user")
        sys.exit(0)
    
    # Run pipeline
    success_count = 0
    total_steps = 3
    
    # Step 1: Preprocessing
    if run_preprocessing():
        success_count += 1
    else:
        print_status("Pipeline failed at preprocessing step", "error")
        sys.exit(1)
    
    # Step 2: BERT training
    if run_bert_training():
        success_count += 1
    else:
        print_status("Pipeline failed at BERT training step", "error")
        sys.exit(1)
    
    # Step 3: Cause training
    if run_cause_training():
        success_count += 1
    else:
        print_status("Pipeline failed at cause training step", "error")
        sys.exit(1)
    
    # Success!
    print_header("Pipeline Complete!")
    print_status(f"Completed {success_count}/{total_steps} steps successfully", "success")
    
    # Open report
    open_report()
    
    print_header("Next Steps")
    print("1. Review the report in Chrome")
    print("2. Check model files in coach/models/")
    print("3. Test Django integration")
    print("4. Ready for deployment!")
    
    print("\n" + "="*60)
    print_status("VerbalCoach NLP System Ready!", "success")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()


