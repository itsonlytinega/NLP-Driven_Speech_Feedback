"""
Install Dependencies for Enhanced Pronunciation Analysis

This script installs the required packages for the enhanced pronunciation
analysis module that includes accent detection, rhythm analysis, and
phonetic accuracy assessment.

Run this before testing the enhanced pronunciation features.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        print(f"[OK] {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to install {package}: {e}")
        return False

def main():
    print("="*60)
    print("INSTALLING ENHANCED PRONUNCIATION DEPENDENCIES")
    print("="*60)
    
    # Required packages for enhanced pronunciation analysis
    packages = [
        'librosa',           # Audio analysis and rhythm detection
        'soundfile',         # Audio file I/O
        'numpy',             # Numerical computations (should already be installed)
        'scipy',             # Scientific computing (for signal processing)
    ]
    
    print(f"\nInstalling {len(packages)} packages...")
    
    success_count = 0
    for package in packages:
        if install_package(package):
            success_count += 1
    
    print(f"\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    print(f"Successfully installed: {success_count}/{len(packages)} packages")
    
    if success_count == len(packages):
        print("\n[SUCCESS] All dependencies installed!")
        print("\nEnhanced pronunciation analysis is now available:")
        print("  ✓ Accent detection (American/British/Australian)")
        print("  ✓ Phonetic accuracy analysis")
        print("  ✓ Rhythm and stress pattern analysis")
        print("  ✓ Specific pronunciation recommendations")
        
        print("\nNext steps:")
        print("1. Restart Django server: python manage.py runserver")
        print("2. Test enhanced analysis: python scripts/test_nlp_pipeline.py")
        print("3. Upload audio and analyze with enhanced pronunciation features!")
        
    else:
        print(f"\n[WARNING] {len(packages) - success_count} packages failed to install")
        print("Enhanced pronunciation analysis may not work properly")
        print("You can still use basic pronunciation analysis")
    
    print("="*60)

if __name__ == '__main__':
    main()
