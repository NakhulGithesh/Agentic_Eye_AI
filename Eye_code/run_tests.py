#!/usr/bin/env python3
"""
Automated Test Runner for Agentic Eye AI
Runs all tests in headless mode without GUI dependency
"""

import sys
import subprocess
import os
from pathlib import Path

def run_test_script(script_name, description):
    """Run a test script and return success status"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"{'='*60}")
    
    try:
        # Get the directory of this script
        script_dir = Path(__file__).parent
        script_path = script_dir / script_name
        
        if not script_path.exists():
            print(f"[ERROR] Test script not found: {script_path}")
            return False
        
        # Run the script with --automated flag
        result = subprocess.run([
            sys.executable, str(script_path), "--automated"
        ], capture_output=True, text=True, cwd=script_dir)
        
        # Print the output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"[PASS] {description} - PASSED")
            return True
        else:
            print(f"[FAIL] {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error running {description}: {e}")
        return False

def main():
    """Run all automated tests"""
    print("AGENTIC EYE AI - AUTOMATED TEST SUITE")
    print("=" * 60)
    
    # List of tests to run
    tests = [
        ("test_calibration.py", "Eye Tracking Calibration Tests"),
        ("test_iris_tracking.py", "Iris Tracking Tests"),
    ]
    
    results = []
    passed = 0
    total = len(tests)
    
    for script, description in tests:
        success = run_test_script(script, description)
        results.append((description, success))
        if success:
            passed += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    for description, success in results:
        status = "[PASSED]" if success else "[FAILED]"
        print(f"{description}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("ALL TESTS PASSED!")
        return 0
    else:
        print("Some tests failed - check output above")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)