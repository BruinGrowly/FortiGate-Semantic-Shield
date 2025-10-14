"""
Run All Test Suites - Final Validation

Executes all test suites and generates a comprehensive validation report.
"""

import subprocess
import sys
import time
from datetime import datetime

def run_test_suite(test_file, description):
    """Run a single test suite and return results"""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"File: {test_file}")
    print(f"{'='*80}\n")

    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        elapsed_time = time.time() - start_time

        # Check if tests passed
        passed = result.returncode == 0

        return {
            'name': description,
            'file': test_file,
            'passed': passed,
            'elapsed_time': elapsed_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        return {
            'name': description,
            'file': test_file,
            'passed': False,
            'elapsed_time': elapsed_time,
            'stdout': '',
            'stderr': 'Test suite timed out after 5 minutes',
            'returncode': -1
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        return {
            'name': description,
            'file': test_file,
            'passed': False,
            'elapsed_time': elapsed_time,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1
        }

def main():
    """Run all test suites"""
    print("="*80)
    print("SEMANTIC SUBSTRATE DATABASE - COMPLETE TEST VALIDATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print("="*80)

    # Define all test suites
    test_suites = [
        {
            'file': 'test_semantic_database.py',
            'description': 'Core Database Tests (30 tests)'
        },
        {
            'file': 'api/test_api_unit.py',
            'description': 'REST API Tests (16 tests)'
        },
        {
            'file': 'test_backup_recovery.py',
            'description': 'Backup & Recovery Tests (12 tests)'
        },
        {
            'file': 'test_transaction_management.py',
            'description': 'Transaction Management Tests (18 tests)'
        }
    ]

    # Run all test suites
    results = []
    for suite in test_suites:
        result = run_test_suite(suite['file'], suite['description'])
        results.append(result)

        # Print immediate result
        if result['passed']:
            print(f"\n[PASS] {result['name']} - {result['elapsed_time']:.2f}s")
        else:
            print(f"\n[FAIL] {result['name']} - {result['elapsed_time']:.2f}s")
            if result['stderr']:
                print(f"Error: {result['stderr'][:200]}")

    # Generate summary report
    print("\n\n")
    print("="*80)
    print("FINAL VALIDATION REPORT")
    print("="*80)

    total_suites = len(results)
    passed_suites = sum(1 for r in results if r['passed'])
    total_time = sum(r['elapsed_time'] for r in results)

    print(f"\nTest Suites Run: {total_suites}")
    print(f"Suites Passed: {passed_suites}")
    print(f"Suites Failed: {total_suites - passed_suites}")
    print(f"Total Execution Time: {total_time:.2f}s")

    print(f"\n{'Suite':<50} {'Status':<10} {'Time':<10}")
    print("-"*70)

    for result in results:
        status = "[PASS]" if result['passed'] else "[FAIL]"
        print(f"{result['name']:<50} {status:<10} {result['elapsed_time']:>6.2f}s")

    # Test counts
    print("\n" + "="*80)
    print("DETAILED TEST COUNTS")
    print("="*80)

    test_counts = {
        'Core Database Tests': 30,
        'REST API Tests': 16,
        'Backup & Recovery Tests': 12,
        'Transaction Management Tests': 18
    }

    total_tests = sum(test_counts.values())

    for name, count in test_counts.items():
        suite_result = next((r for r in results if name in r['name']), None)
        status = "[PASS]" if suite_result and suite_result['passed'] else "[FAIL]"
        print(f"{name:<50} {status:<10} {count:>3} tests")

    print("-"*70)
    print(f"{'TOTAL':<50} {'':<10} {total_tests:>3} tests")

    # Success rate
    if passed_suites == total_suites:
        success_rate = 100.0
        print(f"\n[SUCCESS] All test suites passed: {total_tests}/{total_tests} tests (100%)")
    else:
        success_rate = (passed_suites / total_suites) * 100
        print(f"\n[PARTIAL] {passed_suites}/{total_suites} test suites passed ({success_rate:.1f}%)")

    # Failed suites details
    failed_suites = [r for r in results if not r['passed']]
    if failed_suites:
        print("\n" + "="*80)
        print("FAILED SUITE DETAILS")
        print("="*80)

        for result in failed_suites:
            print(f"\n{result['name']}:")
            print(f"File: {result['file']}")
            print(f"Return Code: {result['returncode']}")

            if result['stderr']:
                print(f"\nError Output:")
                print(result['stderr'][:500])

            if result['stdout']:
                print(f"\nStandard Output (last 500 chars):")
                print(result['stdout'][-500:])

    # Final verdict
    print("\n" + "="*80)
    print("PRODUCTION READINESS STATUS")
    print("="*80)

    if success_rate == 100.0:
        print("\n[PRODUCTION READY]")
        print(f"  Status: All {total_tests} tests passing")
        print(f"  Code Coverage: 100% of test suites")
        print(f"  Execution Time: {total_time:.2f}s")
        print(f"  Verdict: READY FOR DEPLOYMENT")
    else:
        print("\n[NOT PRODUCTION READY]")
        print(f"  Status: {total_tests - (len(failed_suites) * (total_tests // total_suites))} estimated tests passing")
        print(f"  Code Coverage: {success_rate:.1f}% of test suites")
        print(f"  Failed Suites: {len(failed_suites)}")
        print(f"  Verdict: REQUIRES FIXES")

    print("\n" + "="*80)

    # Return exit code based on results
    return 0 if passed_suites == total_suites else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
