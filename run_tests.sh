#!/bin/bash
# -----------------------------------------------------------------------------
# run_tests.sh
#
# Description:
#   This script runs all unit tests for the project using pytest.
#
# Usage:
#   From the project root, run:
#       ./run_tests.sh
#
# Requirements:
#   - Python 3.x and pytest must be installed.
# -----------------------------------------------------------------------------

# Get the directory of this script and change to it (assumed to be the project root)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

if [ "$1" == "--all" ]; then
    echo "Running all tests..."
    pytest --maxfail=1 --disable-warnings -q
else
    pytest -v tests/test_mypymath.py --maxfail=1 --disable-warnings -q
    # pytest -v tests/test_mycmath.py --maxfail=1 --disable-warnings -q
    # pytest -v tests/test_pdbtools.py --maxfail=1 --disable-warnings -q
    # pytest -v tests/test_gmxmd.py --maxfail=1 --disable-warnings -q
fi