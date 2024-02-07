#!/bin/bash

# Run the Python script and capture the output and error
output=$(python -m build 2>&1)

# Define the expected error message
expected_error="Could not load any ArrayFire libraries."

# Check if the output contains the expected error message
if echo "$output" | grep -q "$expected_error"; then
  echo "Expected error received."
  exit 0  # Exit with success as the error is expected
else
  echo "Unexpected output: $output"
  exit 1  # Exit with failure as the output was not expected
fi
