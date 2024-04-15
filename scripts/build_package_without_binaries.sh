#!/bin/bash

# Run the Python script and capture the output or error
output=$(python -m build 2>&1)

# Define the expected output message
expected_output="Successfully built"

# Check if the output contains the expected output message
if echo "$output" | grep -q "$expected_output"; then
  echo "Expected output received."
  exit 0  # Exit with success as the output is expected
else
  echo "Unexpected output: $output"
  exit 1  # Exit with failure as the output was not expected
fi
