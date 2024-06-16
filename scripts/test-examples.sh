#!/usr/bin/env bash

set -eu

rm -rf output
mkdir output

# run all example notebooks
for example in notebooks/examples/*; do
  jupyter nbconvert \
    --execute $example \
    --to html \
    --output-dir output
  echo "✅ $(basename $example)"
done