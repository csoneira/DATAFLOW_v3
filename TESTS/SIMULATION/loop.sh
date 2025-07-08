#!/bin/bash

for value in $(seq 0.001 0.001 0.020); do
    python3 -u new_genedigitana.py "$value"
done

