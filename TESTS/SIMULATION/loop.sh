#!/bin/bash

for value in $(seq 0.001 0.001 0.020); do
    python3 new_genedigitana.py "$value"
done

