#!/bin/bash

nvidia-smi --query-gpu=timestamp,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,compute_mode --format=csv -l 1

