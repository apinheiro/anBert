#!/bin/bash

nvidia-smi \ 
--query-gpu=timestamp,name,temperature.gpu,utilization.gpu,temperature.memory,utilization.memory,memory.used,memory.free,memory.total,compute_mode,pstate,accounting.buffer_size,gom.current,gom.pending,clocks_throttle_reasons.supported,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.hw_slowdown,encoder.stats.sessionCount,encoder.stats.averageFps,encoder.stats.averageLatency,ecc.mode.current,ecc.mode.pending,ecc.errors.corrected.volatile.device_memory,ecc.errors.corrected.volatile.dram,ecc.errors.corrected.volatile.register_file,ecc.errors.corrected.volatile.l1_cache,ecc.errors.corrected.volatile.l2_cache,ecc.errors.corrected.volatile.sram,ecc.errors.corrected.volatile.total,retired_pages.sbe,retired_pages.dbe,retired_pages.pending \ 
--format=csv -l 1