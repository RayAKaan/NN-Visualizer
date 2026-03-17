from __future__ import annotations

from typing import Dict, List


def estimate_throughput(
    batch_sizes: List[int],
    flops_per_sample: int,
    memory_per_sample_bytes: int,
    params_bytes: int,
    optimizer_bytes: int,
    device_gflops: float = 1.0,
) -> List[Dict]:
    results = []
    flops_per_sec = device_gflops * 1_000_000_000.0
    for bs in batch_sizes:
        bs = int(bs)
        total_flops = max(flops_per_sample * bs, 1)
        throughput = flops_per_sec / total_flops
        memory_total = params_bytes + optimizer_bytes + memory_per_sample_bytes * bs
        results.append(
            {
                "batch_size": bs,
                "memory_total_bytes": int(memory_total),
                "estimated_throughput_samples_per_sec": float(throughput * bs),
            }
        )
    return results
