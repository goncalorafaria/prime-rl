"""Summarize completed benchmark phases without importing the training runtime."""
import json
import statistics
import sys
from pathlib import Path

path = Path(sys.argv[1])
rows = json.loads(path.read_text())
lines = ['| Concurrency | Mode | Repeats | Output tok/s | Episode mean (s) | Engine TTFT mean (s) | Prefix hit rate |',
         '|---|---|---|---|---|---|---|']
for concurrency in sorted({r['concurrency'] for r in rows}):
    for mode in ('none', 'weak'):
        values = [r for r in rows if r['concurrency'] == concurrency and r['mode'] == mode]
        if not values:
            continue
        def total(suffix):
            return sum(v for r in values for k, v in r['metrics_delta'].items() if k.endswith(suffix))
        ttft_count = total('time_to_first_token_seconds_count')
        ttft = total('time_to_first_token_seconds_sum') / ttft_count if ttft_count else None
        queries = total('prefix_cache_queries_total')
        hitrate = total('prefix_cache_hits_total') / queries if queries else None
        fmt = lambda x: 'unavailable' if x is None else f'{x:.3f}'
        lines.append(f"| {concurrency} | {mode} | {len(values)} | {statistics.mean(r['output_tokens_per_second'] for r in values):.1f} | {statistics.mean(r['episode_mean_seconds'] for r in values):.2f} | {fmt(ttft)} | {fmt(hitrate)} |")
result = '\n'.join(lines) + '\n'
path.with_name('summary.md').write_text(result)
print(result)
