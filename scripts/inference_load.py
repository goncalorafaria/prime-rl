# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "fire>=0.7", "PyYAML>=6"]
# ///
"""uv run scripts/inference_load.py REXS_ID [--minutes 60] [--bin-minutes 5] [--output DIR].

Reads current-attempt inference logs through Rex, without refreshing/restarting jobs.
Charts describe sampled vLLM load, not hardware GPU utilization. TP/DCP throughput
is attributed equally to allocated GPUs; DCP does not multiply the GPU count.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import subprocess
import sys
from zoneinfo import ZoneInfo

ANSI = re.compile(r'\x1b\[[0-9;]*m')
STAMP = re.compile(r'\b(\d{2})-(\d{2}) (\d{2}:\d{2}:\d{2})\b')
METRIC = re.compile(r'Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%')


def timestamp(line, now):
    match = STAMP.search(line)
    if not match:
        return None
    date = f'{match[1]}-{match[2]} {match[3]}'
    candidates = []
    for year in (now.year - 1, now.year, now.year + 1):
        try:
            candidates.append(datetime.strptime(f'{year}-{date}', '%Y-%m-%d %H:%M:%S').replace(tzinfo=now.tzinfo))
        except ValueError:  # Feb 29 is not present in every candidate year.
            continue
    return min(candidates, key=lambda value: abs((value-now).total_seconds()))


def slurm_jobs(job_ids):
    result = subprocess.run(['sacct', '-X', '-n', '-P', '-j', ','.join(sorted(job_ids)),
                             '--format=JobIDRaw,State,NodeList,AllocTRES%1000'],
                            check=True, capture_output=True, text=True)
    jobs = {}
    for line in result.stdout.splitlines():
        job, state, node, tres, *_ = line.split('|')
        gpu = re.search(r'gres/gpu:([^=,]+)=(\d+)', tres)
        total = re.search(r'(?:^|,)gres/gpu=(\d+)', tres)
        jobs[job] = dict(state=state, node=node, gpu_type=gpu[1].upper() if gpu else 'UNKNOWN',
                         gpus=int(total[1]) if total else int(gpu[2]) if gpu else 0)
    return jobs


def fetch_window(api, identity, task, rank, start, end):
    limit = 2000
    while True:
        record = api.logs(identity, task=task, replica=rank, lines=limit)[0]
        lines = ANSI.sub('', record['content']).splitlines()
        dated = [(timestamp(line, end), line) for line in lines]
        stamps = [date for date, _ in dated if date is not None]
        complete = not record['exists'] or len(lines) < limit or (stamps and min(stamps) <= start)
        if complete or limit >= 256000:
            return record, [(date, line) for date, line in dated if date and start <= date <= end], bool(complete)
        limit *= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    parser.add_argument('--minutes', type=float, help='Lookback; defaults to time since run submission')
    parser.add_argument('--bin-minutes', type=float, default=5)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timezone', default='America/Los_Angeles')
    parser.add_argument('--rexs-dir', type=Path, default=Path(__file__).resolve().parents[2] / 'rexs')
    args = parser.parse_args()
    if args.minutes is not None and args.minutes <= 0:
        parser.error('--minutes must be positive')
    if args.bin_minutes <= 0:
        parser.error('--bin-minutes must be positive')
    sys.path.insert(0, str(args.rexs_dir / 'src'))
    from rexs.cli import Rexs
    api = Rexs()
    info = api.show(args.experiment, refresh=False)
    identity = info['experiment']['id']
    end = datetime.now(ZoneInfo(args.timezone))
    if args.minutes is None:
        submitted = info['experiment'].get('submitted_at') or info['experiment']['created_at']
        start = datetime.fromisoformat(submitted.replace('Z', '+00:00'))
        if start.tzinfo is None:
            raise ValueError('Run submission timestamp must include a timezone; specify --minutes instead')
        start = start.astimezone(end.tzinfo)
        args.minutes = max((end-start).total_seconds()/60, .001)
    else:
        start = end - timedelta(minutes=args.minutes)
    output = args.output or Path('outputs') / f'load-{identity[:8]}-{end:%Y%m%d-%H%M%S}'
    output.mkdir(parents=True, exist_ok=True)
    tasks = [t for t in info['tasks'] if t['name'].startswith('inference')]
    jobs = slurm_jobs({str(t['job_id']) for t in tasks if t.get('job_id')}) if tasks else {}
    rows = []
    observations = []
    for task in tasks:
        rank = task['replica_rank']
        name = f"{task['name']}-{rank}"
        record, lines, complete = fetch_window(api, identity, task['name'], rank, start, end)
        samples = [(date, METRIC.search(line)) for date, line in lines if METRIC.search(line)]
        # Save timestamped vLLM log lines only; un-timestamped lines cannot be
        # reliably assigned to the requested window.
        (output / f'{name}.log').write_text('\n'.join(line for _, line in lines) + '\n')
        job = str(record.get('job_id') or '')
        row = dict(replica=name, job=job, **jobs.get(job, dict(state='UNKNOWN', node='', gpu_type='UNKNOWN', gpus=0)),
                   samples=len(samples), tail_covers_window=complete,
                   mean_running=None, max_running=None, mean_waiting=None, mean_tokens_s=None,
                   mean_kv_percent=None, last_sample=None)
        if samples:
            row.update(mean_running=sum(int(m[2]) for _, m in samples)/len(samples),
                       max_running=max(int(m[2]) for _, m in samples),
                       mean_waiting=sum(int(m[3]) for _, m in samples)/len(samples),
                       mean_tokens_s=sum(float(m[1]) for _, m in samples)/len(samples),
                       mean_kv_percent=sum(float(m[4]) for _, m in samples)/len(samples),
                       last_sample=samples[-1][0].isoformat())
        observations.extend(dict(replica=name, gpu_type=row['gpu_type'], timestamp=date.isoformat(),
                                 running=int(m[2]), waiting=int(m[3]), tokens_s=float(m[1]))
                            for date, m in samples)
        rows.append(row)
    (output/'summary.json').write_text(json.dumps(dict(experiment=identity, start=start.isoformat(), end=end.isoformat(),
        note='Current attempt only; missing samples are not zero load. Per-GPU throughput is attribution, not hardware utilization.', replicas=rows), indent=2))
    (output/'samples.json').write_text(json.dumps(observations, indent=2))
    if rows:
        with (output/'replicas.csv').open('w') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f'{identity}: {start:%H:%M:%S}–{end:%H:%M:%S %Z}')
    print(f'{"REPLICA":24} {"TYPE":8} {"GPUs":>4} {"RUNNING":>8} {"TOK/S":>9} {"SAMPLES":>7}')
    for row in rows:
        active = f"{row['mean_running']:.1f}" if row['samples'] else 'N/A'
        rate = f"{row['mean_tokens_s']:.1f}" if row['samples'] else 'N/A'
        print(f"{row['replica']:24} {row['gpu_type']:8} {row['gpus']:4} {active:>8} {rate:>9} {row['samples']:7}")
    valid = [r for r in rows if r['samples'] and r['gpus']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'A40':'#e69f00', 'L40':'#56b4e9', 'L40S':'#009e73', 'H200':'#cc79a7', 'UNKNOWN':'#999999'}
    from matplotlib.patches import Patch
    legend = [Patch(facecolor=colors.get(kind, '#999999'), label=kind)
              for kind in sorted({r['gpu_type'] for r in rows if r['gpus']})]
    gpu_rows = [(f"{r['replica']}/gpu{i}",r['mean_tokens_s']/r['gpus'],r['gpu_type']) for r in valid for i in range(r['gpus'])]
    fig, ax = plt.subplots(figsize=(13, max(4, len(gpu_rows)*0.19)))
    if gpu_rows:
        ax.barh([r[0] for r in gpu_rows],[r[1] for r in gpu_rows],color=[colors.get(r[2],'#999999') for r in gpu_rows])
        ax.invert_yaxis(); ax.tick_params(axis='y', labelsize=7)
    else:
        ax.text(.5,.5,'No recent inference metrics',ha='center',transform=ax.transAxes)
    ax.set_xlabel('Attributed output tokens/s per GPU (replica throughput ÷ allocated GPUs)')
    ax.set_title(f'Inference load · last {args.minutes:g} minutes\n{start:%H:%M}–{end:%H:%M %Z}; not GPU utilization')
    if legend: ax.legend(handles=legend, title='GPU type')
    fig.tight_layout();fig.savefig(output/'gpu_load.png',dpi=150);plt.close(fig)
    work = defaultdict(float); capacity = defaultdict(int); requests = defaultdict(float)
    for row in valid:
        work[row['gpu_type']] += row['mean_tokens_s']
        requests[row['gpu_type']] += row['mean_running']
    for row in rows:
        if row['state']=='RUNNING': capacity[row['gpu_type']] += row['gpus']
    fig, axes = plt.subplots(1,3,figsize=(16,5))
    for ax, values, title in zip(axes,[requests,work,capacity],['Mean active-request share','Mean output-throughput share','Allocated GPU share (running jobs)']):
        values = {k:v for k,v in sorted(values.items()) if v>0}
        if values: ax.pie(list(values.values()),labels=list(values),autopct='%1.1f%%',colors=[colors.get(k,'#999999') for k in values])
        else: ax.text(.5,.5,'No data',ha='center',transform=ax.transAxes)
        ax.set_title(title)
    fig.suptitle('Inference distribution by GPU type');fig.tight_layout();fig.savefig(output/'gpu_types.png',dpi=150);plt.close(fig)
    # Requests belong to a replica: all TP GPUs jointly serve the same requests.
    # Do not divide counts by TP or count the same request four times.
    fig, ax = plt.subplots(figsize=(12, max(4, len(rows)*.32)))
    for i, row in enumerate(rows):
        if row['samples']:
            color = colors.get(row['gpu_type'], '#999999')
            ax.barh(i, row['mean_running'], color=color)
            ax.barh(i, row['mean_waiting'], left=row['mean_running'], color=color,
                    hatch='///', edgecolor='black')
            ax.plot(row['max_running'], i, 'k|', markersize=10)
        else:
            ax.text(0, i, 'No samples', va='center', color='gray')
    ax.set_yticks(range(len(rows)), [r['replica'] for r in rows]); ax.invert_yaxis()
    ax.set_xlabel('Requests per replica (shared by all its TP GPUs)')
    ax.set_title(f'Request occupancy · last {args.minutes:g} minutes; black tick = peak active')
    ax.legend(handles=legend + [Patch(facecolor='white', edgecolor='black', hatch='///', label='Queued')],
              title='GPU type / queue')
    fig.tight_layout(); fig.savefig(output/'request_load.png', dpi=150); plt.close(fig)

    import numpy as np
    from matplotlib.colors import Normalize
    # Missing bins stay missing instead of being interpreted as idle.
    bin_seconds = args.bin_minutes * 60
    count = max(1, int(np.ceil((end-start).total_seconds()/bin_seconds)))
    indices = {r['replica']: i for i, r in enumerate(rows)}
    buckets = defaultdict(list)
    for sample in observations:
        column = min(count-1, int((datetime.fromisoformat(sample['timestamp'])-start).total_seconds()/bin_seconds))
        buckets[indices[sample['replica']], column].append(sample)
    fig, axes = plt.subplots(2, 1, figsize=(14, max(7, len(rows)*.55)), sharex=True)
    for ax, field, title in zip(axes, ['running', 'waiting'], ['Active requests', 'Queued requests']):
        matrix = np.full((max(1,len(rows)), count), np.nan)
        for (i,j), samples in buckets.items():
            matrix[i,j] = sum(sample[field] for sample in samples)/len(samples)
        cmap = plt.get_cmap('viridis').copy(); cmap.set_bad('#dddddd')
        peak = max([sample[field] for sample in observations], default=1)
        plot = ax.imshow(matrix, aspect='auto', interpolation='nearest', cmap=cmap,
                         norm=Normalize(0,max(1,peak)), extent=[0,args.minutes,max(1,len(rows))-.5,-.5])
        ax.set_yticks(range(len(rows)), [r['replica']+' · '+r['gpu_type'] for r in rows])
        ax.set_title(title); fig.colorbar(plot, ax=ax, label=f'Requests ({args.bin_minutes:g}-minute bin mean)')
    axes[-1].set_xlabel(f'Minutes since {start:%H:%M:%S %Z}')
    fig.suptitle('Request load over time · gray = no sample; occupancy, not dispatch counts')
    fig.tight_layout(); fig.savefig(output/'requests_over_time.png', dpi=150); plt.close(fig)
    # Compare replica occupancy rather than dividing TP requests across GPUs.
    # Each replica contributes once per bin, even if it emitted several samples.
    types = sorted({r['gpu_type'] for r in valid})
    trend = []
    for column in range(count):
        present = [(rows[i]['gpu_type'], samples) for (i,j), samples in buckets.items() if j == column]
        total = sum(sum(s['running'] for s in samples)/len(samples) for _, samples in present)
        for kind in types:
            group = [samples for gpu_type, samples in present if gpu_type == kind]
            running = [sum(s['running'] for s in samples)/len(samples) for samples in group]
            tokens = [sum(s['tokens_s'] for s in samples)/len(samples) for samples in group]
            trend.append(dict(timestamp=(start+timedelta(seconds=column*bin_seconds)).isoformat(),
                              gpu_type=kind, sampled_replicas=len(group),
                              mean_active_requests=sum(running)/len(group) if group else None,
                              active_request_share_pct=100*sum(running)/total if group and total else None,
                              equal_replica_share_pct=100*len(group)/len(present) if present and group else None,
                              mean_tokens_s=sum(tokens)/len(group) if group else None))
    with (output/'gpu_type_timeseries.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=['timestamp','gpu_type','sampled_replicas',
            'mean_active_requests','active_request_share_pct','equal_replica_share_pct','mean_tokens_s'])
        writer.writeheader(); writer.writerows(trend)
    fig, axes = plt.subplots(4, 1, figsize=(13,12), sharex=True)
    x = [(min((column+.5)*bin_seconds, (end-start).total_seconds()))/60 for column in range(count)]
    fields = ['mean_active_requests','active_request_share_pct','mean_tokens_s','sampled_replicas']
    titles = ['Active requests per reporting replica',
              'Share of active requests (dashed: equal load per reporting replica)',
              'Output tokens/s per reporting replica', 'Reporting replicas (coverage)']
    for kind in types:
        series = [item for item in trend if item['gpu_type'] == kind]
        color = colors.get(kind, '#999999')
        for ax, field in zip(axes, fields):
            ax.plot(x, [item[field] if item[field] is not None else np.nan for item in series],
                    color=color, label=kind, marker='.', linewidth=2)
        axes[1].plot(x, [item['equal_replica_share_pct'] if item['equal_replica_share_pct'] is not None
                        else np.nan for item in series], color=color, linestyle='--', alpha=.6)
    for ax, title in zip(axes, titles):
        ax.set_title(title); ax.grid(alpha=.2); ax.set_ylim(bottom=0)
        if types: ax.legend(title='GPU type', loc='upper right', ncol=len(types))
    axes[0].set_ylabel('Requests / replica'); axes[1].set_ylabel('% of sampled fleet')
    axes[2].set_ylabel('Tokens/s / replica'); axes[3].set_ylabel('Replicas')
    axes[-1].set_xlabel(f'Minutes since {start:%H:%M:%S %Z} ({args.bin_minutes:g}-minute bins)')
    fig.suptitle(f'GPU-type load over {args.minutes:g} minutes · current replica layouts\n'
                 f'{args.bin_minutes:g}-minute means; last bin may be partial · occupancy, not request arrivals')
    fig.tight_layout(); fig.savefig(output/'gpu_type_load_over_time.png', dpi=150); plt.close(fig)
    print(f'Report: {output.resolve()}')


if __name__ == '__main__':
    main()
