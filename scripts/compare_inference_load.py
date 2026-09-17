# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9", "fire>=0.7", "PyYAML>=6"]
# ///
"""Compare Rex runs using five-minute vLLM output-throughput averages.
Usage: uv run scripts/compare_inference_load.py RUN_ID RUN_ID ...
Includes retained allocation attempts; reads metadata/logs without refreshing jobs.
"""
import argparse
from collections import defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo
from inference_load import ANSI, METRIC, timestamp, slurm_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--output', type=Path, default=Path('outputs/run-throughput-comparison'))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'rexs/src'))
    from rexs.cli import Rexs
    api = Rexs()
    now = datetime.now(ZoneInfo('America/Los_Angeles'))
    runs = [api.show(run, refresh=False)['experiment'] for run in args.runs]
    sources = []
    for run in runs:
        for allocation in run['allocations']:
            for task in allocation['tasks']:
                if not task['name'].startswith('inference'):
                    continue
                jobs = {str(a['job_id']) for a in allocation.get('attempts', []) if a.get('job_id')}
                if allocation.get('job_id'): jobs.add(str(allocation['job_id']))
                for job in jobs:
                    path = Path(allocation.get('run_root',run['run_root']))/job/'logs'/f"{task['name']}.{task['local_rank']}.log"
                    sources.append((run['id'],f"{task['name']}-{task['rank']}",job,path))
    hardware = slurm_jobs({job for _,_,job,_ in sources})
    buckets = defaultdict(dict)
    missing = []
    for run, replica, job, path in sources:
        if not path.is_file():
            missing.append(str(path)); continue
        with path.open(errors='replace') as file:
            for line in file:
                match = METRIC.search(line)
                if not match: continue
                date = timestamp(ANSI.sub('',line),now)
                if date is None: continue
                bucket = int(date.timestamp()//300)*300
                # Dedupe a repeated metric at the same timestamp for this attempt.
                buckets[run,replica,job,bucket][date.isoformat()] = (float(match[1]),int(match[2]))
    cutoffs = {r['id']: min(now.timestamp(), datetime.fromisoformat(r['finished_at']).timestamp())
               if r.get('finished_at') else now.timestamp() for r in runs}
    rows = []
    for (run,replica,job,bucket), samples in sorted(buckets.items()):
        if bucket+300 > cutoffs[run]: continue  # exclude unfinished wall-clock bin
        rows.append(dict(run=run,replica=replica,job=job,gpu_type=hardware.get(job,{}).get('gpu_type','UNKNOWN'),
                         bin_start=bucket,samples=len(samples),
                         tokens_s=sum(x[0] for x in samples.values())/len(samples),
                         active_requests=sum(x[1] for x in samples.values())/len(samples),
                         included=len(samples)>=24))
    args.output.mkdir(parents=True,exist_ok=True)
    with (args.output/'replica_bins.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=['run','replica','job','gpu_type','bin_start','samples','tokens_s','active_requests','included'])
        writer.writeheader();writer.writerows(rows)
    # Only bins with >=80% of the expected 30 samples contribute to comparisons.
    # Absent/restarting replicas are excluded, never presented as measured idle GPUs.
    usable=[r for r in rows if r['included']]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    colors={'A40':'#e69f00','L40':'#56b4e9','L40S':'#009e73','H200':'#cc79a7','UNKNOWN':'#999999'}
    kinds=sorted({r['gpu_type'] for r in usable})
    legend=[Line2D([0],[0],color=colors.get(k,'#999999'),label=k,linewidth=3) for k in kinds]
    fig,axes=plt.subplots(3,len(runs),figsize=(6*len(runs),11),squeeze=False,sharey='row')
    summary=[]
    for col,run in enumerate(runs):
        selected=[r for r in usable if r['run']==run['id']]
        bins=sorted({r['bin_start'] for r in selected})
        if not bins: continue
        origin=bins[0]
        x=[(b-origin)/60 for b in bins]
        total=[sum(r['tokens_s'] for r in selected if r['bin_start']==b) for b in bins]
        axes[0,col].plot(x,total,color='black',marker='.',label='All types')
        for kind in kinds:
            y=[sum(r['tokens_s'] for r in selected if r['bin_start']==b and r['gpu_type']==kind)
               if any(r['bin_start']==b and r['gpu_type']==kind for r in selected) else float('nan') for b in bins]
            axes[0,col].plot(x,y,color=colors[kind],label=kind)
        for replica, job in sorted({(r['replica'],r['job']) for r in selected}):
            data={r['bin_start']:r for r in selected if r['replica']==replica and r['job']==job}
            # An allocation change can change GPU type, so draw each type separately.
            for kind in kinds:
                y=[data[b]['tokens_s'] if b in data and data[b]['gpu_type']==kind else float('nan') for b in bins]
                axes[1,col].plot(x,y,color=colors[kind],alpha=.7,marker='.',linewidth=1)
        for kind in kinds:
            axes[2,col].plot(x,[sum(r['bin_start']==b and r['gpu_type']==kind for r in selected) for b in bins],color=colors[kind])
        axes[0,col].set_title(f"{run['id'][:8]}\n{run['name']}",fontsize=9,wrap=True)
        axes[2,col].set_xlabel('Minutes since first covered five-minute bin')
        # Final three contiguous completed bins, not the last three arbitrary samples.
        recent=[r for r in selected if r['bin_start']>=bins[-1]-600]
        recent_bins=list(range(bins[-1]-600,bins[-1]+1,300))
        covered=[b for b in recent_bins if any(r['bin_start']==b for r in recent)]
        item=dict(run=run['id'],name=run['name'],window_start=datetime.fromtimestamp(recent_bins[0],now.tzinfo).isoformat(),
                  window_end=datetime.fromtimestamp(bins[-1]+300,now.tzinfo).isoformat(),covered_bins=len(covered),
                  aggregate_tokens_s=sum(r['tokens_s'] for r in recent)/len(covered),
                  mean_reporting_replicas=len(recent)/len(covered),by_type={})
        for kind in kinds:
            values=[r['tokens_s'] for r in recent if r['gpu_type']==kind]
            if values: item['by_type'][kind]=dict(tokens_s_per_replica=sum(values)/len(values),replica_bins=len(values))
        summary.append(item)
    axes[0,0].set_ylabel('Aggregate output tokens/s')
    axes[1,0].set_ylabel('Output tokens/s per replica\nEach line = one replica process')
    axes[2,0].set_ylabel('Reporting replicas')
    for ax in axes.flat:
        ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    fig.legend(handles=[Line2D([0],[0],color='black',label='Fleet total')]+legend,loc='upper center',ncol=len(legend)+1)
    fig.suptitle('Run comparison · five-minute means · only replica bins with ≥24 of 30 expected samples',y=.96)
    fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(args.output/'throughput_comparison.png',dpi=150)
    fig, axes = plt.subplots(3, len(runs), figsize=(6*len(runs), 10), squeeze=False, sharey='row')
    for col, run in enumerate(runs):
        selected = [r for r in usable if r['run'] == run['id']]
        bins = sorted({r['bin_start'] for r in selected})
        if not bins:
            continue
        x = [(b-bins[0])/60 for b in bins]
        axes[0,col].plot(x, [sum(r['active_requests'] for r in selected if r['bin_start']==b) for b in bins], color='black')
        for kind in kinds:
            group = [r for r in selected if r['gpu_type']==kind]
            axes[0,col].plot(x, [sum(r['active_requests'] for r in group if r['bin_start']==b) if any(r['bin_start']==b for r in group) else float('nan') for b in bins], color=colors[kind])
            axes[2,col].plot(x, [sum(r['bin_start']==b for r in group) for b in bins], color=colors[kind])
        for replica, job in sorted({(r['replica'],r['job']) for r in selected}):
            data = {r['bin_start']:r for r in selected if r['replica']==replica and r['job']==job}
            for kind in kinds:
                axes[1,col].plot(x, [data[b]['active_requests'] if b in data and data[b]['gpu_type']==kind else float('nan') for b in bins], color=colors[kind], alpha=.7, linewidth=1)
        axes[0,col].set_title(f"{run['id'][:8]}\n{run['name']}", fontsize=9)
        axes[2,col].set_xlabel('Minutes since first covered five-minute bin')
    for ax in axes.flat:
        ax.set_ylim(bottom=0); ax.grid(alpha=.2)
    for ax, label in zip(axes[:,0], ['Fleet active requests', 'Active requests per replica', 'Reporting replicas']):
        ax.set_ylabel(label)
    fig.legend(handles=[Line2D([0],[0],color='black',label='Fleet total')]+legend, loc='upper center', ncol=len(legend)+1)
    fig.suptitle('Request occupancy comparison · five-minute means · not request arrival counts', y=.96)
    fig.tight_layout(rect=[0,0,1,.93]); fig.savefig(args.output/'requests_comparison.png', dpi=150); plt.close(fig)
    report=dict(note='Observational comparison, not controlled benchmark. Latest 15 minutes with covered bins per run. Missing replicas excluded; counts shown. Retained attempts included; overlapping replacement processes counted separately.',runs=summary,missing_logs=missing)
    (args.output/'summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
