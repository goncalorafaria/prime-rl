"""Keep an RLTracer view with policy and per-rubric judge splits current."""
import argparse
import fcntl
import json
import time
from pathlib import Path


def export(source, output):
    count = 0
    for path in sorted(source.glob('step_*/train/all/traces.jsonl')):
        step = path.parents[2].name
        directory = output / 'rollouts' / step
        directory.mkdir(parents=True, exist_ok=True)
        policy = directory / 'policy'
        if policy.is_symlink():
            policy.unlink()
        (policy / 'all').mkdir(parents=True, exist_ok=True)
        policy_trace = policy / 'all' / 'traces.jsonl'
        if not policy_trace.is_symlink():
            policy_trace.symlink_to(path.resolve())
        target = directory / 'judge' / 'all' / 'traces.jsonl'
        stamp = directory / 'judge' / 'source.json'
        stat = path.stat()
        signature = [stat.st_size, stat.st_mtime_ns]
        if stamp.exists() and json.loads(stamp.read_text())['signature'] == signature:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        pending = target.with_suffix('.pending')
        rows = 0
        with path.open() as incoming, pending.open('w') as outgoing:
            for line in incoming:
                if not line.endswith('\n'):
                    break  # The active writer has not finished this record yet.
                record = json.loads(line)
                judge = record.get('info', {}).get('jtc_rubrichub_judge', {})
                rubrics = judge.get('selected_rubrics', [])
                for i, judgment in enumerate(judge.get('judgments', [])):
                    trace = judgment.get('trace') or {}
                    messages = trace.get('messages') or []
                    if isinstance(messages, str):
                        messages = json.loads(messages)
                    if not messages:
                        continue
                    metadata = trace.get('workflow_metadata') or {}
                    label = judgment.get('label')
                    rubric_index = judgment.get('rubric_index', i)
                    rubric = rubrics[rubric_index] if isinstance(rubric_index, int) and 0 <= rubric_index < len(rubrics) else None
                    view = {
                        'id': f"{record.get('id')}-judge-{i}",
                        'nodes': [{'message': m} for m in messages],
                        'metrics': {'criterion_pass': int(label == 'pass')},
                        'info': {'policy_rollout_id': record.get('id'),
                                 'source': str(path), 'rubric': rubric,
                                 'judgment': {k: v for k, v in judgment.items() if k != 'trace'},
                                 'judge': {k: v for k, v in trace.items() if k != 'messages'}},
                        'is_completed': metadata.get('finished'),
                        'stop_condition': metadata.get('stop_condition'),
                        'timing': metadata.get('timing', {}),
                        'errors': trace.get('errors', []),
                    }
                    outgoing.write(json.dumps(view, ensure_ascii=False) + '\n')
                    rows += 1
        pending.replace(target)
        stamp.write_text(json.dumps({'signature': signature, 'judge_traces': rows}))
        count += rows
        print(f'{step}: {rows} judge traces', flush=True)
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.export.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            export(args.source, args.output)
            if not args.watch:
                break
            time.sleep(30)


if __name__ == '__main__':
    main()
