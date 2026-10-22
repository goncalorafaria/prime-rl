"""Run on CUDA vLLM 0.29.0 with VLLM_PRIME_COMPACT_LOGPROBS=1."""

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch
import vllm
from vllm.sampling_params import SamplingParams
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.sample import sampler as sampler_module
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores as native_scores
from vllm.v1.worker.gpu.states import RequestState

from prime_rl.inference.vllm import compact_logprobs


def measure(fn, iterations):
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    wall = time.perf_counter()
    start.record()
    for _ in range(iterations):
        _result = fn()
    end.record()
    end.synchronize()
    elapsed = time.perf_counter() - wall
    return dict(
        cuda_ms=start.elapsed_time(end) / iterations,
        wall_ms=elapsed * 1000 / iterations,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Compare vLLM sampler latency with no logprobs, native top-128, and compact top-128."
    )
    parser.add_argument("output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    if args.iterations < 1 or args.repeats < 1:
        parser.error("iterations and repeats must be positive")
    compact_logprobs.patch_compact_logprobs()
    patched_scores = sampler_module.compute_topk_scores
    assert patched_scores._prime_compact_logprobs
    torch.manual_seed(271828)
    device = torch.device("cuda:0")
    with args.output.open("w", buffering=1) as stream:

        def emit(row):
            line = json.dumps(row)
            stream.write(line + "\n")
            print(line, flush=True)

        emit(
            dict(
                kind="metadata",
                scope="Full native vLLM Sampler.__call__, no model or host transfer",
                gpu=torch.cuda.get_device_name(),
                vllm=vllm.__version__,
                temperature=0.8,
                top_k=-1,
                top_p=1.0,
                iterations=args.iterations,
                repeats=args.repeats,
                source_sha256=hashlib.sha256(Path(compact_logprobs.__file__).read_bytes()).hexdigest(),
            )
        )
        for batch in (1, 8, 64, 256):
            vocab = 151936
            req = RequestState(batch, 32, batch, 0, vocab, device)
            for i in range(batch):
                req.add_request(str(i), 1, [1], 0, 16)
            req.apply_staged_writes()
            buffers = InputBuffers(batch, batch, device)
            ib = InputBatch.make_dummy(batch, batch, buffers)
            ib.input_ids.fill_(1)
            ib.positions.fill_(1)
            logits = 3 * torch.randn(batch, vocab, device=device)
            samplers = {}
            for k in (None, 128):
                sampler = sampler_module.Sampler(batch, vocab, device, req, logprobs_mode="processed_logprobs")
                for i in range(batch):
                    sampler.add_request(i, 1, SamplingParams(temperature=0.8, top_k=-1, top_p=1.0, logprobs=k))
                sampler.apply_staged_writes()
                samplers[k] = sampler

            def run(k, patched):
                sampler_module.compute_topk_scores = patched_scores if patched else native_scores
                return samplers[k](logits, ib)

            functions = dict(
                no_logprobs=lambda: run(None, False),
                native_top128=lambda: run(128, False),
                patched_top128=lambda: run(128, True),
            )
            baseline = functions["no_logprobs"]()
            assert baseline.logprobs_tensors is None
            a = functions["native_top128"]()
            b = functions["patched_top128"]()
            torch.testing.assert_close(a.sampled_token_ids, b.sampled_token_ids, atol=0, rtol=0)
            for name in ("selected_token_ranks",):
                torch.testing.assert_close(
                    getattr(a.logprobs_tensors, name), getattr(b.logprobs_tensors, name), atol=0, rtol=0
                )
            torch.testing.assert_close(a.logprobs_tensors.logprobs, b.logprobs_tensors.logprobs, atol=5e-5, rtol=1e-5)
            reference = (logits / 0.8).log_softmax(-1)
            torch.testing.assert_close(
                b.logprobs_tensors.logprobs,
                reference.gather(1, b.logprobs_tensors.logprob_token_ids),
                atol=5e-5,
                rtol=1e-5,
            )
            del reference
            assert (b.num_sampled == 1).all()
            del a, b, baseline
            for fn in functions.values():
                for _ in range(10):
                    fn()
            samples = {name: [] for name in functions}
            for repeat in range(args.repeats):
                names = list(functions)
                offset = repeat % 3
                for name in names[offset:] + names[:offset]:
                    samples[name].append(measure(functions[name], args.iterations))
            medians = {name: statistics.median(x["wall_ms"] for x in vals) for name, vals in samples.items()}
            emit(
                dict(
                    kind="case", batch=batch, vocab=vocab, correctness="passed", median_wall_ms=medians, samples=samples
                )
            )
        emit(dict(kind="complete"))
    sampler_module.compute_topk_scores = patched_scores


if __name__ == "__main__":
    main()
