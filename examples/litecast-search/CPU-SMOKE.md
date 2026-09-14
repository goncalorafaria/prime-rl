# Two-middle CPU smoke

Passed on 2026-09-14: `1 passed in 9.48s` (transfer scenario: 6.67 seconds).

The test starts its own Redis process, one LiteCast origin and two real CPU
MiddleNode HTTP servers. PrimeRL's thin adapter uses LiteRegistry to discover
the origin and register completed middle caches. A client discovers sources
through LiteRegistry and uses the normal PrimeRL `fetch_publication` path.

Verified with a 1,048,722-byte PEFT bundle containing real safetensors, split
into 17 shards:

1. Both middle nodes cache the published bundle.
2. Stop the origin and remove its registration; download and verify the bundle
   through the registered middle tier.
3. Stop and deregister one middle; download the identical bundle through the
   surviving middle, with the origin still offline.
4. Stop and deregister the final middle; downloading fails explicitly.
5. Shut down the test's servers and Redis process.

Reproduce from the repository root in an environment with the LiteCast extra
and test dependencies installed:

```bash
LITECAST_TEST_REDIS_SERVER=/path/to/redis-server PYTHONPATH=src \
  uv run --no-sync pytest --confcutdir=tests/unit/litecast \
  -q -s tests/unit/litecast/test_two_middles.py
```

This is a same-host CPU transfer test. The test itself coordinates middle
registration and graceful withdrawal; it does not implement a production
middle supervisor or prove crash-expiry handling, cross-node networking,
Qwen3.5 training, GPU LoRA loading, or Rex scheduling. Each middle independently
caches the entire bundle here; origin-bandwidth savings from cooperative shard
replication are not measured by this test.
