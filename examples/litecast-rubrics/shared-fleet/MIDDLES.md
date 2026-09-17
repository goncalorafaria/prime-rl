# Shared CPU middles

A middle supervisor can subscribe to multiple independent publisher run IDs in the
same LiteRegistry service registry. Each publisher has a separate cache, origin
connection, transfer server, verification, registrations, and retry loop. Clients
discover the appropriate transfer endpoint through LiteRegistry. No LiteRegistry
code changes are required.

For a reloadable subscription list, create `publishers.json`:

```json
["trainer-a", "trainer-b"]
```

Use this command in a Rex CPU task (retain the fleet's runtime/container setup):

```bash
uv run --no-sync python -m prime_rl.litecast.middle \
  --registry "head+sqlite:///shared/fleet/head.sqlite3" \
  --publishers /shared/fleet/publishers.json \
  --advertise-host "$HOSTNAME" --port 0 \
  --max-publishers 8 --max-adapter-bytes 536870912 --max-versions 8
```

The SQLite head discovers Redis; Redis remains the service registry. Publishers
must use unique run IDs in that registry. The list is explicit: middles do not
subscribe automatically to every trainer that can reach Redis.

Replace the JSON file atomically to attach or detach publishers; it is read every
second. `[]` detaches everyone while leaving the supervisor available. Invalid,
duplicate, over-limit, or unreadable replacements keep the previous subscriptions
and emit `MIDDLE_SUBSCRIPTIONS_REJECTED`. The initial file must be valid. Detaching
withdraws registrations and shuts down that publisher's cache, so existing
transfers from that cache may need to retry another source.

For a fixed list, repeat `--run-id trainer-a --run-id trainer-b`. The existing
single `--run-id` invocation still uses the specified port. Multiple publishers
and reloadable lists use an independent OS-assigned port for each publisher,
regardless of `--port`. No stable public transfer port is needed: use registry
service names, never hard-code the publisher's cache port.

Logs `MIDDLE_READY` and `MIDDLE_STATUS` include `run=...`. Attach, detach, and retry
logs identify the publisher. A publisher exception retries its session after five
seconds; other publisher sessions remain alive. A changed publisher owner resets
only that publisher's local version identities.

Cache limits are per publisher. With the defaults, each publisher's checkpoint
store is bounded at 4 GiB; eight publishers can therefore allocate 32 GiB for those
stores alone. Downloader caches, temporary payloads, verification buffers, and
Python overhead require additional RAM. This is not a process RSS limit. Use the
fleet's 16 CPU / 64 GiB middle allocations and choose lower limits if necessary.

This implements middle subscriptions only. Existing processes need an initial
upgrade to this supervisor. It does not attach inference workers, launch trainers,
or change who owns or stops shared services. The current deployment is unchanged.

## Publishers joining later

Start the supervisor with `--publishers` and a file containing `[]` before any
trainer exists. Add run IDs whenever new trainers join. A listed run need not
have a live publisher yet: its session waits for that publisher's descriptor in
LiteRegistry, then starts transferring weights when they appear. Adding another
run while the first is training does not restart or rebind the first run's cache.

The existing toy `role.py middle` launch path still supplies a fixed `--run-id`.
For a shared middle deployment, use the `--publishers` command above in its Rex
task. Editing a file cannot change an already running fixed-list supervisor.
Membership currently requires updating this explicit shared file; publishers do
not automatically enroll themselves merely by appearing in Redis.
