import asyncio
import json
import os
import sys
import time
from datetime import datetime

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from openai.lib.streaming.chat import AsyncChatCompletionStream

BASE_URL = "http://localhost:10001/v1"

TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "run_bash",
        "description": "Run a bash command",
        "parameters": {
            "type": "object",
            "properties": {
                "script":               {"type": "string",  "description": "The bash command to run"},
                "timeout":              {"type": "integer", "description": "Seconds before timeout"},
                "description":          {"type": "string",  "description": "Clear description"},
                "risk":                 {"type": "string",  "enum": ["low", "medium", "high"]},
                "mutation_kind":        {"type": "string",  "enum": ["none", "create", "edit", "overwrite", "append", "delete"]},
                "mutation_targets":     {"type": "array",   "items": {"type": "string"}, "description": "Expected mutated file paths"},
                "affects_shared_state": {"type": "boolean", "description": "True when changing remote/shared/external state"},
                "enable_llm_parser":    {"type": "boolean", "description": "Enable LLM-based output summarization"},
            },
            "required": ["script", "timeout", "description", "risk", "mutation_kind", "mutation_targets", "affects_shared_state", "enable_llm_parser"],
        },
    },
}

PROMPTS = [
    # 0 — Game engine (C++) codebase analysis
    (
        "You are a senior game engine engineer reverse-engineering a large open-source 3D game engine "
        "written in C++ with an integrated editor and scripting runtime. Generate exactly 3 run_bash "
        "tool calls. CRITICAL REQUIREMENT: each individual tool call MUST be extremely long. "
        "Specifically:\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments explaining every single command, variable, pipe, redirect, and flag in detail. "
        "Use heredocs, multi-stage pipelines, associative arrays, trap handlers, temp file management, "
        "and advanced bash features. Each script should be a complete, self-contained analysis tool.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "that covers: (a) the exact purpose and motivation for this exploration step, (b) a line-by-line "
        "walkthrough of what each command in the script does and why, (c) the expected output format and "
        "how to interpret it, (d) edge cases and failure modes, (e) how this step fits into the broader "
        "engine exploration workflow, (f) relevant game engine concepts (scene graph, ECS architecture, "
        "render passes, material systems, shader compilation, physics broadphase/narrowphase, spatial "
        "partitioning, animation blend trees, skeletal mesh skinning, audio mixing, asset hot-reload, "
        "serialization, reflection/RTTI, job systems, fiber-based task scheduling) that this step is "
        "designed to uncover, (g) the historical context of why these patterns evolved from early engines "
        "like Quake/Unreal to modern architectures, (h) detailed comparison with alternative approaches "
        "and why this approach was chosen, (i) performance considerations and scalability analysis, and "
        "(j) potential follow-up investigations suggested by this step's results.\n\n"
        "3. The 'mutation_targets' array should list at least 20 specific example file paths with "
        "realistic game engine project directory structures.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: Rendering pipeline — Vulkan/D3D12 abstraction layer, render graph, shader permutation "
        "  system, GPU resource management and descriptor binding model\n"
        "- Call 2: Entity-Component-System architecture — archetype storage, query iteration, change "
        "  detection, command buffers, world serialization and scene prefab instantiation\n"
        "- Call 3: Asset pipeline — mesh/texture import, shader cross-compilation, dependency tracking, "
        "  hot-reload mechanism, build artifact caching and distribution\n\n"
        "Remember: each individual tool call must be VERY LONG — at least 8192 tokens each. "
        "The description alone should be 8000+ words. The script should be 100+ lines. "
        "Do NOT be brief. Expand every single point exhaustively with full technical depth. "
        "Repeat and elaborate on concepts from multiple angles. Include historical context, "
        "theoretical foundations, practical implications, and worked examples within the description."
    ),
    # 1 — Linux kernel driver audit
    (
        "You are a senior Linux kernel engineer performing a comprehensive audit of a large out-of-tree "
        "device driver subsystem. Generate exactly 3 run_bash tool calls. CRITICAL REQUIREMENT: each "
        "individual tool call MUST be extremely long. Specifically:\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments explaining every command, variable, pipe, redirect, and flag. Use heredocs, "
        "multi-stage pipelines, associative arrays, trap handlers, temp file management, and advanced "
        "bash features. Each script must be a complete, self-contained audit tool.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "covering: (a) purpose and motivation, (b) line-by-line walkthrough, (c) expected output format, "
        "(d) edge cases and failure modes, (e) how this fits the broader audit workflow, (f) relevant "
        "kernel concepts (kconfig, device tree, probe/remove lifecycle, DMA mapping, IRQ handling, "
        "workqueues, power management, sysfs/debugfs interfaces, etc.), (g) historical context of why "
        "these patterns matter for driver stability, (h) comparison with alternative approaches, "
        "(i) performance considerations, and (j) follow-up investigations.\n\n"
        "3. The 'mutation_targets' array should list at least 20 specific file paths with realistic "
        "kernel source tree directory structures.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: Module init/exit paths, PCI/platform device registration, probe error unwinding analysis\n"
        "- Call 2: Concurrency audit — spinlock/mutex usage, RCU patterns, atomic operations, race conditions\n"
        "- Call 3: DMA buffer management, IOMMU integration, memory-mapped I/O register access patterns\n\n"
        "Each tool call must be at least 8192 tokens. The description alone should be 8000+ words. "
        "The script should be 100+ lines. Do NOT be brief. Expand every single point exhaustively."
    ),
    # 2 — Kubernetes operator codebase
    (
        "You are a principal platform engineer reverse-engineering a complex Kubernetes operator written "
        "in Go. Generate exactly 3 run_bash tool calls. CRITICAL REQUIREMENT: each individual tool call "
        "MUST be extremely long. Specifically:\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments. Use heredocs, pipelines, associative arrays, trap handlers, temp files, and "
        "advanced bash features. Each script must be a self-contained analysis tool.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "covering: (a) purpose and motivation, (b) line-by-line walkthrough, (c) expected output, "
        "(d) edge cases, (e) broader workflow context, (f) relevant Kubernetes concepts (CRDs, "
        "controller-runtime, informers, work queues, leader election, finalizers, admission webhooks, "
        "status subresources, owner references, garbage collection), (g) historical context of the "
        "operator pattern and its evolution, (h) alternative approaches comparison, (i) performance "
        "and scalability analysis, (j) follow-up investigations.\n\n"
        "3. The 'mutation_targets' array should list at least 20 file paths with realistic Go operator "
        "project directory structures.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: CRD schema extraction, controller reconciliation loop mapping, RBAC requirement analysis\n"
        "- Call 2: Finalizer chains, status condition management, event recording, retry/backoff strategies\n"
        "- Call 3: Webhook configurations, certificate management, HA/leader-election setup, metrics exposition\n\n"
        "Each tool call must be at least 8192 tokens. Description 8000+ words. Script 100+ lines. "
        "Do NOT be brief. Expand exhaustively with full technical depth."
    ),
    # 3 — Distributed database internals (Rust)
    (
        "You are a database internals expert analyzing a distributed key-value store written in Rust. "
        "Generate exactly 3 run_bash tool calls. CRITICAL REQUIREMENT: each individual tool call MUST "
        "be extremely long. Specifically:\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments. Use heredocs, pipelines, associative arrays, trap handlers, temp files, and "
        "advanced bash features. Self-contained analysis tools.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "covering: (a) purpose and motivation, (b) line-by-line walkthrough, (c) expected output, "
        "(d) edge cases, (e) broader workflow, (f) relevant concepts (LSM-trees, WAL, MVCC, Raft "
        "consensus, snapshot isolation, compaction strategies, bloom filters, block cache, column "
        "families, range partitioning, gossip protocol, clock synchronization), (g) historical context "
        "of distributed storage evolution, (h) comparison with RocksDB/TiKV/CockroachDB approaches, "
        "(i) performance and latency analysis, (j) follow-up investigations.\n\n"
        "3. The 'mutation_targets' array should list at least 20 file paths with realistic Rust project "
        "directory structures for a distributed database.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: Storage engine layer — LSM compaction, WAL recovery, memtable flush, SST file format\n"
        "- Call 2: Raft consensus implementation — log replication, leader election, snapshot transfer, "
        "  configuration changes, pre-vote protocol\n"
        "- Call 3: Transaction layer — MVCC versioning, lock management, deadlock detection, distributed "
        "  2PC coordinator, timestamp oracle\n\n"
        "Each tool call must be at least 8192 tokens. Description 8000+ words. Script 100+ lines. "
        "Expand exhaustively."
    ),
    # 4 — Compiler/language toolchain (LLVM-based)
    (
        "You are a compiler engineer investigating a custom LLVM-based compiler frontend and optimization "
        "pipeline for a domain-specific language. Generate exactly 3 run_bash tool calls. CRITICAL "
        "REQUIREMENT: each individual tool call MUST be extremely long. Specifically:\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments, heredocs, pipelines, associative arrays, trap handlers, and advanced bash. "
        "Each script must be a self-contained analysis tool.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "covering: (a) purpose, (b) line-by-line walkthrough, (c) expected output, (d) edge cases, "
        "(e) broader workflow, (f) relevant concepts (lexer/parser architecture, AST design, semantic "
        "analysis, type inference, SSA form, LLVM IR generation, pass managers, optimization passes, "
        "instruction selection, register allocation, machine IR, object file emission, LTO, PGO), "
        "(g) historical context of LLVM's evolution and pass infrastructure, (h) comparison with GCC "
        "and other compiler frameworks, (i) compile-time performance analysis, (j) follow-up work.\n\n"
        "3. The 'mutation_targets' array should list at least 20 file paths with realistic compiler "
        "project directory structures.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: Frontend — lexer/parser extraction, AST node hierarchy, type system mapping\n"
        "- Call 2: Middle-end — custom LLVM passes, IR transformations, optimization pipeline ordering\n"
        "- Call 3: Backend — target machine description, instruction patterns, ABI lowering, linker integration\n\n"
        "Each tool call must be at least 8192 tokens. Description 8000+ words. Script 100+ lines."
    ),
    # 5 — Large-scale ML training infrastructure (Python/C++)
    (
        "You are an ML infrastructure engineer auditing a large-scale distributed training framework "
        "that mixes Python orchestration with C++/CUDA kernels. Generate exactly 3 run_bash tool calls. "
        "CRITICAL REQUIREMENT: each individual tool call MUST be extremely long. Specifically:\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments, heredocs, pipelines, associative arrays, trap handlers, temp files, and "
        "advanced bash features. Self-contained analysis tools.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "covering: (a) purpose, (b) line-by-line walkthrough, (c) expected output, (d) edge cases, "
        "(e) broader workflow, (f) relevant concepts (data parallelism, tensor/pipeline/expert "
        "parallelism, gradient accumulation, mixed precision, loss scaling, NCCL collectives, RDMA, "
        "NVLink topology, CUDA graph capture, custom autograd functions, checkpoint activation "
        "recomputation, ZeRO optimization stages), (g) historical context of distributed training "
        "evolution from data-parallel to 3D parallelism, (h) comparison with Megatron-LM/DeepSpeed/"
        "FSDP approaches, (i) throughput and memory analysis, (j) follow-up investigations.\n\n"
        "3. The 'mutation_targets' array should list at least 20 file paths with realistic ML training "
        "framework directory structures.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: Parallelism strategy — sharding annotations, device mesh, communication topology\n"
        "- Call 2: Custom CUDA kernels — fused attention, quantized GEMM, memory-efficient backward pass\n"
        "- Call 3: Checkpointing/fault tolerance — async save, elastic scaling, preemption recovery\n\n"
        "Each tool call must be at least 8192 tokens. Description 8000+ words. Script 100+ lines."
    ),
    # 6 — Network protocol stack (embedded C)
    (
        "You are an embedded systems engineer reverse-engineering a proprietary real-time network "
        "protocol stack written in C for an ARM Cortex-M microcontroller. Generate exactly 3 run_bash "
        "tool calls. CRITICAL REQUIREMENT: each individual tool call MUST be extremely long.\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments, heredocs, pipelines, associative arrays, trap handlers, temp files, and "
        "advanced bash features. Self-contained analysis tools.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "covering: (a) purpose, (b) line-by-line walkthrough, (c) expected output, (d) edge cases, "
        "(e) broader workflow, (f) relevant concepts (zero-copy buffer pools, scatter-gather DMA, "
        "interrupt-driven vs polled I/O, priority inversion, watchdog timers, MPU regions, "
        "linker script sections, startup code, ISR vector tables, CMSIS-RTOS, lwIP/FreeRTOS+TCP, "
        "EtherCAT/PROFINET/TSN), (g) historical context of real-time networking evolution, "
        "(h) comparison with Linux network stack vs bare-metal approaches, (i) timing and jitter "
        "analysis, (j) follow-up investigations.\n\n"
        "3. The 'mutation_targets' array should list at least 20 file paths with realistic embedded "
        "firmware project directory structures.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: Memory layout — linker script, MPU config, stack/heap placement, DMA descriptor rings\n"
        "- Call 2: Protocol state machines — frame parsing, retransmission timers, flow control, CRC\n"
        "- Call 3: ISR architecture — vector table, priority grouping, nested interrupt handling, "
        "  deferred processing via PendSV\n\n"
        "Each tool call must be at least 8192 tokens. Description 8000+ words. Script 100+ lines."
    ),
    # 7 — Monorepo CI/CD and build system (Bazel + GitHub Actions)
    (
        "You are a developer productivity engineer analyzing a massive polyglot monorepo using Bazel "
        "for builds and GitHub Actions for CI/CD. Generate exactly 3 run_bash tool calls. CRITICAL "
        "REQUIREMENT: each individual tool call MUST be extremely long. Specifically:\n\n"
        "1. The 'script' field must contain a FULL multi-line bash script (50+ lines) with extensive "
        "inline comments, heredocs, pipelines, associative arrays, trap handlers, temp files, and "
        "advanced bash features. Self-contained analysis tools.\n\n"
        "2. The 'description' field must be an EXTREMELY detailed technical essay (at least 8000 words) "
        "covering: (a) purpose, (b) line-by-line walkthrough, (c) expected output, (d) edge cases, "
        "(e) broader workflow, (f) relevant concepts (Bazel action graph, Starlark macros, remote "
        "execution, remote caching, persistent workers, sandboxing, aspects, transitions, platforms/"
        "toolchains, repository rules, bzlmod, GitHub Actions workflow syntax, composite actions, "
        "reusable workflows, matrix strategies, concurrency groups, OIDC tokens, artifact attestations), "
        "(g) historical context of build system evolution from Make to Bazel, (h) comparison with "
        "Buck2/Pants/Gradle approaches, (i) build performance analysis, (j) follow-up work.\n\n"
        "3. The 'mutation_targets' array should list at least 20 file paths with realistic Bazel "
        "monorepo directory structures.\n\n"
        "Each of the 3 tool calls should explore a DIFFERENT aspect:\n"
        "- Call 1: Dependency graph analysis — target fan-out, circular dependency detection, visibility audit\n"
        "- Call 2: Remote cache hit analysis — action key computation, input hashing, cache miss diagnosis\n"
        "- Call 3: CI pipeline optimization — critical path analysis, test sharding, flaky test quarantine\n\n"
        "Each tool call must be at least 8192 tokens. Description 8000+ words. Script 100+ lines."
    ),
]


async def run_one(idx, client, model, prompt, outdir, results):
    """Stream a single request and measure TPOT / ITL matching vLLM/SGLang methodology.

    Uses continuous_usage_stats to get per-chunk completion_tokens from the
    server, so speculative-decoding batches are counted correctly.
    Per-request tok/s = (output_len - 1) / decode_time  (inverse of TPOT).
    """
    print(f"[{idx}] Sending request...", file=sys.stderr)
    t_start = time.monotonic()

    try:
        raw_stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            tools=[TOOL_SCHEMA],
            stream=True,
            stream_options={
                "include_usage": True,
                "continuous_usage_stats": True,
            },
        )
    except Exception as e:
        print(f"[{idx}] Error: {e}", file=sys.stderr)
        return

    accumulator = AsyncChatCompletionStream(
        raw_stream=raw_stream,
        response_format=NOT_GIVEN,
        input_tools=NOT_GIVEN,
    )

    t_first_token = None
    chunk_count = 0
    prev_completion_tokens = 0
    has_continuous_usage = False
    prev_token_time = None
    itls = []

    try:
        async for event in accumulator:
            if not hasattr(event, "chunk"):
                continue
            chunk = event.chunk

            if not chunk.choices:
                if chunk.usage and chunk.usage.completion_tokens:
                    prev_completion_tokens = chunk.usage.completion_tokens
                continue
            delta = chunk.choices[0].delta
            if delta.content or delta.tool_calls:
                now = time.monotonic()
                if t_first_token is None:
                    t_first_token = now
                chunk_count += 1

                new_tokens = 1
                if chunk.usage and chunk.usage.completion_tokens:
                    new_tokens = chunk.usage.completion_tokens - prev_completion_tokens
                    prev_completion_tokens = chunk.usage.completion_tokens
                    if new_tokens > 1:
                        has_continuous_usage = True
                    new_tokens = max(new_tokens, 1)

                if prev_token_time is not None:
                    chunk_itl = now - prev_token_time
                    per_token_itl = chunk_itl / new_tokens
                    itls.extend([per_token_itl] * new_tokens)
                prev_token_time = now

        final = await accumulator.get_final_completion()
    finally:
        await accumulator.close()

    t_end = time.monotonic()

    total_tokens = chunk_count
    if final and final.usage and final.usage.completion_tokens:
        total_tokens = final.usage.completion_tokens

    total_time = t_end - t_start
    ttft = (t_first_token - t_start) if t_first_token else total_time
    decode_time = total_time - ttft

    tpot = 0.0
    avg_tps = 0.0
    if total_tokens > 1 and decode_time > 0:
        tpot = decode_time / (total_tokens - 1)
        avg_tps = (total_tokens - 1) / decode_time

    win_samples = _windowed_tps(itls)
    min_tps = min(win_samples) if win_samples else avg_tps
    max_tps = max(win_samples) if win_samples else avg_tps

    results[idx] = {
        "total_tokens": total_tokens,
        "total_time_s": round(total_time, 2),
        "ttft_s": round(ttft, 3),
        "tpot_ms": round(tpot * 1000, 2),
        "avg_tps": round(avg_tps, 2),
        "min_tps": round(min_tps, 2),
        "max_tps": round(max_tps, 2),
        "itls": itls,
        "has_continuous_usage": has_continuous_usage,
    }

    outfile = os.path.join(outdir, f"req_{idx}.json")
    if final:
        with open(outfile, "w") as f:
            f.write(final.model_dump_json(indent=2))

    method = "continuous" if has_continuous_usage else "chunk-count"
    print(
        f"[{idx}] Done -> {outfile}  |  "
        f"{total_tokens} tok, "
        f"tok/s min={min_tps:.1f} avg={avg_tps:.1f} max={max_tps:.1f}, "
        f"TTFT {ttft:.3f}s  ({method})",
        file=sys.stderr,
    )


WINDOW_TOKENS = 64


def _windowed_tps(itls, window=WINDOW_TOKENS):
    """Slide a non-overlapping window over ITLs and return tok/s per window."""
    samples = []
    for start in range(0, len(itls) - window + 1, window):
        w = itls[start:start + window]
        elapsed = sum(w)
        if elapsed > 0:
            samples.append(window / elapsed)
    tail = len(itls) % window
    if tail >= window // 2:
        w = itls[-tail:]
        elapsed = sum(w)
        if elapsed > 0:
            samples.append(tail / elapsed)
    return samples


def _percentile(sorted_vals, p):
    """Compute p-th percentile from pre-sorted list (linear interpolation)."""
    if not sorted_vals:
        return 0.0
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = f + 1
    if c >= len(sorted_vals):
        return sorted_vals[-1]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def print_summary(results, wall_time, concurrency):
    """Print per-request details and vLLM/SGLang-compatible summary metrics."""
    active_reqs = concurrency

    print("\n" + "=" * 80, file=sys.stderr)
    print("PER-REQUEST RESULTS", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(
        f"{'Req':>3}  {'Tokens':>7}  "
        f"{'min':>8}  {'avg':>8}  {'max':>8}  "
        f"{'TTFT(s)':>8}  {'Total(s)':>8}",
        file=sys.stderr,
    )
    for i in sorted(results):
        r = results[i]
        print(
            f"{i:3d}  {r['total_tokens']:7d}  "
            f"{r['min_tps']:8.2f}  {r['avg_tps']:8.2f}  {r['max_tps']:8.2f}  "
            f"{r['ttft_s']:8.3f}  {r['total_time_s']:8.2f}",
            file=sys.stderr,
        )

    all_win_samples = []
    for r in results.values():
        all_win_samples.extend(_windowed_tps(r["itls"]))
    avg_tps_vals = [r["avg_tps"] for r in results.values() if r["avg_tps"] > 0]
    tpot_vals = [r["tpot_ms"] for r in results.values() if r["tpot_ms"] > 0]
    ttft_vals = [r["ttft_s"] for r in results.values()]
    all_itl_ms = []
    for r in results.values():
        all_itl_ms.extend(itl * 1000 for itl in r["itls"])

    total_tokens_all = sum(r["total_tokens"] for r in results.values())
    output_throughput = total_tokens_all / wall_time if wall_time > 0 else 0

    sorted_itl = sorted(all_itl_ms)
    sorted_tpot = sorted(tpot_vals)
    sorted_ttft_ms = sorted(t * 1000 for t in ttft_vals)

    min_tps = min(all_win_samples) if all_win_samples else 0
    avg_tps = sum(avg_tps_vals) / len(avg_tps_vals) if avg_tps_vals else 0
    max_tps = max(all_win_samples) if all_win_samples else 0

    print("\n" + "=" * 80, file=sys.stderr)
    print(f"SUMMARY  (window={WINDOW_TOKENS} tokens)", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(f"  Concurrency             : {active_reqs}", file=sys.stderr)
    print(f"  Output throughput (tok/s): {output_throughput:.2f}", file=sys.stderr)
    print(f"  Total output tokens     : {total_tokens_all}", file=sys.stderr)
    print(f"  Wall clock (s)          : {wall_time:.2f}", file=sys.stderr)

    print(f"  {'---':30s}", file=sys.stderr)
    print(f"  Decode tok/s  min       : {min_tps:.2f}  (windowed)", file=sys.stderr)
    print(f"  Decode tok/s  avg       : {avg_tps:.2f}  (aggregate)", file=sys.stderr)
    print(f"  Decode tok/s  max       : {max_tps:.2f}  (windowed)", file=sys.stderr)

    print(f"  {'---':30s}", file=sys.stderr)
    if sorted_ttft_ms:
        print(f"  Mean TTFT (ms)          : {sum(sorted_ttft_ms)/len(sorted_ttft_ms):.2f}", file=sys.stderr)
        print(f"  Median TTFT (ms)        : {_percentile(sorted_ttft_ms, 50):.2f}", file=sys.stderr)
        print(f"  P99 TTFT (ms)           : {_percentile(sorted_ttft_ms, 99):.2f}", file=sys.stderr)

    print(f"  {'---':30s}", file=sys.stderr)
    if sorted_tpot:
        print(f"  Mean TPOT (ms)          : {sum(sorted_tpot)/len(sorted_tpot):.2f}", file=sys.stderr)
        print(f"  Median TPOT (ms)        : {_percentile(sorted_tpot, 50):.2f}", file=sys.stderr)
        print(f"  P99 TPOT (ms)           : {_percentile(sorted_tpot, 99):.2f}", file=sys.stderr)

    print(f"  {'---':30s}", file=sys.stderr)
    if sorted_itl:
        print(f"  Mean ITL (ms)           : {sum(sorted_itl)/len(sorted_itl):.2f}", file=sys.stderr)
        print(f"  Median ITL (ms)         : {_percentile(sorted_itl, 50):.2f}", file=sys.stderr)
        print(f"  P95 ITL (ms)            : {_percentile(sorted_itl, 95):.2f}", file=sys.stderr)
        print(f"  P99 ITL (ms)            : {_percentile(sorted_itl, 99):.2f}", file=sys.stderr)
        print(f"  Max ITL (ms)            : {sorted_itl[-1]:.2f}", file=sys.stderr)
    else:
        print("  (no ITL samples collected)", file=sys.stderr)

    print("=" * 80, file=sys.stderr)

    print("\nMarkdown row (paste into README):", file=sys.stderr)
    print(
        f"| {active_reqs} "
        f"| {min_tps:.2f} "
        f"| {avg_tps:.2f} "
        f"| {max_tps:.2f} "
        f"| {output_throughput:.2f} "
        f"| {_percentile(sorted_ttft_ms, 50):.2f} |",
        file=sys.stderr,
    )


async def main(model, concurrency):
    client = AsyncOpenAI(base_url=BASE_URL, api_key="unused")

    tag = f"c{concurrency}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = f"/tmp/{model}_{tag}_{timestamp}"
    os.makedirs(outdir, exist_ok=True)

    prompts = PROMPTS[:concurrency]
    print(f"Model      : {model}", file=sys.stderr)
    print(f"Concurrency: {concurrency}", file=sys.stderr)
    print(f"Prompts    : {len(prompts)}", file=sys.stderr)
    print(f"Output     : {outdir}/", file=sys.stderr)
    print(file=sys.stderr)

    results = {}

    t_wall_start = time.monotonic()
    await asyncio.gather(*(
        run_one(idx, client, model, prompt, outdir, results)
        for idx, prompt in enumerate(prompts)
    ))
    t_wall_end = time.monotonic()

    wall_time = t_wall_end - t_wall_start
    if results:
        print_summary(results, wall_time, concurrency)

        summary_file = os.path.join(outdir, "summary.json")
        all_win = []
        for r in results.values():
            all_win.extend(_windowed_tps(r["itls"]))
        avg_tps_vals = [r["avg_tps"] for r in results.values() if r["avg_tps"] > 0]
        tpot_vals = [r["tpot_ms"] for r in results.values() if r["tpot_ms"] > 0]
        ttft_vals = [r["ttft_s"] for r in results.values()]
        all_itl_ms = []
        for r in results.values():
            all_itl_ms.extend(itl * 1000 for itl in r["itls"])
        sorted_itl = sorted(all_itl_ms)
        sorted_tpot = sorted(tpot_vals)
        total_tokens_all = sum(r["total_tokens"] for r in results.values())

        with open(summary_file, "w") as f:
            per_req = {}
            for k, v in sorted(results.items()):
                entry = {key: val for key, val in v.items() if key != "itls"}
                entry["n_itl_samples"] = len(v["itls"])
                per_req[str(k)] = entry
            json.dump({
                "model": model,
                "concurrency": concurrency,
                "num_prompts": len(results),
                "window_tokens": WINDOW_TOKENS,
                "wall_time_s": round(wall_time, 2),
                "output_throughput_tok_s": round(total_tokens_all / wall_time, 2) if wall_time > 0 else 0,
                "total_tokens": total_tokens_all,
                "min_decode_tps": round(min(all_win), 2) if all_win else 0,
                "avg_decode_tps": round(sum(avg_tps_vals) / len(avg_tps_vals), 2) if avg_tps_vals else 0,
                "max_decode_tps": round(max(all_win), 2) if all_win else 0,
                "mean_tpot_ms": round(sum(sorted_tpot) / len(sorted_tpot), 2) if sorted_tpot else 0,
                "median_tpot_ms": round(_percentile(sorted_tpot, 50), 2),
                "p99_tpot_ms": round(_percentile(sorted_tpot, 99), 2),
                "mean_ttft_ms": round(sum(ttft_vals) / len(ttft_vals) * 1000, 2) if ttft_vals else 0,
                "mean_itl_ms": round(sum(sorted_itl) / len(sorted_itl), 2) if sorted_itl else 0,
                "median_itl_ms": round(_percentile(sorted_itl, 50), 2),
                "p95_itl_ms": round(_percentile(sorted_itl, 95), 2),
                "p99_itl_ms": round(_percentile(sorted_itl, 99), 2),
                "max_itl_ms": round(sorted_itl[-1], 2) if sorted_itl else 0,
                "per_request": per_req,
            }, f, indent=2)
        print(f"Summary JSON: {summary_file}", file=sys.stderr)
    else:
        print("\nNo successful requests.", file=sys.stderr)

    print(f"\nAll done. Results in: {outdir}", file=sys.stderr)
    print(outdir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark LLM serving throughput")
    parser.add_argument("model", help="Model name to benchmark")
    parser.add_argument("--concurrency", "-c", type=int, default=8,
                        help="Max concurrent requests (default: 8)")
    parsed = parser.parse_args()
    asyncio.run(main(parsed.model, parsed.concurrency))
