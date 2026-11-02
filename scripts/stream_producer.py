"""High-Throughput Streaming Simulator for SentinelStream Fraud Detection.

Simulates real-world financial transaction streams against the SentinelStream API,
supporting:
- Configurable concurrent throughput (RPS) via httpx.AsyncClient connection pooling.
- 99.8% legitimate baseline and 0.2% ambient fraud distribution.
- Automated "Data Drift / Fraud Spike Attack" injection after N requests
  (burst micro-transactions, elevated velocity, abnormal location deltas).
- Real-time console telemetry: throughput (req/sec), p50/p95/p99 latency, and fraud alerts.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import random
import sys
import time
from typing import Any, Dict, List, Optional
import httpx
import numpy as np

# Ensure stdout handles UTF-8 on Windows cp1252
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Color formatting for terminal telemetry (ANSI)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

# Standard merchant categories recognized by SentinelStream schema
STANDARD_CATEGORIES = [
    "retail",
    "electronics",
    "travel",
    "dining",
    "groceries",
    "entertainment",
    "utilities",
    "services",
]

HIGH_RISK_CATEGORIES = ["crypto", "gambling", "electronics"]
ATTACK_CATEGORIES = ["crypto", "gambling", "services"]


class StreamMetricsCollector:
    """Thread-safe telemetry accumulator for client and server performance stats."""

    def __init__(self, window_size: int = 2000) -> None:
        self.total_sent: int = 0
        self.total_success: int = 0
        self.total_errors: int = 0
        self.frauds_flagged: int = 0
        self.reviews_flagged: int = 0
        self.declines_flagged: int = 0
        self.client_latencies: List[float] = []
        self.server_latencies: List[float] = []
        self.recent_client_latencies: List[float] = []
        self.recent_server_latencies: List[float] = []
        self.start_time: float = time.monotonic()
        self.last_report_time: float = self.start_time
        self.last_report_count: int = 0
        self.lock = asyncio.Lock()

    async def record_response(
        self,
        client_lat_ms: float,
        server_lat_ms: Optional[float],
        is_fraud: bool,
        decision: str,
    ) -> None:
        async with self.lock:
            self.total_sent += 1
            self.total_success += 1
            if is_fraud:
                self.frauds_flagged += 1
            if decision == "FLAG_REVIEW":
                self.reviews_flagged += 1
            elif decision == "DECLINE":
                self.declines_flagged += 1

            self.client_latencies.append(client_lat_ms)
            self.recent_client_latencies.append(client_lat_ms)
            if server_lat_ms is not None:
                self.server_latencies.append(server_lat_ms)
                self.recent_server_latencies.append(server_lat_ms)

    async def record_error(self, client_lat_ms: float) -> None:
        async with self.lock:
            self.total_sent += 1
            self.total_errors += 1
            self.client_latencies.append(client_lat_ms)


class TransactionGenerator:
    """Generates synthetic transaction streams with user profiles and attack injection."""

    def __init__(self, n_users: int = 2000) -> None:
        self.user_ids = [f"usr_{i:04d}" for i in range(n_users)]
        # Assign persistent historical profiles to users
        self.user_profiles: Dict[str, Dict[str, float]] = {
            uid: {
                "avg_amount": float(random.lognormvariate(3.8, 0.45)),  # ~$45 median spend
                "std_amount": 0.0,
            }
            for uid in self.user_ids
        }
        for profile in self.user_profiles.values():
            profile["std_amount"] = profile["avg_amount"] * random.uniform(0.18, 0.35)

    def generate(self, tx_index: int, is_attack_mode: bool) -> Dict[str, Any]:
        """Generates single transaction payload according to current operational mode."""
        user_id = random.choice(self.user_ids)
        profile = self.user_profiles[user_id]
        avg_amt = profile["avg_amount"]
        std_amt = profile["std_amount"]

        # Attack Mode: Data Drift & Fraud Spike
        if is_attack_mode and random.random() < 0.65:
            # High-velocity micro-transaction credential stuffing / bot swarm attack
            if random.random() < 0.60:
                amount = round(random.uniform(1.49, 9.99), 2)
                velocity = random.randint(8, 24)  # Extreme velocity burst
                location_delta = round(random.uniform(150.0, 1800.0), 1)  # Proxy/VPN jump
                category = random.choice(ATTACK_CATEGORIES)
            else:
                # High-ticket account takeover (ATO) attack
                amount = round(avg_amt * random.uniform(4.5, 12.0) + random.uniform(100, 500), 2)
                velocity = random.randint(4, 10)
                location_delta = round(random.uniform(300.0, 3500.0), 1)
                category = random.choice(HIGH_RISK_CATEGORIES)

            return {
                "transaction_id": f"tx_sim_atk_{tx_index:07d}",
                "user_id": user_id,
                "amount": amount,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "merchant_category": category,
                "location_delta": location_delta,
                "user_avg_amount_30d": round(avg_amt, 2),
                "user_std_amount_30d": round(std_amt, 2),
                "user_tx_count_1h": velocity,
            }

        # Ambient Fraud (0.2% probability)
        is_ambient_fraud = random.random() < 0.002
        if is_ambient_fraud:
            amount = round(avg_amt * random.uniform(3.5, 8.0) + random.uniform(50, 200), 2)
            velocity = random.randint(4, 8)
            location_delta = round(random.uniform(80.0, 600.0), 1)
            category = random.choice(HIGH_RISK_CATEGORIES)
        else:
            # 99.8% Legitimate Baseline
            # Normal amount fluctuation around user mean
            raw_amt = random.gauss(avg_amt, std_amt)
            amount = round(max(3.50, raw_amt), 2)
            velocity = 1 if random.random() < 0.85 else random.randint(2, 3)
            # Normal daily spatial movement
            location_delta = round(random.expovariate(0.25), 2)  # ~4 km average
            category = random.choice(STANDARD_CATEGORIES)

        return {
            "transaction_id": f"tx_sim_norm_{tx_index:07d}",
            "user_id": user_id,
            "amount": amount,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "merchant_category": category,
            "location_delta": location_delta,
            "user_avg_amount_30d": round(avg_amt, 2),
            "user_std_amount_30d": round(std_amt, 2),
            "user_tx_count_1h": velocity,
        }


async def send_single_transaction(
    client: httpx.AsyncClient,
    url: str,
    payload: Dict[str, Any],
    collector: StreamMetricsCollector,
    semaphore: asyncio.Semaphore,
) -> None:
    """Dispatches a single transaction payload and records performance metrics."""
    async with semaphore:
        t0 = time.perf_counter()
        try:
            response = await client.post(url, json=payload)
            client_lat_ms = (time.perf_counter() - t0) * 1000.0

            if response.status_code == 200:
                data = response.json()
                is_fraud = bool(data.get("is_fraud", False))
                decision = str(data.get("decision", "APPROVE"))
                server_lat_ms = float(data.get("inference_time_ms", 0.0))
                await collector.record_response(client_lat_ms, server_lat_ms, is_fraud, decision)
            else:
                await collector.record_error(client_lat_ms)
        except Exception:
            client_lat_ms = (time.perf_counter() - t0) * 1000.0
            await collector.record_error(client_lat_ms)


async def telemetry_reporter_task(
    collector: StreamMetricsCollector,
    target_rps: float,
    drift_after: int,
    total_requests: int,
    stop_event: asyncio.Event,
) -> None:
    """Background coroutine printing real-time console telemetry dashboard every second."""
    print(f"\n{BOLD}{CYAN}{'=' * 95}{RESET}")
    print(f"{BOLD}{CYAN} SentinelStream Real-Time Stream Simulator — Live Telemetry Dashboard{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 95}{RESET}\n")

    while not stop_event.is_set():
        await asyncio.sleep(1.0)
        async with collector.lock:
            current_time = time.monotonic()
            elapsed_total = current_time - collector.start_time
            dt = current_time - collector.last_report_time
            count_delta = collector.total_sent - collector.last_report_count

            current_rps = count_delta / dt if dt > 0 else 0.0
            collector.last_report_time = current_time
            collector.last_report_count = collector.total_sent

            # Compute recent window latencies
            recent_client = list(collector.recent_client_latencies)
            recent_server = list(collector.recent_server_latencies)
            collector.recent_client_latencies.clear()
            collector.recent_server_latencies.clear()

            total_sent = collector.total_sent
            frauds_flagged = collector.frauds_flagged
            errors = collector.total_errors

        fraud_rate = (frauds_flagged / total_sent * 100.0) if total_sent > 0 else 0.0
        is_attack_active = total_sent >= drift_after

        if recent_client:
            c_p50 = float(np.percentile(recent_client, 50))
            c_p95 = float(np.percentile(recent_client, 95))
            c_p99 = float(np.percentile(recent_client, 99))
        else:
            c_p50 = c_p95 = c_p99 = 0.0

        if recent_server:
            s_p50 = float(np.percentile(recent_server, 50))
            s_p95 = float(np.percentile(recent_server, 95))
        else:
            s_p50 = s_p95 = 0.0

        # Status badge
        if is_attack_active:
            status_badge = f"{RED}{BOLD}[!] ATTACK DRIFT ACTIVE{RESET}"
        else:
            status_badge = f"{GREEN}[OK] NORMAL (99.8% Legit){RESET}"

        total_display = f"{total_sent}/{total_requests}" if total_requests > 0 else f"{total_sent}"

        print(
            f"[{BOLD}T+{elapsed_total:04.1f}s{RESET}] "
            f"Sent: {total_display:>9} | "
            f"RPS: {BOLD}{current_rps:5.1f}{RESET} | "
            f"Frauds: {RED}{frauds_flagged:4d}{RESET} ({fraud_rate:5.2f}%) | "
            f"Errors: {errors:2d} | "
            f"Client: p50={c_p50:4.1f}ms p95={c_p95:4.1f}ms p99={c_p99:4.1f}ms | "
            f"Server: p50={s_p50:4.1f}ms | "
            f"Status: {status_badge}"
        )


def print_final_summary(collector: StreamMetricsCollector, elapsed_sec: float) -> None:
    """Renders final executive performance & telemetry benchmark table."""
    total = collector.total_sent
    success = collector.total_success
    errors = collector.total_errors
    frauds = collector.frauds_flagged
    reviews = collector.reviews_flagged
    declines = collector.declines_flagged
    fraud_pct = (frauds / total * 100.0) if total > 0 else 0.0
    overall_rps = total / elapsed_sec if elapsed_sec > 0 else 0.0

    c_p50 = float(np.percentile(collector.client_latencies, 50)) if collector.client_latencies else 0.0
    c_p95 = float(np.percentile(collector.client_latencies, 95)) if collector.client_latencies else 0.0
    c_p99 = float(np.percentile(collector.client_latencies, 99)) if collector.client_latencies else 0.0
    c_max = float(np.max(collector.client_latencies)) if collector.client_latencies else 0.0

    s_p50 = float(np.percentile(collector.server_latencies, 50)) if collector.server_latencies else 0.0
    s_p95 = float(np.percentile(collector.server_latencies, 95)) if collector.server_latencies else 0.0
    s_p99 = float(np.percentile(collector.server_latencies, 99)) if collector.server_latencies else 0.0
    s_max = float(np.max(collector.server_latencies)) if collector.server_latencies else 0.0

    sla_passed = s_p95 < 30.0

    print(f"\n{BOLD}{CYAN}{'=' * 75}{RESET}")
    print(f"{BOLD}{CYAN}             SENTINELSTREAM BENCHMARK & SIMULATION SUMMARY             {RESET}")
    print(f"{BOLD}{CYAN}{'=' * 75}{RESET}")
    print(f" Total Requests Transmitted : {BOLD}{total:,}{RESET}")
    print(f" Successful Inferences (200) : {GREEN}{success:,}{RESET}")
    print(f" Failed / Errored Requests   : {RED if errors else GREEN}{errors:,}{RESET}")
    print(f" Total Simulation Runtime    : {elapsed_sec:.2f} seconds")
    print(f" Effective Average Throughput: {BOLD}{overall_rps:.2f} RPS{RESET}")
    print(f"{'-' * 75}")
    print(f" Total Frauds Flagged        : {RED}{BOLD}{frauds:,}{RESET} ({fraud_pct:.2f}% of traffic)")
    print(f"   ↳ Action: DECLINE (High)  : {declines:,}")
    print(f"   ↳ Action: REVIEW (Medium) : {reviews:,}")
    print(f"{'-' * 75}")
    print(f" {BOLD}LATENCY PERFORMANCE METRICS:{RESET}")
    print(f"   Client Round-Trip Latency : p50={c_p50:.2f}ms  p95={c_p95:.2f}ms  p99={c_p99:.2f}ms  Max={c_max:.2f}ms")
    print(f"   Server Inference Latency  : p50={s_p50:.2f}ms  p95={s_p95:.2f}ms  p99={s_p99:.2f}ms  Max={s_max:.2f}ms")
    print(f"{'-' * 75}")
    if sla_passed:
        print(f" Production Sub-30ms SLA     : {GREEN}{BOLD}PASSED (p95 = {s_p95:.2f}ms < 30ms SLA target){RESET}")
    else:
        print(f" Production Sub-30ms SLA     : {RED}{BOLD}BREACHED (p95 = {s_p95:.2f}ms >= 30ms SLA target){RESET}")
    print(f"{BOLD}{CYAN}{'=' * 75}{RESET}\n")


async def main_async(args: argparse.Namespace) -> None:
    """Asynchronous orchestrator for rate-governed stream simulation."""
    generator = TransactionGenerator(n_users=1500)
    collector = StreamMetricsCollector()
    semaphore = asyncio.Semaphore(args.concurrency)
    stop_reporter = asyncio.Event()

    limits = httpx.Limits(
        max_keepalive_connections=args.concurrency + 10,
        max_connections=args.concurrency * 2,
    )
    timeout = httpx.Timeout(args.timeout, connect=5.0)

    # Launch background telemetry reporting task
    reporter_task = asyncio.create_task(
        telemetry_reporter_task(
            collector=collector,
            target_rps=args.rps,
            drift_after=args.drift_after,
            total_requests=args.total_requests,
            stop_event=stop_reporter,
        )
    )

    t_start = time.monotonic()
    attack_banner_printed = False

    async with httpx.AsyncClient(limits=limits, timeout=timeout) as client:
        # Verify server availability before starting high-throughput stream
        try:
            health_url = args.url.replace("/api/v1/predict", "/health")
            h_resp = await client.get(health_url)
            if h_resp.status_code != 200 or not h_resp.json().get("model_loaded"):
                print(f"{YELLOW}Warning: SentinelStream at {health_url} reported degraded state: {h_resp.text}{RESET}")
        except Exception as exc:
            print(f"{RED}Error connecting to service at {args.url}: {exc}{RESET}")
            print("Ensure SentinelStream API is running (e.g. uvicorn app.main:app --port 8000)")
            stop_reporter.set()
            await reporter_task
            return

        tx_index = 0
        tasks: List[asyncio.Task] = []

        try:
            while args.total_requests <= 0 or tx_index < args.total_requests:
                tx_index += 1
                is_attack_mode = tx_index >= args.drift_after

                if is_attack_mode and not attack_banner_printed:
                    attack_banner_printed = True
                    print(
                        f"\n{RED}{BOLD}>>> [ATTACK SIMULATION TRIGGERED at Request #{tx_index}] "
                        f"Injecting data drift & high-velocity micro-transaction burst swarm! <<<{RESET}\n"
                    )

                payload = generator.generate(tx_index=tx_index, is_attack_mode=is_attack_mode)

                # Dispatch async request task
                task = asyncio.create_task(
                    send_single_transaction(
                        client=client,
                        url=args.url,
                        payload=payload,
                        collector=collector,
                        semaphore=semaphore,
                    )
                )
                tasks.append(task)

                # Rate control: Sleep to pace next dispatch to target RPS
                target_interval = 1.0 / args.rps
                await asyncio.sleep(target_interval)

                # Periodically prune completed tasks to bound memory
                if len(tasks) > 200:
                    tasks = [t for t in tasks if not t.done()]

            # Await all remaining in-flight tasks
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except (asyncio.CancelledError, KeyboardInterrupt):
            print(f"\n{YELLOW}Stream interrupted by user. Finalizing telemetry...{RESET}")
        finally:
            stop_reporter.set()
            await reporter_task
            t_total = time.monotonic() - t_start
            print_final_summary(collector, t_total)


def parse_arguments() -> argparse.Namespace:
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="SentinelStream High-Throughput Streaming Simulator with Data Drift / Fraud Spike Injection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000/api/v1/predict",
        help="Target SentinelStream prediction endpoint",
    )
    parser.add_argument(
        "--rps",
        type=float,
        default=30.0,
        help="Target request throughput in Requests Per Second (RPS)",
    )
    parser.add_argument(
        "--total-requests",
        type=int,
        default=500,
        help="Total transactions to transmit (0 or negative for infinite streaming)",
    )
    parser.add_argument(
        "--drift-after",
        type=int,
        default=250,
        help="Request index at which Data Drift & Fraud Spike Attack is injected",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=25,
        help="Maximum concurrent in-flight HTTP connections",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="HTTP request timeout in seconds",
    )
    return parser.parse_args()


def main() -> None:
    """CLI execution entrypoint."""
    args = parse_arguments()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\nExiting stream simulator.")
        sys.exit(0)


if __name__ == "__main__":
    main()
