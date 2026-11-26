#!/usr/bin/env python3
"""Exercise public tensor additions without interception-specific application code.

Run the identical workload natively and with an externally supplied preload.
Results alone do not prove slicing; validate opt-in physical traces separately.
This is a correctness/ordering fixture, not a throughput benchmark.
"""

import argparse
import time

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=32769)
    parser.add_argument("--arrival-delay-us", type=int, default=0)
    options = parser.parse_args()
    if not 1 <= options.count <= 4194306 or not 0 <= options.arrival_delay_us <= 1000000:
        parser.error("count or arrival delay outside fixture limits")
    count, guard = options.count, 8
    host = torch.arange(count + guard, dtype=torch.float32)
    first, second = host.cuda(), (host * 2).cuda()
    low_output = torch.full_like(first, -99)
    high_output = torch.full_like(first, -99)
    low = torch.cuda.Stream(priority=0)
    high = torch.cuda.Stream(priority=-1)
    torch.cuda.synchronize()
    for repetition in range(3):
        with torch.cuda.stream(high):
            torch.add(first[:1024], second[:1024], out=high_output[:1024])
        high.synchronize()
        with torch.cuda.stream(low):
            torch.add(first[:count], second[:count], out=low_output[:count])
        if options.arrival_delay_us:
            time.sleep(options.arrival_delay_us / 1000000)
        with torch.cuda.stream(high):
            torch.add(first[:1024], second[:1024], out=high_output[:1024])
        high.synchronize()
        pending = not low.query()
        low.synchronize()
        torch.testing.assert_close(low_output[:count].cpu(), host[:count] * 3,
                                   rtol=0, atol=0)
        torch.testing.assert_close(high_output[:1024].cpu(), host[:1024] * 3,
                                   rtol=0, atol=0)
        assert torch.all(low_output[count:].cpu() == -99).item()
        assert torch.all(high_output[1024:].cpu() == -99).item()
        torch.testing.assert_close(first.cpu(), host, rtol=0, atol=0)
        torch.testing.assert_close(second.cpu(), host * 2, rtol=0, atol=0)
        print(f"tensor_add repetition={repetition} results=ok guards=ok "
              f"high_completed=1 low_pending={int(pending)}")
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
