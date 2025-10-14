"""Example: run the FortiGate Semantic Shield simulator."""

import asyncio

from fortigate_semantic_shield.simulation import FortiGateSimulator


def main() -> None:
    simulator = FortiGateSimulator()
    asyncio.run(simulator.run_simulation(max_waves=3))


if __name__ == "__main__":
    main()
