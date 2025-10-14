# SSE + SSD FortiGate Intelligence<br/>Technical Whitepaper

**Document version:** 1.0  
**Applies to:** `FortiGateSemanticShield v1.0.0` and lab simulator (`fortigate_semantic_shield.simulation`)  
**Prepared for:** Network engineering team – VM FortiGate lab rollout

---

## 1. Executive Summary

This whitepaper explains how to deploy the Semantic Substrate Engine (SSE) and Semantic Substrate Database (SSD) alongside a FortiGate firewall. The combined system adds a cognitive layer that:

- Performs ICE (Intent–Context–Execution) semantic analysis for each threat event.
- Learns continuously and persists results (signatures, confidence, justifications) in SQLite.
- Applies FortiGate policies (blocking, quarantine, routing tweaks) via REST APIs.
- Enforces resource guardrails based on live FortiGate telemetry and exports auditable snapshots.

The solution **complements** FortiGate; the appliance continues to enforce traffic while SSE + SSD supplies explainable AI, memory, and orchestration.

---

## 2. System Components

| Component | Description | Location |
|-----------|-------------|----------|
| `semantic_substrate_engine/` | ICE-based semantic reasoning (Biblical coordinates, SSE APIs). | `semantic_substrate_engine/` |
| `semantic_substrate_database/` | Persistent learning store (SQLite schemas, semantic queries). | `semantic_substrate_database/` |
| `fortigate_semantic_shield/intelligence.py` | Core SSE + SSD orchestrator (learning, guardrails, snapshots). | package |
| `fortigate_semantic_shield/device_interface.py` | FortiGate policy, telemetry, and snapshot adapters. | package |
| `fortigate_semantic_shield/simulation.py` | Constrained FortiGate stress-test/lab simulator. | package |
| `learning_snapshots/` | Rotating database exports (`learning_snapshot_*.db`). | generated |

Supporting modules (e.g., `meaning_scaffold_demo.py`, `truth_scaffold_revelation.py`) extend semantics and require no extra deployment steps.

---

## 3. High-Level Architecture

```
FortiGate Appliance (VM/Lab)
  ├─ Policy table / firewall
  ├─ Routing / sessions
  └─ REST API (token-based)
        ▲
        │ (REST/CLI calls from FortiGatePolicyApplier)
        │
FortiGatePolicyApplier ───► applies block/quarantine/routing/healing
FortiGateTelemetryCollector ─► feeds CPU/memory/session metrics
FortiGateSemanticShield (SSE + SSD brain)
  ├─ Semantic analysis (ICE Semantic Engine)
  ├─ Persistent learning (Semantic Substrate Database)
  ├─ Confidence, patterns, teaching sessions
  └─ LearningPersistenceManager snapshots → learning_snapshots/
```

---

## 4. Prerequisites

### Software
- Python 3.10 or newer (3.11+ recommended).
- `pip install -r requirements.txt` (installs `numpy`, `sympy`, `requests`).
- FortiGate VM appliance (FortiOS 6.4/7.x) with REST API enabled.

### FortiGate API Access
1. Create an API token (System ▶ Administrators ▶ REST API Admin).
2. Record the management IP/port and VDOM.
3. Ensure the SSE host can reach HTTPS on the FortiGate.

### Environment
- VM or bare-metal host for the intelligence stack (2 CPU cores, 4 GB RAM sufficient for lab).
- Optional: dedicated subnet/virtual switch for synthetic traffic during simulation.

---

## 5. Deployment Steps (VM Lab)

### 5.1 Clone & Install
```bash
git clone https://github.com/BruinGrowly/FortiGate-Semantic-Shield.git
cd FortiGate-Semantic-Shield
python -m venv .venv
source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 5.2 Configure Device Integration
Edit `fortigate_semantic_shield/device_interface.py` and replace the placeholder `apply_*` methods with FortiOS REST/CLI calls. Example (pseudo):
```python
import requests

def apply_blocking_rules(self, rules):
    for rule in rules:
        payload = {...}  # map semantic rule to FortiOS policy
        requests.post(f"{self.config.host}/api/v2/cmdb/firewall/policy",
                      headers={"Authorization": f"Bearer {self.config.token}"},
                      json=payload, timeout=self.config.timeout, verify=self.config.verify_ssl)
```

### 5.3 Feed Telemetry
Poll FortiGate stats (`/api/v2/monitor/system/resource/usage`, `/api/v2/monitor/firewall/session`) and call:
```python
telemetry.update_metrics({
    "cpu_usage": cpu,
    "memory_usage": mem,
    "session_queue": sessions,
})
```
The engine pauses learning when thresholds are exceeded, preventing appliance overload.

### 5.4 Run the Lab Simulation
```bash
python -c "import asyncio; from fortigate_semantic_shield.simulation import FortiGateSimulator; asyncio.run(FortiGateSimulator().run_simulation(max_waves=5))"
```

Outputs include success metrics, system health, teaching sessions, and a snapshot under `learning_snapshots/`.

### 5.5 Integrate Live Traffic
1. Stream FortiGate threat events (syslog, REST log monitor, or SIEM forwarder) into `FortiGateSemanticShield.process_threat_with_intelligence`.
2. Map incoming data to the expected `threat_data` format (source/destination, threat type, severity, context).
3. Allow the policy applier to enforce actions or run in advisory mode by logging them for manual approval.

---

## 6. FortiGate REST API Cheat Sheet

| Action | Endpoint | Method | Notes |
|--------|----------|--------|-------|
| List policies | `/api/v2/cmdb/firewall/policy` | GET | Filter via query parameters |
| Create policy | `/api/v2/cmdb/firewall/policy` | POST | Embed src/dst, action, service |
| Update policy | `/api/v2/cmdb/firewall/policy/{id}` | PUT | Adjust temporary rules |
| Delete policy | `/api/v2/cmdb/firewall/policy/{id}` | DELETE | Remove mitigations |
| Quarantine host | `/api/v2/monitor/system/quarantine` | POST | `{"ip": "...", "action": "add"}` |
| Release host | `/api/v2/monitor/system/quarantine` | POST | `{"ip": "...", "action": "delete"}` |
| System stats | `/api/v2/monitor/system/resource/usage` | GET | CPU, memory, sessions |
| Session count | `/api/v2/monitor/firewall/session` | GET | Include `count-only=1` |

Always include the API token (`Authorization: Bearer <token>`). Respect FortiOS rate limits during attack bursts.

---

## 7. Operations & Monitoring

### 7.1 Guardrails
Learning/automation pauses when:
- `cpu_usage ≥ cpu_threshold`
- `memory_usage ≥ memory_threshold`
- `session_queue ≥ queue_threshold`

The engine logs a warning and returns fallback responses until resources recover.

### 7.2 Snapshots & Recovery
- Snapshots live in `learning_snapshots/learning_snapshot_<timestamp>.db`.
- Restore by replacing the active SQLite database before starting the engine.
- Archive snapshots off-box for compliance and disaster recovery.

### 7.3 Audit Trail
`defense_history` contains a JSON payload for every defense (strategy, confidence, justification). Pair with FortiGate logs to show why a policy was created, updated, or removed.

---

## 8. Security Considerations

- **API token hygiene**: store tokens in vaults or environment variables; rotate regularly.
- **Least privilege**: assign a FortiGate admin profile limited to required endpoints.
- **Segmentation**: run the intelligence service on a management network/VLAN.
- **TLS**: enable certificate verification once lab certificates are trusted.
- **Rate limiting**: add retry/back-off logic in the policy applier for large attack waves.
- **Snapshot handling**: snapshots contain threat intelligence—treat them as sensitive data.

---

## 9. Toward Zero-Touch Operation

1. **Automate deployment** (systemd, Docker, health checks).
2. **CI/CD**: execute the brutal simulator on every change; accept only passing builds.
3. **Telemetry integration**: continuously feed FortiGate metrics to keep guardrails accurate.
4. **Model drift detection**: schedule reports on success/confidence; alert if they drop.
5. **Fallback playbooks**: keep low-confidence responses under human review while high-confidence cases auto-enforce.

---

## 10. Future Enhancements

- Replace logging placeholders with production FortiOS REST wrappers (including rollback IDs).
- Integrate FortiAnalyzer/SIEM feeds for cross-validation.
- Extend simulator scenarios (e.g., SWIFT fraud, insider transfers, API tampering).
- Add dashboards (Grafana/Prometheus) for confidence, pattern evolution, and system health.

---

## 11. Conclusion

FortiGate Semantic Shield delivers a transparent, continuously learning brain for FortiGate appliances. Configuring API access, wiring telemetry and policy hooks, and validating with the lab simulator enables a powerful, explainable defense fabric that grows smarter with every incident while keeping operational risk controlled.

For collaboration or questions, review the repository modules and open an issue or discussion on GitHub.
