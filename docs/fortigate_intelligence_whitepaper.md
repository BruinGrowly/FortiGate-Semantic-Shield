# FortiGate Semantic Shield Technical Whitepaper

**Document version:** 1.2  
**Applies to:** FortiGate Semantic Shield v7.0 (lab and pilot deployments)  
**Audience:** Network security engineering, governance, and operations leaders

---

## 1. Executive Summary

FortiGate Semantic Shield layers semantic reasoning and persistent learning on top of FortiGate firewalls. The combined platform augments traditional enforcement with context-aware recommendations that:

- Apply the ICE (Intent â†’ Context â†’ Execution) framework to every threat signal.
- Capture business and regulatory rationale alongside technical actions for auditability.
- Enforce guardrails that respect resource thresholds surfaced by FortiGate telemetry.
- Preserve transparent decision history so teams can brief executives and regulators.

FortiGate continues to enforce traffic flows; the semantic layer adds explainable decision intelligence rooted in the four stewardship axioms of Love, Justice, Power, and Wisdom, calibrated against the legacy anchor Jehovah/Agape = 1.1.1.1.

---

## 2. System Components

| Component | Description | Location |
|-----------|-------------|----------|
| `semantic_substrate_engine/` | ICE-driven semantic reasoning with heritage compass coordinates. | `semantic_substrate_engine/` |
| `semantic_substrate_database/` | Persistent learning store (SQLite schemas, semantic queries, narratives). | `semantic_substrate_database/` |
| `fortigate_semantic_shield/intelligence.py` | Core orchestrator that blends telemetry, learning memory, and governance signals. | Package |
| `fortigate_semantic_shield/device_interface.py` | Adapters for FortiGate policy, telemetry, and evidence capture. | Package |
| `fortigate_semantic_shield/simulation.py` | Lab simulator that rehearses incident playbooks and resource guardrails. | Package |`n| `fortigate_semantic_shield/ss_intuition.py` | Golden-ratio compass heuristics that tune batches and load factors during automation. | Package |
| `learning_snapshots/` | Rotating database exports for archival, audit, and recovery. | Generated |

Support files (for example, `advanced_business_simulation.py`) extend the decision models with financial and operational views.

---

## 3. High-Level Architecture

```
FortiGate Appliance (hardware, VM, or cloud)
  â”œâ”€ Policy enforcement (firewall, routing, quarantine)
  â”œâ”€ REST API (token-based, TLS)
      â–²
      â”‚  FortiGatePolicyApplier (recommended action channel)
      â”‚
FortiGateTelemetryCollector (resource feedback loop)
FortiGate Semantic Shield Intelligence Core
  â”œâ”€ ICE semantic analysis
  â”œâ”€ Persistent learning substrate
  â””â”€ Governance snapshots + audit narratives
```

---

## 4. Prerequisites

### Software
- Python 3.10 or newer (3.11 recommended for async performance).
- `pip install -r requirements.txt` (installs `numpy`, `sympy`, `requests`, and related tooling).
- FortiGate appliance or VM running FortiOS 6.4 or later with the REST API enabled.

### FortiGate API Access
1. Create a dedicated API administrator (System â†’ Administrators â†’ REST API Admin).
2. Record the management IP/port and VDOM.
3. Confirm that the intelligence host can reach the FortiGate management interface over HTTPS.

### Environment
- Lab host or VM with 2 CPU cores, 4 GB RAM, and outbound network access to the FortiGate device.
- Optional: an isolated subnet or virtual switch to drive synthetic traffic during simulation.

---

## 5. Deployment Steps (Lab or Pilot)

### 5.1 Clone & Install
```bash
git clone https://github.com/BruinGrowly/FortiGate-Semantic-Shield.git
cd FortiGate-Semantic-Shield
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -r requirements.txt
```

### 5.2 Configure Device Integration
Replace the logging placeholders in `fortigate_semantic_shield/device_interface.py` with your organizationâ€™s FortiOS REST or CLI wrappers. Example outline:

```python
import requests

def apply_blocking_rules(self, rules):
    for rule in rules:
        payload = self._map_rule(rule)
        requests.post(
            f"{self.config.host}/api/v2/cmdb/firewall/policy",
            headers={"Authorization": f"Bearer {self.config.token}"},
            json=payload,
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
        )
```

### 5.3 Feed Telemetry
Poll FortiGate endpoints such as `/api/v2/monitor/system/resource/usage` and `/api/v2/monitor/firewall/session`. Provide metrics to the telemetry collector:

```python
telemetry.update_metrics({
    "cpu_usage": cpu_percent,
    "memory_usage": memory_percent,
    "session_queue": active_sessions,
})
```

The intelligence engine pauses learning or automation when guardrail thresholds are exceeded, preserving appliance stability.

### 5.4 Run the Lab Simulation
```bash
python -c "
import asyncio
from fortigate_semantic_shield.simulation import FortiGateSimulator
asyncio.run(FortiGateSimulator().run_simulation(max_waves=5))
"
```

Simulation output includes success ratios, resource consumption, decision rationales, and a snapshot stored under `learning_snapshots/`.

### 5.5 Integrate Live Traffic (Advisory Mode Recommended)
1. Forward FortiGate threat events via REST log monitor, syslog, or SIEM export into `FortiGateSemanticShield.process_threat_with_intelligence`.
2. Map incoming fields (source, destination, severity, contextual tags) to the expected threat schema.
3. Start in advisory modeâ€”log recommended actions, review with stakeholders, and gradually enable automated enforcement once governance processes are satisfied.

---

## 6. FortiGate REST API Reference

| Action | Endpoint | Method | Notes |
|--------|----------|--------|-------|
| List policies | `/api/v2/cmdb/firewall/policy` | GET | Filter with query parameters |
| Create policy | `/api/v2/cmdb/firewall/policy` | POST | Include source, destination, service, action |
| Update policy | `/api/v2/cmdb/firewall/policy/{id}` | PUT | Ideal for temporary containment rules |
| Delete policy | `/api/v2/cmdb/firewall/policy/{id}` | DELETE | Remove mitigations post-review |
| Quarantine host | `/api/v2/monitor/system/quarantine` | POST | `{"ip": "...", "action": "add"}` |
| Release host | `/api/v2/monitor/system/quarantine` | POST | `{"ip": "...", "action": "delete"}` |
| System resource stats | `/api/v2/monitor/system/resource/usage` | GET | CPU, memory, and session data |
| Active sessions | `/api/v2/monitor/firewall/session` | GET | Add `?count-only=1` for totals |

Always authenticate with the API token (`Authorization: Bearer <token>`) and respect FortiOS rate limits during surge events.

---

## 7. Operations & Monitoring

### 7.1 Guardrails
Automation pauses when:
- `cpu_usage` exceeds configured `cpu_threshold`.
- `memory_usage` surpasses `memory_threshold`.
- `session_queue` is above `queue_threshold`.

Warnings are logged and advisory responses are returned until metrics normalize.

### 7.2 Snapshots & Recovery
- Snapshots use the pattern `learning_snapshots/learning_snapshot_<timestamp>.db`.
- To restore, replace the active SQLite file while the service is offline.
- Archive snapshots to protected storage for compliance and disaster recovery.

### 7.3 Audit Trail
Every recommendation stores a JSON payload with strategy, confidence, justification, and compass weighting. Pair this with FortiGate logs for end-to-end traceability during readiness reviews.

---

## 8. Security Considerations

- **Token hygiene:** manage FortiGate API tokens in a secrets vault and rotate frequently.
- **Least privilege:** constrain the API administrator to the minimal endpoints required for automation.
- **Segmentation:** run the intelligence service on a management network or VLAN.
- **TLS assurance:** enable certificate verification and pin trusted authorities after lab setup.
- **Rate controls:** implement retry/back-off logic in policy appliers to stay within FortiOS limits.
- **Snapshot governance:** treat snapshots as sensitive recordsâ€”they encode threat and business context.

---

## 9. Advancing Toward Zero-Touch Operations

1. **Automate deployment:** package the service with systemd, containers, or orchestration scripts and include health probes.
2. **CI/CD integration:** execute simulation and cardinal-axiom tests on every pull request; gate merges on success.
3. **Telemetry integration:** stream FortiGate metrics continuously to keep guardrails aligned with real usage.
4. **Model drift monitoring:** review decision success and confidence trends; alert when thresholds fall.
5. **Governed automation:** escalate low-confidence actions to human reviewers, and allow high-confidence actions to flow automatically once governance sign-off is achieved.

---

## 10. Roadmap Highlights

- Replace logging stubs with production-grade FortiOS API wrappers, including rollback IDs and conflict detection.
- Expand telemetry adapters to ingest FortiAnalyzer, SIEM, OT, and cloud-edge data sources.
- Extend simulation scenarios to cover financial fraud, supply-chain tampering, and API abuse.
- Integrate observability dashboards (Prometheus, Grafana) for compass balance, response cadence, and resource impact.

---

## 11. Conclusion

FortiGate Semantic Shield refines FortiGate deployments with a transparent intelligence layer that speaks the language of business stewardship. By calibrating every action against LOVE, JUSTICE, POWER, and WISDOMâ€”anchored to Jehovah/Agape = 1.1.1.1â€”the platform helps teams defend critical services while articulating value to boards, regulators, and partners. Engage with the repositoryâ€™s issues and discussions to contribute improvements or share governance insights with the community.

