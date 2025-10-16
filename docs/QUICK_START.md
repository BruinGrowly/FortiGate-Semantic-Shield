# FortiGate Semantic Shield v7.0 Quick Start

This guide accelerates evaluation in a lab or pilot environment while keeping the four stewardship axioms—LOVE, JUSTICE, POWER, and WISDOM—front of mind during configuration.

---

## Prerequisites
- FortiGate appliance or VM running firmware 7.0 or later with API access enabled.
- Python 3.8 or newer.
- Administrative API token for the target FortiGate instance.
- TLS-enabled management channel.

---

## 1. Clone and Install
```bash
git clone https://github.com/BruinGrowly/FortiGate-Semantic-Shield.git
cd FortiGate-Semantic-Shield
python -m venv .venv
.venv\Scripts\activate  # On Windows
# source .venv/bin/activate  # On macOS/Linux
pip install -r requirements_v7.txt
```

---

## 2. Configure Environment

Create a minimal configuration file that captures device connectivity and business context:

```bash
cp config.yaml config.local.yaml
```

Edit `config.local.yaml`:
- `fortigate.host`, `fortigate.token`, and `fortigate.verify_ssl`
- `business.risk_tolerance`, `business.priority_services`, `business.reporting_contacts`
- `compass_weights` if your organization emphasizes a specific axiom

> Tip: Keep LOVE, JUSTICE, POWER, and WISDOM weightings documented so governance teams can audit rationale.

---

## 3. Validate the Stack

```bash
# Fast semantic pipeline check
python simple_production_test.py --config config.local.yaml

# Confirm compass balance
python test_cardinal_axioms.py --config config.local.yaml
```

If `test_suite.py` is used, ensure the script begins with a Python docstring at the top before execution to avoid syntax errors.

---

## 4. Connect to FortiGate (Dry Run)

```bash
python deploy_to_fortigate.py \
  --device YOUR_FORTIGATE_IP \
  --token YOUR_API_TOKEN \
  --config config.local.yaml \
  --dry-run
```

Review the generated plan, paying attention to how recommendations are justified against the four axioms and mapped to business impacts.

---

## 5. Optional: Simulate Business Impact

```bash
python financial_stress_test.py --config config.local.yaml --report reports/financial_stress_preview.md
python advanced_business_simulation.py --config config.local.yaml --scenario resilience_playbook
```

These simulations translate technical outcomes into business narratives that can be shared with finance, risk, or operations stakeholders before live deployment.

---

## Troubleshooting Essentials
- `simple_validation.py`: smoke test for configuration drift.
- `simple_cardinal_test.py`: confirms compass weights after configuration changes.
- `test_cardinal_axioms.py --explain`: outputs narrative reasoning for each axiom to assist review boards.
- `logs/` directory (created on first run) captures semantic processing traces for debugging.

---

## Where to Go Next
1. Align `config.local.yaml` with production risk appetite and service-tier commitments.
2. Run `production_ready_deployment.py` in staging to validate orchestration timing and audit logging.
3. Share findings or enhancements via GitHub issues or pull requests so the community can refine business-aligned guardrails collectively.

---

## License Reminder
FortiGate Semantic Shield is open sourced under the MIT License. Refer to [`LICENSE`](LICENSE) for permissions and obligations.

