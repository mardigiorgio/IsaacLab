# Calibration probe output, 2026-08-18

Raw `--out` JSON from `scripts/probes/probe_contact_compliance.py`. These are
outputs, not inputs: nothing reads them but the probe's own `--verdict` mode, and
they are superseded the moment the task's contact authoring, `sim.dt` or the
`icf_warp` checkout changes. Re-run rather than cite.

Machine: RTX 5090, 16 envs, 300 env steps, tail 150, settle window 60.

| file | arm | gravity | ICF k [N/m] |
|---|---|---|---|
| `cal_mujoco.json` | fixed MuJoCo | 1g | - |
| `cal_icf_k5200.json` | fixed ICF | 1g | 5200 (uncalibrated starting point) |
| `cal_icf_kstar.json` | fixed ICF | 1g | 318.543 (one-shot rescale; moved n from 4 to 5) |
| `cal_icf_k248.json` | fixed ICF | 1g | 248.2 |
| `cal_icf_k289.json` | fixed ICF | 1g | 289.2 (accepted: AC1 PASS, AC2 PASS) |
| `cal_mujoco_3g.json` | fixed MuJoCo | 3g | - (NOT SETTLED) |
| `cal_icf_k289_3g.json` | fixed ICF | 3g | 289.2 (NOT SETTLED, scene collapsed) |

Reproduce the verdict:

```
./isaaclab.sh -p scripts/probes/probe_contact_compliance.py --verdict scripts/probes/results/*.json
```
