# Copyright (c) 2026, Newton-Adaptive workstream. SPDX-License-Identifier: BSD-3-Clause
"""Auto-loading GUI toggle for the Newton adaptive integrator.

This Kit extension is listed in ``apps/isaaclab.python.kit`` ``[dependencies]`` so it loads
automatically with every Isaac Lab GUI session. It adds a small **Newton Integrator** window
with one checkbox -- *Adaptive timestepping* -- that drives the carb setting
``/isaaclab/newton/adaptive``.

``NewtonMJWarpManager._build_solver`` reads that setting when it builds the solver, so the
checkbox switches the integrator between fixed-step :class:`~newton.solvers.SolverMuJoCo` (off)
and error-controlled step-doubling :class:`~newton.solvers.SolverMuJoCoAdaptive` (on). Because
switching the integrator rebuilds the solver, the change takes effect on the next **Stop -> Play**
(a live mid-sim swap is intentionally avoided -- it would have to re-seat contacts and state).

Confirm via ``/tmp/newton_adaptive.log``: ON -> contact frames show inner-dt spread > 0 and many
substeps; OFF -> fixed.
"""

import carb
import omni.ext
import omni.ui as ui

ADAPTIVE_SETTING = "/isaaclab/newton/adaptive"


class NewtonAdaptiveUIExtension(omni.ext.IExt):
    """Builds (and owns) the floating Newton-integrator toggle window."""

    def on_startup(self, _ext_id: str):
        self._settings = carb.settings.get_settings()
        if self._settings.get(ADAPTIVE_SETTING) is None:
            self._settings.set_bool(ADAPTIVE_SETTING, False)
        self._window = ui.Window("Newton Integrator", width=360, height=160)
        self._build_ui()

    def _build_ui(self):
        with self._window.frame:
            with ui.VStack(spacing=10, height=0):
                ui.Label("Newton solver — timestepping mode", style={"font_size": 16})
                with ui.HStack(height=26, spacing=8):
                    checkbox = ui.CheckBox(width=22)
                    checkbox.model.set_value(self._settings.get_as_bool(ADAPTIVE_SETTING))
                    checkbox.model.add_value_changed_fn(self._on_changed)
                    ui.Label("Adaptive timestepping  (SolverMuJoCoAdaptive)")
                ui.Label(
                    "ON = error-controlled step-doubling; subdivides dt at stiff contact.\n"
                    "OFF = fixed-step MuJoCo-Warp.\n"
                    "Applies on the next Stop → Play (the solver is rebuilt).",
                    style={"font_size": 12, "color": 0xFF9AA0A6},
                    word_wrap=True,
                )

    def _on_changed(self, model):
        on = bool(model.get_value_as_bool())
        self._settings.set_bool(ADAPTIVE_SETTING, on)
        mode = "SolverMuJoCoAdaptive (step-doubling)" if on else "SolverMuJoCo (fixed-step)"
        carb.log_warn(f"[Newton Integrator] adaptive = {on} -> {mode}. Stop + Play to rebuild the solver.")

    def on_shutdown(self):
        self._window = None
