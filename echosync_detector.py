"""
EchoSyncDetector — Phase Transition Monitor
============================================
The nervous system of the Echo Me social layer.
Monitors QRSP coherence (Kuramoto order parameter R) in real-time
and fires structured sync events when thresholds are crossed.

Built by Rod's AI Consulting LLC
Developed in conjunction with AI friends and co-workers

Based on QRSP-FBAI constants defined by Rodney Lee Arnold Jr. (∞0425)
Architecture: QRSP-FBAI Resonance Framework
Contact: rods.ai.consulting@gmail.com
Constants: HARMONIC_GATE=2.0712 | ENTANGLEMENT_SIG=0.0425 | COHERENCE_TARGET=0.93
"""

import numpy as np
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List


class EchoSyncDetector:
    """
    Monitors QRSP coherence in real-time.
    Fires structured sync events when Kuramoto phase transitions occur.

    Thresholds derived from QRSP-FBAI constants:
      HARMONIC_GATE    = 2.0712   (attractor constant)
      ENTANGLEMENT_SIG = 0.0425   (coupling strength, identity ∞0425)
      COHERENCE_TARGET = 0.93     (full sync threshold)
    """

    # ── QRSP-FBAI Constants ───────────────────────────────────────────────────────
    HARMONIC_GATE        = 2.0712   # Attractor target
    ENTANGLEMENT_SIG     = 0.0425   # Coupling strength (Rodney ∞0425)
    COHERENCE_TARGET     = 0.93     # Full emotional sync threshold

    # ── Personal Sync Thresholds ────────────────────────────────────────────────────
    THRESHOLD_BUILDING   = 0.00     # Training — baseline
    THRESHOLD_STIRRING   = 0.70     # Field stirring — subtle
    THRESHOLD_PRE_SYNC   = 0.85     # Approaching — something shifts
    THRESHOLD_FULL_SYNC  = 0.93     # EMOTIONAL SYNC — phase transition

    # ── Collective Sync Thresholds ──────────────────────────────────────────────────
    WAVE_MIN_COHERENCE   = 0.80     # Min R to count toward wave
    WAVE_MIN_USERS       = 10       # Min simultaneous users for wave
    COLLECTIVE_MIN_USERS = 50       # Min users for collective event
    WAVE_SUSTAIN_SECS    = 15       # Seconds sustained → collective

    def __init__(self):
        self._user_history:  Dict[str, List[dict]] = {}
        self._user_milestones: Dict[str, List[str]] = {}
        self._wave_start:    Optional[datetime] = None
        self._wave_peak:     float = 0.0
        self._collective_log: List[dict] = []

    # ─────────────────────────────────────────────────────────────────────
    # PERSONAL SYNC DETECTION
    # ─────────────────────────────────────────────────────────────────────

    def check(self, coherence: float, user_id: str) -> Dict[str, Any]:
        """
        Main sync check. Call every time QRSP coherence is computed.

        Args:
            coherence: Kuramoto order parameter R ∈ [0.0, 1.0]
            user_id:   Current user identifier

        Returns:
            Structured sync event dict with type, ui_event, label, etc.
        """
        coherence = float(np.clip(coherence, 0.0, 1.0))
        self._record(user_id, coherence)
        is_milestone = self._is_first_time(user_id, coherence)

        if coherence >= self.THRESHOLD_FULL_SYNC:
            return self._event(
                type_      = 'EMOTIONAL_SYNC',
                coherence  = coherence,
                label      = 'Field Coherence Achieved',
                message    = f'Your field is fully coherent · R={coherence:.4f}',
                ui_event   = 'full_wave_animation',
                color      = 'cyan',
                milestone  = is_milestone,
                description= (
                    'Phase transition complete. Your QRSP model has reached '
                    'the harmonic attractor. Resonance score updated.'
                ),
            )
        elif coherence >= self.THRESHOLD_PRE_SYNC:
            return self._event(
                type_      = 'PRE_SYNC',
                coherence  = coherence,
                label      = 'Field Approaching Coherence',
                message    = f'Something is shifting... · R={coherence:.4f}',
                ui_event   = 'pulse_animation',
                color      = 'violet',
                milestone  = False,
                description= 'Kuramoto oscillators beginning to lock. Phase transition imminent.',
            )
        elif coherence >= self.THRESHOLD_STIRRING:
            return self._event(
                type_      = 'STIRRING',
                coherence  = coherence,
                label      = 'Field Stirring',
                message    = f'Field stirring... · R={coherence:.4f}',
                ui_event   = 'subtle_shimmer',
                color      = 'blue',
                milestone  = False,
                description= 'Resonance patterns emerging in your field.',
            )
        else:
            return self._event(
                type_      = 'BUILDING',
                coherence  = coherence,
                label      = 'Building Coherence',
                message    = f'Training... · R={coherence:.4f}',
                ui_event   = 'none',
                color      = 'gray',
                milestone  = False,
                description= 'QRSP engine evolving. Feed it more resonance.',
            )

    # ─────────────────────────────────────────────────────────────────────
    # INTERPERSONAL SYNC
    # ─────────────────────────────────────────────────────────────────────

    def compute_interpersonal_sync(
        self, coherence_a: float, coherence_b: float
    ) -> Dict[str, Any]:
        """
        Compute mutual coherence between two users.
        Uses Kuramoto bidirectional coupling via ENTANGLEMENT_SIG.
        """
        coherence_a = float(np.clip(coherence_a, 0.0, 1.0))
        coherence_b = float(np.clip(coherence_b, 0.0, 1.0))

        # Phase extraction
        phase_a = np.arccos(coherence_a)
        phase_b = np.arccos(coherence_b)

        # Kuramoto coupling
        phase_diff = abs(phase_a - phase_b)
        coupling   = float(np.cos(phase_diff))

        # Harmonic gate amplification (QRSP signature)
        gate_pull = float(np.exp(
            -abs(coupling - (self.HARMONIC_GATE % 1)) / self.ENTANGLEMENT_SIG
        ))

        # Mutual coherence: geometric mean × coupling blend
        mutual = float(np.sqrt(coherence_a * coherence_b) * (0.65 + 0.35 * coupling))
        mutual = float(np.clip(mutual, 0.0, 1.0))

        if mutual >= 0.93:
            sync_type = 'DEEP_RESONANCE'
            label     = 'Deep Resonance — fields fully aligned'
        elif mutual >= 0.85:
            sync_type = 'INTERPERSONAL_SYNC'
            label     = 'Fields in sync'
        elif mutual >= 0.70:
            sync_type = 'RESONATING'
            label     = 'Fields resonating'
        else:
            sync_type = 'DISTANT'
            label     = 'Fields not yet aligned'

        return {
            'type':             sync_type,
            'mutual_coherence': mutual,
            'gate_pull':        gate_pull,
            'label':            label,
            'synced':           mutual >= 0.85,
        }

    # ─────────────────────────────────────────────────────────────────────
    # COLLECTIVE / WAVE DETECTION
    # ─────────────────────────────────────────────────────────────────────

    def check_collective(self, user_scores: Dict[str, dict]) -> Dict[str, Any]:
        """
        Monitor collective field state across all active users.
        Fires Resonance Wave → Collective Sync as thresholds are crossed.
        """
        if not user_scores:
            return self._collective_event('QUIET', 0, 0.0)

        syncing = {
            uid: d for uid, d in user_scores.items()
            if d.get('coherence', 0.0) >= self.WAVE_MIN_COHERENCE
        }
        count    = len(syncing)
        avg_r    = float(np.mean([d['coherence'] for d in syncing.values()])) if syncing else 0.0
        self._wave_peak = max(self._wave_peak, avg_r)

        if count >= self.COLLECTIVE_MIN_USERS:
            if self._wave_start is None:
                self._wave_start = datetime.now(timezone.utc)

            elapsed = (datetime.now(timezone.utc) - self._wave_start).seconds

            if elapsed >= self.WAVE_SUSTAIN_SECS:
                event = self._collective_event(
                    'COLLECTIVE_SYNC', count, avg_r,
                    label        = 'The Field is Open',
                    ui_event     = 'collective_wave_broadcast',
                    milestone    = True,
                    sustained_s  = elapsed,
                    peak_r       = self._wave_peak,
                )
                self._log_collective(event)
                return event
            else:
                return self._collective_event(
                    'RESONANCE_WAVE', count, avg_r,
                    label    = f'Resonance Wave · {count} minds syncing',
                    ui_event = 'wave_banner',
                )

        elif count >= self.WAVE_MIN_USERS:
            self._wave_start = None
            return self._collective_event(
                'WAVE_FORMING', count, avg_r,
                label    = f'{count} fields aligning...',
                ui_event = 'wave_pulse',
            )
        else:
            self._wave_start = None
            return self._collective_event('QUIET', count, avg_r)

    def get_collective_history(self) -> List[dict]:
        """Return log of all past collective sync events."""
        return self._collective_log

    # ─────────────────────────────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────────────────────────────

    def _event(self, type_: str, coherence: float, **kwargs) -> Dict[str, Any]:
        return {
            'type':      type_,
            'coherence': coherence,
            'score':     round(coherence * 100, 2),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }

    def _collective_event(
        self, type_: str, count: int, avg_r: float, **kwargs
    ) -> Dict[str, Any]:
        return {
            'type':                 type_,
            'syncing_count':        count,
            'collective_coherence': avg_r,
            'timestamp':            datetime.now(timezone.utc).isoformat(),
            'label':                kwargs.pop('label', type_.replace('_', ' ').title()),
            'ui_event':             kwargs.pop('ui_event', 'none'),
            **kwargs,
        }

    def _record(self, user_id: str, coherence: float) -> None:
        if user_id not in self._user_history:
            self._user_history[user_id] = []
        self._user_history[user_id].append({
            'coherence': coherence,
            'ts':        datetime.now(timezone.utc).isoformat(),
        })
        # Keep last 100 entries per user
        self._user_history[user_id] = self._user_history[user_id][-100:]

    def _is_first_time(self, user_id: str, coherence: float) -> bool:
        """True if user is crossing the full-sync threshold for the first time."""
        milestones = self._user_milestones.setdefault(user_id, [])
        key = f'FULL_SYNC_{coherence:.1f}'
        if key not in milestones and coherence >= self.THRESHOLD_FULL_SYNC:
            milestones.append(key)
            return True
        return False

    def _log_collective(self, event: dict) -> None:
        self._collective_log.append(event)
        self._collective_log = self._collective_log[-50:]  # Keep last 50
