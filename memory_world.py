"""memory_world.py - FlyWorld with hidden food after reveal window.

Memory test design (delayed navigation):
  Steps 0-4: food visible (sensors 7/9/10 carry signal)
  Steps 5+: food HIDDEN (sensors 7/9/10 = 0)

Brain must navigate to food using only the memory of what it saw
during the reveal window.

No memory -> random wandering after step 5 -> far from food.
Memory  -> retains direction estimate -> approaches food.

This isolates the "internal state maintenance" capacity of the brain.
Kathara's leaky integration (state_out = state*(1-leak) + new*leak)
naturally provides short-term memory if leak is small.
"""
import numpy as np
from kathara_brain_sim_v8 import FlyWorldV1


class MemoryFlyWorld(FlyWorldV1):
    """Food visible only during reveal window, then hidden."""

    def __init__(self, reveal_steps=5, total_steps=40, **kwargs):
        super().__init__(**kwargs)
        self.reveal_steps = reveal_steps
        self.total_steps = total_steps
        self.step_count = 0

    def reset(self):
        sensors = super().reset()
        self.step_count = 0
        return sensors

    def _get_sensors(self):
        if self.step_count >= self.reveal_steps:
            # Food sensors (7, 9, 10) -> zero. Only olfactory decays.
            return np.zeros(12)
        return super()._get_sensors()

    def step(self, nav, cen):
        self.step_count += 1
        sensors, dist, reached, caught = super().step(nav, cen)
        # After reveal window, recompute sensors as zero
        if self.step_count >= self.reveal_steps:
            sensors = np.zeros(12)
        return sensors, dist, reached, caught


if __name__ == "__main__":
    world = MemoryFlyWorld(seed=42, reveal_steps=5)
    s = world.reset()
    print(f"step 0 (visible): s7={s[7]:.2f} s9={s[9]:.2f} s10={s[10]:.2f}")
    for i in range(10):
        s, d, r, c = world.step(0.5, 0.5)
        vis = any(s[k] > 0 for k in [7, 9, 10])
        print(f"step {i+1}: s7={s[7]:.2f} s10={s[10]:.2f} dist={d:.2f} visible={vis}")
