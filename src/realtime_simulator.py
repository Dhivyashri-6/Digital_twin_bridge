from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class RealtimeSimulator:
    data: pd.DataFrame
    index: int = 0
    state: str = "STOPPED"

    def start(self): self.state = "RUNNING"
    def pause(self): self.state = "PAUSED"
    def stop(self): self.state = "STOPPED"
    def reset(self): self.index, self.state = 0, "STOPPED"
    @property
    def finished(self): return self.index >= len(self.data)
    def next_window(self):
        if self.state != "RUNNING" or self.finished: return None
        row = self.data.iloc[self.index]; self.index += 1
        if self.finished: self.state = "STOPPED"
        return row
