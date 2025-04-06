from abc import ABC, abstractmethod
import numpy as np
from typing import List, Optional

class Scheduler(ABC):
    """Abstract base class for all learning rate schedulers."""
    def __init__(self, optimizer, last_epoch: int = -1) -> None:
        """
        Args:
            optimizer: Wrapped optimizer.
            last_epoch: The index of last epoch. Default: -1.
        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        self.step_count = 0

    @abstractmethod
    def step(self) -> None:
        """Update the learning rate."""
        pass

    def get_last_lr(self) -> float:
        """Return the current learning rate."""
        return self.optimizer.init_lr

class StepLR(Scheduler):
    """Decays the learning rate by gamma every step_size epochs."""
    def __init__(self, optimizer, step_size: int = 30, gamma: float = 0.1, last_epoch: int = -1) -> None:
        """
        Args:
            optimizer: Wrapped optimizer.
            step_size: Period of learning rate decay.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch. Default: -1.
        """
        super().__init__(optimizer, last_epoch)
        self.step_size = step_size
        self.gamma = gamma

    def step(self) -> None:
        """Update the learning rate and reset step count if needed."""
        self.last_epoch += 1
        if self.last_epoch % self.step_size == 0 and self.last_epoch != 0:
            self.optimizer.init_lr *= self.gamma

class MultiStepLR(Scheduler):
    """Decays the learning rate by gamma when the epoch reaches one of the milestones."""
    def __init__(self, optimizer, milestones: List[int], gamma: float = 0.1, last_epoch: int = -1) -> None:
        """
        Args:
            optimizer: Wrapped optimizer.
            milestones: List of epoch indices. Must be increasing.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch. Default: -1.
        """
        super().__init__(optimizer, last_epoch)
        self.milestones = milestones
        self.gamma = gamma

    def step(self) -> None:
        """Update the learning rate when reaching a milestone."""
        self.last_epoch += 1
        if self.last_epoch in self.milestones:
            self.optimizer.init_lr *= self.gamma

class ExponentialLR(Scheduler):
    """Decays the learning rate by gamma every epoch."""
    def __init__(self, optimizer, gamma: float = 0.95, last_epoch: int = -1) -> None:
        """
        Args:
            optimizer: Wrapped optimizer.
            gamma: Multiplicative factor of learning rate decay.
            last_epoch: The index of last epoch. Default: -1.
        """
        super().__init__(optimizer, last_epoch)
        self.gamma = gamma

    def step(self) -> None:
        """Update the learning rate exponentially."""
        self.last_epoch += 1
        self.optimizer.init_lr *= self.gamma