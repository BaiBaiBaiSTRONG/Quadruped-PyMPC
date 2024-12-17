from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional

class SafetyLevel(Enum):
    """Safety level enumeration"""
    EMERGENCY = 0
    DANGER = 1
    WARNING = 2
    SAFE = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

class ConstraintPriority(Enum):
    """Constraint priority enumeration"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class SafetyStatus:
    """Overall safety status"""
    level: SafetyLevel
    violations: Dict[str, float]  # Constraint name -> violation value
    active_constraints: List[str]  # List of currently active constraints
    recommended_action: str
    details: Optional[Dict] = None  # Additional information if needed
    
    def __str__(self) -> str:
        return (f"Safety Level: {self.level.name}\n"
                f"Violations: {len(self.violations)}\n"
                f"Active Constraints: {len(self.active_constraints)}\n"
                f"Recommended Action: {self.recommended_action}")

@dataclass
class ConstraintStatus:
    """Status of a single constraint check"""
    name: str
    enabled: bool
    priority: ConstraintPriority
    value: float
    threshold: float
    violation: float
    is_violated: bool
    weight: float = 1.0

    def __str__(self) -> str:
        return (f"Constraint {self.name}: "
                f"{'VIOLATED' if self.is_violated else 'SATISFIED'} "
                f"(value: {self.value:.3f}, threshold: {self.threshold:.3f})")