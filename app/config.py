from dataclasses import dataclass
from datetime import datetime
from typing import Optional


DEFAULT_AMOUNT_TOL = 1.0
DEFAULT_DATE_TOL = 2


@dataclass
class ReconParams:
    amount_tol: float = DEFAULT_AMOUNT_TOL
    date_tol: int = DEFAULT_DATE_TOL
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None

    @classmethod
    def from_strings(
        cls,
        amount_tol: Optional[str] = None,
        date_tol: Optional[str] = None,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
    ) -> "ReconParams":
        amt = float(amount_tol) if amount_tol not in (None, "") else DEFAULT_AMOUNT_TOL
        d_tol = int(date_tol) if date_tol not in (None, "") else DEFAULT_DATE_TOL
        df = datetime.fromisoformat(date_from) if date_from else None
        dt = datetime.fromisoformat(date_to) if date_to else None
        return cls(amount_tol=amt, date_tol=d_tol, date_from=df, date_to=dt)
