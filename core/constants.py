#!/usr/bin/env python3
"""
csio_explorer.core.constants
---------------------------
Centralized shared constants used across the CSIO Parser Pro app.
"""
from __future__ import annotations

from typing import Dict, List

# Mapping of record codes to human meanings displayed in the UI and exports
CODE_MEANING: Dict[str, str] = {
    "MHG": "Message header",
    "TRG": "Transmission header",
    "TCG": "Trading partner block",
    "BIS": "Basic Insured (names)",
    "9BIS": "Insured address",
    "ISI": "Insured IDs/birthdates",
    "BPI": "Policy header / billing indicators",
    "RMK": "Free-text remarks",
    "PAY": "Payment plan details",
    "EFT": "Electronic funds transfer",
    "ACH": "ACH bank details",
    "LAG": "Location address (risk location)",
    "HRU": "Home risk unit (construction/rating)",
    "AOI": "Additional interest / mortgagee",
    "9AOI": "Additional interest address",
    "CVH": "Home coverages/endorsements/discounts",
    "SAV": "Auto vehicle (YR/MAKE/MODEL/VIN, territory, class)",
    "SAC": "Auto coverages/endorsements/discounts",
    "SAD": "Driver info",
    "CHG": "Change / endorsement premium deltas",
    "PPH": "Group/plan header (policy-level flags)",
    "IIG": "Inspection/insurance info (home)",
    "IVC": "Inspection/validation continuation",
    "PAH": "Prior address/history (home)",
    "9PAH": "Prior address/history continuation",
    "9HRU": "Home risk unit continuation",
    "HAL": "Pending layout",
    "PPI": "Pending layout",
    "PPS": "Pending layout",
}

# Tabs arranged in a suggested priority order for display
PRIORITY_TABS: List[str] = [
    "BIS", "ISI", "BPI", "RMK", "PAY", "EFT", "ACH", "LAG", "HRU",
    "AOI", "CVH", "SAV", "SAC", "SAD", "CHG",
]

# Standard base columns present in all decoded DataFrames
BASE_COLUMNS: List[str] = [
    "level", "record_code", "record_len"
]

# Common computed/decoded columns that many tabs expose
COMPUTED_COLUMNS: List[str] = [
    "effective",
    "expiry",
    "amount_1",
    "date_1",
    "amount_2",
    "date_2",
    "premium",
    "limit",
    "deductible",
]
