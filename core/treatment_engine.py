"""
WoundLens — Treatment Recommendation Engine
=============================================
Member 4 · Phase 1 Deliverable (Research) + Phase 3 Skeleton

Evidence-based treatment recommendations mapped to wound stages
and tissue composition. Based on clinical guidelines from:
- AAFP (American Academy of Family Physicians)
- NPIAP/EPUAP wound care protocols
- TIME framework (Tissue, Infection, Moisture, Edge)

==========================================================================
TREATMENT SELECTION PRINCIPLES
==========================================================================

Dressing choice depends on TWO primary factors:
  1. Wound STAGE (depth of tissue damage)
  2. EXUDATE LEVEL (amount of drainage)

The TIME Framework guides overall treatment:
  T = Tissue management   → Debride non-viable tissue
  I = Infection control    → Monitor/treat infection signs
  M = Moisture balance     → Choose dressings to hydrate or absorb
  E = Edge advancement     → Monitor epithelialization progress

==========================================================================
DRESSING TYPES REFERENCE
==========================================================================

| Dressing Type    | Best For              | How It Works                          |
|------------------|-----------------------|---------------------------------------|
| Transparent Film | Stage 1, low exudate  | Protects, retains moisture, see-thru  |
| Hydrocolloid     | Stage 2, low-mod exu  | Occlusive, moist healing, autolytic   |
| Foam             | Stage 2-3, mod exu    | Absorbent, cushioning, breathable     |
| Alginate         | Stage 3-4, heavy exu  | Highly absorbent, forms gel, seaweed  |
| Hydrogel         | Dry wounds, necrotic  | Adds moisture, softens eschar         |
| Hydrofiber       | Stage 3-4, heavy exu  | Gelling fiber, locks in exudate       |
| Silver dressings | Infected wounds       | Antimicrobial, broad-spectrum         |

==========================================================================
DEBRIDEMENT METHODS
==========================================================================

| Method      | Description                              | When to Use           |
|-------------|------------------------------------------|-----------------------|
| Autolytic   | Body's enzymes under occlusive dressing  | Mild slough, fragile  |
| Enzymatic   | Topical agents (e.g., collagenase)       | Moderate slough       |
| Mechanical  | Irrigation, wet-to-moist                 | Moderate debris       |
| Sharp       | Scalpel/scissors by trained clinician    | Heavy necrosis/eschar |

IMPORTANT: Stable, dry eschar on heels should generally NOT be debrided
unless signs of infection are present (it acts as a natural barrier).

==========================================================================
"""

# ---------------------------------------------------------------------------
# Treatment Database
# ---------------------------------------------------------------------------

import json
import os

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "treatments.json")

def _load_database():
    """Load the external JSON database containing clinical treatment rules."""
    if not os.path.exists(_DB_PATH):
        raise FileNotFoundError(f"Treatment database not found at {_DB_PATH}")
    with open(_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

_DATA = _load_database()
TREATMENT_DATABASE = _DATA["TREATMENT_DATABASE"]
INFECTION_SIGNS = _DATA["INFECTION_SIGNS"]


# ---------------------------------------------------------------------------
# Treatment Engine
# ---------------------------------------------------------------------------

def get_treatment(stage: str) -> dict:
    """
    Get evidence-based treatment recommendations for a given wound stage.

    Parameters
    ----------
    stage : str
        Wound stage from the staging engine (e.g., "Stage 1", "Stage 3", "Unstageable").

    Returns
    -------
    dict
        {
            "stage": str,
            "urgency": str,
            "primary_actions": list[str],
            "dressings": list[dict],
            "nutrition": str,
            "monitoring": str,
            "referral": str,
            "follow_up_days": int,
            "infection_signs": list[str],
            "debridement": dict | None,
            "disclaimer": str
        }
    """
    if stage not in TREATMENT_DATABASE:
        # Fallback for unknown stages
        stage = "Stage 2"

    treatment = TREATMENT_DATABASE[stage]

    return {
        "stage": stage,
        "urgency": treatment["urgency"],
        "primary_actions": treatment["primary_actions"],
        "dressings": treatment["dressings"],
        "nutrition": treatment["nutrition"],
        "monitoring": treatment["monitoring"],
        "referral": treatment["referral"],
        "follow_up_days": treatment["follow_up_days"],
        "infection_signs": INFECTION_SIGNS,
        "debridement": treatment.get("debridement", None),
        "disclaimer": (
            "DISCLAIMER: These recommendations are AI-generated for clinical "
            "decision SUPPORT only. They do NOT replace professional medical "
            "judgment. Always consult a qualified healthcare provider before "
            "initiating or changing wound care treatment."
        ),
    }


def get_urgency_label(urgency: str) -> dict:
    """Return a formatted urgency label with color and icon for UI display."""
    labels = {
        "LOW": {
            "text": "Low Priority",
            "color": "#4CAF50",
            "icon": "checkmark",
            "action": "Routine care — reassess in 72 hours",
        },
        "MODERATE": {
            "text": "Moderate Priority",
            "color": "#FFC107",
            "icon": "warning",
            "action": "Active treatment required — reassess weekly",
        },
        "HIGH": {
            "text": "High Priority",
            "color": "#FF9800",
            "icon": "alert",
            "action": "Specialist consult recommended — reassess in 1-3 days",
        },
        "CRITICAL": {
            "text": "Critical — Immediate Attention",
            "color": "#F44336",
            "icon": "error",
            "action": "Urgent referral required — daily assessment needed",
        },
    }
    return labels.get(urgency, labels["MODERATE"])


def format_treatment_summary(treatment: dict) -> str:
    """
    Format treatment recommendations into a readable text summary.
    Useful for console output and basic displays.
    """
    lines = []
    lines.append(f"TREATMENT PLAN — {treatment['stage']}")
    lines.append(f"Urgency: {treatment['urgency']}")
    lines.append(f"Follow-up: Every {treatment['follow_up_days']} day(s)")
    lines.append("")

    lines.append("PRIMARY ACTIONS:")
    for i, action in enumerate(treatment["primary_actions"], 1):
        lines.append(f"  {i}. {action}")
    lines.append("")

    lines.append("RECOMMENDED DRESSINGS:")
    for d in treatment["dressings"]:
        lines.append(f"  - {d['name']}")
        lines.append(f"    Reason: {d['reason']}")
        lines.append(f"    Examples: {', '.join(d['examples'])}")
    lines.append("")

    if treatment["debridement"]:
        lines.append("DEBRIDEMENT:")
        lines.append(f"  Method: {treatment['debridement']['method']}")
        lines.append(f"  Details: {treatment['debridement']['details']}")
        lines.append("")

    lines.append(f"NUTRITION: {treatment['nutrition']}")
    lines.append(f"MONITORING: {treatment['monitoring']}")
    lines.append(f"REFERRAL: {treatment['referral']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick Test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 65)
    print("WoundLens Treatment Engine - Test Cases")
    print("=" * 65)

    for stage in ["Stage 1", "Stage 2", "Stage 3", "Stage 4", "Unstageable"]:
        treatment = get_treatment(stage)
        urgency_info = get_urgency_label(treatment["urgency"])

        print(f"\n{'─' * 65}")
        print(f"  {stage} | Urgency: {urgency_info['text']}")
        print(f"  Follow-up: Every {treatment['follow_up_days']} day(s)")
        print(f"  Actions: {len(treatment['primary_actions'])} steps")
        print(f"  Dressings: {len(treatment['dressings'])} options")
        print(f"  Referral: {treatment['referral']}")

    print(f"\n{'=' * 65}")
    print("Treatment engine ready.")
    print("=" * 65)

    # Detailed output for one stage
    print(f"\n\n{'=' * 65}")
    print("DETAILED OUTPUT — Stage 3 Example:")
    print("=" * 65)
    stage3 = get_treatment("Stage 3")
    print(format_treatment_summary(stage3))
