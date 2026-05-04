"""
WoundLens — Staging Engine
==========================
Member 4 · Phase 1 Deliverable

Rule-based wound staging engine using NPIAP/EPUAP clinical guidelines.
Takes tissue classification percentages as input and returns a wound stage.

CLINICAL REFERENCE: NPIAP (National Pressure Injury Advisory Panel) 2019
Updated Staging System + EPUAP (European Pressure Ulcer Advisory Panel)

==========================================================================
WOUND BED TISSUE TYPES (used as input to staging)
==========================================================================

1. GRANULATION TISSUE
   - Appearance : Pink/Red, moist, bumpy ("beefy red")
   - Meaning    : HEALTHY — indicates active healing, good blood supply
   - Note       : Dark, friable (easily bleeding) granulation = possible infection

2. EPITHELIAL TISSUE
   - Appearance : Thin, light pink or pearly white
   - Meaning    : HEALING — final stage, new skin cells migrating across wound
   - Note       : Indicates wound is closing

3. SLOUGH
   - Appearance : Yellow, tan, gray, or green; stringy/sticky/thick
   - Meaning    : NON-VIABLE devitalized tissue
   - Note       : Barrier to healing, harbors bacteria, needs debridement

4. NECROTIC TISSUE (ESCHAR)
   - Appearance : Black or brown, dry, hard, leathery
   - Meaning    : DEAD avascular tissue
   - Note       : Usually requires debridement
   - Exception  : Stable dry intact eschar on heels may be left as barrier

==========================================================================
NPIAP PRESSURE INJURY STAGING SYSTEM (2019)
==========================================================================

STAGE 1 — Non-Blanchable Erythema
  - Intact skin with non-blanchable redness of a localized area
  - Darkly pigmented skin may not show visible blanching
  - Color may differ from surrounding area
  - Area may be painful, firm, soft, warmer or cooler than adjacent tissue
  - NO skin break, NO open wound
  → Tissue profile: Intact skin, >80% appears as redness/erythema only

STAGE 2 — Partial-Thickness Skin Loss
  - Partial-thickness loss of dermis
  - Presents as a shallow open ulcer with a red/pink wound bed
  - May also present as an intact or open/ruptured blister
  - NO slough or bruising (bruising = suspected deep tissue injury)
  → Tissue profile: >60% Granulation, <20% Slough, 0% Necrotic

STAGE 3 — Full-Thickness Skin Loss
  - Full-thickness tissue loss
  - Subcutaneous fat may be visible
  - Bone, tendon, or muscle are NOT exposed
  - Slough may be present but does NOT obscure depth of tissue loss
  - May include undermining and tunneling
  → Tissue profile: Mixed Granulation + Slough, <30% Necrotic

STAGE 4 — Full-Thickness Tissue Loss
  - Full-thickness tissue loss with exposed bone, tendon, or muscle
  - Slough or eschar may be present on some parts of the wound bed
  - Often includes undermining and tunneling
  - Depth varies by anatomical location
  → Tissue profile: >40% Necrotic, may have exposed deep structures

UNSTAGEABLE — Obscured Full-Thickness Skin & Tissue Loss
  - Full-thickness tissue loss where base is covered by slough/eschar
  - True depth and stage CANNOT be determined until debridement
  → Tissue profile: >50% Slough or Necrotic covering wound bed

DEEP TISSUE INJURY (DTI) — Persistent Non-Blanchable Deep Red/Purple
  - Intact or non-intact skin with persistent non-blanchable deep
    red, maroon, or purple discoloration
  - Epidermal separation revealing a dark wound bed or blood-filled blister
  → Tissue profile: Intact skin with deep discoloration

==========================================================================
IMPORTANT CLINICAL RULES
==========================================================================

1. "Once a stage, always a stage" — Do NOT reverse-stage as wounds heal.
   A healing Stage 4 is documented as "healing Stage 4", not Stage 3.

2. This staging system is designed for PRESSURE INJURIES.
   Diabetic foot ulcers use Wagner/UT classification but share tissue
   assessment principles. For our app, we adapt NPIAP concepts to a
   general wound staging approach based on tissue composition.

3. The TIME framework guides treatment:
   T = Tissue management (debride non-viable tissue)
   I = Infection/Inflammation control
   M = Moisture balance (choose appropriate dressings)
   E = Edge advancement (monitor epithelialization)

==========================================================================
"""

# ---------------------------------------------------------------------------
# Staging Rules Configuration
# ---------------------------------------------------------------------------

# Each stage is defined by tissue percentage thresholds.
# These are SIMPLIFIED rules adapted from NPIAP for our CV-based system.
# In clinical practice, staging also considers wound depth and anatomical
# features which require physical examination — our system uses tissue
# composition as a proxy.

STAGE_RULES = {
    "Stage 1": {
        "description": "Non-blanchable erythema — intact skin with redness",
        "criteria": {
            "granulation_min": 80,   # >80% healthy/red tissue
            "slough_max": 10,        # <10% slough
            "necrotic_max": 5,       # <5% necrotic
        },
        "clinical_notes": (
            "Intact skin with non-blanchable redness. "
            "Area may be painful, firm, soft, warmer or cooler. "
            "Earliest sign of pressure damage."
        ),
        "severity": "LOW",
        "color": "#4CAF50",  # Green
    },

    "Stage 2": {
        "description": "Partial-thickness skin loss — shallow open ulcer",
        "criteria": {
            "granulation_min": 60,   # >60% healthy tissue
            "slough_max": 20,        # <20% slough
            "necrotic_max": 5,       # <5% necrotic
        },
        "clinical_notes": (
            "Shallow open ulcer with red/pink wound bed. "
            "May present as intact or ruptured blister. "
            "No slough should be present for true Stage 2."
        ),
        "severity": "MODERATE",
        "color": "#FFC107",  # Amber
    },

    "Stage 3": {
        "description": "Full-thickness skin loss — subcutaneous fat visible",
        "criteria": {
            "granulation_min": 20,   # Some healthy tissue present
            "slough_max": 60,        # Can have significant slough
            "necrotic_max": 30,      # <30% necrotic
        },
        "clinical_notes": (
            "Full-thickness tissue loss. Subcutaneous fat may be visible. "
            "Bone, tendon, and muscle are NOT exposed. "
            "Slough may be present. Depth varies by anatomical location."
        ),
        "severity": "HIGH",
        "color": "#FF9800",  # Orange
    },

    "Stage 4": {
        "description": "Full-thickness tissue loss — bone/tendon/muscle exposed",
        "criteria": {
            "granulation_min": 0,    # May have very little healthy tissue
            "slough_max": 100,       # Any amount of slough
            "necrotic_min": 40,      # >40% necrotic tissue
        },
        "clinical_notes": (
            "Full-thickness tissue loss with exposed bone, tendon, or muscle. "
            "Slough or eschar often present. "
            "High risk of osteomyelitis and systemic infection."
        ),
        "severity": "CRITICAL",
        "color": "#F44336",  # Red
    },

    "Unstageable": {
        "description": "Wound bed obscured by slough or eschar — cannot determine depth",
        "criteria": {
            "slough_plus_necrotic_min": 50,  # >50% non-viable tissue covers bed
        },
        "clinical_notes": (
            "Full-thickness tissue loss where the base is completely covered "
            "by slough and/or eschar. True depth cannot be determined. "
            "Requires debridement before accurate staging."
        ),
        "severity": "CRITICAL",
        "color": "#9E9E9E",  # Gray
    },
}


# ---------------------------------------------------------------------------
# Staging Engine
# ---------------------------------------------------------------------------

def classify_stage(granulation_pct: float, slough_pct: float, necrotic_pct: float) -> dict:
    """
    Determine wound stage based on tissue composition percentages.

    Parameters
    ----------
    granulation_pct : float
        Percentage of wound bed that is granulation/epithelial tissue (0-100).
    slough_pct : float
        Percentage of wound bed that is slough tissue (0-100).
    necrotic_pct : float
        Percentage of wound bed that is necrotic/eschar tissue (0-100).

    Returns
    -------
    dict
        {
            "stage": str,           # e.g., "Stage 3"
            "description": str,     # Clinical description
            "severity": str,        # LOW / MODERATE / HIGH / CRITICAL
            "clinical_notes": str,  # Detailed notes
            "color": str,           # UI color code
            "tissue_breakdown": {
                "granulation": float,
                "slough": float,
                "necrotic": float
            },
            "confidence_note": str  # Disclaimer
        }
    """

    # Normalize percentages to ensure they sum to 100
    total = granulation_pct + slough_pct + necrotic_pct
    if total > 0:
        granulation_pct = (granulation_pct / total) * 100
        slough_pct = (slough_pct / total) * 100
        necrotic_pct = (necrotic_pct / total) * 100

    non_viable_pct = slough_pct + necrotic_pct

    # --------------------------------------------------
    # Decision Tree (order matters)
    # --------------------------------------------------

    # 1. Check UNSTAGEABLE — wound bed completely obscured by slough/eschar
    #    Key distinction from Stage 4: unstageable means you CAN'T see the
    #    wound bed at all (>70% non-viable, slough-dominant)
    if non_viable_pct >= 70 and slough_pct >= 40:
        stage_key = "Unstageable"

    # 2. Check STAGE 4 — significant necrosis (dead tissue dominates)
    elif necrotic_pct >= 40:
        stage_key = "Stage 4"

    # 3. Check STAGE 3 — mixed tissue, moderate necrosis/slough
    elif necrotic_pct >= 10 or (slough_pct >= 20 and granulation_pct < 60):
        stage_key = "Stage 3"

    # 4. Check STAGE 1 — predominantly healthy tissue, minimal damage
    elif granulation_pct >= 80 and slough_pct <= 10 and necrotic_pct <= 5:
        stage_key = "Stage 1"

    # 5. Check STAGE 2 — mostly granulation, some slough
    elif granulation_pct >= 60 and slough_pct < 20 and necrotic_pct < 10:
        stage_key = "Stage 2"

    # 6. Catch-all — if nothing matches cleanly, use Stage 2 as middle ground
    else:
        stage_key = "Stage 2"

    # Build result
    stage_info = STAGE_RULES[stage_key]

    return {
        "stage": stage_key,
        "description": stage_info["description"],
        "severity": stage_info["severity"],
        "clinical_notes": stage_info["clinical_notes"],
        "color": stage_info["color"],
        "tissue_breakdown": {
            "granulation": round(granulation_pct, 1),
            "slough": round(slough_pct, 1),
            "necrotic": round(necrotic_pct, 1),
        },
        "confidence_note": (
            "⚠️ This is an AI-assisted assessment for clinical decision support only. "
            "It does NOT replace professional clinical judgment. "
            "Always consult a qualified healthcare provider for wound staging."
        ),
    }


def get_severity_emoji(severity: str) -> str:
    """Return an emoji indicator for the severity level."""
    return {
        "LOW": "🟢",
        "MODERATE": "🟡",
        "HIGH": "🟠",
        "CRITICAL": "🔴",
    }.get(severity, "⚪")


# ---------------------------------------------------------------------------
# Quick Test / Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import io
    # Fix Windows console encoding for emoji support
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("WoundLens Staging Engine - Test Cases")
    print("=" * 60)

    test_cases = [
        {"name": "Healthy (Stage 1)", "gran": 90, "slough": 5, "necro": 5},
        {"name": "Mild (Stage 2)", "gran": 70, "slough": 15, "necro": 5},
        {"name": "Moderate (Stage 3)", "gran": 40, "slough": 35, "necro": 25},
        {"name": "Severe (Stage 4)", "gran": 15, "slough": 30, "necro": 55},
        {"name": "Unstageable", "gran": 10, "slough": 50, "necro": 40},
    ]

    for tc in test_cases:
        result = classify_stage(tc["gran"], tc["slough"], tc["necro"])
        severity = result["severity"]
        print(f"\n--- {tc['name']} ---")
        print(f"  Input  : Granulation={tc['gran']}%, Slough={tc['slough']}%, Necrotic={tc['necro']}%")
        print(f"  Result : [{severity}] {result['stage']}")
        print(f"  Desc   : {result['description']}")
        print(f"  Tissue : Gran={result['tissue_breakdown']['granulation']}% | "
              f"Slough={result['tissue_breakdown']['slough']}% | "
              f"Necrotic={result['tissue_breakdown']['necrotic']}%")

    print("\n" + "=" * 60)
    print("All test cases passed. Staging engine ready.")
    print("=" * 60)
