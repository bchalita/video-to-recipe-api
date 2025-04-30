# app/utils/rappi_helpers.py
import re, json, logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)

# --- Prompts ---
TRANSLATION_PROMPT = """
You are a translation assistant for a grocery shopping app.
Your job is to take each English ingredient name and produce:
  - A Portuguese translation suitable for Zona Sul.
  - A "search_base" — the generic noun in Portuguese to use for broader searches.
  - A list of any qualifiers (adjectives or preparations) to refine searches.

Instructions:
- Preserve important adjectives (e.g., fresco, integral, vegano).
- Remove true parentheticals (e.g., "(for greasing the dish)").
- Extract preparations like "minced", "diced" into qualifiers.
- If there are no adjectives or prep terms, qualifiers list can be empty.

Output strict JSON array, where each item is:
{
  "original": "<exact English input>",
  "translated": "<full Portuguese phrase>",
  "search_base": "<noun in Portuguese>",
  "qualifiers": ["<qual1>", "<qual2>", ...]
}
Do NOT wrap your output in triple-backtick code fences or annotate it with json.
Return only the raw JSON array.
"""

SEARCH_TERM_PROMPT = """
You are a search-term generator for Zona Sul grocery items.
Input is a JSON object with:
  • "search_base": generic noun, e.g. "alho"
  • "qualifiers": list of qualifiers, e.g. ["fresco", "picado"]

Rules:
1. First term must be the exact full phrase: search_base + qualifiers in natural order.
2. Second term must be only the search_base.
3. Then up to three more combos: search_base + single qualifiers,
   ordered by likely availability.
4. No hard-coded overrides.

Return a JSON array of up to 5 strings.
"""

EVALUATION_PROMPT = """
You are a product evaluator for a grocery shopping app.
You receive:
- candidates: list of objects {id, title, department}
- search_base: the noun expected, e.g. "creme de leite"
- qualifiers: list of descriptors, e.g. ["fresco"]

Instructions:
1. Accept only products whose title contains search_base.
2. If qualifiers exist, prefer those matching at least one qualifier.
3. Department must align (e.g. dairy vs personal care).
4. Reject sponsored or irrelevant items.
5. If multiple, pick best overlap with search_base + qualifiers.
6. If none, chosen_id = null.

Output strict JSON: { "chosen_id": <index|null> }
"""

# --- Helpers ---
def clean_gpt_json_response(text: str) -> str:
    # strip code fences and surrounding text
    text = text.strip()
    # remove triple backticks
    if text.startswith("```"):
        text = text.strip('`')
    return text


def parse_required_quantity(qty_str: str) -> (Optional[float], str):
    if not qty_str:
        return None, ""
    pattern = r"(\d+)(\.?\d*)\s*(g|kg|ml|l|un|unid|unidade|tbsp|tsp|cup|clove)?"
    match = re.match(pattern, qty_str.lower())
    if not match:
        return None, ""
    val = float(match.group(1) + match.group(2))
    unit = match.group(3) or "un"
    factor = {"g":1, "kg":1000, "ml":1, "l":1000, "un":1,
              "unid":1, "unidade":1, "tbsp":1, "tsp":1, "cup":1, "clove":1}[unit]
    return val * factor, unit


def estimate_mass(name: str, unit: str, value: float) -> float:
    table = {
        "un": {"onion":200, "garlic":5, "egg":50},
        "tbsp": {"butter":14, "olive oil":13},
        # add more as needed
    }
    return value * table.get(unit, {}).get(name.lower(), 1)


def format_unit_display(qty: float, unit_type: str) -> str:
    unit_map = {"kg":"kg","g":"g","l":"L","ml":"ml","un":"un"}
    if unit_type == "" and qty >= 50:
        return f"{int(qty)}g"
    return f"{qty}{unit_map.get(unit_type,unit_type)}"
