# app/routers/rappi_cart.py

from fastapi import APIRouter, Body, HTTPException
from typing import List, Optional
import json, logging, requests
from bs4 import BeautifulSoup

from app.utils.rappi_helpers import (
    TRANSLATION_PROMPT,
    clean_gpt_json_response,
    parse_required_quantity,
    estimate_mass,
    format_unit_display,
)
from openai_client import client  # however you import your OpenAI client

router = APIRouter(prefix="/rappi-cart", tags=["rappi-cart"])
logger = logging.getLogger(__name__)

# simple in-memory cache
_cached = {
    "result": None,
    "last_payload": None,
}

@router.post("/", summary="Run a shopping-cart scrape")
def rappi_cart_search(
    ingredients: List[str] = Body(..., embed=True),
    quantities: Optional[List[str]] = Body(None),
    user_id: Optional[str] = Body(None)
):
    # … code that runs GPT translation into `translations` list …

    # build our in-memory cache
    _cached["last_payload"] = dict(ingredients=ingredients, quantities=quantities, user_id=user_id)

    # ── REQUIRED CONTEXT ────────────────────────────────────────────────────────────
    store_urls  = {"Zona Sul": "https://www.zonasul.com.br"}
    store_carts = {store: [] for store in store_urls}
    seen        = set()
    headers     = {"User-Agent": "Mozilla/5.0"}
    # ────────────────────────────────────────────────────────────────────────────────

    # 1️⃣ Loop each translated ingredient
    for idx, trans in enumerate(translations):
        orig        = trans["original"]
        full_pt     = trans["translated"]
        search_base = trans["search_base"]
        qualifiers  = trans.get("qualifiers", [])

        # skip water
        if orig.lower() in ("water", "água"):
            continue

        # build the terms we’ll try
        search_terms = [full_pt, search_base]

        # parse needed mass & estimate grams
        qty_raw                  = quantities[idx] if quantities and idx < len(quantities) else ""
        quantity_needed_val, quantity_needed_unit = parse_required_quantity(qty_raw)
        estimated_needed_val     = estimate_mass(orig, quantity_needed_unit, quantity_needed_val) \
                                    if quantity_needed_val else None

        # ── HERE GOES YOUR SCRAPING & FILTERING BLOCK ───────────────────────────────
        for store, url in store_urls.items():
            found = False
            added = False

            for term in search_terms:
                if found:
                    break
                product_candidates = []
        
                if "zonasul.com.br" in url:
                    # 1️⃣ Build & log the search URL
                    search_url = f"https://www.zonasul.com.br/{term.replace(' ','%20')}?_q={term.replace(' ','%20')}&map=ft"
                    logger.info(f"[rappi-cart][{orig} @ {store}] ➤ Full URL: {search_url}")
                
                    r = requests.get(search_url, headers=headers, timeout=10)
                    soup = BeautifulSoup(r.text, "html.parser")
                
                    # 2️⃣ Grab the VTEX cards
                    cards = soup.select("article.vtex-product-summary-2-x-element")
                    logger.info(f"[rappi-cart][{orig} @ {store}] 🧱 Found {len(cards)} product cards")
                
                    if not cards:
                        logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ No cards at all – page may be JS-rendered")
                        # let fallback or next term handle it
                    for idx, card in enumerate(cards[:5]):
                        # 3️⃣ Extract name
                        name_el = (
                            card.select_one("span.vtex-product-summary-2-x-brandName") 
                            or card.select_one("h2.vtex-product-summary-2-x-productNameContainer span")
                        )
                        logger.debug(f"[{orig} @ {store}] card #{idx} — raw HTML snippet:\n{card.prettify()}")

                        if not name_el:
                            logger.debug(f"[{orig} @ {store}] card #{idx} → no name element, skipping")
                            continue
                        name = name_el.get_text(strip=True)
                        logger.debug(f"[{orig} @ {store}] card #{idx} → name: {name!r}")
                
                        # 4️⃣ Extract price parts (Zona Sul custom first, then VTEX fallback)
                        int_el = (
                            card.select_one("span.zonasul-zonasul-store-1-x-currencyInteger")
                            or card.select_one("span.vtex-product-summary-2-x-currencyInteger")
                        )
                        frac_el = (
                            card.select_one("span.zonasul-zonasul-store-1-x-currencyFraction")
                            or card.select_one("span.vtex-product-summary-2-x-currencyFraction")
                        )
                        if not int_el:
                            logger.warning(f"[{orig} @ {store}] card #{idx} → missing integer price part, skipping")
                            continue
                        int_txt = int_el.get_text(strip=True)
                        frac_txt = frac_el.get_text(strip=True) if frac_el else "00"
                        try:
                            price = float(f"{int_txt}.{frac_txt}")
                        except Exception as e:
                            logger.error(f"[{orig} @ {store}] card #{idx} → bad price parse '{int_txt}.{frac_txt}': {e}")
                            continue
                        logger.debug(f"[{orig} @ {store}] card #{idx} → price: R$ {price:.2f}")
                
                        # 5️⃣ Extract image
                        img_el = card.select_one("img.vtex-product-summary-2-x-imageNormal")
                        img = img_el["src"] if (img_el and img_el.has_attr("src")) else None
                
                        # 6️⃣ Add to candidates
                        product_candidates.append({
                            "name": name,
                            "price": f"R$ {price:.2f}",
                            "description": name.lower(),
                            "image_url": img,
                            "raw": card
                        })
                        logger.info(f"[{orig} @ {store}] candidate #{idx}: {name} — R$ {price:.2f}")
                
                    if not product_candidates:
                        logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ no candidates after scraping")
                        continue



                    
                    # — FILTER BY search_base + qualifiers —
                    filtered = []
                    sb = search_base.lower()
                    quals = [q.lower() for q in qualifiers]
                    for c in product_candidates:
                        n = c["name"].lower()
                        if sb not in n:
                            logger.debug(f"[{orig} @ {store}] dropping '{c['name']}' (no '{sb}' in name)")
                            continue
                        if quals and not any(q in n for q in quals):
                            logger.debug(f"[{orig} @ {store}] dropping '{c['name']}' (none of {quals} present)")
                            continue
                        filtered.append(c)
                    logger.info(f"[{orig} @ {store}] ▶️ {len(filtered)}/{len(product_candidates)} remain after filtering")
                    product_candidates = filtered

        
                else:
                    # — Rappi / Pão de Açúcar JSON logic goes here —
                    resp = requests.get(url, params={"term": term}, headers=headers, timeout=10)
                    json_data = extract_next_data_json(BeautifulSoup(resp.text, "html.parser"))
                    if not json_data:
                        continue
                    for p in iterate_fallback_products(json_data["props"]["pageProps"]["fallback"]):
                        name = p["name"].strip()
                        price = float(str(p["price"]).replace(",", "."))
                        image_raw = p.get("image", "")
                        image_url = (
                            image_raw
                            if image_raw.startswith("http")
                            else f"https://images.rappi.com.br/products/{image_raw}?e=webp&q=80&d=130x130"
                        )
                        product_candidates.append({
                            "name": name,
                            "price": f"R$ {price:.2f}",
                            "description": name.lower(),
                            "image_url": image_url
                        })
        
                if not product_candidates:
                    logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ no candidates for term '{term}'")
                    continue


                phase1 = [
                    c for c in product_candidates
                    if search_base.lower() in c["name"].lower()
                ]
                #   * only on the exact full-phrase pass do we enforce qualifiers
                if term == full_pt and qualifiers:
                    phase2 = [
                        c for c in phase1
                        if any(q.lower() in c["name"].lower() for q in qualifiers)
                    ]
                    product_candidates = phase2 or phase1
                else:
                    product_candidates = phase1
        
                # 2️⃣ Use the same evaluator for either source:
                eval_messages = [
                    {"role": "system", "content": EVALUATION_PROMPT},
                    {"role": "user",   "content": json.dumps({
                        "candidates": [
                            {"id": i, "title": c["name"], "department": c.get("department", "")}
                            for i, c in enumerate(product_candidates)
                        ],
                        "search_base": search_base,
                        "qualifiers": qualifiers
                    }, ensure_ascii=False)}
                ]
                eval_resp = client.chat.completions.create(
                    model="gpt-4o", messages=eval_messages, temperature=0, max_tokens=300
                )
                raw_eval = eval_resp.choices[0].message.content.strip()
                try:
                    evj = json.loads(clean_gpt_json_response(raw_eval))
                    idx = evj.get("chosen_id")
                    chosen_idx = int(idx) if idx is not None else None
                    if chosen_idx is None or not (0 <= chosen_idx < len(product_candidates)):
                        raise ValueError("no valid choice")
                    chosen_product = product_candidates[chosen_idx]
                except Exception:
                    logger.warning(f"[{orig} @ {store}] ❌ Eval failed or null – falling back to top result")
                    chosen_product = product_candidates[0]

                    # ─── extract quantity_per_unit via regex ────────────────────────────────────
                qm = re.search(r"(\d+(?:[.,]\d+)?)(kg|g|unidade|un)", chosen_product["name"].lower())
                if qm:
                    val = float(qm.group(1).replace(",", "."))
                    unit = qm.group(2)
                    factor = {"kg": 1000, "g": 1, "unidade": 1, "un": 1}.get(unit, 1)
                    quantity_per_unit = int(val * factor)
                else:
                    quantity_per_unit = 500
            
                # ─── compute how many units to buy & display string ─────────────────────────
                if estimated_needed_val is not None:
                    units_needed = max(1, int(estimated_needed_val // quantity_per_unit + 0.999))
                    needed_display = (
                        format_unit_display(quantity_needed_val, quantity_needed_unit)
                        + f" (~{int(estimated_needed_val)}g)"
                    )
                else:
                    units_needed = 1
                    needed_display = qty_raw or ""
            
                # ─── totals & excess ───────────────────────────────────────────────────────
                total_cost = units_needed * float(chosen_product["price"]
                                                  .replace("R$", "")
                                                  .replace(",", "."))
                total_quantity = units_needed * quantity_per_unit
                if estimated_needed_val is not None:
                    excess = total_quantity - estimated_needed_val
                else:
                    excess = None

                key = (store, orig, chosen_product['name'])
                if key in seen: break
                seen.add(key)

                store_carts[store].append({
                    "ingredient":orig,
                    "translated":full_pt,
                    "product_name":chosen_product['name'],
                    "price":chosen_product['price'],
                    "image_url":chosen_product['image_url'],
                    "quantity_needed":qty_raw,
                    "quantity_needed_display":f"{units_needed} x {quantity_per_unit}g",
                    "quantity_per_unit":quantity_per_unit,
                    "units_to_buy":units_needed,
                    "total_quantity_added":units_needed*quantity_per_unit,
                    "total_cost":f"R$ {total_cost:.2f}",
                    "excess_quantity":excess
                })

                logger.info(f"[rappi-cart][{orig} @ {store}] ✅ Added: {chosen_product['name']}")
                added = True
                break
              
            if not added:
                logger.warning(f"[rappi-cart][{orig} @ {store}] ❌ No match found")

            if found:
                seen.add((store, orig))
    
    result = {"carts_by_store": store_carts}
    _cached["result"] = result
    return result

@router.get("/view", summary="Get last cart")
def view_rappi_cart():
    if not _cached["result"]:
        raise HTTPException(404, "No cart cached")
    return _cached["result"]

@router.post("/reset", summary="Clear the cart cache")
def reset_rappi_cart():
    _cached["result"] = None
    _cached["last_payload"] = None
    return {"status":"cleared"}

@router.post("/resend", summary="Re-run with last payload")
def resend_rappi_cart():
    if not _cached["last_payload"]:
        raise HTTPException(400, "No previous payload")
    return rappi_cart_search(**_cached["last_payload"])
