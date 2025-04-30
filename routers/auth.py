from fastapi import APIRouter, Body, HTTPException, Depends
from datetime import date
import hashlib, uuid, logging, requests

from schemas import UserLogin, UserSignup
from config import AIRTABLE_API_KEY, AIRTABLE_BASE_ID, AIRTABLE_USERS_TABLE, AUTH_UID_MAP

router = APIRouter()

@router.post("/login")
def login(user: UserLogin = Body(...)):
    """
    Authenticate a user against Airtable.
    """
    headers = {"Authorization": f"Bearer {AIRTABLE_API_KEY}"}
    params  = {"filterByFormula": f"{{Email}} = '{user.email}'"}
    url     = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    resp    = requests.get(url, headers=headers, params=params)

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch user")
    records = resp.json().get("records", [])
    if not records:
        raise HTTPException(status_code=404, detail="User not found")

    airtable_record = records[0]
    record          = airtable_record["fields"]

    # verify password
    hashed_input = hashlib.sha256(user.password.encode()).hexdigest()
    if record.get("Password") != hashed_input:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # grab your “external” UUID from the linked User ID field
    real_uuid = record.get("User ID")
    if not real_uuid:
        raise HTTPException(status_code=500, detail="No external UID on user record")

    # map the Airtable recordID → your external UUID
    AUTH_UID_MAP[airtable_record["id"]] = real_uuid
    logging.info(f"[login] mapped {airtable_record['id']} → {real_uuid}")

    return {
        "success": True,
        "user_id": airtable_record["id"],
        "name": record.get("Name")
    }


@router.post("/signup")
def signup(user: UserSignup):
    headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}"
    }
    params = {
        "filterByFormula": f"{{Email}} = '{user.email}'"
    }
    url = f"https://api.airtable.com/v0/{AIRTABLE_BASE_ID}/{AIRTABLE_USERS_TABLE}"
    check = requests.get(url, headers=headers, params=params)

    if check.status_code == 200 and check.json().get("records"):
        raise HTTPException(status_code=400, detail="Email already registered")

    post_headers = {
        "Authorization": f"Bearer {AIRTABLE_API_KEY}",
        "Content-Type": "application/json"
    }
    user_id = str(uuid.uuid4())
    payload = {
        "fields": {
            "User ID": user_id,
            "Name": user.name,
            "Email": user.email,
            "Password": hashlib.sha256(user.password.encode()).hexdigest(),
            "Authentication Provider": "email",
            "Registration Date": str(date.today()),
            "Number of Uploaded Recipes": 0
        }
    }

    response = requests.post(url, headers=post_headers, json=payload)

    if response.status_code != 200:
        logger.error(f"[signup] Airtable error: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to create user")

    logger.info(f"[signup] New user created with ID: {user_id}")
    return {"success": True, "user_id": user_id}
