from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from datetime import datetime, timedelta
from geopy.distance import geodesic
from google.oauth2 import service_account
from google.cloud import aiplatform
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain_chroma import Chroma
from langchain.docstore.document import Document
import motor.motor_asyncio
import random
import httpx
from pathlib import Path
# add middlewares
from fastapi.middleware.cors import CORSMiddleware

# Load blood bank data
with open("bloodbank.json", "r") as f:
    blood_bank_data = json.load(f)

# Initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file("vertical-setup-450217-n2-8904fd8695bd.json")
aiplatform.init(project="strategy-agent", location="us-central1", credentials=credentials)

# Initialize Vertex AI components
llm = VertexAI(model_name="gemini-1.5-flash", credentials=credentials, max_output_tokens=3000, temperature=0.7)
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003", credentials=credentials)

# Create vector store
documents = [Document(page_content=f"Pincode {zone['pincode']}: {json.dumps(zone)}", metadata={"pincode": zone["pincode"]}) for zone in blood_bank_data]
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")
vector_store.persist_directory = "./chroma_db"  # Ensure persistence

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class BloodRequest(BaseModel):
    blood_group: str
    
class PredictShortages(BaseModel):
    shortages: str

class DonationAppointment(BaseModel):
    donor_name: str
    donor_location: List[float]  
    blood_group: str

def find_blood_banks_by_group(blood_group: str):
    matching_banks = [
        bank for bank in blood_bank_data
        if blood_group in bank["available_blood_groups"] and bank["available_blood_groups"][blood_group] > 0
    ]
    if not matching_banks:
        # Gather all available blood groups in the data for debugging (optional)
        all_blood_groups = set()
        for bank in blood_bank_data:
            all_blood_groups.update(bank["available_blood_groups"].keys())
        return {
            "message": f"No blood banks found with {blood_group} availability.",
            "available_blood_groups_in_data": sorted(list(all_blood_groups))
        }
    return {"blood_banks": matching_banks}

def predict_blood_shortages(threshold=5):
    shortages = [
        {
            "name": bank["name"],
            "district": bank["district"],
            "state": bank["state"],
            "low_stock": [blood_group for blood_group, units in bank["available_blood_groups"].items() if units < threshold]
        }
        for bank in blood_bank_data
        if any(units < threshold for units in bank["available_blood_groups"].values())
    ]
    return {"shortages": shortages} if shortages else {"message": "All blood banks have sufficient stock."}

def schedule_donation_appointment(donor_name: str, donor_location: List[float], blood_group: str):
    if not blood_bank_data:
        return {"message": "No blood banks available in the system."}
    
    nearest_bank = min(
        blood_bank_data,
        key=lambda bank: geodesic(donor_location, (bank["latitude"], bank["longitude"])).km
    )

    appointment_time = datetime.now() + timedelta(days=1, hours=2)
    return {
        "donor": donor_name,
        "blood_group": blood_group,
        "blood_bank": nearest_bank["name"],
        "location": f"{nearest_bank['district']}, {nearest_bank['state']}",
        "appointment_time": appointment_time.strftime('%Y-%m-%d %H:%M')
    }

@app.get("/find_blood_banks")
def get_blood_banks(blood_group: str):
    """
    Find blood banks with available blood group
    """
    return find_blood_banks_by_group(blood_group)

@app.get("/predict_shortages")
def get_blood_shortages(threshold: int = 5):
    """
    Predict blood shortages based on threshold
    """
    return predict_blood_shortages(threshold)

@app.post("/schedule_appointment")
def book_appointment(appointment: DonationAppointment):
    """
    Schedule a blood donation appointment
    """
    return schedule_donation_appointment(
        appointment.donor_name,
        appointment.donor_location,
        appointment.blood_group
    )

# MongoDB connection
MONGO_DETAILS = "mongodb+srv://Jayanth:HAKUNAmatata123@jayanth.7ackfrz.mongodb.net/?retryWrites=true&w=majority&appName=Jayanth"
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_DETAILS)
database = client["blog_database"]
bloodbank_collection = database.get_collection("bloodbanks")

bloodgroup = database.get_collection("bloodgroups")
bloodcomponents = database.get_collection("bloodcomponents")


import json

# Load the test.bloodbanks.json data (assuming it's in a file)
with open('test.bloodbanks.json', 'r') as file:
    data = json.load(file)

# Function to convert blood group totals into positive/negative split
def split_blood_groups(bloodgroup):
    # 80% positive, 20% negative assumption
    result = {}
    for group in ['A', 'B', 'O', 'AB']:
        total = bloodgroup.get(group, 0)
        positive = round(total * 0.8)  # 80% for positive
        negative = total - positive    # Remaining 20% for negative
        result[f"{group}+"] = positive
        result[f"{group}-"] = negative
    return result

# Convert the data into bloodbank.json format
converted_data = []
for entry in data:
    new_entry = {
        "name": entry["blood_bank_name"],
        "pincode": entry["pincode"],
        "district": entry["district"],
        "state": entry["state"],
        "latitude": entry["latitude"],
        "longitude": entry["longitude"],
        "available_blood_groups": split_blood_groups(entry["bloodgroup"])
    }
    converted_data.append(new_entry)

# Save the converted data to a new file
with open('converted_bloodbank.json', 'w') as outfile:
    json.dump(converted_data, outfile, indent=4)

# get api to get bloodbanks data in json file
@app.get("/get_bloodbanks")
def get_bloodbanks():
    data_path = Path("converted_bloodbank.json")  # Change if your file is named differently
    with data_path.open() as f:
        data = json.load(f)  # data is now a list of documents
    return data

@app.get("/bloodbanks")
def get_bloodbanks():
    """
    Reads data from 'data.json' and returns it as valid JSON.
    Converts MongoDB '$oid' and '$date' fields into normal strings.
    """
    data_path = Path("test.bloodbanks.json")  # Change if your file is named differently
    with data_path.open() as f:
        data = json.load(f)  # data is now a list of documents

    cleaned_data = []
    for doc in data:
        # 1. Convert top-level "_id" if present
        if "_id" in doc and "$oid" in doc["_id"]:
            doc["id"] = doc["_id"]["$oid"]  # store the string ID
            del doc["_id"]

        # 2. Convert nested date fields
        if "date_license_obtained" in doc and "$date" in doc["date_license_obtained"]:
            doc["date_license_obtained"] = doc["date_license_obtained"]["$date"]
        if "date_of_renewal" in doc and "$date" in doc["date_of_renewal"]:
            doc["date_of_renewal"] = doc["date_of_renewal"]["$date"]

        # 3. Convert nested bloodgroup._id if present
        if "bloodgroup" in doc and "_id" in doc["bloodgroup"]:
            if "$oid" in doc["bloodgroup"]["_id"]:
                doc["bloodgroup"]["id"] = doc["bloodgroup"]["_id"]["$oid"]
            del doc["bloodgroup"]["_id"]

        # 4. Convert nested blood_component._id if present
        if "blood_component" in doc and "_id" in doc["blood_component"]:
            if "$oid" in doc["blood_component"]["_id"]:
                doc["blood_component"]["id"] = doc["blood_component"]["_id"]["$oid"]
            del doc["blood_component"]["_id"]

        cleaned_data.append(doc)

    return cleaned_data


@app.get("/bloodgroups")
async def get_bloodgroups():
    """
    Returns all documents from the 'bloodgroups' collection.
    """
    results = []
    async for doc in db.bloodgroups.find():
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results

@app.get("/bloodcomponents")
async def get_bloodcomponents():
    """
    Returns all documents from the 'bloodcomponents' collection.
    """
    results = []
    async for doc in database.bloodcomponents.find():
        doc["_id"] = str(doc["_id"])
        results.append(doc)
    return results
# Pydantic models
class BloodGroup(BaseModel):
    A: int
    B: int
    AB: int
    O: int
    total: int

class BloodComponents(BaseModel):
    wholeBlood: int = 0
    singleDonorPlatelets: int = 0
    singleDonorPlasma: int = 0
    sagmPackedRbc: int = 0
    randomDonorPlatelets: int = 0
    plaletRichPlasma: int = 0
    platetConcentrate: int = 0
    plasma: int = 0
    packedRbc: int = 0
    cryoprecipitate: int = 0
    leukoreducedRbc: int = 0
    freshFrozenPlasma: int = 0
    irediatedRbc: int = 0
    cryopoorPlasma: int = 0
    total: int = 0

class BloodBank(BaseModel):
    sr_no: int
    blood_bank_name: str
    state: str = None
    district: str = None
    city: str = None
    address: str = None
    pincode: str = None
    contact_no: str = None
    mobile: str = None
    helpline: str = None
    fax: str = None
    email: str = None
    website: str = None
    nodal_officer: str = None
    contact_nodal_officer: str = None
    mobile_nodal_officer: str = None
    email_nodal_officer: str = None
    qualification_nodal_officer: str = None
    category: str = None
    blood_component_available: str = None
    apheresis: str = None
    service_time: str = None
    license: str = None
    date_license_obtained: datetime = None
    date_of_renewal: datetime = None
    latitude: float = None
    longitude: float = None
    bloodgroup: BloodGroup
    blood_component: BloodComponents

# Helper functions
def parse_date(date_str: str):
    """Convert a date string like '14.6.1996' to a datetime object."""
    if not date_str:
        return None
    parts = date_str.split('.')
    if len(parts) != 3:
        return None
    try:
        day, month, year = parts
        return datetime(int(year), int(month), int(day))
    except Exception:
        return None

def get_random_int(min_val: int, max_val: int) -> int:
    return random.randint(min_val, max_val)

def generate_random_blood_group() -> BloodGroup:
    A = get_random_int(10, 100)
    B = get_random_int(10, 100)
    AB = get_random_int(10, 100)
    O = get_random_int(10, 100)
    return BloodGroup(A=A, B=B, AB=AB, O=O, total=A + B + AB + O)

def generate_random_blood_components() -> BloodComponents:
    wholeBlood = get_random_int(5, 50)
    singleDonorPlatelets = get_random_int(5, 50)
    singleDonorPlasma = get_random_int(5, 50)
    sagmPackedRbc = get_random_int(5, 50)
    randomDonorPlatelets = get_random_int(5, 50)
    plaletRichPlasma = get_random_int(5, 50)
    platetConcentrate = get_random_int(5, 50)
    plasma = get_random_int(5, 50)
    packedRbc = get_random_int(5, 50)
    cryoprecipitate = get_random_int(5, 50)
    leukoreducedRbc = get_random_int(5, 50)
    freshFrozenPlasma = get_random_int(5, 50)
    irediatedRbc = get_random_int(5, 50)
    cryopoorPlasma = get_random_int(5, 50)
    total = (wholeBlood + singleDonorPlatelets + singleDonorPlasma + sagmPackedRbc +
             randomDonorPlatelets + plaletRichPlasma + platetConcentrate + plasma +
             packedRbc + cryoprecipitate + leukoreducedRbc + freshFrozenPlasma +
             irediatedRbc + cryopoorPlasma)
    return BloodComponents(
        wholeBlood=wholeBlood,
        singleDonorPlatelets=singleDonorPlatelets,
        singleDonorPlasma=singleDonorPlasma,
        sagmPackedRbc=sagmPackedRbc,
        randomDonorPlatelets=randomDonorPlatelets,
        plaletRichPlasma=plaletRichPlasma,
        platetConcentrate=platetConcentrate,
        plasma=plasma,
        packedRbc=packedRbc,
        cryoprecipitate=cryoprecipitate,
        leukoreducedRbc=leukoreducedRbc,
        freshFrozenPlasma=freshFrozenPlasma,
        irediatedRbc=irediatedRbc,
        cryopoorPlasma=cryopoorPlasma,
        total=total
    )

@app.post("/store_blood_banks")
async def store_blood_banks():
    url = (
        "https://api.data.gov.in/resource/fced6df9-a360-4e08-8ca0-f283fc74ce15?"
        "api-key=579b464db66ec23bdd000001603eb0cc38324dd768735197a75609f5&format=json&limit=2823"
    )
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            raise HTTPException(status_code=500, detail="Error fetching data from URL")
    
    records = data.get("records", [])
    transformed_records = []
    for record in records:
        transformed = {
            "sr_no": int(record.get("sr_no")),
            "blood_bank_name": record.get("_blood_bank_name"),
            "state": record.get("_state"),
            "district": record.get("_district"),
            "city": record.get("_city"),
            "address": record.get("_address"),
            "pincode": record.get("pincode"),
            "contact_no": record.get("_contact_no"),
            "mobile": record.get("_mobile"),
            "helpline": record.get("_helpline"),
            "fax": record.get("_fax"),
            "email": record.get("_email"),
            "website": record.get("_website"),
            "nodal_officer": record.get("_nodal_officer_"),
            "contact_nodal_officer": record.get("_contact_nodal_officer"),
            "mobile_nodal_officer": record.get("_mobile_nodal_officer"),
            "email_nodal_officer": record.get("_email_nodal_officer"),
            "qualification_nodal_officer": record.get("_qualification_nodal_officer"),
            "category": record.get("_category"),
            "blood_component_available": record.get("_blood_component_available"),
            "apheresis": record.get("_apheresis"),
            "service_time": record.get("_service_time"),
            "license": record.get("_license__"),
            "date_license_obtained": parse_date(record.get("_date_license_obtained")),
            "date_of_renewal": parse_date(record.get("_date_of_renewal")),
            "latitude": float(record.get("_latitude")) if record.get("_latitude") else None,
            "longitude": float(record.get("_longitude")) if record.get("_longitude") else None,
            "bloodgroup": generate_random_blood_group().dict(),
            "blood_component": generate_random_blood_components().dict()
        }
        transformed_records.append(transformed)
    
    if transformed_records:
        result = await bloodbank_collection.insert_many(transformed_records)
        inserted_count = len(result.inserted_ids)
        return {"message": "Records inserted successfully", "count": inserted_count}
    else:
        raise HTTPException(status_code=404, detail="No records found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app,  port=8000)