import json
from langchain_google_vertexai import VertexAI, VertexAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain.docstore.document import Document
from google.oauth2 import service_account
from google.cloud import aiplatform
from datetime import datetime, timedelta
from geopy.distance import geodesic  # More accurate distance calculation

# Load blood bank data
with open("bloodbank.json", "r") as f:
    blood_bank_data = json.load(f)

# Initialize Vertex AI
credentials = service_account.Credentials.from_service_account_file("vertical-setup-450217-n2-8904fd8695bd.json")
aiplatform.init(project="strategy-agent", location="us-central1", credentials=credentials)
print("Vertex AI initialized successfully!")

# Initialize Gemini LLM
llm = VertexAI(model_name="gemini-1.5-flash", credentials=credentials, max_output_tokens=3000, temperature=0.7)

# Initialize Vertex AI Embeddings
embeddings = VertexAIEmbeddings(model_name="textembedding-gecko@003", credentials=credentials)

# Create vector store
documents = [Document(page_content=f"Pincode {zone['pincode']}: {json.dumps(zone)}", metadata={"pincode": zone["pincode"]}) for zone in blood_bank_data]
vector_store = Chroma.from_documents(documents=documents, embedding=embeddings, persist_directory="./chroma_db")

# Function to find blood banks by blood group
def find_blood_banks_by_group(blood_group, blood_bank_data):
    matching_banks = [
        bank for bank in blood_bank_data
        if blood_group in bank["available_blood_groups"] and bank["available_blood_groups"][blood_group] > 0
    ]
    
    if not matching_banks:
        return f"No blood banks found with {blood_group} availability."
    
    response = f"Blood banks with {blood_group} availability:\n"
    for bank in matching_banks:
        response += (
            f"\n{bank['name']} - {bank['district']}, {bank['state']}\n"
            f"   {blood_group}: {bank['available_blood_groups'][blood_group]} units\n"
            f"   Location: ({bank['latitude']}, {bank['longitude']})\n"
        )
    return response

# Predict blood shortages
def predict_blood_shortages(blood_bank_data, threshold=5):
    shortage_alerts = [
        f"⚠ Alert: {bank['name']} in {bank['district']}, {bank['state']} "
        f"has a low stock of {blood_group} ({units} units left)."
        for bank in blood_bank_data
        for blood_group, units in bank["available_blood_groups"].items()
        if units < threshold
    ]
    
    return "\n".join(shortage_alerts) if shortage_alerts else "✅ All blood banks have sufficient stock."

# Schedule a blood donation appointment
def schedule_donation_appointment(donor_name, donor_location, blood_group, blood_bank_data):
    # Find the nearest blood bank
    nearest_bank = min(
        blood_bank_data,
        key=lambda bank: geodesic(donor_location, (bank["latitude"], bank["longitude"])).km,
        default=None
    )
    
    if not nearest_bank:
        return "❌ No nearby blood banks found for scheduling an appointment."

    # Generate an appointment time
    appointment_time = datetime.now() + timedelta(days=1, hours=2)

    return (
        f"✅ Appointment Confirmed!\n"
        f"👤 Donor: {donor_name}\n"
        f"🩸 Blood Group: {blood_group}\n"
        f"🏥 Location: {nearest_bank['name']} ({nearest_bank['district']}, {nearest_bank['state']})\n"
        f"⏰ Date & Time: {appointment_time.strftime('%Y-%m-%d %H:%M')}"
    )

# Example usage
blood_group_required = "O-"
print(find_blood_banks_by_group(blood_group_required, blood_bank_data))

print(predict_blood_shortages(blood_bank_data))

donor_location = (17.4, 78.5)  # Example location (Hyderabad)
print(schedule_donation_appointment("Vinuthna", donor_location, "O+", blood_bank_data))
