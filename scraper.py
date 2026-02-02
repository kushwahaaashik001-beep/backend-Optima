import os
import json
from duckduckgo_search import DDGS # Nayi library logic
from groq import Groq
from supabase import create_client

# Setup
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def get_leads():
    # Thoda broad query taaki results pakka milein
    query = 'site:twitter.com "looking for" "video editor"'
    results = []
    try:
        with DDGS() as ddgs:
            # Naya format: ddgs.text ki jagah sirf ddgs().text
            for r in ddgs.text(query, max_results=5):
                results.append(r)
    except Exception as e:
        print(f"Search Error: {e}")
    return results

def filter_with_groq(content):
    prompt = f"Extract freelance job details from this text: {content}. Return JSON with keys: title, category, sub_category, budget, source_url. If not a job, return null."
    try:
        completion = groq_client.chat.completions.create(
            model="llama3-8b-8192", # Fast model for speed
            messages=[{"role": "user", "content": prompt}],
            response_format={ "type": "json_object" }
        )
        return json.loads(completion.choices[0].message.content)
    except:
        return None

# Execution
raw_leads = get_leads()
print(f"Found {len(raw_leads)} potential links.") # Debugging ke liye

for lead in raw_leads:
    clean_data = filter_with_groq(lead['body'])
    if clean_data and clean_data.get('title'):
        clean_data['source_url'] = lead['href']
        # Upsert taaki duplicate na ho
        supabase.table("leads").upsert(clean_data).execute()

print("Mission Successful: Leads updated!")
