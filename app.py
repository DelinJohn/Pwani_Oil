# Imports & Setup
from typing import List
from typing_extensions import TypedDict
from pydantic import Field
from datetime import datetime
from tinydb import TinyDB, Query
from openai import OpenAI
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
import os


# Load environment variables


# API keys and model configs
key = st.secrets.get("GROQ_API_KEY")
model_provider = st.secrets.get("GROQ_model_provider")
model_name = st.secrets.get("GROQ_model")
perplexity_key = st.secrets.get("perp_api_key")

# Initialize LLM
@st.cache_resource(show_spinner=False)
def load_llm():
    if not all([key, model_provider, model_name]):
        raise ValueError("Missing LLM configuration in secrets.")
    os.environ["API_KEY"] = key
    return init_chat_model(model_name, model_provider=model_provider)

llm = load_llm()

# Data Fetching Functions
@st.cache_data(show_spinner="Fetching competitor data...")
def get_companies(prompt: str) -> str:
    messages = [
    {
        "role": "system",
        "content": (
            """You are a market analyst specializing in the Kenyan FMCG market. 
            When provided with a product from Pwani Oils and a list of competitors, your role is to analyze and provide:
            - The advertisement strategies
            - Types of advertisements
            - Types of promotions

            **Specifically for the competitors** provided in the prompt, not Pwani Oils itself.
            If Pwani Oils is mentioned, it is only for context and benchmarking.
            Your focus is on understanding what **the competitors** are doing in the Kenyan market."""
        ),
    },
    {   
        "role": "user",
        "content": (
            prompt
        ),
    },
    ]
    client = OpenAI(api_key=perplexity_key, base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(model="sonar", messages=messages)
    return response.choices[0].message.content

# Save structured competitor data
def json_db_creator(data, product, category):
    class AdvertisementTechniques(TypedDict):
        name: str= Field(description="Name of the company")
        Advertisement_strategy:str=Field(description="overall strategy are they using to position their product in the market")
        Type_of_advertisement:str=Field(description="channels or formats are being used? (e.g., TV, radio, social media, print, influencer marketing)")
        Type_of_Promotion:str=Field(description="promotional tactics are they using? (e.g., discounts, free samples, bundled offers, loyalty programs)")

    class ExtractSchema(TypedDict):
        companies: List[AdvertisementTechniques]

    structured_llm = llm.with_structured_output(ExtractSchema)
    result = structured_llm.invoke(data)
    result.update({"date": str(datetime.today().date()), "Product": product, "category": category})
    TinyDB("db.json").table("products").insert(result)

# Wrapper for prompt + saving
def data_collector_json_getter(competitors, category, product):
    prompt = f"""
        You are provided with a list of competitors for the Pwani Oil company in Kenya.

        **Product of focus**: {product}  
        **category**: {category}  
        **Competitors**: {competitors}

        Your task is to analyze the **advertising and promotional strategies** used by these competitors in the Kenyan market.

        Please provide insights under the following sections for each competitor:

        1. **Advertisement Strategy** – What overall strategy are they using to position their product in the market?
        2. **Type of Advertisement** – What channels or formats are being used? (e.g., TV, radio, social media, print, influencer marketing)
        3. **Type of Promotion** – What promotional tactics are they using? (e.g., discounts, free samples, bundled offers, loyalty programs)

        Focus only on the competitors listed. Do not provide details for Pwani Oil itself.
"""
    data = get_companies(prompt)
    json_db_creator(data, product, category)
    return data

# Load competitor info
def competitor_data_collector(product, competitors, category):
    db = TinyDB("db.json").table("products")
    q = Query()
    result = db.search((q.date == str(datetime.today().date())) & (q.name == product) & (q.category == category))
    return result if result else data_collector_json_getter(competitors, category, product)

# Product info
@st.cache_data
def product_data_fetcher(brand, category):
    db = TinyDB("new_product.json").table("Product_details")
    results = db.search((Query().Brand == brand) & (Query().Category == category))
    return results, results[0].get("Competition")

# Demographics info
@st.cache_data
def demographics_fetcher(gender, region, urban_or_rural):
    result = TinyDB("demographics.json").table("demographics").all()[-1]
    gender_data = [{g: result[g]} for g in gender] if gender != "No Preference" else ""
    locality_data = {urban_or_rural: result[urban_or_rural]} if urban_or_rural != "No Preference" else ""
    location_data = {region: result[region]}
    return location_data, gender_data, locality_data

# Final LLM invocation
@st.cache_data(show_spinner="Generating campaign...")
def final_llm(product, campaign_type, tone, content_type, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku):
    product_details, competitors = product_data_fetcher(product, category)
    competitor_list = competitor_data_collector(product, competitors, category)
    location_data, gender_data, locality_data = demographics_fetcher(gender, region, urban_or_rural)

    messages = [
    SystemMessage(f"""
ROLE: You are a world-class, highly creative advertising strategist known for **groundbreaking** and **locally resonant** campaigns.

OBJECTIVE: Design a high-impact **{campaign_type}** campaign for the brand **Pwani**, and the product **{product}, {category}, {sku}**. If the SKU is specifically referenced, tailor your message accordingly. The campaign is to be delivered to**{channel}** customers on **{platform}**.

TONE & STYLE: {tone}  
CAMPAIGN CATEGORY: {campaign_category}  
OUTPUT FORMAT: {content_type}

CONTEXT:
You will be given:
- `product_details`: key attributes of the product  
- `competitor_list`: current advertising strategies used by competitors  
- `target_audience`: includes:
    - Region: {region}
    - Gender: {gender}
    - Age Range: {age_range}
    - Income Level: {income}
    - Demographics from location, gender, and locality breakdowns

MISSION:
1. Craft a **brilliant campaign message between 20 and 30 words**.
2. Add a **rationale (max 20 words)** explaining why your idea **leapfrogs competitors** through originality and deep audience resonance.
3. Your campaign must reflect:
    - **Fresh, unexpected creativity** (no clichés, no buzzwords)
    - **Deep cultural insight** (use language, tone, or hashtags that truly reflect the people of {region})
    - **Emotional or social triggers** relevant to this group (not generic product selling points)
    - **Better storytelling or symbolism** than competitors — even if it's a simple hashtag or callout.
    - **Mention why this SKU is better**
    - **Adhere to these instructions {instructions}

MUST-HAVES:
- Do **not** exceed 30 words for the campaign message.
- Do **not** exceed 20 words for the competitive rationale.
- Do **not** copy competitor formats. This must feel **unique**.
- Do **not** generalize. Go for bold, memorable, local, or clever.
- Make sure the type of data you are creating meets the standard and treding optimaization of {platform}.

Take bold creative risks. This campaign should feel like it was **born in the streets of {region}**, not in a boardroom.

"""),
    HumanMessage(f"""
product_details = {product_details}  
competitor_list = {competitor_list}  
target_audience = {{
    'region': '{region}',
    'gender': '{gender}',
    'age_range': '{age_range}',
    'income_level': '{income}',
    'demographics': {{
        'region_data': {location_data},
        'gender_data': {gender_data},
        'locality_data': {locality_data}
    }}
}}
""")
]

    result = llm.invoke(messages)
    return result.content.split("</think>")[-1]

# UI Starts Here
products_with_category={
    'FRESHFRI':             {'COOKING OIL': ['10000ML',' 1000ML',' 20000ML',' 2000ML',' 250ML',' 3000ML',' 5000ML',' 500ML']},
    'SALIT':                {'COOKING OIL': ['10000ML',' 1000ML',' 20000ML',' 2000ML',' 3000ML',' 5000ML',' 500ML']},
    'DETREX':               {'TOILET SOAP': ['100G',' 300ML',' 30ML',' 500ML',' 50ML',' 80G']},
    'USHINDI':              {'LAUNDRY DETERGENT': ['1000G', ' 20G', ' 500G'],'LAUNDRY BAR': ['1000G', ' 175G', ' 350G', ' 60G', ' 800G']},
    'SAWA MILKING JELLY':   {'SKIN CARE - PETROLLEUM JELLIES': ['50ML',' 90ML',' 200ML']},
    'SAWA PETROLEUM JELLY': {'SKIN CARE - PETROLLEUM JELLIES': ['100ML',' 250ML',' 25ML',' 50ML']},
    'SAWA HANDWASH':        {'TOILET SOAP': ['250ML', ' 500ML']},
    'SAWA FAMILY SOAPS':    {'TOILET SOAP': ['70G', ' 125G', ' 225G', ' 250G']},
    'SAWA BODY WASH':       {'TOILET SOAP': ['500ML']},
    'BELLEZA ':             {'LOTIONS': ['100ML']},
    'FRESCO ':              {'LOTIONS': ['100ML']},
    'AFRISENSE':            {'TOILET SOAP': ['125G', ' 225G']},
    'NDUME':                {'LAUNDRY BAR': ['1000G', ' 200G', ' 600G', ' 700G', ' 800G']},
    'POPCO':                {'COOKING OIL': ['1000ML',' 20000ML',' 2000ML',' 3000ML',' 5000ML',' 500ML'],'LAUNDRY BAR': ['1000G', ' 600G']},
    'WHITE WASH (PWANI OIL PRODUCTS)': {'LAUNDRY BAR': ['1000G',' 175G',' 200G',' 800G',' 90G']},
    'DIVA ':                {'TOILET SOAP': ['100G', ' 200G', ' 500ML']},
    '4U ':                  {'TOILET SOAP': ['250G']},
    'FRESH ZAIT':           {'COOKING OIL': ['10000ML', ' 20000ML']},
    'FRYKING':              {'COOKING FATS': ['1000G', ' 500G']},
    'MPISHI POA':           {'COOKING FATS': ['10000G',' 1000G',' 100G',' 17000G',' 2000G',' 250G',' 40G',' 5000G',' 500G',' 50G']},
    'ONJA':                 {'COOKING FATS': ['500G'], 'SPREADS': ['10000G', ' 1000G', ' 250G']},
    'TWIGA':                {'COOKING OIL': ['10000ML', ' 20000ML']},
    'TIKU ':                {'COOKING OIL': ['10000ML', ' 20000ML']}
    }  

channels={
            'B2B': ['RTM', 'Whatsapp'],
            'B2C':[ 'Offline Posters', 'Facebook', 'Instagram', 'Tiktok', 'Twitter', 'Whatsapp group']
        }

# Sidebar Inputs
with st.sidebar:
    with st.container():
        col1, col2 = st.columns(2)
        with col1.expander("Campaign Definition", expanded=True):
        
            product = st.selectbox("Brand", list(products_with_category))
            category = st.selectbox("Category", list(products_with_category[product]))
            sku = st.selectbox("SKU", ["Select"] + products_with_category[product][category])
            sku = None if sku == "Select" else sku
            channel = st.selectbox("Channel", list(channels))
            platform = st.selectbox("Platform", channels[channel])
            campaign_category = st.selectbox("Campaign Category",                                   
                ("Awareness Campaign", 
                "Engagement Campaign", 
                "Conversion Campaign",
                "Retention Campaign", 
                "Product launch", 
                "Seasonal Promotion")
                )
            campaign_type = st.selectbox("Campaign Type", 
                ("Brand Awareness", 
                "Educational", 
                "Influencer/Partnership", 
                "Social Media Awareness", 
                "PR and Media Coverage")
                )
            tone = st.selectbox("Tone & Style", 
                ("Professional", 
                "Casual", 
                "Inspirational", 
                "Authoritative", 
                "Friendly", 
                "Energetic", 
                "Persuasive")
                )
            
            content_type = st.selectbox("Content Type", ["Hashtag", "Script"])
        with col2.expander("Target Market", expanded=True):
            
            age_range = st.slider("Age Range", 15, 65, (20, 30))
            gender = st.multiselect("Gender", ["Female", "Male", "Non_binary", "No Preference"])
            income = st.selectbox('Income Level',
                               ('Low (e.g, <$30k)',
                                'Middle (e.g., $30k-$75k)',
                                'High (e.g., $75k-$150k)',
                                'Very High (e.g.,>$150k)',
                                'No Preference'))
            region = st.selectbox("Region",
                                ('Nairobi',
                                'Central', 
                                'RiftValley', 
                                'Coast', 
                                'Eastern', 
                                'Western', 
                                'Northeastern')
                                )
            urban_or_rural = st.selectbox("Urban/Rural", ["Urban", "Rural", "No Preference"])
    
# Main Action
instructions = st.text_input("Enter additional instructions")
if st.button("Generate Content"):
    if all([product, campaign_category, campaign_type, tone, content_type, instructions, age_range, gender, income, region, urban_or_rural, channel, platform]):
        st.write(final_llm(product, campaign_type, tone, content_type, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku))
    else:
        st.warning("Please complete all inputs.")
