# Imports & Setup
from typing import List
import logging
logging.basicConfig(filename="app.log",level=logging.INFO,  # Set the logging level
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.disable()
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
key = st.secrets.get("OPENAI_API_KEY")
model_provider = st.secrets.get("GPT_model_provider")
model_name = st.secrets.get("GPT_model")
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
    logging.info(f"Response: {response.choices[0].message.content}")
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
    gender_data = [{g: result[g]} for g in gender] if gender != ["No Preference"] else ""
    locality_data = {urban_or_rural: result[urban_or_rural]} if urban_or_rural != "No Preference" else ""
    location_data = {region: result[region]} if region != "No Preference" else "Kenya"
    logging.info(f"Dmographics_data: Location data{location_data}, Gender data{gender_data}, locality data{locality_data}")

    return location_data, gender_data, locality_data

# Final LLM invocation

def final_llm(product, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku,language):
    product_details, competitors = product_data_fetcher(product, category)
    competitor_list = competitor_data_collector(product, competitors, category)
    location_data, gender_data, locality_data = demographics_fetcher(gender, region, urban_or_rural)

    messages = [
    SystemMessage(f"""
ROLE: 
    You are a **top-tier advertising strategist** specializing in **original, hyper-local campaigns**.

OBJECTIVE: 
    Design a standout **{campaign_type}** campaign for **Pwani** — promoting **{product}, {category}, {sku}**.
    The campaign will target **{channel}** customers via **{platform}**.

TONE & STYLE: 
    

CAMPAIGN CATEGORY: 
    {campaign_category}

CONTEXT:  
    You will receive from the User:
    - `product_details`: key benefits, unique selling points, and emotional anchors  
    - `competitor_list`: current ad formats and brand messages from competitors  
    - `target_audience`:  
        - Region: {region}  
        - Gender: {gender}  
        - Age Range: {age_range}  
        - Income Level: {income}  
        - Demographics: regional, gender-specific, and locality insights

IMPORTANT MUST-HAVES:
    - **Follow briefing instructions**: {instructions}
    - **Output should be of Type:**{content_type}**  and the language should be:**{language}** of a **{tone}** focuced on {campaign_category}.
    - **Platform-optimized**: Aligned with what’s trending and effective on **{platform}**.
    - **Cultural relevance**: Use local references, humor, or trends that resonate with the target audience.
    - **Unique selling proposition**: Highlight what makes Pwani’s product different from competitors.
    - **Call to action**: Encourage immediate engagement or purchase.
    - **Brand voice**: Align with Pwani’s brand identity and values.
    - **Avoid jargon**: Use clear, relatable language that resonates with the target audience.
Important Instruction:
    - **Output should be in 20-30 word Maximum.**
    - **Use the following format for the output:**
    




       Output Structure:(output should be in this language:{language}(if mentioned local and english it should have both the flavours))

        **Header:**

                [Generate a catchy and impactful title related to the campaign and product.]

        **Caption:**

                [Generate a persuasive caption that highlights the product benefits, includes brand endorsement information, and emphasizes its relevance to the target audience.]

        **{content_type}**

                [If content_type={content_type} is hashtag, generate a set of multiple hashtags related to the campaign and product else provide only and advertising script for the campaign and product.]            
"""),

HumanMessage(f"""
product_details = {product_details}  
competitor_list = {competitor_list}  

target_audience = {{
    "region": "{region}",
    "gender": "{gender}",
    "age_range": "{age_range}",
    "income_level": "{income}",
    "demographics": {{
        "region_data": {location_data},
        "gender_data": {gender_data},
        "locality_data": {locality_data}
    }}
}}
""")
]

    result = llm.invoke(messages)
    logging.info(f"LLM Response: {result.content}")
    logging.info(f"LLM:Prompt: {messages}")
    return result.content

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
    
    with st.expander("Campaign Definition", expanded=True):
        
            product = st.selectbox("Brand", list(products_with_category))
            category = st.selectbox("Category", list(products_with_category[product]))
            sku = st.selectbox("SKU", ["Select"] + products_with_category[product][category])
            sku = None if sku == "Select" else sku
            channel = st.selectbox("Channel", list(channels))
            platform = st.selectbox("Platform", channels[channel])
            campaign_category = st.selectbox("Campaign Category",                                   
                (
   "Awareness",
   "Engagement",
   "Conversion",
   "Retention",
   "Launch",
   "Seasonal / Tactical"
)
                )
            campaign_type = st.selectbox("Campaign Type", (
                
   "Influencer / Partnership",
   "Educational / Thought Leadership",
   "PR & Media Coverage",
   "UGC / Community-Driven",
   "Social Media Series / Challenge",
   "Performance Ads",
   "Email / Newsletter",
   "Content Marketing"
)
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
    with st.expander("Target Market", expanded=True):
            
            age_range = st.slider("Age Range", 15, 65, (20, 30))
            gender = st.multiselect("Gender", ["Female", "Male", "Non_binary", 'No Preference'])
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
                                'Northeastern',
                                'No Preference')
                                )
            urban_or_rural = st.selectbox("Urban/Rural", ["Urban", "Rural", "No Preference"])

    with st.expander('Select the language',expanded=True):

            language=st.selectbox('Language',('English','Local language','English and Local Language'))    
    
# Main Action
instructions = st.text_input("Enter additional instructions")
if st.button("Generate Content"):
    if all([product, campaign_category, campaign_type, tone, content_type, instructions, age_range, gender, income, region, urban_or_rural, channel, platform]):
        st.write(final_llm(product, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku,language))
    else:
        st.warning("Please complete all inputs.")
