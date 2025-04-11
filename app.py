
from typing import List
from  datetime import datetime
from tinydb import TinyDB, Query
import streamlit as st
from openai import OpenAI
import datetime
from typing_extensions import  TypedDict
from pydantic import  Field
from typing import List
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
load_dotenv()



key=os.environ.get("GROQ_API_KEY")
model_provider=os.environ.get('GROQ_model_provider')
model_name=os.environ.get('GROQ_model')


###########################################################

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def model_type(model_name,key,model_provider):

    """This fucntion is responsible for selection of the llm"""
    if not key:
        raise ValueError("API KEY is missing. Please set it in your environment variables.")
    if not model_name:
        raise ValueError("model name is missing")
    
    if not model_provider:
        raise ValueError("model provider is missing")
    os.environ["API_KEY"]=key
    model = init_chat_model(model_name, model_provider=model_provider)
    return model

llm=model_type(model_name,key,model_provider)

##################################################################




########################################################################
def get_companies(prompt):
    """This function is responsible for getting the companies"""

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


    client = OpenAI(api_key="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", base_url="https://api.perplexity.ai")

    # chat completion without streaming
    response = client.chat.completions.create(
        model="sonar",
        messages=messages,
    )
    return response.choices[0].message.content
###############################################################

##############################################################
def json_db_creator(data,product,category):
    """this is responsible for creating jason"""
    class Advertisement_techniques(TypedDict):
        name: str= Field(description="Name of the company")
        Advertisement_strategy:str=Field(description="overall strategy are they using to position their product in the market")
        Type_of_advertisement:str=Field(description="channels or formats are being used? (e.g., TV, radio, social media, print, influencer marketing)")
        Type_of_Promotion:str=Field(description="promotional tactics are they using? (e.g., discounts, free samples, bundled offers, loyalty programs)")


    class ExtractSchema(TypedDict):
        companies: List[Advertisement_techniques]
        


    structured_llm = llm.with_structured_output(ExtractSchema)

    result=structured_llm.invoke(data)    

    result['date']=str(datetime.date.today())
    result['Product']=product
    result['category']=category
    db = TinyDB('db.json')

    table = db.table('products')
    table.insert(result) 
########################################################




#######################################################################
def data_collector_json_getter(cometitors,category,product):
    """This collects the data from perplexity and creates jason"""
    prompt = f"""
You are provided with a list of competitors for the Pwani Oil company in Kenya.

**Product of focus**: {product}  
**category**: {category}  
**Competitors**: {cometitors}

Your task is to analyze the **advertising and promotional strategies** used by these competitors in the Kenyan market.

Please provide insights under the following sections for each competitor:

1. **Advertisement Strategy** – What overall strategy are they using to position their product in the market?
2. **Type of Advertisement** – What channels or formats are being used? (e.g., TV, radio, social media, print, influencer marketing)
3. **Type of Promotion** – What promotional tactics are they using? (e.g., discounts, free samples, bundled offers, loyalty programs)

Focus only on the competitors listed. Do not provide details for Pwani Oil itself.
"""

    data=get_companies(prompt)
    json_db_creator(data,product,category)
    return data
#######################################################################


##################################################################################
def competitor_data_collector(product,cometitors,category):
    """This funciton manages every thing that includes collection of data that include advertisement strategies and comanpanies 
    that compete """
    db = TinyDB('db.json')
    table = db.table('products')
    Product = Query()

     # Find all products in the "Floor Cleaner" category
    if filtered_data := table.search((Product.date == str(datetime.date.today())) & (Product.name==product) & (Product.category==category)):

        
                
        
        return filtered_data
    else:
        return data_collector_json_getter(cometitors,category,product)
#####################################################################################################
    




#####################################################################################################
def product_data_fetcher(brand,category):
    db = TinyDB('new_product.json')
    table = db.table('Product_details')
    Product = Query()
    filtered_data = table.search((Product.Brand==brand) &(Product.Category==category) )
    
    return filtered_data,filtered_data[0]['Competition']

#####################################################################################################


def demographics_fetcher(gender,region,urban_or_rural):
    db = TinyDB('demographics.json')
    table = db.table('demographics')
    result = table.all()
    gender_data=[]
    if gender!='No Preference':
        for i in gender:
            gender_data.append({i:result[-1][i]})
    else:
        gender_data=" "    
    if urban_or_rural!='No Preference':
        locality_data={urban_or_rural:result[-1][urban_or_rural]}
    else:
        locality_data="  "   
    location_data={region:result[-1][region]}    
    return location_data,gender_data,locality_data 


    

#####################################################################################################


def final_llm(product,campaign_type,Tone_and_style,content_type,specific_instructions,category,gender,age_range,income_level,region,urban_or_rural,channel,social_media,sku):
    """Here all the data is assigned to get a final output"""
    
    product_details,cometitors=product_data_fetcher(product,category)
    competitor_list=competitor_data_collector(product,cometitors,category)
    location_data,gender_data,locality_data=demographics_fetcher(gender,region,urban_or_rural)
    if sku :
        messages = [
    SystemMessage(f"""
ROLE: You are a world-class, highly creative advertising strategist known for **groundbreaking** and **locally resonant** campaigns.

OBJECTIVE: Design a high-impact **{campaign_type}** campaign for the brand **Pwani**, and the product **{product}, {category}, {sku}**. If the SKU is specifically referenced, tailor your message accordingly. The campaign is to be delivered through **{channel}** on **{social_media}**.

TONE & STYLE: {Tone_and_style}  
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
    - Income Level: {income_level}
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

MUST-HAVES:
- Do **not** exceed 30 words for the campaign message.
- Do **not** exceed 20 words for the competitive rationale.
- Do **not** copy competitor formats. This must feel **unique**.
- Do **not** generalize. Go for bold, memorable, local, or clever.
- Make sure the type of data you are creating meets the standard and treding optimaization of {social_media}.

Take bold creative risks. This campaign should feel like it was **born in the streets of {region}**, not in a boardroom.

"""),
    HumanMessage(f"""
product_details = {product_details}  
competitor_list = {competitor_list}  
target_audience = {{
    'region': '{region}',
    'gender': '{gender}',
    'age_range': '{age_range}',
    'income_level': '{income_level}',
    'demographics': {{
        'region_data': {location_data},
        'gender_data': {gender_data},
        'locality_data': {locality_data}
    }}
}}
""")
]

    else:
        messages = [
    SystemMessage(f"""
ROLE: You are a world-class, highly creative advertising strategist known for **groundbreaking** and **locally resonant** campaigns.

OBJECTIVE: Design a high-impact **{campaign_type}** campaign for the brand **Pwani**, and the product **{product}, {category}**. If the SKU is specifically referenced, tailor your message accordingly. The campaign is to be delivered through **{channel}** on **{social_media}**.

TONE & STYLE: {Tone_and_style}  
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
    - Income Level: {income_level}
    - Demographics from location, gender, and locality breakdowns

MISSION:
1. Craft a **brilliant campaign message between 20 and 30 words**.
2. Add a **rationale (max 20 words)** explaining why your idea **leapfrogs competitors** through originality and deep audience resonance.
3. Your campaign must reflect:
    - **Fresh, unexpected creativity** (no clichés, no buzzwords)
    - **Deep cultural insight** (use language, tone, or hashtags that truly reflect the people of {region})
    - **Emotional or social triggers** relevant to this group (not generic product selling points)
    - **Better storytelling or symbolism** than competitors — even if it's a simple hashtag or callout.

MUST-HAVES:
- Do **not** exceed 30 words for the campaign message.
- Do **not** exceed 20 words for the competitive rationale.
- Do **not** copy competitor formats. This must feel **unique**.
- Do **not** generalize. Go for bold, memorable, local, or clever.

Take bold creative risks. This campaign should feel like it was **born in the streets of {region}**, not in a boardroom.

"""),
    HumanMessage(f"""
product_details = {product_details}  
competitor_list = {competitor_list}  
target_audience = {{
    'region': '{region}',
    'gender': '{gender}',
    'age_range': '{age_range}',
    'income_level': '{income_level}',
    'demographics': {{
        'region_data': {location_data},
        'gender_data': {gender_data},
        'locality_data': {locality_data}
    }}
}}
""")
]




    result = llm.invoke(messages)
    return result.content.split('</think>')[-1]

   

products_with_category={'FRESHFRI': {'COOKING OIL': ['10000ML',
   ' 1000ML',
   ' 20000ML',
   ' 2000ML',
   ' 250ML',
   ' 3000ML',
   ' 5000ML',
   ' 500ML']},
 'SALIT': {'COOKING OIL': ['10000ML',
   ' 1000ML',
   ' 20000ML',
   ' 2000ML',
   ' 3000ML',
   ' 5000ML',
   ' 500ML']},
 'DETREX': {'TOILET SOAP': ['100G',
   ' 300ML',
   ' 30ML',
   ' 500ML',
   ' 50ML',
   ' 80G']},
 'USHINDI': {'LAUNDRY DETERGENT': ['1000G', ' 20G', ' 500G'],
  'LAUNDRY BAR': ['1000G', ' 175G', ' 350G', ' 60G', ' 800G']},
 'SAWA MILKING JELLY': {'SKIN CARE - PETROLLEUM JELLIES': ['50ML',
   ' 90ML',
   ' 200ML']},
 'SAWA PETROLEUM JELLY': {'SKIN CARE - PETROLLEUM JELLIES': ['100ML',
   ' 250ML',
   ' 25ML',
   ' 50ML']},
 'SAWA HANDWASH': {'TOILET SOAP': ['250ML', ' 500ML']},
 'SAWA FAMILY SOAPS': {'TOILET SOAP': ['70G', ' 125G', ' 225G', ' 250G']},
 'SAWA BODY WASH': {'TOILET SOAP': ['500ML']},
 'BELLEZA ': {'LOTIONS': ['100ML']},
 'FRESCO ': {'LOTIONS': ['100ML']},
 'AFRISENSE': {'TOILET SOAP': ['125G', ' 225G']},
 'NDUME': {'LAUNDRY BAR': ['1000G', ' 200G', ' 600G', ' 700G', ' 800G']},
 'POPCO': {'COOKING OIL': ['1000ML',
   ' 20000ML',
   ' 2000ML',
   ' 3000ML',
   ' 5000ML',
   ' 500ML'],
  'LAUNDRY BAR': ['1000G', ' 600G']},
 'WHITE WASH (PWANI OIL PRODUCTS)': {'LAUNDRY BAR': ['1000G',
   ' 175G',
   ' 200G',
   ' 800G',
   ' 90G']},
 'DIVA ': {'TOILET SOAP': ['100G', ' 200G', ' 500ML']},
 '4U ': {'TOILET SOAP': ['250G']},
 'FRESH ZAIT': {'COOKING OIL': ['10000ML', ' 20000ML']},
 'FRYKING': {'COOKING FATS': ['1000G', ' 500G']},
 'MPISHI POA': {'COOKING FATS': ['10000G',
   ' 1000G',
   ' 100G',
   ' 17000G',
   ' 2000G',
   ' 250G',
   ' 40G',
   ' 5000G',
   ' 500G',
   ' 50G']},
 'ONJA': {'COOKING FATS': ['500G'], 'SPREADS': ['10000G', ' 1000G', ' 250G']},
 'TWIGA': {'COOKING OIL': ['10000ML', ' 20000ML']},
 'TIKU ': {'COOKING OIL': ['10000ML', ' 20000ML']}}


channels={'B2B': ['RTM', 'Whatsapp'],'B2C':[ 'Offline Posters', 'Facebook', 'Instagram', 'Tiktok', 'Twitter', 'Whatsapp group']}

with st.sidebar.expander('Campaign Definition', expanded=True):
    # Sidebar Inputs
    product = st.selectbox(
        "Brand",
        (products_with_category.keys()),  # Convert keys to a list
        key="product_selectbox"  # Optional: Use key for better control
    )

    category= st.selectbox(
        "category", 
        (products_with_category[product].keys()),
        key="category_selectbox"
    )
    sku=st.selectbox("SKU",
                    (['Select']+products_with_category[product][category]))
    if sku=='Select':
        sku=None
    

    channel=st.selectbox('Channels',(channels))

    social_media=st.selectbox('Social Media',(channels[channel]))

    campaign_category = st.selectbox(
        "Campaign Category",
        ("Awareness Campaign", 
         "Engagement Campaign", 
         "Conversion Campaign",
         "Retention Campaign", 
         "Product launch", 
         "Seasonal Promotion"),
        key="campaign_category_selectbox"
    )

    campaign_type = st.selectbox(
        "Campaign Type",
        ("Brand Awareness", 
         "Educational", 
         "Influencer/Partnership", 
         "Social Media Awareness", 
         "PR and Media Coverage"),
        key="campaign_type_selectbox"
    )

    Tone_and_style = st.selectbox(
        "Tone & Style",
        ("Professional", 
         "Casual", 
         "Inspirational", 
         "Authoritative", 
         "Friendly", 
         "Energetic", 
         "Persuasive"),
        placeholder="Select contact method...",
        key="tone_style_selectbox",
      
    )

    content_type = st.selectbox(
        "Content Type",
        ('Hashtag', 'Script'),
        key="content_type_selectbox"
    )

with st.sidebar.expander('Target Market', expanded=True):
    age_range = st.slider("Select a range", min_value=15, max_value=65, value=(20, 30))
    
    gender=st.multiselect('Select the gender',options=['Female', 'Male', 'Non_binary','No Preference'])

    income_level=st.selectbox('Income Level',
                              ('Low (e.g, <$30k)',
                               'Middle (e.g., $30k-$75k)',
                               'High (e.g., $75k-$150k)',
                               'Very High (e.g.,>$150k)',
                               'No Preference'))
    
    region=st.selectbox("Region",('Nairobi', 'Central', 'RiftValley', 'Coast', 'Eastern', 'Western', 'Northeastern'))
    urban_or_rural=st.selectbox("Urban/Rural",
                                ('Urban', 'Rural','No Preference'))

   



specific_instructions = st.text_input('Enter specific instructions or features')
if st.button('Generate Content'):
    # Validate that all required fields are filled
    if product and campaign_category and campaign_type and Tone_and_style and content_type and specific_instructions and age_range and gender and income_level and region and urban_or_rural and channel and social_media:
        # Call the function
        st.write(final_llm(product, campaign_type, Tone_and_style, content_type, specific_instructions, category,gender,age_range,income_level,region,urban_or_rural,channel,social_media,sku))
    else:
        st.warning("Please fill all the fields before submitting.")