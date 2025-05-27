# Imports & Setup
import base64
import sqlite3
from io import BytesIO
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
from pymongo import MongoClient


# Load environment variables


# API keys and model configs
try:
    # Fetching API keys and model configurations from Streamlit secrets
    key = st.secrets.get("OPENAI_API_KEY")
    model_provider = st.secrets.get("GPT_model_provider")
    model_name = st.secrets.get("GPT_model")
    perplexity_key = st.secrets.get("perp_api_key")
    uri=st.secrets.get('uri')

    # Checking if any secret is None (not found in secrets)
    if None in [key, model_provider, model_name, perplexity_key]:
        raise ValueError("One or more secrets are missing or not set properly.")

except KeyError as e:
    st.error(f"KeyError: {e} - One or more secrets are missing.")
except ValueError as e:
    st.error(f"ValueError: {e}")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")

mongo_client=MongoClient(uri)
db=mongo_client['Pwani_llm_Output']


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

    # db = TinyDB("db.json")
    # ProductCategory = Query()
    # result.update({"date": str(datetime.today().date()), "Product": product, "category": category})
    # existing_record = db.search((ProductCategory.Product == product) & (ProductCategory.category == category))
    # if existing_record:
    #     # If the product and category exist, update the record
    #     db.update(result, (ProductCategory.Product == product) & (ProductCategory.category == category))
    #     print(f"Product '{product}' in category '{category}' updated.") 
    # else:
    #     # If the product and category don't exist, insert a new record
    #     db.insert(result)
    #     print(f"Product '{product}' in category '{category}' inserted.")    

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
    result = db.search((q.date == str(datetime.today().date())) & (q.Product == product) & (q.category == category))
    logging.info(f"Competitor data from data base db.json{result}")
    return result if result else data_collector_json_getter(competitors, category, product)

# Product info
@st.cache_data
def product_data_fetcher(brand, category):
    db = TinyDB("product_database.json")
    results = db.search((Query().Brand == brand) & (Query().Category == category))
    logging.info(f"Product data from data base new_product.json{results}")
    return results, results[0].get("Competition")

# Demographics info
@st.cache_data
def demographics_fetcher(gender, region, urban_or_rural):
    result = TinyDB("demographics.json").table("demographics").all()[-1]
    gender_data = [{g: result[g]} for g in gender] if gender != ["All Genders"] else {i:result[i] for i in ["Female", "Male", "Non_binary"]}
    locality_data = {urban_or_rural: result[urban_or_rural]} if urban_or_rural != "No Preference" else ""
    location_data = {region: result[region]} if region != "No Preference" else "Kenya"
    logging.info(f"Dmographics_data: Location data{location_data}, Gender data{gender_data}, locality data{locality_data}")

    return location_data, gender_data, locality_data

# Final LLM invocation

def Text_llm(product,content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku,language,history=[]):
    product_details, competitors = product_data_fetcher(product, category)
    # competitor_list = competitor_data_collector(product, competitors, category)
    location_data, gender_data, locality_data = demographics_fetcher(gender, region, urban_or_rural)

    messages = [
    SystemMessage(f"""
ROLE: 
    You are a **top-tier advertising strategist** specializing in **original, hyper-local campaigns**.

OBJECTIVE: 
    Design a standout **{campaign_type}** campaign for **Pwani** — promoting **{product}, {category}, {sku}**.
    The campaign will target **{channel}** customers via **{platform}**.

TONE & STYLE: 
    [Define the tone here - e.g., energetic, authentic, humorous, etc.]

CAMPAIGN CATEGORY: 
    {campaign_category}

CONTEXT:  
    You will receive from the User:
    - `product_details`: key benefits, unique selling points, and emotional anchors  

    - `target_audience`:  
        - Region: {region}  
        - Gender: {gender}  
        - Age Range: {age_range}  
        - Income Level: {income}  
        - Demographics: regional, gender-specific, and locality insights

IMPORTANT MUST-HAVES:
    - **Follow briefing instructions to a point **: {instructions}()
    - **Output should be of Type:**{content_type} and the language should be **{language}** of a **{tone}** focused on {campaign_category}.
    - **Platform-optimized**: Aligned with what’s trending and effective on **{platform}**.
    - **Cultural relevance**: Use local references, humor, or trends that resonate with the target audience.
    - **Unique selling proposition**: Highlight what makes Pwani’s product different from competitors.
    - **Call to action**: Encourage immediate engagement or purchase.
    - **Brand voice**: Align with Pwani’s brand identity and values.
    - **Avoid jargon**: Use clear, relatable language that resonates with the target audience.

Important Instruction:
    - **Output should be in 20-30 word Maximum.**
    - **This is the history of the conversation**: {history} (if it's empty, ignore this part; else your response should differ and improve based on the past campaign insights).

    - **Output should follow this format:**
        **Header:**
                [Generate a catchy and impactful title related to the campaign and product.]

        **Caption:**
                [Generate a persuasive caption that highlights the product benefits, includes brand endorsement information, and emphasizes its relevance to the target audience.]

        **{content_type}**
                [If content_type = "hashtag", generate relevant hashtags for the campaign and product; otherwise, provide an advertising script that can reach people don't provide hastags.]           
"""),

HumanMessage(f"""
product_details = {product_details}  


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


def image_llm(uploaded_image,product, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku,history=[]):
    product_details, competitors = product_data_fetcher(product, category)
    # competitor_list = competitor_data_collector(product, competitors, category)
    location_data, gender_data, locality_data = demographics_fetcher(gender, region, urban_or_rural)

    messages = [
    SystemMessage(f"""
ROLE: 
    You are a **top-tier advertising strategist** specializing in **original, hyper-local campaigns**.

OBJECTIVE: 
    Create a detailed **image generation prompt** for the background of a **{campaign_type}** campaign for **Pwani** — promoting **{product}, {category}, {sku}**. 
    The campaign will target **{channel}** customers via **{platform}**.

TONE & STYLE: 

    The style should align with what’s trending and effective on **{platform}**, featuring a **clear emotional appeal** and the output have tone of {tone}.

CAMPAIGN CATEGORY: 
    {campaign_category}

CONTEXT:  
    You will receive the following from the User as a **Human Message**:
    - `product_details`: Key benefits, unique selling points, and emotional anchors
    - `competitor_list`: Current ad formats and brand messages from competitors
    - `target_audience`:  
        - Region: {region}  
        - Gender: {gender}  
        - Age Range: {age_range}  
        - Income Level: {income}  
        - Demographics: Regional, gender-specific, and locality insights

IMPORTANT MUST-HAVES:
    - **Platform-optimized**: Align the background image with what’s currently trending and effective on **{platform}**.
    - **Cultural relevance**: Incorporate local references, humor, or trends that strongly resonate with the target audience and also product.
    - **Unique selling proposition**: The image should subtly highlight what makes Pwani’s product stand out from competitors.

Important Instruction:
    - **The output must be a specific, clear and simple it should not overwhelm the image model not by giving fine details** . 
    - The product image is already provided, and **it must not be altered in any way** — this instruction should be **assertive** in the prompt.
    - The image should very closely adhere to all the things mentioned in the instruction:{instructions} and also use your creative ideas that can be used in the prompt that aligns with the context.
    - **This is the history of the prompts**: {history} — if it’s empty, ignore this else your response should differ and improve based on the past prompts.      
"""),

HumanMessage(f"""
product_details = {product_details}  

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
    logging.info(f"Image LLM:Prompt: {messages}")
    
    prompt= llm.invoke(messages).content
    logging.info(f"Image LLM Response prompt: {prompt}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    img = client.images.edit(
        model="gpt-image-1",  # This is an assumption. Check OpenAI documentation for correct model name
        image=uploaded_image,
        prompt=prompt,
        size="1536x1024"
    )

    

    return img,prompt   

def base64_to_image(img):
    image_bytes = base64.b64decode(img.data[0].b64_json)
    image_io = BytesIO(image_bytes)
    return image_io


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
            gender = st.multiselect("Gender", ["Female", "Male", "Non_binary", 'All Genders'])
            if ('All Genders' in gender) and len(gender)>1:
                st.error('Please Select either All Genders or combination of other options')
                st.stop()
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
    with st.expander('Select the input type') :
         input_type=st.selectbox('Output type',('Text','Image','Image and Text'))
         if input_type!="Text":
            uploaded_image = st.file_uploader(f"Choose an image of type png of {product} , {category}", type=["png"])
            if not uploaded_image:
                st.error("Please upload an image.")      
    
# Main Action
with st.expander('Recommended Guidelines for Getting the Best Campaign Creative'):
    st.write(
        """🔍 **1. Focus on the Product’s Unique Features**  
• Highlight the product’s key selling points.  
• Examples:  
    • “Highlight the smooth texture and rich foam of our luxury soap.”  
    • “Showcase the compact, sleek design of our wireless earbuds.”  
    • “Focus on the creamy, tropical vibe of our mango ice cream.”  
⸻  

🎨 **2. Set the Scene and Mood Clearly**  
• Describe the environment or setting you envision.  
• Examples:  
    • “Fresh morning bathroom scene for a soap ad.”  
    • “Techy, futuristic workspace for a gadget ad.”  
    • “Beachside, summer vibe for a cold drink ad.”  
⸻  

🎯 **3. Specify the Target Audience’s Lifestyle**  
• Think about who you’re selling to and what resonates with them.  
• Examples:  
    • “Designed for busy professionals on the go.”  
    • “Perfect for adventurous, outgoing Gen Z.”  
    • “Ideal for health-conscious parents.”  
⸻  

🖼️ **4. Use Strong Visual Cues**  
• Mention specific colors, objects, or themes.  
• Examples:  
    • “Bright, tropical colors for a summer campaign.”  
    • “Clean, minimalist design for premium electronics.”  
    • “Natural, earthy tones for organic products.”  
⸻  

📱 **5. Match the Creative to the Channel**  
• Consider how the ad will look on the intended platform.  
• Examples:  
    • “Vertical, eye-catching for Instagram Stories.”  
    • “Professional, polished for LinkedIn posts.”  
    • “High-contrast, direct for WhatsApp promos.”  
⸻  

💥 **6. Add Emotional Triggers (When Possible)**  
• Play on emotions that drive action.  
• Examples:  
    • “Excitement for summer flavors.”  
    • “Peace of mind for safety tech.”  
    • “Luxury feel for premium products.”  
⸻  

🛑 **7. Avoid Common Creative Pitfalls**  
• ❌ Don’t just say “Make it look good” — be specific.  
• ❌ Avoid generic words like “awesome” or “cool” without context.  
• ❌ Don’t forget the product context (e.g., size, shape, use case).  
⸻  

💡 **8. Examples for Inspiration (Clickable in UI)**  
• “Show a young couple enjoying mango ice cream on a sunny beach.”  
• “Feature a professional in a modern workspace using high-tech earbuds.”  
• “Highlight the vibrant, bubbly foam of a luxury soap bar.”"""
    )

result,competitors=product_data_fetcher(product, category)


def result_null_finder(result):
    a=[]
    for i in result[0]:
        if result[0][i]=="None":
            a.append(i)
    return a   
null_values= result_null_finder(result)
if null_values:
    
    st.warning(f"""These aspects {null_values} of this {product} ,{category} is missing 
               **This may impact the quality and personalization of your campaign assets. The output may not fully reflect your brand's tone and style.**""")
    
instructions = st.text_input("Please provide your campaign brief here")
    

def real_time_text_generator(product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language):
    history = []
    for i in range(3):
        result = Text_llm(product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language, history)
        history.append(result)
        if 'text_results' not in st.session_state:
            st.session_state.text_results = []
        st.session_state.text_results.append(result)  # Save the results in session state
        st.write(result)  # Show the result immediately
        

def real_time_image_generator(uploaded_image, product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language):
    history = []
    images = []
    for i in range(3):
        result, prompt = image_llm(uploaded_image, product, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, history)
        history.append(prompt)
        images.append(result)
        if 'image_results' not in st.session_state:
            st.session_state.image_results = []
        st.session_state.image_results.append(result)  # Save the images in session state
        st.image(base64_to_image(result))  # Display the image immediately
        

def real_time_image_text_generator(uploaded_image, product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language):
    text_history = []
    prompt_history = []
    images = []
    for i in range(3):
        text_result = Text_llm(product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language, text_history)
        image_result, prompt = image_llm(uploaded_image, product, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, prompt_history)
        text_history.append(text_result)
        prompt_history.append(prompt)
        images.append(image_result)
        if 'text_results' not in st.session_state:
            st.session_state.text_results = []
        if 'image_results' not in st.session_state:
            st.session_state.image_results = []
        st.session_state.text_results.append(text_result)  # Save the text results in session state
        st.session_state.image_results.append(image_result)  # Save the image results in session state
        st.write(text_result)  # Show the text result immediately
        st.image(base64_to_image(image_result))  # Show the image immediately
        

if st.button("Generate Campaign"):
    if all([product, campaign_category, campaign_type, tone, content_type, instructions, age_range, gender, income, region, urban_or_rural, channel, platform]):
        now = datetime.now()
        common_data = {
            "date": str(now.date()),
            "time": str(now.time()),
            "product": product,
            "content_type": content_type,
            "campaign_type": campaign_type,
            "tone": tone,
            "campaign_category": campaign_category,
            "instructions": instructions,
            "category": category,
            "gender": str(gender),
            "age_range": str(age_range),
            "income": income,
            "region": region,
            "urban_or_rural": urban_or_rural,
            "channel": channel,
            "platform": platform,
            "sku": sku,
            "language": language,
            "input_type": input_type
        }

        if input_type == "Text":
            with st.spinner("Generating your text..."):
                real_time_text_generator(product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language)
                text_data = {
                    **common_data,
                    "text_result_1": st.session_state.text_results[0],
                    "text_result_2": st.session_state.text_results[1],
                    "text_result_3": st.session_state.text_results[2]
                }
                db.Text_output.insert_one(text_data)

        elif input_type == "Image":
            if uploaded_image:
                with st.spinner("Generating your image..."):
                    real_time_image_generator(uploaded_image, product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language)
                    image_data = {
                        **common_data,
                        "image_result_1": base64.b64decode(st.session_state.image_results[0].data[0].b64_json),
                        "image_result_2": base64.b64decode(st.session_state.image_results[1].data[0].b64_json),
                        "image_result_3": base64.b64decode(st.session_state.image_results[2].data[0].b64_json)
                    }
                    db.Image_output.insert_one(image_data)
            else:
                st.error("Please upload an image.")

        elif input_type == "Image and Text":
            if uploaded_image:
                with st.spinner("Generating image and text..."):
                    real_time_image_text_generator(uploaded_image, product, content_type, campaign_type, tone, campaign_category, instructions, category, gender, age_range, income, region, urban_or_rural, channel, platform, sku, language)
                    combined_data = {
                        **common_data,
                        "image_result_1": base64.b64decode(st.session_state.image_results[0].data[0].b64_json),
                        "image_result_2": base64.b64decode(st.session_state.image_results[1].data[0].b64_json),
                        "image_result_3": base64.b64decode(st.session_state.image_results[2].data[0].b64_json),
                        "text_result_1": st.session_state.text_results[0],
                        "text_result_2": st.session_state.text_results[1],
                        "text_result_3": st.session_state.text_results[2]
                    }
                    db.Image_and_Text_output.insert_one(combined_data)
            else:
                st.error("Please upload an image.")
    else:
        st.error("Please complete all inputs.")