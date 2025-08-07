import streamlit as st
from supabase import create_client
import pickle
import requests
import base64
import json

SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
st.set_page_config(page_title="LILA", page_icon=":material/design_services:")
st.title("Hi, I'm LILA.")
st.header("Your Lifetime Interior Layout Assistant")

@st.cache_resource
def load_model():
    with open('decision-tree.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

def file_to_data_url(file):
    data = file.read()
    encoded = base64.b64encode(data).decode()
    mime = file.type
    return f"data:{mime};base64,{encoded}"

def query_groq(image_url):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = """
        "You are a interior designer that specializes in finding the perfect rug to fit a space."
        "The user will give you an image of their room."
        Respond with a JSON object with fields: {
            'style': The rug's primary style (e.g. traditional, contemporary, outdoor),
            'color': The rug's primary color (one word),
            'pattern': The rug's pattern (one word)
        }.
        Only respond with a JSON object. Do not include comments or text.
        """
    body = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    response = requests.post(GROQ_API_URL, headers=headers, json=body)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(content)
        return parsed
    except json.JSONDecodeError:
        st.error("Model did not return valid JSON.")
        st.text(content)
        return None

def search_rugs(style, color, pattern):
    result = supabase.table("rugs").select("*") \
        .ilike("color", f"%{color}%") \
        .ilike("style", f"%{style}%") \
        .ilike("pattern", f"%{pattern}%") \
        .limit(1) \
        .execute()
    return result.data[0] if result.data else None

with st.container():
    user_input = st.chat_input("What would you like to ask?")
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload a photo of your room", type=["jpg", "jpeg", "png"])
    with col2:
        image_url = st.text_input("Or paste an image URL")

if uploaded_file or image_url:
    if uploaded_file:
        image_source = file_to_data_url(uploaded_file)
    else:
        image_source = image_url

    st.image(image_source, caption="Room Image", use_column_width=True)
    st.info("Thinking...")

    result = query_groq(image_source)

    if result:
        st.success("Here's your perfect rug!")
        st.json(result)

        rug = search_rugs(result["style"], result["color"], result["pattern"])
        if rug:
            st.subheader("Recommended Rug:")
            st.markdown(f"**{rug['name']}** — *{rug['style']} style, {rug['color']} color*")
            st.image(rug["image_url"], width=400)
        else:
            st.warning("No matching rug found in database. Try changing your image.")