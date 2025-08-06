import streamlit as st
import os
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
from groq import Groq
from supabase import create_client, Client
import logging

# Page configuration
st.set_page_config(
    page_title="LILA",
    page_icon=":material/design_services:",
    layout="wide"
)

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load secrets and initialize clients
@st.cache_resource
def initialize_clients():
    """Initialize Groq and Supabase clients"""
    try:
        # Get API keys from secrets
        groq_api_key = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        supabase_url = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        supabase_key = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not all([groq_api_key, supabase_url, supabase_key]):
            st.error("Missing required API keys. Please check your secrets configuration.")
            st.stop()
        
        # Initialize clients
        groq_client = Groq(api_key=groq_api_key)
        supabase_client = create_client(supabase_url, supabase_key)
        
        return groq_client, supabase_client
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        st.stop()

@st.cache_resource
def load_decision_tree():
    """Load the decision tree model"""
    try:
        with open('decision_tree.pkl', 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error("Decision tree model not found. Please ensure 'decision_tree.pkl' is in your repository.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading decision tree model: {str(e)}")
        st.stop()

def encode_image(image):
    """Convert PIL Image to base64 string for API"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def analyze_room_image(groq_client, image):
    """Analyze room image using Llama 4 Scout model"""
    try:
        # Encode image
        base64_image = encode_image(image)
        
        # Prompt for room analysis
        prompt = """
        You are an expert interior designer specializing in rug selection. Analyze this living room image and extract the following features:
        
        1. Room size (small/medium/large)
        2. Color scheme (warm/cool/neutral/mixed)
        3. Style (modern/traditional/eclectic/minimalist/rustic/industrial)
        4. Lighting (bright/moderate/dim)
        5. Furniture style (contemporary/vintage/mixed)
        6. Floor type (hardwood/carpet/tile/laminate)
        7. Dominant colors (list up to 3 main colors)
        8. Room mood (cozy/elegant/casual/formal/bohemian)
        
        Please respond in JSON format with these exact keys:
        {
            "room_size": "",
            "color_scheme": "",
            "style": "",
            "lighting": "",
            "furniture_style": "",
            "floor_type": "",
            "dominant_colors": [],
            "room_mood": "",
            "analysis_summary": "Brief description of the room"
        }
        
        Only respond with the JSON, no other text.
        """
        
        # Make API call to Groq
        response = groq_client.chat.completions.create(
            model="llama-3.2-90b-vision-preview",  # Using vision model for image analysis
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=1000,
            temperature=0.3
        )
        
        # Parse response
        response_text = response.choices[0].message.content.strip()
        
        # Clean response if it has markdown formatting
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        
        room_features = json.loads(response_text)
        return room_features
        
    except Exception as e:
        st.error(f"Error analyzing room image: {str(e)}")
        return None

def chat_with_designer(groq_client, message, room_features=None, chat_history=None):
    """Chat with LILA"""
    try:
        # System prompt for interior designer
        system_prompt = """
        You are an expert interior designer specializing in rug selection for living rooms. 
        
        IMPORTANT RESTRICTIONS:
        - Only discuss topics related to interior design, rugs, home decor, and room aesthetics
        - If asked about unrelated topics, politely redirect to interior design
        - Be helpful, creative, and professional in your responses
        - Provide specific, actionable advice
        
        Your expertise includes:
        - Rug styles, patterns, and materials
        - Color coordination and room harmony
        - Spatial planning and rug sizing
        - Style matching and aesthetic principles
        - Maintenance and care advice
        """
        
        # Add room context if available
        if room_features:
            system_prompt += f"""
            
            CURRENT ROOM CONTEXT:
            - Room size: {room_features.get('room_size', 'unknown')}
            - Color scheme: {room_features.get('color_scheme', 'unknown')}
            - Style: {room_features.get('style', 'unknown')}
            - Lighting: {room_features.get('lighting', 'unknown')}
            - Floor type: {room_features.get('floor_type', 'unknown')}
            - Dominant colors: {', '.join(room_features.get('dominant_colors', []))}
            - Room mood: {room_features.get('room_mood', 'unknown')}
            
            Use this context to provide personalized recommendations.
            """
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history
        if chat_history:
            messages.extend(chat_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Check if message is interior design related
        if not is_interior_design_related(message):
            return "I'm specialized in interior design and rug selection. Let's talk about how I can help you find the perfect rug for your living room! What specific questions do you have about rugs, colors, styles, or room design?"
        
        # Make API call
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def is_interior_design_related(message):
    """Check if message is related to interior design"""
    interior_keywords = [
        'rug', 'carpet', 'decor', 'design', 'room', 'color', 'style', 'furniture',
        'living', 'home', 'interior', 'decoration', 'aesthetic', 'pattern',
        'texture', 'size', 'placement', 'match', 'coordinate'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in interior_keywords)

def predict_rug_features(model_data, room_features):
    """Use decision tree to predict rug features"""
    try:
        model = model_data['model']
        feature_names = model_data['feature_names']
        
        # Convert room features to numerical format for decision tree
        # This is a simplified example - you'll need to adapt based on your model
        feature_vector = encode_room_features(room_features, feature_names)
        
        # Make prediction
        prediction = model.predict([feature_vector])[0]
        probabilities = model.predict_proba([feature_vector])[0]
        
        return {
            'predicted_rug_type': prediction,
            'confidence': max(probabilities),
            'feature_vector': feature_vector
        }
        
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return None

def encode_room_features(room_features, feature_names):
    """Convert room features to numerical vector"""
    # This is a simplified encoding - adapt based on your actual model features
    feature_vector = []
    
    # Example encoding (you'll need to match your training data)
    size_map = {'small': 0, 'medium': 1, 'large': 2}
    style_map = {'modern': 0, 'traditional': 1, 'eclectic': 2, 'minimalist': 3, 'rustic': 4}
    
    for feature_name in feature_names:
        if 'size' in feature_name.lower():
            feature_vector.append(size_map.get(room_features.get('room_size', ''), 1))
        elif 'style' in feature_name.lower():
            feature_vector.append(style_map.get(room_features.get('style', ''), 0))
        # Add more feature encodings as needed
        else:
            feature_vector.append(0)  # Default value
    
    return feature_vector

def search_rugs_in_database(supabase_client, rug_prediction):
    """Search for similar rugs in Supabase database"""
    try:
        # Example query - adapt based on your database schema
        response = supabase_client.table('rugs').select('*').eq('type', rug_prediction['predicted_rug_type']).limit(5).execute()
        
        return response.data
        
    except Exception as e:
        st.error(f"Error searching database: {str(e)}")
        return []

# Initialize clients and model
groq_client, supabase_client = initialize_clients()
model_data = load_decision_tree()

# App header
st.title("LILA, Your Lifetime Interior Layout Assistant")
st.markdown("Upload a photo of your living room and chat with our AI designer to find the perfect rug!")

# Sidebar for room analysis
st.sidebar.header("Room Analysis")

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'room_features' not in st.session_state:
    st.session_state.room_features = None
if 'rug_recommendations' not in st.session_state:
    st.session_state.rug_recommendations = []

# Main layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Upload Room Photo")
    uploaded_file = st.file_uploader(
        "Choose an image of your living room",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of your living room for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Living Room", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Room", type="primary"):
            with st.spinner("Analyzing your room..."):
                room_features = analyze_room_image(groq_client, image)
                
                if room_features:
                    st.session_state.room_features = room_features
                    st.success("Room analysis complete!")
                    
                    # Make prediction with decision tree
                    with st.spinner("Finding perfect rug matches..."):
                        rug_prediction = predict_rug_features(model_data, room_features)
                        
                        if rug_prediction:
                            # Search database for recommendations
                            rug_recommendations = search_rugs_in_database(supabase_client, rug_prediction)
                            st.session_state.rug_recommendations = rug_recommendations

with col2:
    st.header("💬 Chat with AI Designer")
    
    # Display room features if available
    if st.session_state.room_features:
        with st.expander("📋 Room Analysis Results"):
            features = st.session_state.room_features
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write(f"**Size:** {features.get('room_size', 'Unknown')}")
                st.write(f"**Style:** {features.get('style', 'Unknown')}")
                st.write(f"**Lighting:** {features.get('lighting', 'Unknown')}")
                st.write(f"**Floor:** {features.get('floor_type', 'Unknown')}")
            
            with col_b:
                st.write(f"**Color Scheme:** {features.get('color_scheme', 'Unknown')}")
                st.write(f"**Mood:** {features.get('room_mood', 'Unknown')}")
                st.write(f"**Furniture Style:** {features.get('furniture_style', 'Unknown')}")
                colors = features.get('dominant_colors', [])
                st.write(f"**Dominant Colors:** {', '.join(colors) if colors else 'Unknown'}")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**Designer:** {message['content']}")
    
    # Chat input
    user_message = st.text_input(
        "Ask me about rugs, colors, styles, or room design:",
        placeholder="What size rug should I get for my living room?",
        key="chat_input"
    )
    
    if user_message:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Get designer response
        with st.spinner("Designer is thinking..."):
            response = chat_with_designer(
                groq_client, 
                user_message, 
                st.session_state.room_features,
                st.session_state.chat_history
            )
            
            # Add response to history
            st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Rerun to update chat display
        st.rerun()

# Display rug recommendations
if st.session_state.rug_recommendations:
    st.header("🎯 Recommended Rugs")
    
    if len(st.session_state.rug_recommendations) > 0:
        for i, rug in enumerate(st.session_state.rug_recommendations):
            with st.expander(f"Rug Option {i+1}: {rug.get('name', 'Unnamed Rug')}"):
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    if rug.get('image_url'):
                        st.image(rug['image_url'], use_column_width=True)
                    else:
                        st.write("No image available")
                
                with col_b:
                    st.write(f"**Price:** ${rug.get('price', 'N/A')}")
                    st.write(f"**Size:** {rug.get('size', 'N/A')}")
                    st.write(f"**Material:** {rug.get('material', 'N/A')}")
                    st.write(f"**Style:** {rug.get('style', 'N/A')}")
                    st.write(f"**Description:** {rug.get('description', 'N/A')}")
                    
                    if rug.get('purchase_url'):
                        st.link_button("View Product", rug['purchase_url'])
    else:
        st.info("Upload a room photo and analyze it to get personalized rug recommendations!")

# Footer
st.markdown("---")
st.markdown("*Powered by Llama 4 Scout via Groq and advanced ML algorithms*")