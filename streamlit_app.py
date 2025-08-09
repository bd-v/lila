import streamlit as st
import os
import json
import pickle
from PIL import Image
import base64
from io import BytesIO
from groq import Groq
from supabase import create_client
import logging
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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
        GROQ_API_KEY = st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        SUPABASE_URL = st.secrets.get("SUPABASE_URL") or os.getenv("SUPABASE_URL")
        SUPABASE_KEY = st.secrets.get("SUPABASE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not all([GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY]):
            st.error("Missing required API keys. Please check your secrets configuration.")
            st.stop()
        
        # Initialize clients
        groq_client = Groq(api_key=GROQ_API_KEY)
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        return groq_client, supabase_client
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        st.stop()

@st.cache_resource
def load_decision_tree():
    """Load the decision tree model in a consistent format."""
    with open('decision_tree.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

    #     # If pickle already contains a dict with a model
    #     if isinstance(loaded_obj, dict) and 'model' in loaded_obj:
    #         return loaded_obj

    #     # If pickle contains just the model
    #     if hasattr(loaded_obj, "predict"):
    #         return {
    #             "model": loaded_obj,
    #             "feature_names": getattr(loaded_obj, "feature_names_in_", []),
    #             "target_classes": list(getattr(loaded_obj, "classes_", []))
    #         }

    #     # Fallback if structure is unexpected
    #     st.warning("Decision tree file loaded but structure was unexpected. Using fallback.")
    #     return {
    #         'model': None,
    #         'feature_names': ['style1', 'style2', 'colorMain', 'colorMinor', 'Brightness', 'hue'],
    #         'target_classes': ['Persian', 'Modern', 'Traditional', 'Boho']
    #     }

    # except FileNotFoundError:
    #     st.warning("Decision tree model not found. Using fallback mode.")
    #     return {
    #         'model': None,
    #         'feature_names': ['style1', 'style2', 'colorMain', 'colorMinor', 'Brightness', 'hue'],
    #         'target_classes': ['Persian', 'Modern', 'Traditional', 'Boho']
    #     }
    # except Exception as e:
    #     st.error(f"Error loading decision tree model: {str(e)}")
    #     return {
    #         'model': None,
    #         'feature_names': ['style1', 'style2', 'colorMain', 'colorMinor', 'Brightness', 'hue'],
    #         'target_classes': ['Persian', 'Modern', 'Traditional', 'Boho']
    #     }
    
def encode_image(image):
    """Convert PIL Image to base64 string for API"""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def create_fallback_analysis():
    """Create a fallback room analysis when API fails"""
    return {
        "style1": "Contemporary",
        "style2": "Modern",
        "colorMain": "Grey",
        "colorMinor": "White",
        "Brightness": "Light",
        "hue": "Cool"
    }

def analyze_room_image(groq_client, image):
    """Analyze room image using Groq vision model"""
    try:
        # Encode image
        base64_image = encode_image(image)
        
        # Prompt for room analysis
        prompt = """
        You are an expert interior designer specializing in rug selection. Analyze this living room image and extract the following features.
        
        Respond ONLY with valid JSON in this exact format (no other text, no markdown, no explanations):
        
        {
            "style1": "",
            "style2": "",
            "colorMain": "",
            "colorMinor": "",
            "Brightness": "",
            "hue": ""
        }
        
        Choose from these options only:
        - style1: Traditional, Contemporary, Boho, Outdoor
        - style2: Ornate, Modern, Classic, Solid, Geometric, Persian, Abstract
        - colorMain: Blue, Grey, Red, Multi, Ivory, Tan, Taupe
        - colorMinor: Ivory, Multi, White, Grey, Black, Blue
        - Brightness: Light, Dark
        - hue: Warm, Cool
        """
        
        # Make API call to Groq
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Correct model name
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
            max_tokens=500,
            temperature=0.1
        )
        
        # Get response text
        response_text = response.choices[0].message.content
        
        if not response_text or response_text.strip() == "":
            st.warning("Empty response from Groq API, using fallback analysis")
            return create_fallback_analysis()
        
        # Debug: Show raw response (remove this in production)
        st.write("Debug - Raw API Response:", response_text[:200] + "...")
        
        # Clean response
        response_text = response_text.strip()
        
        # Remove markdown formatting if present
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()
        
        # Remove any text before the first {
        json_start = response_text.find('{')
        if json_start > 0:
            response_text = response_text[json_start:]
        
        # Remove any text after the last }
        json_end = response_text.rfind('}')
        if json_end > 0:
            response_text = response_text[:json_end + 1]
        
        try:
            room_features = json.loads(response_text)
            
            # Validate required keys
            required_keys = ['style1', 'style2', 'colorMain', 'colorMinor', 'Brightness', 'hue']
            
            for key in required_keys:
                if key not in room_features:
                    st.warning(f"Missing key in response: {key}")
                    # Set default values based on the key
                    if key == 'style1':
                        room_features[key] = "Contemporary"
                    elif key == 'style2':
                        room_features[key] = "Modern"
                    elif key == 'colorMain':
                        room_features[key] = "Grey"
                    elif key == 'colorMinor':
                        room_features[key] = "White"
                    elif key == 'Brightness':
                        room_features[key] = "Light"
                    elif key == 'hue':
                        room_features[key] = "Cool"
            
            return room_features
            
        except json.JSONDecodeError as je:
            st.error(f"JSON parsing error: {str(je)}")
            st.error(f"Response that failed to parse: {response_text}")
            return create_fallback_analysis()
        
    except Exception as e:
        st.error(f"Error analyzing room image: {str(e)}")
        return create_fallback_analysis()

def chat_with_designer(groq_client, message, room_features=None, chat_history=None):
    """Chat with LILA"""
    try:
        # System prompt for interior designer
        system_prompt = """
        You are LILA, a Lifetime Interior Layout Assistant specializing in rug selection for living rooms. 
        
        IMPORTANT RESTRICTIONS:
        - Only discuss topics related to interior design, rugs, home decor, and room aesthetics
        - If asked about unrelated topics, politely redirect to interior design
        - Be helpful, creative, and professional in your responses
        - Provide specific, actionable advice
        - Keep responses concise but informative
        
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
            - Primary style: {room_features.get('style1', 'unknown')}
            - Secondary style: {room_features.get('style2', 'unknown')}
            - Main color: {room_features.get('colorMain', 'unknown')}
            - Minor color: {room_features.get('colorMinor', 'unknown')}
            - Brightness: {room_features.get('Brightness', 'unknown')}
            - Hue: {room_features.get('hue', 'unknown')}
            
            Use this context to provide personalized recommendations.
            """
        
        # Build messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add chat history (last 10 messages to avoid context overflow)
        if chat_history:
            messages.extend(chat_history[-10:])
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Check if message is interior design related
        if not is_interior_design_related(message):
            return "I'm LILA, your Lifetime Interior Layout Assistant specialized in interior design and rug selection. Let's talk about how I can help you find the perfect rug for your living room! What specific questions do you have about rugs, colors, styles, or room design?"
        
        # Make API call
        response = groq_client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",  # Correct model name
            messages=messages,
            max_tokens=500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Let me help you with interior design questions when you're ready!"

def is_interior_design_related(message):
    """Check if message is related to interior design"""
    interior_keywords = [
        'rug', 'carpet', 'decor', 'design', 'room', 'color', 'style', 'furniture',
        'living', 'home', 'interior', 'decoration', 'aesthetic', 'pattern',
        'texture', 'size', 'placement', 'match', 'coordinate', 'layout', 'space',
        'wall', 'floor', 'couch', 'sofa', 'table', 'lighting'
    ]
    
    message_lower = message.lower()
    return any(keyword in message_lower for keyword in interior_keywords)

def predict_rug_features(model, room_features):
    """Use decision tree to predict rug features."""
    try:
        if model is None:
            print("Model is NONE")
            return predict_rug_rule_based(room_features), 1

        feature_vector = encode_room_features(room_features)
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]

        return {
            'predicted_rug_type': prediction,
            'confidence': float(max(probabilities)),
            'feature_vector': feature_vector
        }, 0

    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        return predict_rug_rule_based(room_features)

def predict_rug_rule_based(room_features):
    """Rule-based rug prediction when ML model is not available"""
    style1 = room_features.get('style1', '').lower()
    style2 = room_features.get('style2', '').lower()
    
    # Simple rule-based prediction
    if 'traditional' in style1 or 'persian' in style2:
        predicted_type = 'Persian'
    elif 'boho' in style1 or 'ornate' in style2:
        predicted_type = 'Boho'
    elif 'contemporary' in style1 or 'modern' in style2:
        predicted_type = 'Modern'
    else:
        predicted_type = 'Traditional'
    
    return {
        'predicted_rug_type': predicted_type,
        'confidence': 0.8,
        'method': 'rule_based'
    }, 0

def encode_room_features(room_features):
    """Convert room features to numerical vector"""
    feature_vector = []
    
    rugs = pd.read_csv("projectDatabase.csv")
    test_case = pd.DataFrame([room_features])
    rugs = pd.concat([rugs, test_case])

    X = rugs[['style1','style2','colorMain','colorMinor','Brightness','hue']]
    y = rugs['design']

    # Define which columns are categorical
    categorical_features = ['style1','style2','colorMain','colorMinor','Brightness','hue']

    # Create ColumnTransformer to apply OneHotEncoder to categorical columns only
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features) # Add handle_unknown='ignore'
        ],
        remainder='passthrough'  # keep the other columns (e.g., 'weight') as is
    )

    # Transform the features
    X_encoded = preprocessor.fit_transform(X)

    # Optional: convert to DataFrame with column names
    feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    # Remove 'weight' from final_feature_names as it's not in X
    final_feature_names = list(feature_names)
    X_encoded_df = pd.DataFrame(X_encoded.toarray(), columns=final_feature_names)

    feature_vector = X_encoded_df.tail(1)
    
    return feature_vector

def search_rugs_in_database(supabase_client, rug_prediction, flag):
    """Search for similar rugs in Supabase database"""
    try:
        predicted_type = rug_prediction['predicted_rug_type']
        
        # Search for rugs matching the predicted type
        if flag == 0:
            response = supabase_client.table('rugs').select('*').eq('design', predicted_type).limit(5).execute()
        elif flag == 1:
            response = supabase_client.table('rugs').select('*').eq('style1', predicted_type).limit(5).execute()
        
        # If no exact matches, search for similar styles
        if not response.data:
            # Try broader search
            response = supabase_client.table('rugs').select('*').limit(5).execute()
        
        return response.data
        
    except Exception as e:
        st.error(f"Error searching database: {str(e)}")
        return []

# Initialize clients and model
try:
    groq_client, supabase_client = initialize_clients()
    model = load_decision_tree()
except Exception as e:
    st.error(f"Failed to initialize: {str(e)}")
    st.stop()

# App header
st.title("🏠 LILA - Your Lifetime Interior Layout Assistant")
st.markdown("Upload a photo of your living room and chat with LILA to find the perfect rug!")

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
    st.header("Upload Room Photo")
    uploaded_file = st.file_uploader(
        "Choose an image of your living room",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear photo of your living room for analysis"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Your Living Room", use_container_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Room", type="primary"):
            with st.spinner("LILA is analyzing your room..."):
                room_features = analyze_room_image(groq_client, image)
                print(room_features)

                if room_features:
                    st.session_state.room_features = room_features
                    st.success("Room analysis complete!")
                    
                    # Make prediction with decision tree
                    with st.spinner("Finding perfect rug matches..."):
                        rug_prediction, flag = predict_rug_features(model, room_features)
                        
                        if rug_prediction:
                            # Search database for recommendations
                            print(rug_prediction['predicted_rug_type'])
                            rug_recommendations = search_rugs_in_database(supabase_client, rug_prediction, flag)
                            print(rug_recommendations)
                            st.session_state.rug_recommendations = rug_recommendations

with col2:
    st.header("Chat with LILA")
    
    # Display room features if available
    if st.session_state.room_features:
        with st.expander("📋 Room Analysis Results"):
            features = st.session_state.room_features
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.write(f"**Primary Style:** {features.get('style1', 'Unknown')}")
                st.write(f"**Secondary Style:** {features.get('style2', 'Unknown')}")
                st.write(f"**Main Color:** {features.get('colorMain', 'Unknown')}")
            
            with col_b:
                st.write(f"**Minor Color:** {features.get('colorMinor', 'Unknown')}")
                st.write(f"**Brightness:** {features.get('Brightness', 'Unknown')}")
                st.write(f"**Hue:** {features.get('hue', 'Unknown')}")
    
    # Chat interface
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        if not st.session_state.chat_history:
            st.markdown("**LILA:** Hello! I'm your Lifetime Interior Layout Assistant. Upload a photo of your room and I'll help you find the perfect rug! You can also ask me any questions about interior design.")
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**LILA:** {message['content']}")
    
    # Chat input
    user_message = st.text_input(
        "Ask LILA about rugs, colors, styles, or room design:",
        placeholder="What size rug should I get for my living room?",
        key="chat_input"
    )
    
    if user_message:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_message})
        
        # Get designer response
        with st.spinner("LILA is thinking..."):
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
    st.header("LILA's Rug Recommendations")
    
    if len(st.session_state.rug_recommendations) > 0:
        for i, rug in enumerate(st.session_state.rug_recommendations):
            with st.expander(f"Rug Option {i+1}: {rug.get('design', 'Unnamed Rug')}"):
                col_a, col_b = st.columns([1, 2])
                
                with col_a:
                    if rug.get('photo'):
                        st.image(rug['photo'], use_container_width=True)
                    else:
                        st.write("No image available")
                
                with col_b:
                    st.write(f"**Collection:** {rug.get('collection', 'N/A')}")
                    st.write(f"**Style:** {rug.get('style1', 'N/A')}")
                    st.write(f"**Color:** {rug.get('colorMain', 'N/A')}")
                    st.write(f"**Design:** {rug.get('style2', 'N/A')}")
                    
                    if rug.get('purchase_url'):
                        st.link_button("View Product", rug['purchase_url'])
    else:
        st.info("Upload a room photo and analyze it to get personalized rug recommendations from LILA!")

# Footer
st.markdown("---")
st.markdown("*Powered by LILA - Your AI Interior Design Assistant*")