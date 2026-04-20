import streamlit as st
from PIL import Image
import torch
import numpy as np
import pickle
import faiss
import os
import sys
from utils import load_clip_model, get_clip_embedding
from download_models import ensure_segment_anything, download_sam_model
from sam_utils import get_card_crops
import plotly.graph_objects as go
from io import BytesIO
import pandas as pd
import json
import google.generativeai as genai
from extract_card_details import (
    WebSearchPriceProvider,
    build_price_search_query,
    extract_card_text_details,
)

# Define constants
CLIP_INDEX_PATH = "clip.index"
CARD_DB_PATH = "cards.csv"
SAM_MODEL_TYPE = "vit_b"
MASKS_IMAGE_PATH = "all_masks_with_info.png"
GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-05-20'

GEMINI_PROMPT = """This is an image of Pokémon cards.
Please return a list of cards detected, with the following info per card:
- Card name
- Set name (if visible)
- Price (if present)
- Condition (if present, choose from: Mint, Near Mint, Lightly Played, Moderately Played, Heavily Played, Damaged, N/A)
Return the result as a JSON array like this:
[
  {
    "name": "Charizard",
    "set": "Base Set",
    "price": "350",
    "condition": "Lightly Played"
  }
]
Only include cards you are confident about."""

# This must be the first Streamlit command
st.set_page_config(layout="wide")

# Initialize models on startup
@st.cache_resource
def initialize_models():
    # Ensure segment_anything is available
    if not ensure_segment_anything():
        st.error("Failed to load segment_anything. Please check your installation.")
        return False
    
    # Download SAM model if needed
    model_path = download_sam_model(SAM_MODEL_TYPE)
    if not model_path or not os.path.exists(model_path):
        st.error(f"Failed to download SAM model. Please check your internet connection and try again.")
        return False
    
    return True

# Load card database with price history and parse JSON
def load_card_db():
    card_db = pd.read_csv(CARD_DB_PATH)
    # Ensure price_history is parsed correctly as a dictionary
    card_db["price_history"] = card_db["price_history"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    return card_db

# Update the existing card_db loading logic
@st.cache_resource
def load_index():
    index = faiss.read_index(CLIP_INDEX_PATH)
    card_db = load_card_db()
    return index, card_db


# Function to generate an interactive price history chart using Plotly
def generate_price_chart(card_name, card_db):
    card = card_db[card_db["name"] == card_name]
    if card.empty:
        return None

    price_history = card.iloc[0]["price_history"]
    dates = price_history["dates"]
    prices = price_history["prices"]

    # Ensure all data points are included in the chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=prices,
        mode='lines',  # Use smooth lines without markers
        line=dict(shape='spline',color='orange', width=2),  # Set line color to orange
        name='Price',
        hovertemplate='<b>Date:</b> %{x}<br><b>Price:</b> $%{y}<extra></extra>'
    ))
    fig.update_layout(
        title={
            'text': f"Price History: {card_name}",
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Date",
        yaxis_title="Price ($)",
        xaxis=dict(showgrid=True, gridcolor='lightgrey', tickformat='%b %d, %Y'),
        yaxis=dict(showgrid=True, gridcolor='lightgrey'),
        template="plotly_white",
        height=400,
        width=400,  # Set the chart width to 400
        margin=dict(l=40, r=40, t=60, b=40),
        hovermode="x unified"
    )
    return fig

# Function to display card details in a list style
def display_card_details(card):
    # Add a title for the card details
    st.markdown("### Card Details")

    # Display each card detail with the attribute in orange
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Card Name:</span> {card['name']}", unsafe_allow_html=True)
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Set:</span> {card['set']}", unsafe_allow_html=True)
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Current Market Price:</span> CAD$ {card['value']}", unsafe_allow_html=True)
    st.markdown(f"<span style='color: orange; font-weight: bold;'>Condition:</span> {card['condition']}", unsafe_allow_html=True)

# Initialize models first
initialize_models()

# Configure Gemini API key
# Make sure to set your GOOGLE_API_KEY environment variable
# or use Streamlit secrets: st.secrets["GOOGLE_API_KEY"]
try:
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY") or st.secrets["GOOGLE_API_KEY"])
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}. Please ensure your API key is set.")
    st.stop()

# Function to call Gemini API
def get_gemini_response(image_bytes, prompt):
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    image_parts = [
        {
            "mime_type": "image/jpeg",
            "data": image_bytes
        }
    ]
    prompt_parts = [
        image_parts[0],
        prompt,
    ]
    try:
        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return None

# Reusable function for image input
def get_image_input(key_prefix: str, help_text: str = "Upload an image or use your camera."):
    st.markdown(f"#### {help_text}")

    # Ensure session state keys for the image and its raw source ID exist
    if f"{key_prefix}_image" not in st.session_state:
        st.session_state[f"{key_prefix}_image"] = None
    if f"{key_prefix}_raw_input_id" not in st.session_state:
        st.session_state[f"{key_prefix}_raw_input_id"] = None

    input_method = st.selectbox("Select input method:", ("Upload Image", "Use Camera"), key=f"{key_prefix}_input_method")

    raw_input_data = None 

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key=f"{key_prefix}_uploader")
        raw_input_data = uploaded_file
    elif input_method == "Use Camera":
        camera_image_bytes = st.camera_input("Take a picture", key=f"{key_prefix}_camera")
        raw_input_data = camera_image_bytes
    
    new_raw_input_id = id(raw_input_data) if raw_input_data is not None else None
    
    # If the raw input data has changed (new upload, new photo, or cleared input)
    if new_raw_input_id != st.session_state[f"{key_prefix}_raw_input_id"]:
        if raw_input_data is not None:
            try:
                # Process and store the new image
                st.session_state[f"{key_prefix}_image"] = Image.open(raw_input_data).convert("RGB")
            except Exception as e:
                st.error(f"Error opening image: {e}")
                st.session_state[f"{key_prefix}_image"] = None
        else:
            # Input was cleared
            st.session_state[f"{key_prefix}_image"] = None
        # Update the stored ID of the raw input
        st.session_state[f"{key_prefix}_raw_input_id"] = new_raw_input_id
    
    # This function no longer directly returns the image.
    # It manages the UI and updates st.session_state[f"{key_prefix}_image"].
    # The calling code will retrieve the image from session_state.

st.title("TCG Card Recognition and Matching Tool")

# Create tabs
tab1, tab2 = st.tabs(["Main App", "Gemini Vision"])

with tab1:
    st.markdown("""
    ### Key Features:
    - **:orange[Multi-Item Detection]**: Capable of detecting and processing multiple cards in a single image.
    - **:orange[Versatile Product Matching]**: Supports recognition and matching of various types of trading cards and collectibles.

    ### Powered by:
    - **:orange[SAM (Segment Anything Model)]**: Used for precise segmentation and cropping of cards from the uploaded image.
    - **:orange[CLIP (Contrastive Language–Image Pretraining)]**: Utilized for generating embeddings to match card images with the database.
    - **:orange[FAISS (Facebook AI Similarity Search)]**: Enables efficient similarity search to find the closest match in the card database.
    """)

    @st.cache_resource
    def load_models_tab1(): # Renamed to avoid conflict
        return load_clip_model()

    index, card_db = load_index()
    clip_model, clip_processor = load_models_tab1()
    web_search_provider = WebSearchPriceProvider(engine="google")

    # Replace the radio button with a segment control for input method selection
    get_image_input(key_prefix="tab1", help_text="Choose an input method for card recognition")
    image_tab1 = st.session_state.get("tab1_image")


    if image_tab1: # Check image_tab1
        # Display the captured or uploaded image
        cols_tab1 = st.columns(2)
        with cols_tab1[0]:
            st.image(image_tab1, caption="Input Image", use_column_width=True)

        # Segment and crop cards using SAM
        with cols_tab1[1]:
            with st.container():
                sub_cols_tab1 = st.columns([1, 2, 1])  # Add horizontal spacing with columns
                with sub_cols_tab1[1]:
                    with st.spinner("#### 🪄 Segmenting cards...\nThis might take a moment depending on the image complexity."): # Enhanced spinner message
                        crops = get_card_crops(image_tab1) # This function might be slow, spinner is good.
            if os.path.exists(MASKS_IMAGE_PATH): # Use constant
                masks_image = Image.open(MASKS_IMAGE_PATH) # Use constant
                st.image(masks_image, caption="All Masks with Info", use_column_width=True)
            else:
                st.warning("⚠️ Masks visualization not found.")

        num_crops = len(crops)
        st.write(f"#### 📦 Detected {num_crops} card(s)")

        if num_crops > 0:
            for i, crop in enumerate(crops):
                extraction = extract_card_text_details(crop)
                query_result = build_price_search_query(extraction, provider=web_search_provider)

                # Create two columns for the images (cropped card and matched card)
                img_col1, img_col2 = st.columns(2)

                with img_col1:
                    st.write(f"**Card {i+1}**")
                    st.image(crop, use_column_width=True)

                ocr_col, link_col = st.columns(2)
                with ocr_col:
                    st.markdown("#### OCR extraction (price search)")
                    st.markdown(f"**Card name:** {extraction.card_name or 'N/A'}")
                    if extraction.collector_number and extraction.set_total:
                        st.markdown(f"**Collector number:** {extraction.collector_number}/{extraction.set_total}")
                    else:
                        st.markdown("**Collector number:** N/A")
                    st.caption(f"Raw name OCR: {extraction.raw_name_text or '(empty)'}")
                    st.caption(f"Raw number OCR: {extraction.raw_number_text or '(empty)'}")
                    st.code(query_result.query, language="text")
                with link_col:
                    st.markdown("#### Web price lookup")
                    st.link_button("Search card price", query_result.url)

                try:
                    # Get the embedding and search for the match
                    emb = get_clip_embedding(crop.resize((224, 224)), clip_model, clip_processor)
                    D, I = index.search(np.array([emb], dtype="float32"), k=1)
                    match = card_db.iloc[I[0][0]]

                    with img_col2:
                        st.write(f"matched with {match['name']}")
                        st.image(match['image_path'], use_column_width=True)

                    # Create two columns for card details and price history chart
                    details_col, chart_col = st.columns(2)

                    with details_col:
                        display_card_details(match)

                    with chart_col:
                        try:
                            chart_fig = generate_price_chart(match['name'], card_db)
                            if chart_fig:
                                st.plotly_chart(chart_fig, use_container_width=True)
                            else:
                                st.warning("⚠️ No price history available.")
                        except Exception as e:
                            st.error(f"❌ Chart generation failed: {e}")

                except Exception as e:
                    st.error(f"❌ Match failed: {e}")
        else:
            st.warning("⚠️ No cards detected. Please try another image.")

with tab2:
    st.header("Card Analysis with Gemini Vision")
    st.markdown("""
    Upload an image or use your camera to capture Pokémon cards.
    Gemini Vision will attempt to identify the cards and provide details based on the following prompt:
    """)
    st.code(GEMINI_PROMPT, language="text") # Use global constant

    get_image_input(key_prefix="tab2", help_text="Choose an input method for Gemini Vision")
    image_tab2 = st.session_state.get("tab2_image")


    if image_tab2:
        # Display the captured or uploaded image
        st.image(image_tab2, caption="Image for Gemini Analysis", use_column_width=True) # Display original image

        if st.button("Analyze with Gemini", key="gemini_button"):
            with st.spinner("#### 🧠 Gemini is analyzing the image..."):
                # Convert PIL Image to bytes using the original full-resolution image
                buffered = BytesIO()
                image_tab2.save(buffered, format="JPEG") # Use original image_tab2
                img_bytes = buffered.getvalue()

                gemini_result = get_gemini_response(img_bytes, GEMINI_PROMPT) # Use global constant

                if gemini_result:
                    st.subheader("Gemini Vision API Response:")
                    # Attempt to parse and display as JSON, otherwise show raw text
                    try:
                        # Clean the response if it's wrapped in markdown
                        if gemini_result.startswith("```json"):
                            gemini_result = gemini_result.replace("```json", "").replace("```", "").strip()
                        parsed_json = json.loads(gemini_result)
                        st.json(parsed_json)
                    except json.JSONDecodeError:
                        st.text_area("Raw Response", gemini_result, height=300)
                    except Exception as e:
                        st.error(f"Could not parse or display Gemini response: {e}")
                        st.text_area("Raw Response", gemini_result, height=300)
