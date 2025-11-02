"""
Mental Health Assessment Chatbot - Streamlit App
Interactive questionnaire with conversational flow
Uses production_mental_health_model.pkl for predictions
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any
import re
import string
import spacy
import contractions
import wordninja
import emoji
from bs4 import BeautifulSoup
from cleantext import clean
from autocorrect import Speller
import torch
from sentence_transformers import SentenceTransformer
import io

# ================================================================
# CUSTOM CLASSES (MUST BE DEFINED BEFORE MODEL LOADING)
# ================================================================

class AdvancedTextPreprocessor:
    """
    Complete text preprocessing pipeline:
    1. Lowercase
    2. HTML removal
    3. URL/email removal
    4. Hashtag removal
    5. Slang expansion
    6. Contraction expansion
    7. Emoji to text
    8. Punctuation removal
    9. Word splitting
    10. Repeated chars
    11. Number removal
    12. Spelling correction
    13. Extra space
    14. Lemmatization
    """
    
    def __init__(self, slang_dict=None):
        print("Initializing Advanced Text Preprocessor...")
        self.spell = Speller(lang="en")
        self.punct = string.punctuation
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.slang_abbr_dict = slang_dict if slang_dict else self.get_default_slang_dict()
        print("✅ Text Preprocessor initialized!")
    
    def get_default_slang_dict(self):
        return {
            "idk": "i do not know",
            "tbh": "to be honest",
            "imo": "in my opinion",
            "rn": "right now",
            "ngl": "not gonna lie",
            "ur": "your",
            "u": "you",
            "ppl": "people",
            "thx": "thanks",
            "plz": "please",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "cant": "cannot",
            "wont": "will not",
            "dont": "do not",
            "didnt": "did not",
            "isnt": "is not",
            "arent": "are not"
        }
    
    def clean_text(self, text):
        if not text or str(text).strip() == "":
            return ""
        
        # Lowercase
        text = str(text).lower()
        
        # Remove HTML
        text = BeautifulSoup(text, "lxml").get_text()
        
        # Remove URLs/emails
        text = clean(text, no_urls=True, replace_with_url="", no_emails=True, replace_with_email="")
        
        # Remove hashtags
        text = re.sub(r"#\w+", "", text)
        
        # Expand slang
        words = [self.slang_abbr_dict.get(w, w) for w in text.split()]
        text = " ".join(words)
        
        # Expand contractions
        text = contractions.fix(text)
        
        # Emojis to text
        text = emoji.demojize(text, delimiters=(" ", " "))
        
        # Remove punctuation
        text = text.translate(str.maketrans("", "", self.punct))
        
        # Split compound words
        text = " ".join(wordninja.split(text))
        
        # Remove repeated chars
        text = re.sub(r"(.)\1{2,}", r"\1\1", text)
        
        # Remove numbers
        text = re.sub(r"\d+", "", text)
        
        # Spell check
        text = self.spell(text)
        
        # Lemmatization
        doc = self.nlp(text)
        text = " ".join([token.lemma_ for token in doc])
        
        # Remove extra spaces
        text = " ".join(text.split())
        
        return text
    
    def transform(self, texts):
        return [self.clean_text(text) for text in texts]


class BERTVectorizer:
    """BERT-based semantic embedding generator using sentence-transformers"""
    
    def __init__(self, model_name='bert-base-uncased'):
        self.model_name = model_name
        # Force CPU for compatibility
        self.device = 'cpu'
        self.model = None

    def fit(self, X=None, y=None):
        print(f"Loading {self.model_name} on {self.device.upper()}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        print("✅ Model loaded on CPU!")
        return self

    def transform(self, texts):
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        print(f"Encoding {len(texts)} texts to BERT embeddings...")
        embeddings = self.model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False,
            device=self.device, 
            batch_size=32
        )
        print(f"✅ Embeddings shape: {embeddings.shape}")
        return embeddings

    def fit_transform(self, texts, y=None):
        self.fit()
        return self.transform(texts)


# ================================================================
# PAGE CONFIGURATION
# ================================================================

st.set_page_config(
    page_title="Mental Health Assessment",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ================================================================
# CUSTOM CSS FOR BEAUTIFUL UI WITH BETTER READABILITY
# ================================================================

st.markdown("""
<style>
    /* Main container - solid color background instead of gradient */
    .main {
        background: #f5f7fa;
        padding: 2rem;
    }
    
    /* Card-like containers */
    .stApp {
        background: white;
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        max-width: 800px;
        margin: 0 auto;
    }
    
    /* Headers with better contrast */
    h1 {
        color: #667eea !important;
        text-align: center;
        font-weight: 700;
        margin-bottom: 1rem;
        text-shadow: none;
    }
    
    h2 {
        color: #764ba2 !important;
        font-weight: 600;
    }
    
    h3 {
        color: #2d3748 !important;
        font-weight: 500;
    }
    
    /* Regular text - ensure good contrast */
    p, span, div {
        color: #2d3748 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        transform: translateY(-2px);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Text input and text area - white background with dark text */
    .stTextInput>div>div>input,
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.75rem;
        font-size: 1rem;
        background-color: white !important;
        color: #2d3748 !important;
    }
    
    /* Selectbox - white background with dark text */
    .stSelectbox>div>div {
        background-color: white !important;
    }
    
    .stSelectbox>div>div>div {
        background-color: white !important;
        color: #2d3748 !important;
    }
    
    /* Selectbox input field */
    .stSelectbox input {
        background-color: white !important;
        color: #2d3748 !important;
    }
    
    /* Selectbox dropdown menu */
    [data-baseweb="popover"] {
        background-color: white !important;
    }
    
    [data-baseweb="menu"] {
        background-color: white !important;
    }
    
    /* Dropdown options */
    [role="option"] {
        background-color: white !important;
        color: #2d3748 !important;
    }
    
    /* Dropdown option on hover */
    [role="option"]:hover {
        background-color: #f5f7fa !important;
        color: #2d3748 !important;
    }
    
    /* Dropdown option selected */
    [aria-selected="true"] {
        background-color: #e6f0ff !important;
        color: #2d3748 !important;
    }
    
    /* Radio and select labels */
    .stRadio label,
    .stSelectbox label {
        color: #2d3748 !important;
    }
    
    /* Radio button options */
    .stRadio>div {
        background-color: white !important;
    }
    
    .stRadio>div>label>div {
        color: #2d3748 !important;
    }
    
    /* Success/Info boxes with white text */
    .stSuccess {
        background-color: #d4edda !important;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stSuccess p {
        color: #155724 !important;
    }
    
    .stInfo {
        background-color: #d1ecf1 !important;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stInfo p {
        color: #0c5460 !important;
    }
    
    .stWarning {
        background-color: #fff3cd !important;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
    }
    
    .stWarning p {
        color: #856404 !important;
    }
    
    /* Slider labels */
    .stSlider label {
        color: #2d3748 !important;
    }
    
    /* Number input labels */
    .stNumberInput label {
        color: #2d3748 !important;
    }
    
    /* Ensure all streamlit elements have readable text */
    [data-testid="stMarkdownContainer"] {
        color: #2d3748;
    }
</style>
""", unsafe_allow_html=True)
# ==============================================================
# LOAD MODEL WITH CPU MAPPING
# ==============================================================

@st.cache_resource
def load_production_model():
    """Load the trained production model and all components, forcing CPU for torch tensors"""
    try:
        # Monkey-patch torch.load to always use CPU mapping
        original_torch_load = torch.load
        
        def torch_load_cpu(*args, **kwargs):
            """Wrapper to force CPU mapping for all torch.load calls"""
            kwargs['map_location'] = torch.device('cpu')
            return original_torch_load(*args, **kwargs)
        
        # Temporarily replace torch.load
        torch.load = torch_load_cpu
        
        # Now load the model - all torch tensors will be mapped to CPU
        model_data = joblib.load("production_mental_health_model_1.pkl")
        
        # Restore original torch.load
        torch.load = original_torch_load
        
        # If the BERT vectorizer has a loaded model, ensure it's on CPU
        if 'bert_vectorizer' in model_data:
            if hasattr(model_data['bert_vectorizer'], 'model') and model_data['bert_vectorizer'].model is not None:
                model_data['bert_vectorizer'].model = model_data['bert_vectorizer'].model.to('cpu')
                model_data['bert_vectorizer'].device = 'cpu'
        
        print("✅ Model loaded successfully on CPU!")
        return model_data
        
    except FileNotFoundError:
        st.error("❌ Model file not found! Please ensure 'production_mental_health_model.pkl' is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

model_components = load_production_model()

# Extract components
hybrid_model = model_components['hybrid_model']
text_preprocessor = model_components['text_preprocessor']
bert_vectorizer = model_components['bert_vectorizer']
column_transformer = model_components['column_transformer']
text_scaler = model_components['text_scaler']
all_features = model_components['all_features']
numeric_features = model_components['numeric_features']
categorical_features = model_components['categorical_features']
target_names = model_components['target_names']
STRUCTURED_WEIGHT = model_components['weights']['structured']
TEXT_WEIGHT = model_components['weights']['text']

# ================================================================
# SESSION STATE INITIALIZATION
# ================================================================

if 'step' not in st.session_state:
    st.session_state.step = 0
    st.session_state.responses = {}
    st.session_state.completed = False

# ================================================================
# QUESTION DEFINITIONS
# ================================================================

questions = [
    {
        "key": "Age",
        "question": "👤 Let's start with the basics. How old are you?",
        "type": "number",
        "min": 10,
        "max": 100,
        "default": 25,
        "help": "Your age helps us understand life stage factors"
    },
    {
        "key": "Gender",
        "question": "🚻 What is your gender?",
        "type": "select",
        "options": ["Male", "Female", "Non-Binary", "Other"],
        "help": "This information helps personalize our assessment"
    },
    {
        "key": "Education_Level",
        "question": "🎓 What's your highest level of education?",
        "type": "select",
        "options": ["High School", "Bachelor's", "Master's", "PhD", "Other"],
        "help": "Educational background can influence stress patterns"
    },
    {
        "key": "Employment_Status",
        "question": "💼 What's your current employment status?",
        "type": "select",
        "options": ["Employed", "Unemployed", "Student", "Retired"],
        "help": "Work status affects daily routines and stress"
    },
    {
        "key": "Sleep_Hours",
        "question": "😴 On average, how many hours do you sleep per night?",
        "type": "slider",
        "min": 0.0,
        "max": 12.0,
        "default": 7.0,
        "step": 0.5,
        "help": "Sleep is crucial for mental health"
    },
    {
        "key": "Physical_Activity_Hrs",
        "question": "🏃 How many hours per week do you engage in physical activity?",
        "type": "slider",
        "min": 0.0,
        "max": 20.0,
        "default": 3.0,
        "step": 0.5,
        "help": "Exercise has proven mental health benefits"
    },
    {
        "key": "Social_Support_Score",
        "question": "👥 How would you rate your social support system?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "0 = No support, 10 = Excellent support network"
    },
    {
        "key": "Stress_Level",
        "question": "😰 How would you rate your overall stress level?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "0 = No stress, 10 = Extreme stress"
    },
    {
        "key": "Chronic_Illnesses",
        "question": "🏥 Do you have any chronic illnesses?",
        "type": "radio",
        "options": {"No": 0, "Yes": 1},
        "help": "Physical health impacts mental wellbeing"
    },
    {
        "key": "Therapy",
        "question": "💭 Are you currently in therapy or counseling?",
        "type": "radio",
        "options": {"No": 0, "Yes": 1},
        "help": "Professional support is valuable"
    },
    {
        "key": "Meditation",
        "question": "🧘 Do you practice meditation or mindfulness?",
        "type": "radio",
        "options": {"No": 0, "Yes": 1},
        "help": "Mindfulness practices support mental health"
    },
    {
        "key": "Financial_Stress",
        "question": "💰 How stressed are you about finances?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "0 = No financial stress, 10 = Extreme financial stress"
    },
    {
        "key": "Work_Stress",
        "question": "💼 How stressed are you about work or school?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "0 = No work stress, 10 = Extreme work stress"
    },
    {
        "key": "Self_Esteem_Score",
        "question": "🪞 How would you rate your self-esteem?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "0 = Very low, 10 = Very high"
    },
    {
        "key": "Life_Satisfaction_Score",
        "question": "😊 How satisfied are you with your life overall?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "0 = Very dissatisfied, 10 = Very satisfied"
    },
    {
        "key": "Loneliness_Score",
        "question": "😔 How often do you feel lonely?",
        "type": "slider",
        "min": 0,
        "max": 10,
        "default": 5,
        "help": "0 = Never lonely, 10 = Always lonely"
    },
    {
        "key": "Family_History_Mental_Illness",
        "question": "🧬 Is there a family history of mental illness?",
        "type": "radio",
        "options": {"No": 0, "Yes": 1},
        "help": "Genetic factors can influence mental health"
    },
    {
        "key": "Medication_Use",
        "question": "💊 Do you use any psychiatric medications?",
        "type": "select",
        "options": ["None", "Occasional", "Regular"],
        "help": "Medication can be part of effective treatment"
    },
    {
        "key": "Substance_Use",
        "question": "🍺 How would you describe your substance use?",
        "type": "select",
        "options": ["None", "Occasional", "Frequent"],
        "help": "Substance use patterns affect mental health"
    },
    {
        "key": "emotional_text",
        "question": "💬 Finally, please describe how you've been feeling lately. Share your thoughts, mood, worries, or anything on your mind.",
        "type": "text_area",
        "help": "Your words help us understand your emotional state better. Be as open as you'd like."
    }
]

# ================================================================
# PREDICTION FUNCTION
# ================================================================

def predict_mental_health(responses: Dict[str, Any]) -> Dict:
    """
    Make mental health prediction from user responses
    
    Args:
        responses: Dictionary with all user answers
        
    Returns:
        Dictionary with prediction and confidence scores
    """
    # Extract emotional text
    emotional_text = responses.get('emotional_text', '')
    
    # Clean and process text
    cleaned_text = text_preprocessor.clean_text(emotional_text)
    text_embedding = bert_vectorizer.transform([cleaned_text])
    text_normalized = text_scaler.transform(text_embedding)
    text_weighted = text_normalized * TEXT_WEIGHT
    
    # Prepare structured data
    structured_data = {key: responses.get(key, 0) for key in all_features}
    df_user = pd.DataFrame([structured_data])
    structured_encoded = column_transformer.transform(df_user[all_features])
    structured_weighted = structured_encoded * STRUCTURED_WEIGHT
    
    # Combine features
    combined_features = np.hstack([structured_weighted, text_weighted])
    
    # Make prediction
    prediction = hybrid_model.predict(combined_features)[0]
    probabilities = hybrid_model.predict_proba(combined_features)[0]
    
    return {
        'prediction': target_names[prediction],
        'probabilities': {
            'Normal': probabilities[0],
            'Anxious': probabilities[1],
            'Depressed': probabilities[2]
        }
    }

# ================================================================
# UI RENDERING
# ================================================================

def render_header():
    """Render the app header"""
    st.markdown("<h1>Mental Health Assessment</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666; font-size: 1.1rem;'>A compassionate AI-powered mental health screening tool</p>", unsafe_allow_html=True)
    st.markdown("---")

def render_question(question_data: Dict):
    """Render a single question based on its type"""
    question_type = question_data["type"]
    key = question_data["key"]
    
    st.markdown(f"### {question_data['question']}")
    
    if question_data.get("help"):
        st.info(f"ℹ️ {question_data['help']}")
    
    if question_type == "number":
        value = st.number_input(
            "",
            min_value=question_data["min"],
            max_value=question_data["max"],
            value=question_data["default"],
            key=f"input_{key}"
        )
    
    elif question_type == "slider":
        value = st.slider(
            "",
            min_value=question_data["min"],
            max_value=question_data["max"],
            value=question_data["default"],
            step=question_data.get("step", 1),
            key=f"input_{key}"
        )
    
    elif question_type == "select":
        value = st.selectbox(
            "",
            options=question_data["options"],
            key=f"input_{key}"
        )
    
    elif question_type == "radio":
        options = question_data["options"]
        selected = st.radio(
            "",
            options=list(options.keys()),
            key=f"input_{key}",
            horizontal=True
        )
        value = options[selected]
    
    elif question_type == "text_area":
        value = st.text_area(
            "",
            height=150,
            placeholder="Take your time... Share how you've been feeling...",
            key=f"input_{key}"
        )
    
    return value

def render_progress():
    """Render progress bar"""
    progress = (st.session_state.step + 1) / len(questions)
    st.progress(progress)
    st.markdown(f"<p style='text-align: center; color: #888;'>Question {st.session_state.step + 1} of {len(questions)}</p>", unsafe_allow_html=True)

def render_result(result: Dict):
    """Render the final prediction result"""
    prediction = result['prediction']
    probabilities = result['probabilities']
    
    # Color mapping
    color_map = {
        'Normal': '#28a745',
        'Anxious': '#ffc107',
        'Depressed': '#dc3545'
    }
    
    emoji_map = {
        'Normal': '😊',
        'Anxious': '😰',
        'Depressed': '😔'
    }
    
    st.markdown("---")
    st.markdown("## 📊 Your Assessment Results")
    
    # Main prediction
    st.markdown(f"""
    <div style='background: {color_map[prediction]}; color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;'>
        <h1 style='color: white; margin: 0;'>{emoji_map[prediction]} {prediction}</h1>
        <p style='color: white; font-size: 1.2rem; margin-top: 0.5rem;'>Based on your responses, this is your current mental health status</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confidence scores
    st.markdown("### Confidence Breakdown")
    
    for category, prob in probabilities.items():
        st.markdown(f"**{emoji_map[category]} {category}**")
        st.progress(float(prob))
        st.markdown(f"<p style='text-align: right; color: #666;'>{prob:.1%}</p>", unsafe_allow_html=True)
    
    # Disclaimer
    # st.warning("""
    # ⚠️ **Important Disclaimer**
    
    # This is an AI-powered screening tool and **NOT a medical diagnosis**. 
    
    # If you're experiencing mental health concerns:
    # - Please consult a licensed mental health professional
    # - Contact a crisis helpline if you're in immediate distress
    # - This tool is meant to provide insights, not replace professional care
    # """)
    
    # # Resources
    # st.markdown("### 📞 Mental Health Resources")
    # st.info("""
    # - **National Suicide Prevention Lifeline**: 988
    # - **Crisis Text Line**: Text HOME to 741741
    # - **SAMHSA National Helpline**: 1-800-662-4357
    # """)

# ================================================================
# MAIN APP LOGIC
# ================================================================

def main():
    render_header()
    
    if not st.session_state.completed:
        # Show progress
        render_progress()
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Get current question
        current_question = questions[st.session_state.step]
        
        # Render question
        answer = render_question(current_question)
        
        # Navigation buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.session_state.step > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.step -= 1
                    st.rerun()
        
        with col3:
            if st.session_state.step < len(questions) - 1:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.responses[current_question["key"]] = answer
                    st.session_state.step += 1
                    st.rerun()
            else:
                if st.button("Get Results", use_container_width=True):
                    st.session_state.responses[current_question["key"]] = answer
                    
                    # Validate emotional text
                    if not st.session_state.responses.get('emotional_text', '').strip():
                        st.error("Please share your feelings before getting results.")
                    else:
                        st.session_state.completed = True
                        st.rerun()
    
    else:
        # Show results
        with st.spinner("🔮 Analyzing your responses..."):
            result = predict_mental_health(st.session_state.responses)
        
        render_result(result)
        
        # Reset button
        if st.button("🔄 Take Assessment Again", use_container_width=True):
            st.session_state.step = 0
            st.session_state.responses = {}
            st.session_state.completed = False
            st.rerun()

# ================================================================
# RUN APP
# ================================================================

if __name__ == "__main__":
    main()
