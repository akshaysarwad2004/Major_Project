
# -----------------------------------------------------------------------------
# Project: Interactive Sentiment Analysis Dashboard
# Author: Gemini
# Date: 17-09-2025
#
# Description:
# FINAL STABLE VERSION - HYBRID ONLINE/OFFLINE USE
# This script uses local AI models for most tasks and an online service for
# translating native Kannada script to enable deep analysis.
# All features and bug fixes, including a fail-safe for summarization, are included.
# Sarcasm detection model added for nuanced emotion analysis.
# -----------------------------------------------------------------------------

# --- 1. SETUP AND IMPORTS ---
import streamlit as st
import re
import pandas as pd
from transformers import pipeline
import warnings
import time
import nltk
from textblob import TextBlob
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from bertopic import BERTopic
import os
import plotly.express as px
import emoji
from collections import Counter
from googletrans import Translator

# Suppress warnings and set configurations
warnings.filterwarnings("ignore", category=UserWarning)
DetectorFactory.seed = 0

# --- Define the path to your local models ---
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'local_models')

# --- 2. ONE-TIME SETUP AND DATA LOADING (CACHED FOR PERFORMANCE) ---

@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data if not already present."""
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        nltk.download('stopwords')

@st.cache_resource
def load_sentiment_pipeline():
    """Loads the sentiment classification AI model from the local folder."""
    model_path = os.path.join(LOCAL_MODEL_PATH, "cardiffnlp_twitter-roberta-base-sentiment-latest")
    if not os.path.exists(model_path) or not os.listdir(model_path): return None
    try:
        return pipeline("sentiment-analysis", model=model_path)
    except Exception as e:
        st.error(f"Error loading sentiment model: {e}")
        return None

@st.cache_resource
def load_summarizer_pipeline():
    """Loads the summarization AI model from the local folder."""
    model_path = os.path.join(LOCAL_MODEL_PATH, "facebook_bart-large-cnn")
    if not os.path.exists(model_path) or not os.listdir(model_path): return None
    try:
        return pipeline("summarization", model=model_path)
    except Exception as e:
        st.error(f"Error loading summarizer model: {e}")
        return None

@st.cache_resource
def load_emotion_pipeline():
    """Loads the emotion classification AI model from the local folder."""
    model_path = os.path.join(LOCAL_MODEL_PATH, "SamLowe_roberta-base-go_emotions")
    if not os.path.exists(model_path) or not os.listdir(model_path): return None
    try:
        return pipeline("text-classification", model=model_path)
    except Exception as e:
        st.error(f"Error loading emotion model: {e}")
        return None

@st.cache_resource
def load_sarcasm_pipeline():
    """Loads the sarcasm detection AI model from the local folder."""
    model_path = os.path.join(LOCAL_MODEL_PATH, "helinivan_english-sarcasm-detector")
    if not os.path.exists(model_path) or not os.listdir(model_path): return None
    try:
        return pipeline("text-classification", model=model_path)
    except Exception as e:
        st.error(f"Error loading sarcasm model: {e}")
        return None

@st.cache_resource
def load_safe_list_from_file(filepath):
    """Loads the Kannada word list from a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            words = {line.strip().lower() for line in f if line.strip()}
        return words
    except FileNotFoundError:
        st.warning(f"Vocabulary file not found: {filepath}.")
        return set()

# --- 3. CORE DATA PROCESSING FUNCTIONS (CACHED FOR SPEED) ---

def preprocess_chat_content(file_content):
    start_pattern = re.compile(r'^(?:\u200e)?(?:\[?\d{1,2}[./-]\d{1,2}[./-]\d{2,4},? \d{1,2}:\d{2})')
    lines = file_content.split('\n')
    first_message_line_index = -1
    for i, line in enumerate(lines):
        if start_pattern.match(line):
            first_message_line_index = i
            break
    if first_message_line_index != -1:
        return '\n'.join(lines[first_message_line_index:])
    else:
        return file_content

@st.cache_data
def parse_whatsapp_chat(uploaded_file_content):
    patterns_to_try = [
        re.compile(r'\[(\d{1,2}-\d{1,2}-\d{4}, \d{1,2}:\d{2}\s*?(?:AM|PM|am|pm))\]\s([^:]+): ([\s\S]+?)(?=\n\[\d{1,2}-\d{1,2}-\d{4},|$)'),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s*?(?:AM|PM|am|pm))\s-\s([^:]+): ([\s\S]+?)(?=\n\d{1,2}/\d{1,2}/\d{2,4},|$)'),
        re.compile(r'\[(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}:\d{2}\s*?(?:AM|PM|am|pm))\]\s([^:]+): ([\s\S]+?)(?=\n\[\d{1,2}/\d{1,2}/\d{2,4},|$)'),
        re.compile(r'\[(\d{1,2}\.\d{1,2}\.\d{2,4}, \d{1,2}:\d{2}:\d{2})\]\s([^:]+): ([\s\S]+?)(?=\n\[\d{1,2}\.\d{1,2}\.\d{2,4},|$)'),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2})\s-\s([^:]+): ([\s\S]+?)(?=\n\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}\s-\s|$)'),
        re.compile(r'(\d{4}-\d{2}-\d{2}, \d{1,2}:\d{2})\s-\s([^:]+): ([\s\S]+?)(?=\n\d{4}-\d{2}-\d{2},|$)'),
        re.compile(r'(\d{1,2}\.\d{1,2}\.\d{2,4}, \d{1,2}:\d{2})\s-\s([^:]+): ([\s\S]+?)(?=\n\d{1,2}\.\d{1,2}\.\d{2,4},|$)'),
        re.compile(r'([A-Z][a-z]+ \d{1,2}, \d{4}, \d{1,2}:\d{2}\s*?(?:AM|PM|am|pm))\s-\s([^:]+): ([\s\S]+?)(?=\n[A-Z][a-z]+ \d{1,2}, \d{4},|$)'),
        re.compile(r'(\d{1,2}/\d{1,2}/\d{4}, \d{2}:\d{2})\s-\s([^:]+): ([\s\S]+?)(?=\n\d{1,2}/\d{1,2}/\d{4}, \d{2}:\d{2}\s-\s|$)'),
    ]
    matches = []
    for pattern in patterns_to_try:
        matches = pattern.findall(uploaded_file_content)
        if matches: break
    if not matches: return None
    df = pd.DataFrame(matches, columns=['timestamp_str', 'user', 'message'])
    df['timestamp'] = pd.to_datetime(df['timestamp_str'], errors='coerce', dayfirst=True)
    df = df.dropna(subset=['timestamp'])
    df['message'] = df['message'].str.strip()
    df = df[~df['message'].str.contains("omitted", case=False, na=False)]
    return df[['timestamp', 'user', 'message']]

@st.cache_data
def translate_kannada_messages(_df, message_column='message'):
    df = _df.copy()
    df['text_for_analysis'] = df[message_column]
    kan_mask = df['language'] == 'kn'
    kan_messages = df.loc[kan_mask, message_column].tolist()
    if not kan_messages: return df
    try:
        translator = Translator()
        unique_kan_messages = list(set(kan_messages))
        progress_bar = st.sidebar.progress(0, text="Translating Kannada messages...")
        translation_map = {}
        for i, text in enumerate(unique_kan_messages):
            translation = translator.translate(text, src='kn', dest='en')
            translation_map[text] = translation.text
            time.sleep(0.1)
            progress_bar.progress((i + 1) / len(unique_kan_messages), text=f"Translating Kannada ({i+1}/{len(unique_kan_messages)})...")
        progress_bar.empty()
        df.loc[kan_mask, 'text_for_analysis'] = df.loc[kan_mask, message_column].map(translation_map)
        st.sidebar.info(f"Successfully translated {len(kan_messages)} Kannada messages.")
    except Exception as e:
        st.sidebar.warning(f"Translation failed. Kannada messages will have limited analysis. Error: {e}")
    return df

@st.cache_data
def analyze_emojis(_df):
    _df['emojis'] = _df['message'].apply(lambda text: [e['emoji'] for e in emoji.emoji_list(text)])
    all_emojis = [e for row_emojis in _df['emojis'] for e in row_emojis]
    total_emoji_count = len(all_emojis)
    top_emojis_df = pd.DataFrame(Counter(all_emojis).most_common(10), columns=['Emoji', 'Count'])
    user_emoji_list = [(row['user'], e) for index, row in _df.iterrows() for e in row['emojis']]
    user_emojis_df = pd.DataFrame(user_emoji_list, columns=['User', 'Emoji'])
    user_emoji_counts = user_emojis_df.groupby('User')['Emoji'].count().sort_values(ascending=False).reset_index()
    return total_emoji_count, top_emojis_df, user_emoji_counts

@st.cache_data
def detect_language(_df, _kannada_safe_list):
    df = _df.copy()
    languages = []
    for msg in df['message']:
        words_in_message = set(re.findall(r'\w+', msg.lower()))
        if any(word in _kannada_safe_list for word in words_in_message):
            languages.append('kanglish')
            continue
        try:
            if len(str(msg).split()) > 0: languages.append(detect(msg))
            else: languages.append('unknown')
        except LangDetectException: languages.append('unknown')
    df['language'] = languages
    allowed_languages = ['en', 'kanglish', 'kn']
    df['language'] = df['language'].apply(lambda x: x if x in allowed_languages else 'other')
    return df

@st.cache_data
def intelligent_correct_spelling(_df, user_names, _kannada_safe_list):
    df = _df.copy()
    dynamic_safe_list = set(_kannada_safe_list)
    for name in user_names:
        dynamic_safe_list.add(name.lower())
        for part in name.split(): dynamic_safe_list.add(part.lower())
    corrected_messages = []
    for _, row in df.iterrows():
        if row.get('language', 'unknown') == 'en':
            try:
                words = TextBlob(row['message']).words
                corrected_sentence = [str(word) if word.lower() in dynamic_safe_list else str(word.correct()) for word in words]
                corrected_messages.append(" ".join(corrected_sentence))
            except Exception: corrected_messages.append(row['message'])
        else: corrected_messages.append(row['message'])
    df['message_corrected'] = corrected_messages
    return df

@st.cache_data
def run_sentiment_analysis(_df, _sentiment_pipeline, message_column='text_for_analysis'):
    df = _df.copy()
    text_list = [text[:512] for text in _df[message_column].astype(str).tolist()]
    results = _sentiment_pipeline(text_list, batch_size=16, truncation=True)
    df['sentiment'] = [res['label'].lower() for res in results]
    df['sentiment_score'] = [res['score'] for res in results]
    return df

@st.cache_data
def run_emotion_analysis(_df, _emotion_pipeline, message_column='text_for_analysis'):
    df = _df.copy()
    text_to_analyze = df[df['language'].isin(['en', 'kanglish', 'kn'])][message_column].astype(str).tolist()
    if not text_to_analyze:
        df['emotion'] = 'N/A'
        return df
    truncated_text_to_analyze = [text[:512] for text in text_to_analyze]
    results = _emotion_pipeline(truncated_text_to_analyze, batch_size=16, truncation=True, top_k=1)
    emotion_map = {text: res[0]['label'] for text, res in zip(text_to_analyze, results)}
    df['emotion'] = df[message_column].map(emotion_map).fillna('N/A')
    return df

@st.cache_data
def run_sarcasm_analysis(_df, _sarcasm_pipeline, message_column='text_for_analysis'):
    df = _df.copy()
    text_to_analyze = df[df['language'].isin(['en', 'kanglish'])][message_column].astype(str).tolist()
    if not text_to_analyze:
        df['is_sarcastic'] = False
        return df
    
    truncated_text = [text[:512] for text in text_to_analyze]
    results = _sarcasm_pipeline(truncated_text, batch_size=16, truncation=True)
    
    sarcasm_map = {text: res['label'].lower() == 'sarcasm' for text, res in zip(text_to_analyze, results)}
    df['is_sarcastic'] = df[message_column].map(sarcasm_map).fillna(False)
    return df

# A set of strong negative words that often indicate sarcasm when paired with 'joy'
SARCASM_TRIGGER_WORDS = {'crashed', 'failed', 'error', 'broke', 'disaster', 'ruined', 'worst'}

def reconcile_emotions(row):
    """Corrects emotion based on sentiment, sarcasm detection, and keywords."""
    sentiment = row['sentiment']
    emotion = row['emotion']
    message = row['message'].lower()
    is_sarcastic = row.get('is_sarcastic', False)

    # Primary check: Use the sarcasm model's prediction
    if is_sarcastic and sentiment == 'negative' and emotion == 'joy':
        return 'sarcasm (inferred)'

    # Secondary check (Safety Net): Use keywords if the model missed it
    contains_trigger_word = any(word in message for word in SARCASM_TRIGGER_WORDS)
    if sentiment == 'negative' and emotion == 'joy' and contains_trigger_word:
        return 'sarcasm (inferred)'

    # Otherwise, keep the original emotion
    return emotion

@st.cache_data
def run_bertopic_modeling(_df, message_column='text_for_analysis'):
    df = _df.copy()
    messages = df[message_column].astype(str).tolist()
    messages = [msg[:384] for msg in messages]
    messages = [msg for msg in messages if msg and msg.strip()]
    n_messages = len(messages)
    model_path = os.path.join(LOCAL_MODEL_PATH, "sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2")
    if not os.path.exists(model_path) or not os.listdir(model_path):
        st.warning("BERTopic model not found. Skipping topic modeling.")
        return df, None
    if n_messages < 4:
        st.info(f"Topic modeling requires at least 4 messages to run (found {n_messages}). Skipping.")
        df['topic'] = 'N/A'
        return df, None
    try:
        from sklearn.decomposition import PCA
        dim_model = PCA(n_components=min(5, n_messages - 1))
        topic_model = BERTopic(embedding_model=model_path, umap_model=dim_model, verbose=False, min_topic_size=2)
        topics, _ = topic_model.fit_transform(messages)
        df['topic'] = [topic_model.get_topic_info(topic)['Name'].iloc[0] for topic in topics]
        return df, topic_model
    except Exception as e:
        st.warning(f"BERTopic modeling failed. Error: {e}")
        df['topic'] = 'N/A'
        return df, None

def to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 4. STREAMLIT UI AND APPLICATION FLOW ---
st.set_page_config(layout="wide", page_title="WhatsApp Chat Analyzer")
st.title("🎓 WhatsApp Educational Chat Analyzer")
st.markdown("Upload your exported WhatsApp .txt file for a deep analysis.")

download_nltk_data()

with st.spinner("Loading AI models from local files..."):
    sentiment_pipeline = load_sentiment_pipeline()
    summarizer_pipeline = load_summarizer_pipeline()
    emotion_pipeline = load_emotion_pipeline()
    sarcasm_pipeline = load_sarcasm_pipeline()

kannada_word_file = os.path.join(os.path.dirname(__file__), 'kannada_words.txt')
kannada_safe_list = load_safe_list_from_file(kannada_word_file)

st.sidebar.title("Controls")
uploaded_file = st.sidebar.file_uploader("1. Upload your chat file", type="txt")

if uploaded_file:
    raw_content = uploaded_file.getvalue().decode("utf-8")
    cleaned_content = preprocess_chat_content(raw_content)
    data_df = parse_whatsapp_chat(cleaned_content)
    
    if data_df is None:
        st.error("Could not parse the chat file. Please ensure it's a valid WhatsApp export.")
    else:
        st.sidebar.success(f"Successfully loaded {len(data_df)} messages.")
        st.sidebar.subheader("2. Analysis Options")
        enable_spelling_correction = st.sidebar.checkbox("Enable Spelling Correction (Slower)")
        enable_topic_modeling = st.sidebar.checkbox("Enable Topic Modeling (Slowest)")
        
        if st.sidebar.button("🚀 Run Full Analysis"):
            analysis_df = data_df.copy()
            
            with st.spinner("Detecting languages..."):
                analysis_df = detect_language(analysis_df, kannada_safe_list)

            with st.spinner("Translating Kannada messages (requires internet)..."):
                analysis_df = translate_kannada_messages(analysis_df)
            
            if enable_spelling_correction:
                with st.spinner("Correcting spelling..."):
                    user_names = analysis_df['user'].unique().tolist()
                    analysis_df = intelligent_correct_spelling(analysis_df, user_names, kannada_safe_list)
            
            analysis_col = 'text_for_analysis'
            
            with st.spinner("Analyzing sentiments..."):
                analysis_df = run_sentiment_analysis(analysis_df, sentiment_pipeline, analysis_col)
            
            if emotion_pipeline:
                with st.spinner("Detecting emotions..."):
                    analysis_df = run_emotion_analysis(analysis_df, emotion_pipeline, analysis_col)
            
            if sarcasm_pipeline:
                with st.spinner("Detecting sarcasm..."):
                    analysis_df = run_sarcasm_analysis(analysis_df, sarcasm_pipeline, analysis_col)
                
                # Reconcile emotions after all analyses are done
                analysis_df['emotion_corrected'] = analysis_df.apply(reconcile_emotions, axis=1)

            with st.spinner("Analyzing emojis..."):
                total_emojis, top_emojis, user_emojis = analyze_emojis(analysis_df)
                st.session_state['emoji_stats'] = (total_emojis, top_emojis, user_emojis)

            analysis_df['is_question'] = analysis_df['message'].str.contains(r'\?', regex=True, na=False)
            concern_keywords = {
                'confused', 'help', 'problem', 'issue', 'doubt', 'stuck', 'error', 'unclear', 'difficult', 
                'hard', 'trouble', 'challenge', 'struggling', 'question', 'query', 'clarification', 'explain', 
                'understand', 'bug', 'fail', 'crash', 'not working', 'won\'t run', 'exception', 'traceback', 
                'syntax error', 'logic error', 'doesn\'t work', 'don\'t get', 'do not get', 'am I wrong', 
                'is this right', 'need assistance', 'can someone', 'how to'
            }
            keyword_match = analysis_df['text_for_analysis'].str.contains('|'.join(concern_keywords), case=False, na=False)
            negative_sentiment = analysis_df['sentiment'] == 'negative'
            analysis_df['key_concern'] = keyword_match | negative_sentiment
            
            topic_model = None
            if enable_topic_modeling:
                with st.spinner("Modeling topics..."):
                    analysis_df, topic_model = run_bertopic_modeling(analysis_df, analysis_col)

            st.session_state['analysis_df'] = analysis_df
            st.session_state['topic_model'] = topic_model
            st.success("Analysis Complete!")

if 'analysis_df' in st.session_state:
    analysis_df = st.session_state['analysis_df']
    topic_model = st.session_state.get('topic_model')
    
    st.header("📊 Dashboard")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Messages", len(analysis_df))
    col2.metric("Unique Users", analysis_df['user'].nunique())
    col3.metric("Questions Asked", int(analysis_df['is_question'].sum()))
    col4.metric("Key Concerns", int(analysis_df['key_concern'].sum()))

    st.sidebar.download_button("📥 Download Report (CSV)", to_csv(analysis_df), f"analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    st.markdown("---")

    tabs_list = ["📈 Overview", "🎭 Sentiment Analysis", "👤 User Analysis", "😀 Emoji Analysis", "😲 Emotion Analysis"]
    if topic_model:
        tabs_list.append("📚 Topic Analysis")
    tabs_list.append("📋 Raw Data")
    tabs = st.tabs(tabs_list)

    tab_index = 0
    with tabs[tab_index]: # Overview
        st.subheader("Language Distribution")
        lang_counts = analysis_df['language'].value_counts()
        if not lang_counts.empty:
            fig = px.pie(lang_counts, values=lang_counts.values, names=lang_counts.index, hole=0.4, title="Language Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Chat Activity Heatmap")
        activity = analysis_df.pivot_table(index=analysis_df['timestamp'].dt.day_name(), columns=analysis_df['timestamp'].dt.hour, values='message', aggfunc='count', fill_value=0)
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        activity = activity.reindex(day_order, fill_value=0).astype(int)
        fig_heatmap = px.imshow(activity, text_auto=True, aspect="auto", title="Chat Activity Heatmap", color_continuous_scale='Viridis')
        st.plotly_chart(fig_heatmap, use_container_width=True)

    tab_index += 1
    with tabs[tab_index]: # Sentiment
        st.subheader("Overall Sentiment Distribution")
        sentiment_counts = analysis_df['sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
        st.subheader("Messages Flagged with Key Concerns")
        concern_df = analysis_df[analysis_df['key_concern']][['user', 'message', 'sentiment', 'emotion']]
        st.dataframe(concern_df)

    tab_index += 1
    with tabs[tab_index]: # User
        st.subheader("User Leaderboard")
        user_stats = analysis_df.groupby('user').agg(Total_Messages=('message', 'count'), Questions=('is_question', 'sum'), Key_Concerns=('key_concern', 'sum')).sort_values(by="Total_Messages", ascending=False)
        st.dataframe(user_stats)
        st.subheader("User Sentiment Breakdown")
        for user in user_stats.index:
            with st.expander(f"Sentiments for {user}"):
                user_sentiments = analysis_df[analysis_df['user'] == user]['sentiment'].value_counts()
                if not user_sentiments.empty: st.bar_chart(user_sentiments)
                else: st.write("No sentiments detected.")

    tab_index += 1
    with tabs[tab_index]: # Emoji
        st.subheader("Emoji Analysis")
        if 'emoji_stats' in st.session_state:
            total_emojis, top_emojis, user_emojis = st.session_state['emoji_stats']
            st.metric("Total Emojis Used", total_emojis)
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Most Used Emojis")
                st.dataframe(top_emojis)
            with c2:
                st.subheader("Emoji Leaderboard")
                st.dataframe(user_emojis)

    tab_index += 1
    with tabs[tab_index]: # Emotion
        st.subheader("Emotion Distribution")
        
        emotion_col_to_display = 'emotion_corrected' if 'emotion_corrected' in analysis_df.columns else 'emotion'
        
        if emotion_col_to_display in analysis_df.columns:
            emotion_counts = analysis_df[analysis_df[emotion_col_to_display] != 'N/A'][emotion_col_to_display].value_counts()
            if not emotion_counts.empty:
                fig_emotion = px.bar(emotion_counts, x=emotion_counts.index, y=emotion_counts.values, labels={'x':'Emotion', 'y':'Count'}, title="Overall Emotion Distribution (Corrected for Sarcasm)")
                st.plotly_chart(fig_emotion, use_container_width=True)
            else:
                st.info("No specific emotions were detected.")
            st.subheader("Messages by Detected Emotion")
            available_emotions = [e for e in analysis_df[emotion_col_to_display].unique() if e != 'N/A']
            if available_emotions:
                selected_emotion = st.selectbox("Select an emotion to view messages:", available_emotions)
                emotion_df = analysis_df[analysis_df[emotion_col_to_display] == selected_emotion][['user', 'message']]
                st.dataframe(emotion_df)

    tab_index += 1
    if topic_model:
        with tabs[tab_index]: # Topic
            st.subheader("Discovered Conversation Topics")
            topic_info = topic_model.get_topic_info()
            topic_info = topic_info[topic_info.Topic != -1]
            st.dataframe(topic_info[['Name', 'Count']], hide_index=True)
            if summarizer_pipeline:
                translator = Translator()
                for _, row in topic_info.iterrows():
                    topic_name = row['Name']
                    with st.expander(f"Explore Topic: {topic_name.split('_', 1)[1]}"):
                        topic_df = analysis_df[analysis_df['topic'] == topic_name]
                        st.write(f"*Messages in this topic:* {len(topic_df)}")
                        language_counts = topic_df['language'].value_counts(normalize=True)
                        english_fraction = language_counts.get('en', 0.0) + language_counts.get('kanglish', 0.0)
                        
                        summarization_text = ""
                        if english_fraction > 0.5:
                            message_col_to_use = 'message_corrected' if enable_spelling_correction else 'message'
                            topic_text = " ".join(topic_df[message_col_to_use].astype(str))
                            summarization_text = re.sub(r'[^\x00-\x7F]+', ' ', topic_text)
                        else:
                            with st.spinner("Translating topic for summarization..."):
                                try:
                                    topic_text_to_translate = " ".join(topic_df['message'].astype(str))
                                    translated_topic = translator.translate(topic_text_to_translate[:4096], src='kn', dest='en')
                                    summarization_text = translated_topic.text
                                    st.info("Topic translated to English for summarization.")
                                except Exception as e:
                                    st.warning(f"Could not translate this topic for summarization. Error: {e}")
                        
                        if summarization_text:
                            with st.spinner("Generating summary..."):
                                safe_text = summarization_text[:4096]
                                if safe_text and safe_text.strip() and len(safe_text.split()) > 25:
                                    try:
                                        min_len = 25
                                        word_count = len(safe_text.split())
                                        
                                        dynamic_max_length = min(100, max(min_len + 5, word_count // 2))
                                        
                                        if dynamic_max_length > word_count:
                                            dynamic_max_length = word_count
                                        
                                        summary = summarizer_pipeline(safe_text, max_length=dynamic_max_length, min_length=min_len, do_sample=False)[0]['summary_text']
                                        st.success(f"*Summary:* {summary}")
                                    except IndexError:
                                        st.warning("Could not generate a summary for this topic due to an internal model error.")
                                else:
                                    st.info("Not enough text to summarize.")

                        st.dataframe(topic_df[['user', 'message', 'sentiment', 'emotion']])
        tab_index += 1
                        
    with tabs[tab_index]: # Raw Data
        st.subheader("Full Analyzed Data")
        st.dataframe(analysis_df)
