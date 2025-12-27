import streamlit as st
import pandas as pd
import numpy as np
import json
import re
from collections import defaultdict, Counter
import math
import os
from tqdm import tqdm
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from urllib.parse import quote

# Page Configuration
st.set_page_config(
    page_title="JawaSeek - CLIR Indonesia-Jawa",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 50%, #60a5fa 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #64748b;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .method-title {
        background: linear-gradient(90deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        font-size: 1.3rem;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .translated-query {
        background-color: #f1f5f9;
        padding: 1rem;
        border-left: 4px solid #3b82f6;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        font-weight: bold;
        font-size: 1.1rem;
        padding: 0.75rem 2rem;
        border-radius: 0.5rem;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .dataframe {
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .sidebar-header {
        background: linear-gradient(90deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #dbeafe;
        border-left: 4px solid #3b82f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Dark mode adjustments */
    @media (prefers-color-scheme: dark) {
        body, .stApp {
            background-color: #0b1220 !important;
            color: #e6eef8 !important;
        }
        .main-header {
            -webkit-text-fill-color: unset !important;
        }
        .sub-header { color: #cbd5e1 !important; }
        .method-title { color: #e6eef8 !important; }
        .translated-query {
            background-color: rgba(148,163,184,0.06) !important;
            border-left: 4px solid rgba(59,130,246,0.6) !important;
            color: #e6eef8 !important;
        }
        .info-box {
            background-color: rgba(59,130,246,0.08) !important;
            border-left: 4px solid rgba(59,130,246,0.6) !important;
            color: #e6eef8 !important;
        }
        .warning-box {
            background-color: rgba(245,158,11,0.06) !important;
            border-left: 4px solid rgba(245,158,11,0.6) !important;
            color: #e6eef8 !important;
        }
        .stButton>button {
            background: linear-gradient(90deg, #1f6feb 0%, #0b5ed7 100%) !important;
            color: #fff !important;
        }
        .dataframe, .sidebar-header { color: #e6eef8 !important; }
        a { color: #93c5fd !important; }
        p, div, span { color: #e6eef8 !important; }
    }
    </style>
""", unsafe_allow_html=True)

# Load dataset
DATA_FILE = 'clir_wikidata_id-jv.json'
with open(DATA_FILE, 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# Wikipedia redirect
WIKI_BASE_URL = "https://jv.wikipedia.org/wiki/"

# Preprocessing tools
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopwords_id = stopword_factory.get_stop_words()
stopwords_jv = set([
    'lan', 'ing', 'kang', 'iku', 'ana', 'apa', 'iki', 'kuwi',
    'saka', 'marang', 'menyang', 'amarga', 'dadi', 'nanging',
    'utawa', 'yen', 'karo', 'kanthi', 'supaya', 'dening',
    'wong', 'sing', 'dhéwé', 'kabèh', 'sawetara', 'liyane'
])

def preprocess_text(text, language='id'):
    text = text.lower()
    text = re.sub(r'[^a-zàáâãäåèéêëìíîïòóôõöùúûüýÿāăąćčďđēėęěğģīįķĺľļłńňņŋōőœŕřśšşťţūůűųźżž\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = text.split()
    if language == 'id':
        tokens = [t for t in tokens if t not in stopwords_id and len(t) > 2]
    else:
        tokens = [t for t in tokens if t not in stopwords_jv and len(t) > 2]
    return ' '.join(tokens)

def build_translation_dict_from_csv(dict_df):
    """Build translation dictionary from CSV with Indonesia-Javanese mappings"""
    translation_dict = {}
    for _, row in dict_df.iterrows():
        indo = str(row["Indonesia"]).strip().lower()
        jv = str(row["Javanese"]).strip().lower()
        # Pisahkan jika ada beberapa terjemahan dipisah titik koma atau koma
        jv_words = [w.strip() for w in re.split(r'[;,]', jv) if w.strip()]
        if indo and jv_words:
            translation_dict[indo] = jv_words
    return translation_dict

def translate_query(query, trans_dict):
    processed_query = preprocess_text(query, 'id')
    query_words = processed_query.split()
    translated_words = []
    for word in query_words:
        if word in trans_dict and trans_dict[word]:
            translated_words.extend(trans_dict[word])
        else:
            translated_words.append(word)
    return ' '.join(translated_words)

def search(query, trans_dict, tfidf_vectorizer, tfidf_matrix, df, top_k=5):
    translated_query = translate_query(query, trans_dict)
    query_vector = tfidf_vectorizer.transform([translated_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'doc_id': idx,
            'score': similarities[idx],
            'id_title': df.iloc[idx]['id_title'],
            'jv_title': df.iloc[idx]['jv_title'],
            'category': df.iloc[idx]['category'],
            'jv_text_preview': df.iloc[idx]['jv_text'][:200] + '...'
        })
    return results, translated_query

def build_wikipedia_url(title):
    normalized_title = title.strip().replace(' ', '_')
    return WIKI_BASE_URL + quote(normalized_title, safe='_/')

def render_search_results(section_title, results, translated_query=None):
    st.markdown(f'<div class="method-title">{section_title}</div>', unsafe_allow_html=True)
    if translated_query:
        st.markdown(f'<div class="translated-query"><strong>Query Terjemahan:</strong> {translated_query}</div>', unsafe_allow_html=True)
    if not results:
        st.warning('⚠️ Tidak ada hasil ditemukan')
        return
    for res in results:
        url = build_wikipedia_url(res['jv_title'])
        st.markdown(
            f"""
            <div style='border:1px solid #e2e8f0; border-radius:0.75rem; padding:1rem; margin-bottom:1rem; box-shadow:0 2px 10px rgba(15,23,42,0.05);'>
                <a href='{url}' target='_blank' style='font-size:1.25rem; font-weight:600; color:#1d4ed8; text-decoration:none;'>
                    {res['rank']}. {res['jv_title']}
                </a>
                <div style='margin-top:0.25rem; color:#475569; font-size:0.9rem;'>Kategori: {res['category']} · Skor: {res['score']:.2f}</div>
                <div style='margin-top:0.75rem; color:#0f172a;'>{res['jv_text_preview']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

def get_mbert_embedding(text, tokenizer, model, device, max_length=512):
    inputs = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    embeddings = sum_embeddings / sum_mask
    return embeddings.cpu().numpy()[0]

def search_mbert(query, tokenizer, model, jv_embeddings, df, device, top_k=5):
    query_emb = get_mbert_embedding(query, tokenizer, model, device)
    similarities = cosine_similarity([query_emb], jv_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'doc_id': idx,
            'score': similarities[idx],
            'id_title': df.iloc[idx]['id_title'],
            'jv_title': df.iloc[idx]['jv_title'],
            'category': df.iloc[idx]['category'],
            'jv_text_preview': df.iloc[idx]['jv_text'][:200] + '...'
        })
    return results

def search_labse(query, model, jv_embeddings, df, top_k=5):
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, jv_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'doc_id': idx,
            'score': similarities[idx],
            'id_title': df.iloc[idx]['id_title'],
            'jv_title': df.iloc[idx]['jv_title'],
            'category': df.iloc[idx]['category'],
            'jv_text_preview': df.iloc[idx]['jv_text'][:200] + '...'
        })
    return results

# Load dictionary from GitHub CSV
@st.cache_data()
def load_translation_dict():
    dict_path = "https://raw.githubusercontent.com/nsulistiyawan/sastra-jawa/refs/heads/master/csv/dictionary.csv"
    dict_df = pd.read_csv(dict_path)
    dict_df = dict_df.dropna(subset=["Indonesia", "Javanese"])
    trans_dict = build_translation_dict_from_csv(dict_df)
    return trans_dict

# Preprocess all docs
@st.cache_data()
def preprocess_all():
    df['id_processed'] = df['id_text'].apply(lambda x: preprocess_text(x, 'id'))
    df['jv_processed'] = df['jv_text'].apply(lambda x: preprocess_text(x, 'jv'))
    trans_dict = load_translation_dict()
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
    jv_tfidf_matrix = tfidf_vectorizer.fit_transform(df['jv_processed'])
    return df, trans_dict, tfidf_vectorizer, jv_tfidf_matrix

df, trans_dict, tfidf_vectorizer, jv_tfidf_matrix = preprocess_all()

# Load mBERT
@st.cache_resource()
def load_mbert():
    model_name = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_mbert()

# Encode all jv docs with mBERT
@st.cache_data()
def encode_jv_mbert():
    jv_embeddings = []
    for text in df['jv_text']:
        text_truncated = text[:2000]
        emb = get_mbert_embedding(text_truncated, tokenizer, model, device)
        jv_embeddings.append(emb)
    return np.array(jv_embeddings)

jv_embeddings = encode_jv_mbert()

# Load LaBSE
@st.cache_resource()
def load_labse():
    return SentenceTransformer('sentence-transformers/LaBSE')

labse_model = load_labse()

@st.cache_data()
def encode_jv_labse():
    return np.array(labse_model.encode(df['jv_text'].tolist(), batch_size=32, show_progress_bar=False))

jv_labse_embeddings = encode_jv_labse()

def search_labse_ft(query, model, jv_embeddings, df, top_k=5):
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, jv_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'doc_id': idx,
            'score': similarities[idx],
            'id_title': df.iloc[idx]['id_title'],
            'jv_title': df.iloc[idx]['jv_title'],
            'category': df.iloc[idx]['category'],
            'jv_text_preview': df.iloc[idx]['jv_text'][:200] + '...'
        })
    return results

def search_mbert_ft(query, model, jv_embeddings, df, top_k=5):
    query_emb = model.encode([query])
    similarities = cosine_similarity(query_emb, jv_embeddings).flatten()
    top_indices = similarities.argsort()[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'rank': len(results) + 1,
            'doc_id': idx,
            'score': similarities[idx],
            'id_title': df.iloc[idx]['id_title'],
            'jv_title': df.iloc[idx]['jv_title'],
            'category': df.iloc[idx]['category'],
            'jv_text_preview': df.iloc[idx]['jv_text'][:200] + '...'
        })
    return results

def ensemble_late_fusion(query, top_k=5, weights=None, use_ft=False):
    """Late fusion ensemble: weighted combination of scores from multiple methods"""
    if weights is None:
        weights = {'dict': 0.20, 'mbert': 0.35, 'labse': 0.45}
    
    # Get results from each method
    results_dict, _ = search(query, trans_dict, tfidf_vectorizer, jv_tfidf_matrix, df, top_k=top_k)
    
    if use_ft and mbert_ft_model is not None and jv_mbert_ft_embeddings is not None:
        results_mbert = search_mbert_ft(query, mbert_ft_model, jv_mbert_ft_embeddings, df, top_k=top_k)
    else:
        results_mbert = search_mbert(query, tokenizer, model, jv_embeddings, df, device, top_k=top_k)
    
    if use_ft and labse_ft_model is not None and jv_labse_ft_embeddings is not None:
        results_labse = search_labse_ft(query, labse_ft_model, jv_labse_ft_embeddings, df, top_k=top_k)
    else:
        results_labse = search_labse(query, labse_model, jv_labse_embeddings, df, top_k=top_k)
    
    # Combine scores
    score_map = {}
    meta_map = {}
    
    for r in results_dict:
        score_map[r['doc_id']] = score_map.get(r['doc_id'], 0) + weights.get('dict', 0) * r['score']
        meta_map[r['doc_id']] = r
    
    for r in results_mbert:
        score_map[r['doc_id']] = score_map.get(r['doc_id'], 0) + weights.get('mbert', 0) * r['score']
        meta_map[r['doc_id']] = r
    
    for r in results_labse:
        score_map[r['doc_id']] = score_map.get(r['doc_id'], 0) + weights.get('labse', 0) * r['score']
        meta_map[r['doc_id']] = r
    
    # Rank by combined score
    ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)[:top_k]
    
    # Build output
    output = []
    for rank, (doc_id, score) in enumerate(ranked, 1):
        base = meta_map[doc_id].copy()
        base['rank'] = rank
        base['score'] = score
        output.append(base)
    
    return output

# Load fine-tuned models (before sidebar to avoid "not defined" errors)
@st.cache_resource()
def load_labse_ft():
    try:
        return SentenceTransformer("HyacinthiaIca/LaBSE-Indonesia-Jawa-Wikipedia")
    except Exception as e:
        return None

@st.cache_data()
def encode_jv_labse_ft(_labse_ft_model):
    if _labse_ft_model is not None:
        return np.array(_labse_ft_model.encode(df['jv_text'].tolist(), batch_size=32, show_progress_bar=False))
    return None

@st.cache_resource()
def load_mbert_ft():
    try:
        return SentenceTransformer("HyacinthiaIca/mBERT-Indonesia-Jawa-Wikipedia")
    except Exception as e:
        return None

@st.cache_data()
def encode_jv_mbert_ft(_mbert_ft_model):
    if _mbert_ft_model is not None:
        return np.array(_mbert_ft_model.encode(df['jv_text'].tolist(), batch_size=32, show_progress_bar=False))
    return None

labse_ft_model = load_labse_ft()
jv_labse_ft_embeddings = encode_jv_labse_ft(labse_ft_model)
mbert_ft_model = load_mbert_ft()
jv_mbert_ft_embeddings = encode_jv_mbert_ft(mbert_ft_model)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">⚙️ Pengaturan</div>', unsafe_allow_html=True)
    st.markdown("### 📊 Statistik Dataset")
    st.info(f"""
    - **Total Dokumen**: {len(df):,}
    - **Kategori**: {df['category'].nunique()}
    - **Bahasa**: Indonesia ↔️ Jawa
    """)
    st.markdown("### 🤖 Model Fine-tuned (Otomatis)")
    st.markdown("Model fine-tuned yang digunakan:")
    st.markdown("""
    - 🟣 LaBSE Fine-tuned: `HyacinthiaIca/LaBSE-Indonesia-Jawa-Wikipedia`
    - 🟠 mBERT Fine-tuned: `HyacinthiaIca/mBERT-Indonesia-Jawa-Wikipedia`
    """)
    st.markdown("---")
    st.markdown("### 🎯 Pilih Metode Pencarian")
    
    available_methods = [
        "📖 Dictionary-Based + TF-IDF",
        "🤖 mBERT (Semantic Search)",
        "🟢 LaBSE (Semantic Search)"
    ]
    
    if labse_ft_model is not None and jv_labse_ft_embeddings is not None:
        available_methods.append("🟣 LaBSE Fine-tuned")
    
    if mbert_ft_model is not None and jv_mbert_ft_embeddings is not None:
        available_methods.append("🟠 mBERT Fine-tuned")
    
    available_methods.extend([
        "🔵 Ensemble (Base Models)",
        "🟡 Ensemble (Fine-tuned)"
    ])
    
    selected_methods = st.multiselect(
        "Pilih metode yang ingin ditampilkan:",
        available_methods,
        default=["🟡 Ensemble (Fine-tuned)", "🟢 LaBSE (Semantic Search)", "🔵 Ensemble (Base Models)"]
    )
    
    # Ensemble weights configuration
    if "🔵 Ensemble (Base Models)" in selected_methods or "🟡 Ensemble (Fine-tuned)" in selected_methods:
        st.markdown("### ⚖️ Konfigurasi Ensemble")
        with st.expander("Pengaturan Bobot (Opsional)"):
            w_dict = st.slider("Dictionary", 0.0, 1.0, 0.20, 0.05)
            w_mbert = st.slider("mBERT", 0.0, 1.0, 0.35, 0.05)
            w_labse = st.slider("LaBSE", 0.0, 1.0, 0.45, 0.05)
            
            total = w_dict + w_mbert + w_labse
            if abs(total - 1.0) > 0.01:
                st.warning(f"⚠️ Total bobot: {total:.2f} (sebaiknya = 1.0)")
            
            ensemble_weights = {'dict': w_dict, 'mbert': w_mbert, 'labse': w_labse}
    else:
        ensemble_weights = {'dict': 0.20, 'mbert': 0.35, 'labse': 0.45}
    
    st.markdown("---")
    st.markdown("### ℹ️ Tentang JawaSeek")
    st.markdown("""
    **JawaSeek** adalah sistem Cross-Lingual Information Retrieval (CLIR) 
    yang memungkinkan pencarian dokumen bahasa Jawa menggunakan query bahasa Indonesia.
    
    **Metode yang digunakan:**
    - 📖 Dictionary-Based + TF-IDF
    - 🤖 mBERT (Semantic Search)
    - 🟢 LaBSE (Semantic Search)
    - 🟣 LaBSE Fine-tuned (otomatis)
    - 🟠 mBERT Fine-tuned (otomatis)
    """)

# Main UI
st.markdown('<h1 class="main-header">🔍 JawaSeek</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Pencarian Cross-Lingual Indonesia ↔️ Jawa</p>', unsafe_allow_html=True)

# Info Box
st.markdown("""
<div class="info-box">
    <strong>💡 Cara Penggunaan:</strong><br>
    1. Masukkan query dalam bahasa Indonesia di kolom di bawah ini<br>
    2. Pilih jumlah hasil yang ingin ditampilkan<br>
    3. Klik tombol "🔍 Cari Dokumen" untuk melihat hasil<br>
    4. Sistem akan menampilkan hasil dari berbagai metode pencarian
</div>
""", unsafe_allow_html=True)

# Input Section
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("📝 Masukkan Query Bahasa Indonesia:", 
                         value="teknologi komputer dan internet di Indonesia",
                         placeholder="Contoh: musik tradisional Indonesia")
with col2:
    top_k = st.slider("📊 Jumlah Hasil", 1, 10, 5)

# Search Button
search_button = st.button("🔍 Cari Dokumen")

if search_button and query:
    if not selected_methods:
        st.warning("⚠️ Pilih minimal satu metode pencarian di sidebar!")
    else:
        with st.spinner('🔄 Sedang mencari dokumen yang relevan...'):
            
            # Dictionary-Based Results
            if "📖 Dictionary-Based + TF-IDF" in selected_methods:
                results_dict, translated = search(query, trans_dict, tfidf_vectorizer, jv_tfidf_matrix, df, top_k=top_k)
                render_search_results('📖 Dictionary-Based + TF-IDF', results_dict, translated_query=translated)

            # mBERT Results
            if "🤖 mBERT (Semantic Search)" in selected_methods:
                results_mbert = search_mbert(query, tokenizer, model, jv_embeddings, df, device, top_k=top_k)
                render_search_results('🤖 mBERT (Semantic Search)', results_mbert)

            # LaBSE Results
            if "🟢 LaBSE (Semantic Search)" in selected_methods:
                results_labse = search_labse(query, labse_model, jv_labse_embeddings, df, top_k=top_k)
                render_search_results('🟢 LaBSE (Semantic Search)', results_labse)

            # LaBSE Fine-tuned Results
            if "🟣 LaBSE Fine-tuned" in selected_methods:
                if labse_ft_model is not None and jv_labse_ft_embeddings is not None:
                    results_labse_ft = search_labse_ft(query, labse_ft_model, jv_labse_ft_embeddings, df, top_k=top_k)
                    render_search_results('🟣 LaBSE Fine-tuned', results_labse_ft)
                else:
                    st.error("❌ Model LaBSE Fine-tuned tidak tersedia")

            # mBERT Fine-tuned Results
            if "🟠 mBERT Fine-tuned" in selected_methods:
                if mbert_ft_model is not None and jv_mbert_ft_embeddings is not None:
                    results_mbert_ft = search_mbert_ft(query, mbert_ft_model, jv_mbert_ft_embeddings, df, top_k=top_k)
                    render_search_results('🟠 mBERT Fine-tuned', results_mbert_ft)
                else:
                    st.error("❌ Model mBERT Fine-tuned tidak tersedia")
            
            # Ensemble Base Models
            if "🔵 Ensemble (Base Models)" in selected_methods:
                results_ensemble_base = ensemble_late_fusion(query, top_k=top_k, weights=ensemble_weights, use_ft=False)
                render_search_results('🔵 Ensemble (Base Models) - Late Fusion', results_ensemble_base)
            
            # Ensemble Fine-tuned
            if "🟡 Ensemble (Fine-tuned)" in selected_methods:
                if (labse_ft_model is not None and jv_labse_ft_embeddings is not None) or \
                   (mbert_ft_model is not None and jv_mbert_ft_embeddings is not None):
                    results_ensemble_ft = ensemble_late_fusion(query, top_k=top_k, weights=ensemble_weights, use_ft=True)
                    render_search_results('🟡 Ensemble (Fine-tuned) - Late Fusion', results_ensemble_ft)
                else:
                    st.error("❌ Model Fine-tuned tidak tersedia untuk ensemble")

            st.success("✅ Pencarian selesai!")

elif search_button and not query:
    st.error("❌ Mohon masukkan query terlebih dahulu!")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; padding: 1rem;">
    <p><strong>JawaSeek</strong> - Cross-Lingual Information Retrieval System</p>
    <p>Indonesia ↔️ Jawa | Powered by mBERT & LaBSE</p>
</div>
""", unsafe_allow_html=True)