import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from supabase import create_client, Client
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Audit Anggaran AI", layout="wide")

# --- 1. KONEKSI SUPABASE ---
# Mengambil credential dari Streamlit Secrets (akan disetting nanti di dashboard)
@st.cache_resource
def init_supabase():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

try:
    supabase = init_supabase()
except Exception as e:
    st.error("Gagal koneksi ke Supabase. Pastikan Secrets sudah diatur.")
    st.stop()

# --- 2. LOAD AI (CACHED) ---
# Agar model tidak didownload ulang setiap kali klik tombol
@st.cache_resource
def load_ai_model():
    # Load Base Model dari Internet (HuggingFace)
    # Ini otomatis download jika belum ada di cache server Streamlit
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')
    
    # Load Data Lokal (Wajib ada di folder GitHub)
    try:
        df_ref = pd.read_pickle('database_rekening.pkl')
        embeddings = torch.load('embeddings_tensor.pt', map_location=torch.device('cpu'))
    except FileNotFoundError:
        st.error("File 'database_rekening.pkl' atau 'embeddings_tensor.pt' tidak ditemukan!")
        st.stop()
        
    return model, df_ref, embeddings

with st.spinner('Sedang menyiapkan otak AI... (Tunggu sebentar)'):
    model, df_target, corpus_embeddings = load_ai_model()

# --- 3. FUNGSI EVALUASI ---
def evaluasi_row(uraian, kode_user):
    if pd.isna(uraian): return None
    
    # Encode & Search
    query_vec = model.encode(str(uraian), convert_to_tensor=True)
    cos_scores = util.cos_sim(query_vec, corpus_embeddings)[0]
    best_idx = torch.argmax(cos_scores).item()
    score = cos_scores[best_idx].item()
    
    rekomendasi_kode = df_target.iloc[best_idx]['REKENING BARU']
    rekomendasi_nama = df_target.iloc[best_idx]['NAMA REKENING BELANJA DAERAH']
    
    # Logic Audit
    kode_user_str = str(kode_user).strip()
    kode_ai_str = str(rekomendasi_kode).strip()
    
    status = "🔴 BERISIKO"
    if kode_user_str == kode_ai_str:
        status = "✅ SESUAI"
    elif kode_user_str[:3] == kode_ai_str[:3]: # Cek level jenis (5.1 vs 5.2)
        status = "⚠️ KURANG TEPAT"
        
    return {
        "uraian_input": uraian,
        "kode_user": kode_user_str,
        "kode_ai": kode_ai_str,
        "uraian_ai": rekomendasi_nama,
        "skor_confidence": score,
        "status_evaluasi": status
    }

# --- 4. UI UTAMA ---
st.title("🕵️‍♀️ Sistem Evaluasi Anggaran Mandiri")
st.markdown("Upload file Excel berisi usulan anggaran, AI akan mengecek kesesuaian kode rekeningnya.")

# Upload File
uploaded_file = st.file_uploader("Upload Excel (.xlsx)", type=['xlsx'])

if uploaded_file:
    df_input = pd.read_excel(uploaded_file)
    st.dataframe(df_input.head())
    
    # Mapping Kolom
    col1, col2 = st.columns(2)
    with col1:
        col_uraian = st.selectbox("Pilih Kolom Uraian Belanja", df_input.columns)
    with col2:
        col_kode = st.selectbox("Pilih Kolom Kode Rekening", df_input.columns)
        
    if st.button("Mulai Evaluasi AI"):
        results = []
        progress_bar = st.progress(0)
        
        for index, row in df_input.iterrows():
            # Proses AI
            res = evaluasi_row(row[col_uraian], row[col_kode])
            if res:
                results.append(res)
            
            # Update progress
            progress_bar.progress((index + 1) / len(df_input))
            
        # Tampilkan Hasil
        df_result = pd.DataFrame(results)
        st.success("Evaluasi Selesai!")
        
        # Tampilkan Dataframe dengan highlight
        def highlight_status(val):
            color = 'red' if val == '🔴 BERISIKO' else 'green' if val == '✅ SESUAI' else 'orange'
            return f'background-color: {color}'
            
        st.dataframe(df_result.style.map(highlight_status, subset=['status_evaluasi']))
        
        # Tombol Simpan ke Supabase
        if st.button("Simpan Hasil ke Database Cloud"):
            data_to_insert = df_result.to_dict(orient='records')
            try:
                # Insert data ke tabel 'audit_logs' di Supabase
                supabase.table('audit_logs').insert(data_to_insert).execute()
                st.toast("Data berhasil disimpan ke Supabase!", icon="🚀")
            except Exception as e:
                st.error(f"Gagal menyimpan: {e}")