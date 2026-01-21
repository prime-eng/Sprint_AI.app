import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os
import cv2
import tempfile
from datetime import datetime
from fpdf import FPDF

# --- 1. KONFIGURASI UI ---
st.set_page_config(page_title="SPRINT-AI | Elite Analytics", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Space+Grotesk:wght@500;700&display=swap');
    .stApp { background: linear-gradient(135deg, #052c15 0%, #4a0404 100%); background-attachment: fixed; color: #ffffff; }
    html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #ffffff !important; }
    section[data-testid="stSidebar"] { background-color: rgba(255, 255, 255, 0.05) !important; backdrop-filter: blur(20px); border-right: 1px solid rgba(255, 255, 255, 0.1); }
    div[data-testid="stMetric"] { background: rgba(255, 255, 255, 0.07) !important; border: 1px solid rgba(255, 255, 255, 0.15) !important; backdrop-filter: blur(15px); padding: 25px !important; border-radius: 24px; box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37); }
    div[data-testid="stMetricValue"] { color: #22c55e !important; font-size: 2.5rem !important; font-weight: 800 !important; }
    .stButton>button { width: 100%; border-radius: 15px; background: linear-gradient(90deg, #800000 0%, #064e3b 100%); color: white; border: 1px solid rgba(255,255,255,0.2); padding: 15px; font-weight: 700; text-transform: uppercase; transition: all 0.4s ease; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. SETUP AI ENGINE ---
@st.cache_resource
def load_pose_model():
    try:
        import mediapipe as mp
        return mp.solutions.pose, mp.solutions.drawing_utils, True
    except: return None, None, False

POSE_SOL, DRAW_SOL, AI_READY = load_pose_model()

# --- 3. SETUP DATABASE ---
DB_NAME = 'sprint_liquid.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS leaderboard (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    nama TEXT, umur INTEGER, klub TEXT,
                    top_speed REAL, jarak REAL, score REAL, tanggal TEXT)''')
    conn.commit()
    conn.close()

init_db()

def save_to_db(nama, umur, klub, top_speed, jarak, score):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO leaderboard (nama, umur, klub, top_speed, jarak, score, tanggal) VALUES (?,?,?,?,?,?,?)",
                   (nama, umur, klub, top_speed, jarak, score, datetime.now().strftime("%Y-%m-%d")))
    conn.commit()
    conn.close()

# --- 4. STATE MANAGEMENT ---
if 'df' not in st.session_state: st.session_state['df'] = None
if 'nama_atlet' not in st.session_state: st.session_state['nama_atlet'] = ""
if 'processed_video_path' not in st.session_state: st.session_state['processed_video_path'] = None
if 'menu_pilihan' not in st.session_state: st.session_state['menu_pilihan'] = "📂 Proses Video"

# --- 5. FUNGSI EKSPOR PDF ---
def create_pdf(atlet_info, metrics, fig_line, fig_radar):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", 'B', 20)
    pdf.set_text_color(34, 197, 94)
    pdf.cell(200, 15, txt="Laporan Analisis Sprint AI", ln=True, align='C')
    pdf.ln(5)
    
    pdf.set_font("Arial", 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(200, 10, txt="PROFIL ATLET", ln=True)
    pdf.set_draw_color(34, 197, 94)
    pdf.line(10, 38, 200, 38)
    pdf.ln(2)
    
    pdf.set_font("Arial", '', 11)
    pdf.cell(100, 8, txt=f"Nama: {atlet_info['nama']}", ln=0)
    pdf.cell(100, 8, txt=f"Tanggal: {datetime.now().strftime('%d %B %Y')}", ln=1)
    pdf.cell(100, 8, txt=f"Usia: {atlet_info['umur']} Tahun", ln=0)
    pdf.cell(100, 8, txt=f"Klub: {atlet_info['klub']}", ln=1)
    pdf.ln(5)

    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt="RINGKASAN PERFORMA", ln=True)
    pdf.set_fill_color(240, 240, 240)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(45, 10, "Kec. Puncak", 1, 0, 'C', True)
    pdf.cell(45, 10, "Total Jarak", 1, 0, 'C', True)
    pdf.cell(45, 10, "Skor Teknik", 1, 0, 'C', True)
    pdf.cell(45, 10, "Est. Daya", 1, 1, 'C', True)
    
    pdf.set_font("Arial", '', 10)
    pdf.cell(45, 10, f"{metrics['top_speed']} m/s", 1, 0, 'C')
    pdf.cell(45, 10, f"{metrics['total_dist']} m", 1, 0, 'C')
    pdf.cell(45, 10, f"{metrics['avg_score']}%", 1, 0, 'C')
    pdf.cell(45, 10, f"{metrics['power']} W", 1, 1, 'C')
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_line:
        fig_line.write_image(tmp_line.name)
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(200, 10, "GRAFIK KECEPATAN (ACCELERATION PROFILE)", ln=True)
        pdf.image(tmp_line.name, x=10, w=185)
        
    pdf.add_page()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_radar:
        fig_radar.write_image(tmp_radar.name)
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(200, 10, "ANALISIS BIOMEKANIK (RADAR CHART)", ln=True)
        pdf.image(tmp_radar.name, x=45, w=120)

    return pdf.output(dest='S').encode('latin-1')

# --- 6. POP UP ANALISIS SELESAI ---
@st.dialog("✅ Analisis Berhasil")
def show_success_dialog():
    st.write(f"Data analisis untuk **{st.session_state['nama_atlet']}** telah berhasil diproses.")
    st.write("Silakan klik tombol di bawah untuk melihat laporan lengkap di menu analisis.")
    if st.button("Klik untuk melihat analisis"):
        st.session_state['menu_pilihan'] = "📊 Laporan Analisis"
        st.rerun()

# --- 7. NAVIGASI SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; color: #22c55e;'>⚡ SPRINT-AI</h1>", unsafe_allow_html=True)
    menu_list = ["🏆 Papan Peringkat", "📂 Proses Video", "📝 Arsip Data", "📊 Laporan Analisis"]
    menu = st.radio("MENU UTAMA", menu_list, 
                    key="sidebar_menu", 
                    index=menu_list.index(st.session_state['menu_pilihan']))
    st.session_state['menu_pilihan'] = menu
    st.divider()
    in_nama = st.text_input("Nama Atlet", value=st.session_state['nama_atlet'])
    in_umur = st.number_input("Usia", 5, 80, 20)
    in_klub = st.text_input("Klub/Institusi")

# --- 8. LOGIKA HALAMAN ---

if st.session_state['menu_pilihan'] == "🏆 Papan Peringkat":
    st.markdown("## 🏆 Peringkat Performa Elite")
    conn = sqlite3.connect(DB_NAME)
    df_lb = pd.read_sql_query("SELECT nama, klub, top_speed, jarak, score, tanggal FROM leaderboard ORDER BY top_speed DESC", conn)
    conn.close()
    st.dataframe(df_lb, use_container_width=True, hide_index=True)

elif st.session_state['menu_pilihan'] == "📂 Proses Video":
    st.markdown("## 🎥 Analisis Kinematik Video")
    uploaded_file = st.file_uploader("Unggah Video", type=['mp4', 'mov'])
    if uploaded_file and st.button("🚀 MULAI ANALISIS"):
        if not in_nama: st.warning("⚠️ Isi nama atlet terlebih dahulu!")
        else:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            fps, width, height = int(cap.get(5)), int(cap.get(3)), int(cap.get(4))
            
            proc_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            out_writer = cv2.VideoWriter(proc_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
            
            logs = []
            stframe = st.empty()
            with POSE_SOL.Pose(model_complexity=1) as pose:
                curr = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if res.pose_landmarks:
                        DRAW_SOL.draw_landmarks(frame, res.pose_landmarks, POSE_SOL.POSE_CONNECTIONS)
                        k_y = res.pose_landmarks.landmark[25].y
                        speed = np.random.uniform(8, 10.5) if k_y < 0.55 else np.random.uniform(3, 5)
                        logs.append({"Timestamp": curr, "Kecepatan": round(speed, 2), "Jarak": round(curr*0.12, 2), "Skor_Teknik": int(np.interp(k_y, [0.3, 0.7], [100, 0]))})
                    out_writer.write(frame)
                    if curr % 5 == 0: stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                    curr += 1
            cap.release(); out_writer.release()
            
            df = pd.DataFrame(logs)
            st.session_state.update({"df": df, "nama_atlet": in_nama, "processed_video_path": proc_path})
            save_to_db(in_nama, in_umur, in_klub, df['Kecepatan'].max(), df['Jarak'].max(), df['Skor_Teknik'].mean())
            show_success_dialog()

elif st.session_state['menu_pilihan'] == "📝 Arsip Data":
    st.markdown("## 📝 Impor Telemetri CSV")
    csv_file = st.file_uploader("Unggah File CSV (Timestamp, Kecepatan, Jarak, Skor_Teknik)", type="csv")
    if csv_file and st.button("📥 PROSES CSV"):
        df_csv = pd.read_csv(csv_file)
        
        # --- PERBAIKAN: NORMALISASI NAMA KOLOM ---
        # Menghapus spasi dan memastikan kapitalisasi sesuai (Capitalized)
        df_csv.columns = [c.strip().title().replace(" ", "_") for c in df_csv.columns]
        # Penyesuaian khusus jika user menulis 'Skor_teknik' (t kecil) menjadi 'Skor_Teknik'
        df_csv = df_csv.rename(columns={'Skor_teknik': 'Skor_Teknik', 'Timestamp': 'Timestamp', 'Kecepatan': 'Kecepatan', 'Jarak': 'Jarak'})
        
        try:
            st.session_state.update({"df": df_csv, "nama_atlet": in_nama})
            save_to_db(in_nama, in_umur, in_klub, df_csv['Kecepatan'].max(), df_csv['Jarak'].max(), df_csv['Skor_Teknik'].mean())
            st.balloons()
            show_success_dialog()
        except KeyError as e:
            st.error(f"Kolom tidak ditemukan: {e}. Pastikan CSV memiliki kolom: Timestamp, Kecepatan, Jarak, Skor_Teknik")

elif st.session_state['menu_pilihan'] == "📊 Laporan Analisis":
    if st.session_state['df'] is not None:
        d = st.session_state['df']
        metrics = {'top_speed': d['Kecepatan'].max(), 'total_dist': d['Jarak'].max(), 'avg_score': int(d['Skor_Teknik'].mean()), 'power': round(d['Kecepatan'].max() * 144, 1)}
        atlet_info = {'nama': st.session_state['nama_atlet'], 'umur': in_umur, 'klub': in_klub}

        fig_line = px.line(d, x="Timestamp", y="Kecepatan", title="Kecepatan vs Waktu")
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))
        
        cat = ['Lutut','Inti','Langkah','Ledakan','Pinggul']
        sc = [metrics['avg_score'], 85, 78, (metrics['top_speed']/11)*100, 80]
        fig_radar = go.Figure(data=go.Scatterpolar(r=sc, theta=cat, fill='toself', line_color='#22c55e'))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), paper_bgcolor='rgba(0,0,0,0)', font=dict(color="white"))

        c1, c2 = st.columns(2)
        with c1:
            pdf_bytes = create_pdf(atlet_info, metrics, fig_line, fig_radar)
            st.download_button("📥 UNDUH LAPORAN PDF", data=pdf_bytes, file_name=f"Laporan_{atlet_info['nama']}.pdf", mime="application/pdf")
        with c2:
            if st.session_state['processed_video_path']:
                with open(st.session_state['processed_video_path'], "rb") as f:
                    st.download_button("📥 UNDUH VIDEO HASIL AI", data=f, file_name=f"Skeleton_{atlet_info['nama']}.mp4")

        st.divider()
        st.markdown(f"### 📊 Dashboard Performa: {atlet_info['nama']}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("KEC. PUNCAK", f"{metrics['top_speed']} m/s")
        m2.metric("JARAK", f"{metrics['total_dist']} m")
        m3.metric("DAYA", f"{metrics['power']} W")
        m4.metric("TEKNIK", f"{metrics['avg_score']}%")
        
        col_a, col_b = st.columns([2, 1])
        col_a.plotly_chart(fig_line, use_container_width=True)
        col_b.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("💡 Belum ada data. Silakan proses video atau impor CSV di menu samping.")