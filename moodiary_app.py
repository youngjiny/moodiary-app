# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit.components.v1 as components
from datetime import datetime, timezone, timedelta
from streamlit_calendar import calendar
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# (선택) Spotify SDK
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    spotipy = None
    SpotifyClientCredentials = None
    SPOTIPY_AVAILABLE = False

# --- 2) 기본 설정 ---
EMOTION_MODEL_ID = "JUDONGHYEOK/6-emotion-bert-korean-v6-balanced"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
GSHEET_DB_NAME = "moodiary_db" 
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

EMOTION_META = {
    "기쁨": {"color": "rgba(255, 215, 0, 0.6)", "emoji": "😆", "desc": "웃음이 끊이지 않는 하루!"},
    "분노": {"color": "rgba(255, 80, 80, 0.6)", "emoji": "🤬", "desc": "워워, 진정이 필요해요."},
    "불안": {"color": "rgba(255, 160, 50, 0.6)", "emoji": "😰", "desc": "마음이 조마조마해요."},
    "슬픔": {"color": "rgba(80, 120, 255, 0.6)", "emoji": "😭", "desc": "마음의 위로가 필요해요."},
    "힘듦": {"color": "rgba(150, 150, 150, 0.6)", "emoji": "🤯", "desc": "휴식이 절실한 하루."},
    "중립": {"color": "rgba(80, 180, 120, 0.6)", "emoji": "😐", "desc": "평온하고 무난한 하루."}
}

KST = timezone(timedelta(hours=9))
st.set_page_config(layout="wide", page_title="MOODIARY", page_icon="💖")

# ⭐️ 디자인 고정 (1번 코드의 그라데이션 + 글래스모피즘 유지)
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    if is_dark:
        bg_start, bg_mid, bg_end = "#121212", "#2c2c2c", "#403A4E"
        main_bg, main_text, secondary_text = "rgba(40, 40, 40, 0.9)", "#f0f0f0", "#bbbbbb"
        sidebar_bg, card_bg, card_text_happy = "#1e1e1e", "#3a3a3a", "#ffffff"
    else:
        bg_start, bg_mid, bg_end = "#ee7752", "#e73c7e", "#23d5ab"
        main_bg, main_text, secondary_text = "rgba(255, 255, 255, 0.85)", "#333333", "#666666"
        sidebar_bg, card_bg, card_text_happy = "#f8f9fa", "#fff9c4", "#2c3e50"

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        h1, h2, h3 {{ color: {main_text}; font-weight: 700; }}
        
        /* 배경 애니메이션 유지 */
        .stApp {{ background: linear-gradient(-45deg, {bg_start}, {bg_mid}, {bg_end}); background-size: 400% 400%; animation: gradient 15s ease infinite; }}
        @keyframes gradient {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}
        
        /* 글래스모피즘 컨테이너 유지 */
        .block-container {{ background: {main_bg}; backdrop-filter: blur(15px); border-radius: 25px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15); padding: 3rem !important; margin-top: 2rem; max-width: 1000px; }}
        p, label, .stMarkdown, .stTextarea, .stTextInput {{ color: {main_text} !important; }}
        
        /* ⭐️ 영화 카드 추천 섹션 개선 (요청하신 부분) */
        .movie-card {{
            background: {card_bg if is_dark else 'white'};
            border-radius: 15px; padding: 15px; margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1); display: flex; gap: 15px;
            height: 180px; overflow: hidden; /* 높이 고정 및 넘침 방지 */
            border: 1px solid rgba(128,128,128,0.1);
        }}
        .movie-card img {{ width: 110px; height: 100%; border-radius: 10px; object-fit: cover; }}
        .movie-info {{ display: flex; flex-direction: column; justify-content: flex-start; overflow: hidden; }}
        .movie-title {{ font-weight: bold; font-size: 1.1em; color: {main_text}; margin-bottom: 5px; }}
        .movie-rating {{ color: #f1c40f; font-weight: bold; margin-bottom: 8px; }}
        .movie-overview {{ 
            font-size: 0.85em; color: {secondary_text}; line-height: 1.4;
            display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; overflow: hidden; /* 3줄 말줄임 */
        }}

        /* 행복 저장소 디자인 유지 */
        .happy-card {{ background: {card_bg}; border-left: 6px solid #FFD700; padding: 20px; border-radius: 20px; margin-bottom: 15px; }}
        .animated-title {{ font-size: 3.5rem !important; font-weight: 800; animation: color-shift 5s ease-in-out infinite alternate; }}
        @keyframes color-shift {{ 0% {{ color: #6C5CE7; }} 100% {{ color: #FF7675; }} }}
        header, footer {{visibility: hidden;}}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 3) DB & AI & 추천 로직 (기존 1번 코드와 동일) ---
@st.cache_resource
def get_gsheets_client():
    try:
        creds = st.secrets["connections"]["gsheets"]
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds, scopes=scope)
        return gspread.authorize(credentials)
    except: return None

@st.cache_resource(ttl=3600)
def init_db():
    client = get_gsheets_client()
    if not client: return None
    try: return client.open(GSHEET_DB_NAME)
    except: return None 

def get_user_diaries(_sh, username):
    try:
        rows = _sh.worksheet("diaries").get_all_records()
        return {row['date']: {"emotion": row['emotion'], "text": row['text']} for row in rows if str(row['username']) == str(username)}
    except: return {}

def recommend_music(emotion):
    if not SPOTIPY_AVAILABLE: return []
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        sp = spotipy.Spotify(client_credentials_manager=manager)
        query = random.choice(["Daily Mix", "K-Pop Trend"])
        results = sp.search(q=query, type="playlist", limit=5)
        pl = random.choice(results["playlists"]["items"])
        tracks = sp.playlist_items(pl["id"], limit=10)["items"]
        return [{"id": t["track"]["id"], "title": t["track"]["name"]} for t in tracks if t.get("track")][:3]
    except: return []

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or EMERGENCY_TMDB_KEY
    GENRES = {"기쁨": "35", "분노": "28", "슬픔": "18", "중립": "18"}
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={"api_key": key, "language": "ko-KR", "with_genres": GENRES.get(emotion, "18"), "page": 1})
        results = r.json().get("results", [])
        picks = random.sample(results, min(3, len(results)))
        return [{"title": m["title"], "year": (m.get("release_date") or "")[:4], "rating": m["vote_average"], "overview": m["overview"], "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in picks]
    except: return []

# --- 4) 페이지 구현 ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

apply_custom_css()

def page_recommend(sh):
    st.markdown("## 🎵 음악/영화 추천")
    emo = st.session_state.get("final_emotion", "중립")
    music_recs = st.session_state.get("music_recs", [])
    movie_recs = st.session_state.get("movie_recs", [])
    
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])
    st.markdown(f"<div style='text-align: center; margin-bottom: 2rem;'><h2 style='color: {meta['color'].replace('0.6', '1.0')};'>{meta['emoji']} 감정: {emo}</h2></div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 🎵 추천 음악")
        if st.button("🔄 음악 새로고침"):
            st.session_state.music_recs = recommend_music(emo)
            st.rerun()
        for item in music_recs:
            components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=160)
    with c2:
        st.markdown("#### 🎬 추천 영화")
        if st.button("🔄 영화 새로고침"):
            st.session_state.movie_recs = recommend_movies(emo)
            st.rerun()
        for m in movie_recs:
            st.markdown(f"""
                <div class="movie-card">
                    <img src="{m['poster']}">
                    <div class="movie-info">
                        <div class="movie-title">{m['title']} ({m['year']})</div>
                        <div class="movie-rating">⭐ {m['rating']}</div>
                        <div class="movie-overview">{m['overview']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

# --- (나머지 intro_page, login_page, main_app 등 기존 코드 유지) ---
def main_app():
    sh = init_db()
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        st.session_state.dark_mode = st.checkbox("🌙 야간 모드", value=st.session_state.dark_mode)
        st.divider()
        if st.button("📝 일기 작성", use_container_width=True): st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()
        if st.button("📊 통계 보기", use_container_width=True): st.session_state.page = "stats"; st.rerun()
        if st.button("📂 행복 저장소", use_container_width=True): st.session_state.page = "happy"; st.rerun()
        st.divider()
        if st.button("🚪 로그아웃", use_container_width=True): st.session_state.logged_in = False; st.session_state.page = "intro"; st.rerun()

    if st.session_state.page == "write": pass # 기존 page_write 호출
    elif st.session_state.page == "dashboard": pass # 기존 page_dashboard 호출
    elif st.session_state.page == "result": page_recommend(sh)
    elif st.session_state.page == "stats": pass # 기존 page_stats 호출
    elif st.session_state.page == "happy": pass # 기존 page_happy_storage 호출

# --- 라우팅 ---
if st.session_state.logged_in: main_app()
elif st.session_state.page == "intro": pass # 기존 intro_page 호출
else: pass # 기존 login_page 호출
