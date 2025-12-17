# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit.components.v1 as components
from datetime import datetime, timezone, timedelta  # KST
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

# 비상용 TMDB 키
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

# ⭐️ 커스텀 CSS (달력 꽉 찬 배경 및 영화 카드 수정)
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    if is_dark:
        bg_start, bg_mid, bg_end = "#121212", "#2c2c2c", "#403A4E"
        main_bg, main_text = "rgba(40, 40, 40, 0.9)", "#f0f0f0"
        sidebar_bg, card_bg = "#1e1e1e", "#3a3a3a"
        card_text_happy = "#ffffff"
    else:
        bg_start, bg_mid, bg_end = "#ee7752", "#e73c7e", "#23d5ab"
        main_bg, main_text = "rgba(255, 255, 255, 0.85)", "#333333"
        sidebar_bg, card_bg = "#f8f9fa", "#fff9c4"
        card_text_happy = "#2c3e50"

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        h1, h2, h3 {{ color: {main_text}; font-weight: 700; }}
        .stApp {{ background: linear-gradient(-45deg, {bg_start}, {bg_mid}, {bg_end}); background-size: 400% 400%; animation: gradient 15s ease infinite; }}
        @keyframes gradient {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}
        .block-container {{ background: {main_bg}; backdrop-filter: blur(15px); border-radius: 25px; box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15); padding: 3rem !important; margin-top: 2rem; max-width: 1000px; }}
        p, label, .stMarkdown, .stTextarea, .stTextInput {{ color: {main_text} !important; }}
        
        /* 영화 카드: 줄거리 안짤리게 수정 */
        .movie-card {{
            background: {card_bg if is_dark else 'white'};
            border-radius: 15px; padding: 15px; margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; gap: 15px;
        }}
        .movie-card img {{ width: 120px; border-radius: 10px; object-fit: cover; }}
        .movie-overview {{ font-size: 0.9em; color: #555; line-height: 1.5; overflow: visible; }}

        /* 행복 저장소 카드 */
        .happy-card {{ background: {card_bg}; border-left: 6px solid #FFD700; padding: 20px; border-radius: 20px; margin-bottom: 15px; }}
        .happy-date {{ font-weight: 700; font-size: 0.9em; color: #888; margin-bottom: 5px; }}
        .happy-text {{ font-size: 1.2em; font-weight: 600; color: {card_text_happy}; }}

        /* 사이드바 고정 */
        section[data-testid="stSidebar"] {{ transform: none !important; visibility: visible !important; min-width: 250px !important; background-color: {sidebar_bg} !important; }}
        .animated-title {{ font-size: 3.5rem !important; font-weight: 800; animation: color-shift 5s ease-in-out infinite alternate; }}
        @keyframes color-shift {{ 0% {{ color: #6C5CE7; }} 100% {{ color: #FF7675; }} }}
        header, footer {{visibility: hidden;}}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 3) DB 로직 (생략) ---
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

def get_all_users(sh):
    try: return {str(row['username']): str(row['password']) for row in sh.worksheet("users").get_all_records()}
    except: return {}

def add_user(sh, username, password):
    try: sh.worksheet("users").append_row([str(username), str(password)]); return True
    except: return False

@st.cache_data(ttl=5)
def get_user_diaries(_sh, username):
    try:
        rows = _sh.worksheet("diaries").get_all_records()
        return {row['date']: {"emotion": row['emotion'], "text": row['text']} for row in rows if str(row['username']) == str(username)}
    except: return {}

def add_diary(sh, username, date, emotion, text):
    try:
        ws = sh.worksheet("diaries")
        cell = ws.find(date, in_column=2)
        if cell and str(ws.cell(cell.row, 1).value) == str(username):
            ws.update_cell(cell.row, 3, emotion); ws.update_cell(cell.row, 4, text)
        else: ws.append_row([username, date, emotion, text])
        get_user_diaries.clear(); return True
    except: return False

# --- 4) AI & 추천 로직 (생략) ---
@st.cache_resource
def load_emotion_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        id2label = getattr(model.config, "id2label", {0: "기쁨", 1: "분노", 2: "불안", 3: "슬픔", 4: "중립", 5: "힘듦"})
        return model, tokenizer, device, id2label
    except: return None, None, None, None

def analyze_diary(text, model, tokenizer, device, id2label):
    if not text: return None, 0.0
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad(): logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    return id2label.get(pred_id, "중립"), float(probs[pred_id].cpu().item())

@st.cache_resource
def get_spotify_client():
    if not SPOTIPY_AVAILABLE: return None
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        return spotipy.Spotify(client_credentials_manager=manager)
    except: return None

def recommend_music(emotion):
    sp = get_spotify_client()
    if not sp: return []
    query = random.choice(["Daily Mix", "K-Pop Trend"])
    try:
        results = sp.search(q=query, type="playlist", limit=5)
        pl = random.choice(results.get("playlists", {}).get("items", []))
        tracks = sp.playlist_items(pl["id"], limit=10).get("items", [])
        return [{"id": t["track"]["id"], "title": t["track"]["name"]} for t in tracks if t.get("track")][:3]
    except: return []

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or EMERGENCY_TMDB_KEY
    GENRES = {"기쁨": "35|10749", "분노": "28", "불안": "16", "슬픔": "18", "힘듦": "18|10402", "중립": "35|18"}
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={
            "api_key": key, "language": "ko-KR", "sort_by": "popularity.desc",
            "with_genres": GENRES.get(emotion, "18"), "page": random.randint(1, 3)
        }, timeout=5)
        results = r.json().get("results", [])
        picks = random.sample(results, min(3, len(results)))
        return [{"title": m["title"], "year": (m.get("release_date") or "")[:4], "rating": m["vote_average"], "overview": m["overview"], "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in picks]
    except: return []

# --- 5) 메인 화면 ---
apply_custom_css()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

def intro_page():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div style='text-align: center; padding: 40px;'><h1 class='animated-title'>MOODIARY</h1><h3>오늘 당신의 마음은 어떤가요?</h3></div>", unsafe_allow_html=True)
        if st.button("✨ 내 마음 기록하러 가기", use_container_width=True):
            st.session_state.page = "login"; st.rerun()

def login_page():
    sh = init_db()
    c1, c2 = st.columns([0.6, 0.4])
    with c1: st.markdown("<div style='padding-top: 5rem;'><h1 class='animated-title'>MOODIARY</h1><p>감정을 기록하고 나만을 위한 추천을 받으세요.</p></div>", unsafe_allow_html=True)
    with c2:
        tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
        if not sh: st.error("DB 연결 중..."); return
        with tab1:
            lid, lpw = st.text_input("아이디"), st.text_input("비밀번호", type="password")
            if st.button("로그인", use_container_width=True):
                users = get_all_users(sh)
                if lid in users and users[lid] == str(lpw):
                    st.session_state.logged_in, st.session_state.username = True, lid
                    st.session_state.page = "dashboard"; st.rerun()
        with tab2:
            nid, npw = st.text_input("새 아이디"), st.text_input("새 비밀번호 (4자리)", type="password", max_chars=4)
            if st.button("가입하기", use_container_width=True):
                if add_user(sh, nid, npw): st.success("가입 완료!"); st.rerun()

def main_app():
    sh = init_db()
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        st.session_state.dark_mode = st.checkbox("🌙 야간 모드", value=st.session_state.dark_mode)
        if st.button("📝 일기 작성"): st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력"): st.session_state.page = "dashboard"; st.rerun()
        if st.button("🎵 추천 보기"): st.session_state.page = "result"; st.rerun()
        if st.button("📊 통계 보기"): st.session_state.page = "stats"; st.rerun()
        if st.button("📂 행복 저장소"): st.session_state.page = "happy"; st.rerun()
        st.divider()
        if st.button("🚪 로그아웃"): st.session_state.logged_in = False; st.session_state.page = "intro"; st.rerun()

    if st.session_state.page == "write": page_write(sh)
    elif st.session_state.page == "dashboard": page_dashboard(sh)
    elif st.session_state.page == "result": page_recommend(sh)
    elif st.session_state.page == "stats": page_stats(sh)
    elif st.session_state.page == "happy": page_happy_storage(sh)

def page_write(sh):
    st.markdown("## 📝 오늘의 이야기")
    model, tokenizer, device, id2label = load_emotion_model()
    txt = st.text_area("오늘 하루는 어땠나요?", height=300)
    if st.button("🔍 분석 및 저장", type="primary"):
        with st.spinner("분석 중..."):
            emo, sc = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            add_diary(sh, st.session_state.username, datetime.now(KST).strftime("%Y-%m-%d"), emo, txt)
            st.session_state.page = "result"; st.rerun()

def page_recommend(sh):
    st.markdown("## 🎵 음악/영화 추천")
    emo = st.session_state.get("final_emotion", "중립")
    music_recs = st.session_state.get("music_recs", [])
    movie_recs = st.session_state.get("movie_recs", [])
    
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])
    st.markdown(f"<div style='text-align: center;'><h2 style='color: {meta['color'].replace('0.6', '1.0')};'>{meta['emoji']} 감정: {emo}</h2></div>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("#### 🎵 추천 음악")
        for item in music_recs:
            components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=160)
    with c2:
        st.markdown("#### 🎬 추천 영화")
        for item in movie_recs:
            st.markdown(f"""
            <div class="movie-card">
                <img src="{item['poster']}">
                <div>
                    <div style="font-weight:bold;">{item['title']} ({item['year']})</div>
                    <div style="color:#f1c40f; margin: 5px 0;">⭐ {item['rating']}</div>
                    <div class="movie-overview">{item['overview']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ⭐️ 달력 부분: 칸 꽉 채우기 + 이모지 크기 키우기
def page_dashboard(sh):
    st.markdown("## 📅 감정 달력")
    my_diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    for d, data in my_diaries.items():
        meta = EMOTION_META.get(data['emotion'], EMOTION_META["중립"])
        # display: 'background'를 사용하여 칸 전체 색칠
        events.append({"start": d, "display": "background", "backgroundColor": meta["color"]})
        # 이모지 추가
        events.append({"title": meta["emoji"], "start": d, "allDay": True})
    
    calendar(events=events, options={"initialView": "dayGridMonth"}, custom_css="""
        .fc-event-title { font-size: 2.5em !important; text-align: center; cursor: default; }
        .fc-daygrid-day-frame { min-height: 100px !important; }
        .fc-bg-event { opacity: 1.0 !important; }
    """)

# ⭐️ 통계 부분: 색상 일치 + 한글 똑바로 (가로형)
def page_stats(sh):
    st.markdown("## 📊 감정 통계")
    diaries = get_user_diaries(sh, st.session_state.username)
    if not diaries: st.info("기록이 없습니다."); return
    
    df = pd.DataFrame([{"emotion": d['emotion']} for d in diaries.values()])
    counts = df['emotion'].value_counts().reindex(EMOTION_META.keys(), fill_value=0).reset_index()
    counts.columns = ['emotion', 'count']
    
    # 달력과 동일한 색상 매핑
    color_range = [m['color'].replace('0.6', '1.0') for m in EMOTION_META.values()]
    
    st.vega_lite_chart(counts, {
        "mark": {"type": "bar", "cornerRadius": 5},
        "encoding": {
            "x": {"field": "emotion", "type": "nominal", "axis": {"labelAngle": 0}, "sort": list(EMOTION_META.keys())}, # 한글 똑바로
            "y": {"field": "count", "type": "quantitative"},
            "color": {
                "field": "emotion", 
                "scale": {"domain": list(EMOTION_META.keys()), "range": color_range},
                "legend": None
            }
        }
    }, use_container_width=True)

# ⭐️ 행복 저장소: 날짜 추가
def page_happy_storage(sh):
    st.markdown("## 📂 행복 저장소")
    diaries = get_user_diaries(sh, st.session_state.username)
    happy_list = [(date, d['text']) for date, d in diaries.items() if d['emotion'] == "기쁨"]
    if not happy_list: st.info("아직 기쁜 기록이 없네요."); return
    for date, text in sorted(happy_list, reverse=True):
        st.markdown(f"""
            <div class='happy-card'>
                <div class='happy-date'>{date}</div>
                <div class='happy-text'>{text}</div>
            </div>
        """, unsafe_allow_html=True)

if st.session_state.logged_in: main_app()
elif st.session_state.page == "intro": intro_page()
else: login_page()
