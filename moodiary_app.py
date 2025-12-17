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

# --- 2) 세션 상태 초기화 및 기본 설정 (파일 최상단) ---
st.set_page_config(layout="wide", page_title="MOODIARY", page_icon="💖")

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False
if "username" not in st.session_state: st.session_state.username = ""

EMOTION_MODEL_ID = "JUDONGHYEOK/6-emotion-bert-korean-v6-balanced"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
GSHEET_DB_NAME = "moodiary_db" 
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"
KST = timezone(timedelta(hours=9))

EMOTION_META = {
    "기쁨": {"color": "rgba(255, 215, 0, 0.6)", "emoji": "😆", "desc": "웃음이 끊이지 않는 하루!"},
    "분노": {"color": "rgba(255, 80, 80, 0.6)", "emoji": "🤬", "desc": "워워, 진정이 필요해요."},
    "불안": {"color": "rgba(255, 160, 50, 0.6)", "emoji": "😰", "desc": "마음이 조마조마해요."},
    "슬픔": {"color": "rgba(80, 120, 255, 0.6)", "emoji": "😭", "desc": "마음의 위로가 필요해요."},
    "힘듦": {"color": "rgba(150, 150, 150, 0.6)", "emoji": "🤯", "desc": "휴식이 절실한 하루."},
    "중립": {"color": "rgba(80, 180, 120, 0.6)", "emoji": "😐", "desc": "평온하고 무난한 하루."}
}

# --- 3) 커스텀 CSS (사이드바 열기 버튼 강제 활성화 로직) ---
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    primary_purple = "#7B61FF" 
    
    bg_start, bg_mid, bg_end = ("#121212", "#2c2c2c", "#403A4E") if is_dark else ("#ee7752", "#e73c7e", "#23d5ab")
    main_bg = "rgba(40, 40, 40, 0.9)" if is_dark else "rgba(255, 255, 255, 0.85)"

    # 핵심: 로그인 전/후 사이드바 노출 로직
    if not st.session_state.logged_in or st.session_state.page in ["intro", "login"]:
        sidebar_css = """
            section[data-testid="stSidebar"] { display: none !important; }
            button[data-testid="stSidebarCollapseButton"] { display: none !important; }
        """
    else:
        # 로그인 후: 사이드바를 접었을 때 나오는 '>' 버튼을 강제로 보이게 함
        sidebar_css = """
            section[data-testid="stSidebar"] { display: block !important; }
            button[data-testid="stSidebarCollapseButton"] { 
                display: flex !important; 
                visibility: visible !important; 
                opacity: 1 !important;
                left: 0 !important;
                background-color: white !important; /* 가시성을 위해 배경색 추가 */
                box-shadow: 0 2px 5px rgba(0,0,0,0.2) !important;
            }
        """

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        
        /* 1번의 애니메이션 배경 */
        @keyframes gradient {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}
        .stApp {{
            background: linear-gradient(-45deg, {bg_start}, {bg_mid}, {bg_end});
            background-size: 400% 400%; animation: gradient 15s ease infinite;
        }}

        .block-container {{
            background: {main_bg}; backdrop-filter: blur(15px); border-radius: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15); padding: 3rem !important; margin-top: 2rem; max-width: 1000px;
        }}

        /* 2번의 보라색 버튼 테마 */
        div.stButton > button {{
            background: {primary_purple} !important; color: white !important;
            border-radius: 50px !important; font-weight: 700 !important; border: none !important;
            padding: 0.6rem 2.5rem !important;
        }}
        
        .animated-title {{ font-size: 3.5rem !important; font-weight: 800; color: {primary_purple}; text-align: center; }}
        
        {sidebar_css}
        
        header {{visibility: hidden;}} footer {{visibility: hidden;}}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 4) DB 및 AI 로직 (기존 유지) ---
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

@st.cache_resource
def load_emotion_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device, getattr(model.config, "id2label", {0: "기쁨", 1: "분노", 2: "불안", 3: "슬픔", 4: "중립", 5: "힘듦"})
    except: return None, None, None, None

def analyze_diary(text, model, tokenizer, device, id2label):
    if not text or model is None: return None, 0.0
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad(): logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    return id2label.get(pred_id, "중립"), float(probs[pred_id].cpu().item())

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or EMERGENCY_TMDB_KEY
    GENRES = {"기쁨": "35|10749", "분노": "28", "불안": "16", "슬픔": "18", "힘듦": "18|10402", "중립": "35|18"}
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={"api_key": key, "language": "ko-KR", "sort_by": "popularity.desc", "with_genres": GENRES.get(emotion, "18"), "page": random.randint(1, 3)}, timeout=5)
        results = r.json().get("results", [])
        picks = random.sample(results, min(3, len(results)))
        return [{"title": m["title"], "year": (m.get("release_date") or "")[:4], "rating": m["vote_average"], "overview": m["overview"], "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in picks]
    except: return []

# --- 5) 페이지 함수 ---
def intro_page():
    apply_custom_css()
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.markdown("<h1 class='animated-title'>MOODIARY</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: white;'>오늘 당신의 마음은 어떤가요?</h3>", unsafe_allow_html=True)
    if st.button("✨ 내 마음 기록하러 가기"):
        st.session_state.page = "login"; st.rerun()

def login_page():
    apply_custom_css()
    sh = init_db()
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown("<div style='padding-top: 5rem;'><h1 class='animated-title'>MOODIARY</h1><p style='text-align:center; color:white;'>당신의 감정을 기록하고 추천을 받으세요.</p></div>", unsafe_allow_html=True)
    with c2:
        tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
        with tab1:
            lid = st.text_input("아이디", key="l_id")
            lpw = st.text_input("비밀번호", type="password", key="l_pw")
            if st.button("로그인"):
                users = get_all_users(sh)
                if lid in users and users[lid] == str(lpw):
                    st.session_state.logged_in, st.session_state.username = True, lid
                    st.session_state.page = "dashboard"; st.rerun()
                else: st.error("로그인 실패")
        with tab2:
            nid = st.text_input("새 아이디", key="n_id")
            npw = st.text_input("비번(4자리)", type="password", max_chars=4, key="n_pw")
            if st.button("가입"):
                if add_user(sh, nid, npw): st.success("성공! 로그인하세요."); st.rerun()

def main_app():
    apply_custom_css() # 로그인 상태이므로 사이드바 노출 CSS 적용
    sh = init_db()
    
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        if st.button("📝 일기 작성", use_container_width=True): st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()
        if st.button("📂 행복 저장소", use_container_width=True): st.session_state.page = "happy"; st.rerun()
        st.divider()
        if st.button("🚪 로그아웃", use_container_width=True):
            st.session_state.logged_in = False; st.session_state.page = "intro"; st.rerun()

    if st.session_state.page == "write": page_write(sh)
    elif st.session_state.page == "dashboard": page_dashboard(sh)
    elif st.session_state.page == "result": page_result(sh)
    elif st.session_state.page == "happy": page_happy_storage(sh)

def page_write(sh):
    st.markdown("## 📝 오늘의 이야기")
    model, tokenizer, device, id2label = load_emotion_model()
    txt = st.text_area("오늘 하루는 어땠나요?", height=300)
    if st.button("🔍 분석 및 저장"):
        if txt.strip():
            emo, _ = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            st.session_state.movie_recs = recommend_movies(emo)
            add_diary(sh, st.session_state.username, datetime.now(KST).strftime("%Y-%m-%d"), emo, txt)
            st.session_state.page = "result"; st.rerun()

def page_result(sh):
    emo = st.session_state.get("final_emotion", "중립")
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])
    st.markdown(f"<h1>{meta['emoji']} 감정: {emo}</h1>", unsafe_allow_html=True)
    if st.button("📅 달력 보기"): st.session_state.page = "dashboard"; st.rerun()

def page_dashboard(sh):
    st.markdown("## 📅 감정 달력")
    my_diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    for d, data in my_diaries.items():
        meta = EMOTION_META.get(data['emotion'], EMOTION_META["중립"])
        events.append({"start": d, "display": "background", "backgroundColor": meta["color"]})
        events.append({"title": meta["emoji"], "start": d, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": "#000"})
    calendar(events=events, options={"initialView": "dayGridMonth"})

def page_happy_storage(sh):
    st.markdown("## 📂 행복 저장소")
    diaries = get_user_diaries(sh, st.session_state.username)
    for date, d in diaries.items():
        if d['emotion'] == "기쁨": st.info(f"📅 {date}\n\n{d['text']}")

# --- 6) 최종 실행 로직 ---
if st.session_state.logged_in:
    main_app()
elif st.session_state.page == "intro":
    intro_page()
else:
    login_page()
