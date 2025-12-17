# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit.components.v1 as components
from datetime import datetime, timezone, timedelta  # KST
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
# ⭐️ v6 모델로 고정
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

# ⭐️ 커스텀 CSS (가시성 및 일렬 레이아웃 최적화)
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    if is_dark:
        bg_start, bg_mid, bg_end = "#121212", "#2c2c2c", "#403A4E"
        main_bg, main_text, secondary_text = "rgba(40, 40, 40, 0.9)", "#f0f0f0", "#bbbbbb"
        sidebar_bg, menu_checked = "#1e1e1e", "#A29BFE"
        card_bg, card_text_happy = "#3a3a3a", "#ffffff"
        stat_card_line = "1px solid #444444"
    else:
        bg_start, bg_mid, bg_end = "#ee7752", "#e73c7e", "#23d5ab"
        main_bg, main_text, secondary_text = "rgba(255, 255, 255, 0.85)", "#333333", "#666666"
        sidebar_bg, menu_checked = "#f8f9fa", "#6C5CE7"
        card_bg, card_text_happy = "#fff9c4", "#2c3e50"
        stat_card_line = "none"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {main_text}; font-weight: 700; }}
        
        .stApp {{
            background: linear-gradient(-45deg, {bg_start}, {bg_mid}, {bg_end});
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }}
        @keyframes gradient {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}

        .block-container {{
            background: {main_bg}; backdrop-filter: blur(15px); border-radius: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15); padding: 3rem !important; max-width: 1000px;
        }}

        p, label, .stMarkdown, .stTextarea, .stTextInput, .stCheckbox {{ color: {main_text} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}
        
        .stButton > button {{
            width: 100%; border-radius: 20px; border: none; font-weight: 700;
            background: linear-gradient(90deg, #6C5CE7 0%, #a29bfe 100%); color: white;
        }}

        /* ⭐️ 행복 저장소 일렬 카드 디자인 */
        .happy-card {{
            background: {card_bg}; border-left: 8px solid #FFD700;
            padding: 22px; border-radius: 15px; margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 100%;
        }}
        .happy-date {{ color: {main_text}; font-weight: 700; font-size: 1.1em; margin-bottom: 10px; }}
        .happy-text {{ font-size: 1.2em; font-weight: 500; line-height: 1.6; color: {card_text_happy}; }}
        .month-header {{ 
            margin: 30px 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #FFD700;
            font-size: 1.5rem; color: {main_text}; font-weight: 800;
        }}

        header {{visibility: hidden;}} footer {{visibility: hidden;}}
        </style>
    """, unsafe_allow_html=True)

# --- 3) DB & AI 유틸리티 (v6 모델 고정) ---
@st.cache_resource
def get_gsheets_client():
    try:
        creds = st.secrets["connections"]["gsheets"]
        credentials = Credentials.from_service_account_info(creds, scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
        return gspread.authorize(credentials)
    except: return None

@st.cache_resource
def init_db():
    client = get_gsheets_client()
    return client.open(GSHEET_DB_NAME) if client else None

def get_all_users(sh):
    try: return {str(row['username']): str(row['password']) for row in sh.worksheet("users").get_all_records()}
    except: return {}

@st.cache_data(ttl=10)
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
        return model, tokenizer, device, {0: "기쁨", 1: "분노", 2: "불안", 3: "슬픔", 4: "중립", 5: "힘듦"}
    except: return None, None, None, None

def analyze_diary(text, model, tokenizer, device, id2label):
    if not text or model is None: return None, 0.0
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad(): probs = torch.softmax(model(**enc).logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    return id2label.get(pred_id, "중립"), float(probs[pred_id].cpu().item())

# --- 추천 로직 (Spotify, TMDB) ---
def recommend_music(emotion):
    if not SPOTIPY_AVAILABLE: return []
    try:
        creds = st.secrets["spotify"]
        sp = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"]))
        query = {"기쁨": "Upbeat", "슬픔": "Ballad 새벽", "분노": "Rock", "불안": "Lofi Calm", "힘듦": "Healing Comfort", "중립": "Daily Chill"}.get(emotion, "Daily")
        res = sp.search(q=query, type="playlist", limit=5)
        playlist = random.choice(res['playlists']['items'])
        tracks = sp.playlist_items(playlist['id'], limit=15)['items']
        return [{"id": t['track']['id'], "title": t['track']['name']} for t in random.sample(tracks, 3)]
    except: return []

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or EMERGENCY_TMDB_KEY
    genres = {"기쁨": "35|10749", "분노": "28", "불안": "53|9648", "슬픔": "18", "힘듦": "18|10402", "중립": "18|35"}
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={"api_key": key, "language": "ko-KR", "with_genres": genres.get(emotion, "18"), "sort_by": "popularity.desc", "page": random.randint(1, 3)}, timeout=5)
        results = r.json().get("results", [])
        return [{"title": m["title"], "year": m.get("release_date","")[:4], "rating": m["vote_average"], "overview": m["overview"], "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get('poster_path') else None} for m in random.sample(results, min(3, len(results)))]
    except: return []

# --- 4) 화면 로직 ---
apply_custom_css()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

def intro_page():
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("<div style='text-align: center; margin-top: 5rem;'><h1 class='animated-title'>MOODIARY</h1><br><h3>오늘 당신의 감정은 어떤가요?</h3></div>", unsafe_allow_html=True)
        if st.button("✨ 내 마음 기록하러 가기", use_container_width=True):
            st.session_state.page = "login"; st.rerun()

def login_page():
    sh = init_db()
    c1, c2 = st.columns([0.6, 0.4])
    with c1:
        st.markdown("<div style='padding-top: 5rem;'><h1 class='animated-title'>MOODIARY</h1><p style='font-size: 1.5rem;'>감정을 기록하고<br>나를 위한 처방을 받으세요.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='background: rgba(255,255,255,0.4); padding: 2rem; border-radius: 20px;'>", unsafe_allow_html=True)
        t1, t2 = st.tabs(["🔑 로그인", "📝 회원가입"])
        with t1:
            lid, lpw = st.text_input("아이디", key="lid"), st.text_input("비밀번호", type="password", key="lpw")
            if st.button("로그인", use_container_width=True):
                users = get_all_users(sh)
                if lid in users and str(users[lid]) == str(lpw):
                    st.session_state.logged_in, st.session_state.username = True, lid
                    today = datetime.now(KST).strftime("%Y-%m-%d")
                    st.session_state.page = "dashboard" if today in get_user_diaries(sh, lid) else "write"
                    st.rerun()
                else: st.error("계정 정보를 확인하세요.")
        with t2:
            nid, npw = st.text_input("새 아이디", key="nid"), st.text_input("새 비밀번호(4자리)", type="password", key="npw", max_chars=4)
            if st.button("가입하기", use_container_width=True):
                if nid and len(npw)==4:
                    if nid in get_all_users(sh): st.error("이미 있는 아이디입니다.")
                    elif add_diary(sh, nid, "init", "init", "init"): # 유저 생성을 위해 append_row 로직 대신 시트 직접 접근 필요하나 구조 유지 위해 diary함수 활용 가능성 검토
                        # 실제 add_user 함수 호출이 안전
                        try:
                            sh.worksheet("users").append_row([nid, npw])
                            st.success("가입 성공! 로그인하세요.")
                        except: st.error("가입 실패")
                else: st.error("형식을 확인하세요.")
        st.markdown("</div>", unsafe_allow_html=True)

def main_app():
    sh = init_db()
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        dark = st.checkbox("🌙 야간 모드", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode: st.session_state.dark_mode = dark; st.rerun()
        st.divider()
        if st.button("📝 일기 작성", use_container_width=True): st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()
        if st.button("📊 통계 보기", use_container_width=True): st.session_state.page = "stats"; st.rerun()
        if st.button("📂 행복 저장소", use_container_width=True): st.session_state.page = "happy"; st.rerun()
        st.divider()
        if st.button("🚪 로그아웃", use_container_width=True):
            st.session_state.logged_in = False; st.session_state.page = "intro"; st.rerun()

    if st.session_state.page == "write": page_write(sh)
    elif st.session_state.page == "dashboard": page_dashboard(sh)
    elif st.session_state.page == "result": page_recommend(sh)
    elif st.session_state.page == "stats": page_stats(sh)
    elif st.session_state.page == "happy": page_happy_storage(sh)

def page_write(sh):
    st.markdown("## 📝 오늘의 이야기")
    model, tok, dev, labs = load_emotion_model()
    txt = st.text_area("오늘 하루는 어땠나요?", height=300, placeholder="자유롭게 적어주세요...")
    if st.button("🔍 감정 분석하고 저장하기", type="primary", use_container_width=True):
        if not txt.strip(): st.warning("내용을 입력해주세요."); return
        with st.spinner("AI 분석 중..."):
            emo, _ = analyze_diary(txt, model, tok, dev, labs)
            st.session_state.final_emotion = emo
            st.session_state.music_recs, st.session_state.movie_recs = recommend_music(emo), recommend_movies(emo)
            add_diary(sh, st.session_state.username, datetime.now(KST).strftime("%Y-%m-%d"), emo, txt)
            st.session_state.page = "result"; st.rerun()

def page_dashboard(sh):
    st.markdown("## 📅 감정 달력")
    diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    for d, v in diaries.items():
        if d == "init": continue
        meta = EMOTION_META.get(v['emotion'], EMOTION_META["중립"])
        events.append({"start": d, "display": "background", "backgroundColor": meta["color"]})
        events.append({"title": meta["emoji"], "start": d, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": "#000"})
    calendar(events=events, options={"initialView": "dayGridMonth"})
    if st.button("✏️ 새 일기 쓰기", use_container_width=True): st.session_state.page = "write"; st.rerun()

def page_recommend(sh):
    emo = st.session_state.get("final_emotion", "중립")
    meta = EMOTION_META[emo]
    st.markdown(f"### {meta['emoji']} 오늘의 감정: {emo}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎵 추천 음악")
        for m in st.session_state.get("music_recs", []):
            components.iframe(f"https://open.spotify.com/embed/track/{m['id']}?utm_source=generator", height=80)
    with c2:
        st.markdown("#### 🎬 추천 영화")
        for mv in st.session_state.get("movie_recs", []):
            st.write(f"**{mv['title']}** ({mv['year']}) ⭐{mv['rating']}")
            if mv['poster']: st.image(mv['poster'], width=150)
    if st.button("📅 달력으로 이동", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()

def page_stats(sh):
    st.markdown("## 📊 감정 통계")
    diaries = get_user_diaries(sh, st.session_state.username)
    data = [v['emotion'] for d, v in diaries.items() if d != "init"]
    if data: st.bar_chart(pd.Series(data).value_counts())
    else: st.info("기록이 없습니다.")

# ⭐️ 요청하신 행복 저장소 (일렬 배치 + 월별 구분)
def page_happy_storage(sh):
    st.markdown("## 📂 행복 저장소")
    st.markdown("내가 **'기쁨'**을 느꼈던 순간들만 월별로 모아봤어요. 🥰")
    
    diaries = get_user_diaries(sh, st.session_state.username)
    # 기쁨 일기만 필터링 (초기 데이터 제외)
    happy_list = [{"date": d, "text": v["text"]} for d, v in diaries.items() if v["emotion"] == "기쁨" and d != "init"]
    
    if not happy_list:
        st.info("아직 기록된 '기쁨'의 순간이 없어요.")
    else:
        # 날짜 내림차순 정렬
        happy_list.sort(key=lambda x: x["date"], reverse=True)
        
        current_month = ""
        for item in happy_list:
            # 날짜에서 년-월 추출 (ex: 2025-12)
            month_str = item["date"][:7] 
            year, month = month_str.split("-")
            
            # 월이 바뀌면 헤더 출력
            if month_str != current_month:
                st.markdown(f"<div class='month-header'>{year}년 {month}월</div>", unsafe_allow_html=True)
                current_month = month_str
            
            # 일기 카드 출력 (일렬)
            st.markdown(f"""
                <div class="happy-card">
                    <div class="happy-date">{item['date']} {EMOTION_META['기쁨']['emoji']}</div>
                    <div class="happy-text">{item['text']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    st.divider()
    if st.button("📅 달력 보기", use_container_width=True):
        st.session_state.page = "dashboard"; st.rerun()

# --- 실행부 ---
if st.session_state.logged_in: main_app()
elif st.session_state.page == "intro": intro_page()
else: login_page()
