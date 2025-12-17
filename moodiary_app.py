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

# 감정별 메타 데이터 (달력 및 통계용)
EMOTION_META = {
    "기쁨": {"color": "#FFD700", "emoji": "😆", "desc": "웃음이 끊이지 않는 하루!"},
    "분노": {"color": "#FF5050", "emoji": "🤬", "desc": "워워, 진정이 필요해요."},
    "불안": {"color": "#FFA032", "emoji": "😰", "desc": "마음이 조마조마해요."},
    "슬픔": {"color": "#5078FF", "emoji": "😭", "desc": "마음의 위로가 필요해요."},
    "힘듦": {"color": "#969696", "emoji": "🤯", "desc": "휴식이 절실한 하루."},
    "중립": {"color": "#50B478", "emoji": "😐", "desc": "평온하고 무난한 하루."}
}

KST = timezone(timedelta(hours=9))

st.set_page_config(layout="wide", page_title="MOODIARY", page_icon="💖")

# --- 3) 커스텀 CSS (이미지 스타일 및 달력 수정) ---
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    primary_purple = "#7B61FF"  # 이미지의 메인 보라색
    
    if is_dark:
        bg_color = "#121212"
        main_bg = "rgba(40, 40, 40, 0.95)"
        text_color = "#f0f0f0"
        card_bg = "#3a3a3a"
    else:
        bg_color = "#F8F9FA"
        main_bg = "rgba(255, 255, 255, 1.0)"
        text_color = "#333333"
        card_bg = "#ffffff"

    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        
        /* 배경 설정 */
        .stApp {{ background-color: {bg_color}; }}
        
        /* 메인 컨테이너 (중앙 정렬 및 라운딩) */
        .block-container {{ 
            background: {main_bg}; 
            border-radius: 30px; 
            padding: 4rem !important; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.05);
            margin-top: 2rem;
            max-width: 900px;
        }}

        /* 이미지 스타일 타이틀 */
        .main-title {{
            font-size: 4.5rem !important;
            font-weight: 900 !important;
            color: {primary_purple};
            margin-bottom: 0.5rem;
            letter-spacing: -2px;
            text-align: center;
        }}
        .main-subtitle {{
            font-size: 1.6rem;
            font-weight: 700;
            color: #333333;
            margin-bottom: 3rem;
            text-align: center;
        }}

        /* 이미지 속 보라색 라운드 버튼 */
        div.stButton > button {{
            background-color: {primary_purple} !important;
            color: white !important;
            border-radius: 50px !important;
            padding: 0.7rem 2.5rem !important;
            font-size: 1.1rem !important;
            font-weight: 600 !important;
            border: none !important;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(123, 97, 255, 0.3) !important;
            width: auto;
            margin: 0 auto;
            display: block;
        }}
        div.stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(123, 97, 255, 0.4) !important;
            background-color: #6649FF !important;
        }}

        /* 달력 커스텀 (칸 전체 채우기 및 이모지 중앙화) */
        .fc-daygrid-day-frame {{ min-height: 120px !important; cursor: pointer; }}
        .fc-bg-event {{ opacity: 1.0 !important; border-radius: 5px; }}
        .fc-event-title {{ 
            font-size: 2.5em !important; 
            text-align: center; 
            display: flex; 
            justify-content: center; 
            align-items: center; 
            height: 90px;
        }}
        
        /* 영화 카드 */
        .movie-card {{
            background: {card_bg};
            border-radius: 15px; padding: 15px; margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); display: flex; gap: 15px;
        }}
        .movie-card img {{ width: 110px; border-radius: 10px; object-fit: cover; }}
        
        /* 사이드바 보이지 않게 처리 (인트로/로그인 시) */
        { 'section[data-testid="stSidebar"] { display: none; }' if st.session_state.page in ["intro", "login"] else '' }
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 4) DB 및 AI 로직 (기존 로직 유지) ---
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

def recommend_music(emotion):
    if not SPOTIPY_AVAILABLE: return []
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        sp = spotipy.Spotify(client_credentials_manager=manager)
        query = random.choice(["Daily Mix", "K-Pop Trend"])
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

# --- 5) 각 페이지 구현 ---

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

apply_custom_css()

def intro_page():
    st.markdown("<div style='height: 100px;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='main-title'>MOODIARY</div>", unsafe_allow_html=True)
    st.markdown("<div class='main-subtitle'>오늘 당신의 마음은 어떤가요?</div>", unsafe_allow_html=True)
    if st.button("✨ 내 마음 기록하러 가기"):
        st.session_state.page = "login"
        st.rerun()

def login_page():
    sh = init_db()
    st.markdown("<div class='main-title' style='font-size: 3rem !important;'>MOODIARY</div>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
    if not sh: st.error("DB 연결 실패"); return
    
    with tab1:
        lid = st.text_input("아이디", key="l_id")
        lpw = st.text_input("비밀번호", type="password", key="l_pw")
        if st.button("로그인"):
            users = get_all_users(sh)
            if lid in users and users[lid] == str(lpw):
                st.session_state.logged_in, st.session_state.username = True, lid
                st.session_state.page = "dashboard"; st.rerun()
            else: st.error("정보가 올바르지 않습니다.")
            
    with tab2:
        nid = st.text_input("새 아이디", key="n_id")
        npw = st.text_input("새 비밀번호 (4자리)", type="password", max_chars=4, key="n_pw")
        if st.button("가입하기"):
            if add_user(sh, nid, npw): st.success("가입 완료! 로그인 해주세요."); st.rerun()

def main_app():
    sh = init_db()
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        st.divider()
        if st.button("📝 일기 작성", use_container_width=True): st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()
        if st.button("📊 감정 통계", use_container_width=True): st.session_state.page = "stats"; st.rerun()
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
    model, tokenizer, device, id2label = load_emotion_model()
    txt = st.text_area("오늘 하루는 어땠나요?", height=250, placeholder="여기에 당신의 마음을 적어보세요.")
    if st.button("🔍 분석 및 저장"):
        if not txt.strip(): st.warning("내용을 입력해주세요."); return
        with st.spinner("감정을 분석하고 있습니다..."):
            emo, sc = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            add_diary(sh, st.session_state.username, datetime.now(KST).strftime("%Y-%m-%d"), emo, txt)
            st.session_state.page = "result"; st.rerun()

def page_recommend(sh):
    emo = st.session_state.get("final_emotion", "중립")
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])
    st.markdown(f"<h1 style='text-align:center;'>{meta['emoji']} 오늘의 감정은 <span style='color:{meta['color']}'>{emo}</span></h1>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎵 추천 음악")
        for item in st.session_state.get("music_recs", []):
            components.iframe(f"https://open.spotify.com/embed/track/{item['id']}", height=80)
    with c2:
        st.markdown("#### 🎬 추천 영화")
        for item in st.session_state.get("movie_recs", []):
            st.markdown(f"""<div class="movie-card"><img src="{item['poster']}"><div><b>{item['title']}</b><br><small>{item['year']}</small><br>⭐{item['rating']}</div></div>""", unsafe_allow_html=True)

def page_dashboard(sh):
    st.markdown("## 📅 감정 달력")
    my_diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    for d, data in my_diaries.items():
        meta = EMOTION_META.get(data['emotion'], EMOTION_META["중립"])
        # 배경색 이벤트 (칸 전체 색칠)
        events.append({"start": d, "display": "background", "backgroundColor": meta["color"]})
        # 이모지 이벤트 (중앙 표시, 투명 배경으로 파란 선 방지)
        events.append({
            "title": meta["emoji"], 
            "start": d, 
            "allDay": True,
            "backgroundColor": "rgba(0,0,0,0)", 
            "borderColor": "rgba(0,0,0,0)",
            "textColor": "#000"
        })
    
    calendar(events=events, options={"initialView": "dayGridMonth", "locale": "ko"})

def page_stats(sh):
    st.markdown("## 📊 감정 통계")
    diaries = get_user_diaries(sh, st.session_state.username)
    if not diaries: st.info("아직 기록이 없습니다."); return
    df = pd.DataFrame([{"emotion": d['emotion']} for d in diaries.values()])
    counts = df['emotion'].value_counts().reindex(EMOTION_META.keys(), fill_value=0).reset_index()
    counts.columns = ['emotion', 'count']
    color_range = [m['color'] for m in EMOTION_META.values()]
    st.vega_lite_chart(counts, {
        "mark": {"type": "bar", "cornerRadius": 5},
        "encoding": {
            "x": {"field": "emotion", "type": "nominal", "axis": {"labelAngle": 0}, "sort": list(EMOTION_META.keys())},
            "y": {"field": "count", "type": "quantitative"},
            "color": {"field": "emotion", "scale": {"domain": list(EMOTION_META.keys()), "range": color_range}, "legend": None}
        }
    }, use_container_width=True)

def page_happy_storage(sh):
    st.markdown("## 📂 행복 저장소")
    diaries = get_user_diaries(sh, st.session_state.username)
    happy_list = [(date, d['text']) for date, d in diaries.items() if d['emotion'] == "기쁨"]
    if not happy_list: st.info("아직 기쁜 기록이 없네요."); return
    for date, text in sorted(happy_list, reverse=True):
        st.info(f"📅 **{date}**\n\n{text}")

# --- 6) 라우팅 ---
if st.session_state.logged_in: main_app()
elif st.session_state.page == "intro": intro_page()
else: login_page()
