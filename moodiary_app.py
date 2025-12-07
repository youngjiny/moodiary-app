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
EMOTION_MODEL_ID = "JUDONGHYEOK/6-emotion-bert-korean-v2"
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

# ⭐️ 커스텀 CSS (글래스모피즘 + Sidebar 복구)
def apply_custom_css():
    st.markdown("""
        <style>
        /* 1. 폰트 설정 */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
        
        /* 2. 배경 애니메이션 */
        @keyframes gradient {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .stApp {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }

        /* 3. 메인 컨테이너 (글래스모피즘) */
        .block-container {
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(15px);
            border-radius: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            padding: 3rem !important;
            margin-top: 2rem;
            max-width: 1000px;
        }
        
        /* 4. 버튼 스타일 (메인) */
        .stButton > button {
            width: 100%;
            border-radius: 20px;
            border: none;
            background: linear-gradient(90deg, #6C5CE7 0%, #a29bfe 100%);
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
            filter: brightness(1.1);
            color: white;
        }
        
        /* 5. 사이드바 메뉴 스타일링 (라디오 버튼을 메뉴처럼 보이게) */
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] {
            border: none;
            padding: 0;
        }
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label {
            background: #f8f9fa; /* 배경색 */
            border-radius: 10px;
            margin-bottom: 5px;
            padding: 10px 15px;
            box-shadow: none;
            transition: background-color 0.1s;
        }
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label:hover {
            background: #eee;
        }
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked='true'] {
            background: #6C5CE7;
            color: white !important;
        }
        section[data-testid="stSidebar"] .stRadio > div[role="radiogroup"] label[data-checked='true'] p {
            color: white !important;
        }

        /* 6. 표지(Intro) 스타일 */
        .intro-title {
            font-size: 6rem;
            font-weight: 800;
            background: linear-gradient(to right, #6C5CE7, #FF7675);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            text-align: center;
        }
        .intro-subtitle {
            font-size: 2.5rem;
            color: #555;
            text-align: center;
            margin-bottom: 3rem;
            font-family: 'Gamja Flower', cursive !important;
        }

        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# =========================================
# 🔐 3) 구글 시트 데이터베이스
# =========================================
@st.cache_resource
def get_gsheets_client():
    try:
        creds = st.secrets["connections"]["gsheets"]
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds, scopes=scope)
        return gspread.authorize(credentials)
    except Exception as e:
        return None

@st.cache_resource(ttl=3600)
def init_db():
    client = get_gsheets_client()
    if not client: return None
    try:
        sh = client.open(GSHEET_DB_NAME)
        sh.worksheet("users")
        sh.worksheet("diaries")
        return sh
    except:
        return None 

def get_all_users(sh):
    if not sh: return {}
    try:
        rows = sh.worksheet("users").get_all_records()
        return {str(row['username']): str(row['password']) for row in rows}
    except: return {}

def add_user(sh, username, password):
    if not sh: return False
    try:
        sh.worksheet("users").append_row([str(username), str(password)])
        return True
    except: return False

@st.cache_data(ttl=10)
def get_user_diaries(_sh, username):
    if not _sh: return {}
    try:
        rows = _sh.worksheet("diaries").get_all_records()
        user_diaries = {}
        for row in rows:
            if str(row['username']) == str(username):
                user_diaries[row['date']] = {"emotion": row['emotion'], "text": row['text']}
        return user_diaries
    except: return {}

def add_diary(sh, username, date, emotion, text):
    if not sh: return False
    try:
        ws = sh.worksheet("diaries")
        cell = ws.find(date, in_column=2)
        if cell and str(ws.cell(cell.row, 1).value) == str(username):
            ws.update_cell(cell.row, 3, emotion)
            ws.update_cell(cell.row, 4, text)
        else:
            ws.append_row([username, date, emotion, text])
        get_user_diaries.clear()
        return True
    except: return False

# =========================================
# 🧠 4) AI & 추천 로직
# =========================================
@st.cache_resource
def load_emotion_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        cfg_id2label = getattr(model.config, "id2label", None)
        if isinstance(cfg_id2label, dict) and cfg_id2label: id2label = {int(k): v for k, v in cfg_id2label.items()}
        else: id2label = {0: "기쁨", 1: "분노", 2: "불안", 3: "슬픔", 4: "중립", 5: "힘듦"}
        return model, tokenizer, device, id2label
    except Exception as e: return None, None, None, None

def analyze_diary(text, model, tokenizer, device, id2label):
    if not text or model is None: return None, 0.0
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(device)
    with torch.no_grad(): logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    score = float(probs[pred_id].cpu().item())
    return id2label.get(pred_id, "중립"), score

@st.cache_resource
def get_spotify_client():
    if not SPOTIPY_AVAILABLE: return "라이브러리 없음"
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        sp = spotipy.Spotify(client_credentials_manager=manager, retries=3, backoff_factor=0.3)
        sp.search(q="test", limit=1)
        return sp
    except: return "로그인 실패"

def recommend_music(emotion):
    sp = get_spotify_client()
    if not isinstance(sp, spotipy.Spotify): return [{"error": sp}]
    SEARCH_KEYWORDS = {
        "기쁨": ["신나는 K-Pop", "Upbeat", "Happy Hits"], "슬픔": ["Ballad", "Sad Songs", "새벽 감성"],
        "분노": ["Rock", "Hip Hop", "Workout"], "불안": ["Lofi", "Piano", "Calm"],
        "힘듦": ["Healing", "Acoustic", "Comfort"], "중립": ["Chill", "K-Pop", "Daily"]
    }
    query = random.choice(SEARCH_KEYWORDS.get(emotion, SEARCH_KEYWORDS["중립"]))
    try:
        results = sp.search(q=query, type="playlist", limit=10, market="KR")
        playlists = results.get("playlists", {}).get("items", [])
        if not playlists: return [{"error": "검색 실패"}]
        valid_tracks = []
        random.shuffle(playlists)
        for pl in playlists:
            try:
                tracks = sp.playlist_items(pl["id"], limit=30)
                items = tracks.get("items", []) if tracks else []
                for it in items:
                    t = it.get("track")
                    if t and t.get("id"): valid_tracks.append({"id": t["id"], "title": t["name"]})
                if len(valid_tracks) >= 10: break
            except: continue
        if not valid_tracks: return [{"error": "곡 없음"}]
        seen = set(); unique = []
        for v in valid_tracks:
            if v["id"] not in seen: unique.append(v); seen.add(v["id"])
        return random.sample(unique, k=min(3, len(unique)))
    except Exception as e: return [{"error": f"오류: {e}"}]

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or st.secrets.get("TMDB_API_KEY") or EMERGENCY_TMDB_KEY
    if not key: return [{"text": "API 키 없음", "poster": None}]
    GENRES = {"기쁨": "35|10749", "분노": "28|12", "불안": "16|10751", "슬픔": "18", "힘듦": "18|10402", "중립": "35|18"}
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={
            "api_key": key, "language": "ko-KR", "sort_by": "popularity.desc",
            "with_genres": GENRES.get(emotion, "18"), "without_genres": "16",
            "page": random.randint(1, 5), "vote_count.gte": 500, "primary_release_date.gte": "2000-01-01"
        }, timeout=5)
        results = r.json().get("results", [])
        if not results: return [{"text": "영화 없음", "poster": None}]
        picks = random.sample(results, min(3, len(results)))
        return [{"title": m["title"], "year": (m.get("release_date") or "")[:4], "rating": m["vote_average"], "overview": m["overview"], "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in picks]
    except Exception as e: return [{"text": f"오류: {e}", "poster": None}]

# =========================================
# 🖥️ 화면 및 네비게이션 로직
# =========================================
apply_custom_css()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 

# 0. 표지 (Intro) 페이지
def intro_page():
    st.write("")
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
            <div style='text-align: center; padding: 40px; border-radius: 20px;'>
                <h1 class='intro-title'>MOODIARY</h1>
                <h3 class='intro-subtitle'>당신의 감정은?</h3>
                <br>
            </div>
        """, unsafe_allow_html=True)
        if st.button("✨ 내 마음 기록하러 가기", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

# 1. 로그인 페이지
def login_page():
    sh = init_db()
    
    c1, c2 = st.columns([0.6, 0.4])

    with c1:
        st.markdown("""
            <div style='padding-top: 5rem;'>
                <h1 style='font-size: 4rem; color:#333;'>MOODIARY</h1>
                <p style='font-size: 1.5rem; color:#555;'>오늘의 감정을 기록하고<br>나를 위한 처방을 받아보세요.</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
        
        if sh is None:
            st.warning("⚠️ DB 연결 중...")
            if st.button("🔄 새로고침"): st.rerun()
            return

        with tab1:
            lid = st.text_input("아이디", key="lid")
            lpw = st.text_input("비밀번호", type="password", key="lpw")
            if st.button("로그인", use_container_width=True):
                users = get_all_users(sh)
                if str(lid) in users and str(users[str(lid)]) == str(lpw):
                    st.session_state.logged_in = True
                    st.session_state.username = lid
                    
                    today_str = datetime.now(KST).strftime("%Y-%m-%d")
                    diaries = get_user_diaries(sh, lid)
                    if today_str in diaries: st.session_state.page = "dashboard"
                    else: st.session_state.page = "write"
                    st.rerun()
                else: st.error("아이디/비밀번호 오류")
                
            with tab2:
                nid = st.text_input("새 아이디", key="nid")
                npw = st.text_input("새 비밀번호 (4자리)", type="password", key="npw", max_chars=4)
                if st.button("가입하기", use_container_width=True):
                    users = get_all_users(sh)
                    if str(nid) in users: st.error("이미 존재함")
                    elif len(nid)<1 or len(npw)!=4: st.error("형식 확인 (비번 4자리)")
                    else:
                        if add_user(sh, nid, npw): st.success("가입 성공! 로그인하세요.")
                        else: st.error("가입 실패")
        st.markdown("</div>", unsafe_allow_html=True)

# 2. 메인 앱
def main_app():
    sh = init_db()
    if sh is None:
        st.error("데이터베이스 연결 끊김. 새로고침 해주세요.")
        if st.button("🔄 새로고침"): st.rerun()
        return

    # --- 사이드바 ---
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        st.write("")
        
        # ⭐️ 사이드바 메뉴 (목차 복구)
        if st.button("📝 일기 작성", use_container_width=True): st.session_state.page = "write"; st.rerun()
        if st.button("📅 달력 보기", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()
        if st.button("🎵 음악/영화 추천", use_container_width=True): st.session_state.page = "result"; st.rerun()
        if st.button("📊 통계 보기", use_container_width=True): st.session_state.page = "stats"; st.rerun()
        if st.button("📂 행복 저장소", use_container_width=True): st.session_state.page = "happy"; st.rerun()
        
        st.divider()
        if st.button("🚪 로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "intro"
            st.rerun()

    # --- 라우팅 ---
    if st.session_state.page == "write": page_write(sh)
    elif st.session_state.page == "dashboard": page_dashboard(sh)
    elif st.session_state.page == "result": page_recommend(sh)
    elif st.session_state.page == "stats": page_stats(sh)
    elif st.session_state.page == "happy": page_happy_storage(sh)

# --- 페이지 함수들 ---
def page_write(sh):
    st.markdown("## 📝 오늘의 이야기")
    model, tokenizer, device, id2label = load_emotion_model()
    if not model: st.error("AI 로드 실패"); return

    if "diary_input" not in st.session_state: st.session_state.diary_input = ""
    txt = st.text_area("오늘 하루는 어땠나요?", value=st.session_state.diary_input, height=300, placeholder="오늘 있었던 일과 감정을 자유롭게 적어주세요...")
    
    if st.button("🔍 감정 분석하고 저장하기", type="primary", use_container_width=True):
        if not txt.strip(): st.warning("내용을 입력해주세요."); return
        with st.spinner("분석 중..."):
            emo, sc = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            today = datetime.now(KST).strftime("%Y-%m-%d")
            add_diary(sh, st.session_state.username, today, emo, txt)
            st.session_state.page = "result"
            st.rerun()

def page_dashboard(sh):
    st.markdown("## 📅 감정 달력")
    cols = st.columns(6)
    for i, (k, v) in enumerate(EMOTION_META.items()):
        cols[i].markdown(f"<span style='color:{v['color'].replace('0.6','1')}; font-size:1.5em;'>●</span> {k}", unsafe_allow_html=True)
    
    my_diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    for date_str, data in my_diaries.items():
        emo = data.get("emotion", "중립")
        if emo not in EMOTION_META: emo = "중립"
        meta = EMOTION_META[emo]
        events.append({"start": date_str, "display": "background", "backgroundColor": meta["color"]})
        events.append({"title": meta["emoji"], "start": date_str, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": "#000000"})
    
    calendar(events=events, options={"headerToolbar": {"left": "prev,next today", "center": "title", "right": ""}, "initialView": "dayGridMonth"},
             custom_css="""
             .fc-event-title { font-size: 3em !important; display: flex; justify-content: center; align-items: center; height: 100%; transform: translateY(-25px); text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
             .fc-daygrid-event { border: none !important; background-color: transparent !important; }
             .fc-daygrid-day-number { z-index: 10 !important; color: black; font-weight: bold; }
             .fc-bg-event { opacity: 1.0 !important; }
             .fc-col-header-cell-cushion { color: #333; font-weight: bold; }
             """)
    
    st.write("")
    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    if today_str in my_diaries:
        st.success(f"오늘의 기록 완료! ({my_diaries[today_str]['emotion']})")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✏️ 일기 수정하기", use_container_width=True):
                st.session_state.diary_input = my_diaries[today_str]["text"]
                st.session_state.page = "write"
                st.rerun()
        with c2:
            if st.button("🎵 오늘의 추천 보기", type="primary", use_container_width=True):
                emo = my_diaries[today_str]["emotion"]
                st.session_state.final_emotion = emo
                st.session_state.music_recs = recommend_music(emo)
                st.session_state.movie_recs = recommend_movies(emo)
                st.session_state.page = "result"
                st.rerun()
    else:
        if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", use_container_width=True):
            st.session_state.diary_input = ""
            st.session_state.page = "write"
            st.rerun()

def page_recommend(sh):
    st.markdown("## 🎵 음악/영화 추천")

    if "final_emotion" not in st.session_state:
        today = datetime.now(KST).strftime("%Y-%m-%d")
        diaries = get_user_diaries(sh, st.session_state.username)
        if today in diaries:
            st.session_state.final_emotion = diaries[today]['emotion']
            st.session_state.music_recs = recommend_music(st.session_state.final_emotion)
            st.session_state.movie_recs = recommend_movies(st.session_state.final_emotion)
        else:
            st.info("작성된 일기가 없습니다.")
            if st.button("일기 쓰러 가기", type="primary"):
                st.session_state.page = "write"
                st.rerun()
            return

    emo = st.session_state.final_emotion
    if emo not in EMOTION_META: emo = "중립"
    meta = EMOTION_META[emo]
    st.markdown(f"""<div style='text-align: center; padding: 2rem;'><h2 style='color: {meta['color'].replace('0.6', '1.0').replace('0.5', '1.0')}; font-size: 3rem;'>{meta['emoji']} 오늘의 감정: {emo}</h2><h4 style='color: #555;'>{meta['desc']}</h4></div>""", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎵 추천 음악")
        if st.button("🔄 음악 새로고침", use_container_width=True):
            st.session_state.music_recs = recommend_music(emo)
            st.rerun()
        for item in st.session_state.get("music_recs", []):
            if item.get('id'): components.iframe(f"https://open.spotify.com/embed/track/{item['id']}", height=250, width="100%")
    with c2:
        st.markdown("#### 🎬 추천 영화")
        if st.button("🔄 영화 새로고침", use_container_width=True):
            st.session_state.movie_recs = recommend_movies(emo)
            st.rerun()
        for item in st.session_state.get("movie_recs", []):
            if item.get('poster'):
                ic, tc = st.columns([1, 2])
                ic.image(item['poster'], use_container_width=True)
                tc.markdown(f"**{item['title']} ({item['year']})**\n⭐ {item['rating']}\n\n*{item.get('overview','')}*")

    st.divider()
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📅 달력 보기", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()
    with b2:
        if st.button("📊 통계 보기", use_container_width=True): st.session_state.page = "stats"; st.rerun()
    with b3:
        if st.button("📂 행복 저장소", use_container_width=True): st.session_state.page = "happy"; st.rerun()

def page_stats(sh):
    st.markdown("## 📊 나의 감정 통계")
    
    if "stats_year" not in st.session_state:
        now = datetime.now(KST)
        st.session_state.stats_year = now.year
        st.session_state.stats_month = now.month

    c1, c2, c3 = st.columns([0.2, 0.6, 0.2])
    with c1:
        if st.button("◀️", use_container_width=True, key="prev_stats"):
            if st.session_state.stats_month == 1:
                st.session_state.stats_year -= 1
                st.session_state.stats_month = 12
            else: st.session_state.stats_month -= 1
            st.rerun()
    with c2:
        st.markdown(f"<h3 style='text-align: center; margin:0; color: #333;'>{st.session_state.stats_year}년 {st.session_state.stats_month}월</h3>", unsafe_allow_html=True)
    with c3:
        if st.button("▶️", use_container_width=True, key="next_stats"):
            if st.session_state.stats_month == 12:
                st.session_state.stats_year += 1
                st.session_state.stats_month = 1
            else: st.session_state.stats_month += 1
            st.rerun()
    st.write("")

    my_diaries = get_user_diaries(sh, st.session_state.username)
    target_prefix = f"{st.session_state.stats_year}-{st.session_state.stats_month:02d}"
    
    month_data = []
    for date, d in my_diaries.items():
        if date.startswith(target_prefix):
            e = d['emotion']
            if e in EMOTION_META: month_data.append(e)
    
    df = pd.DataFrame(month_data, columns=['emotion'])
    counts = df['emotion'].value_counts().reindex(EMOTION_META.keys(), fill_value=0)
    
    chart_data = counts.reset_index()
    chart_data.columns = ['emotion', 'count']
    domain = list(EMOTION_META.keys())
    range_ = [m['color'].replace('0.6', '1.0').replace('0.5', '1.0') for m in EMOTION_META.values()] 
    
    max_val = int(chart_data['count'].max()) if not chart_data.empty else 5
    y_values = list(range(0, max_val + 2))

    st.vega_lite_chart(chart_data, {
        "mark": {"type": "bar", "cornerRadius": 10},
        "encoding": {
            "x": {"field": "emotion", "type": "nominal", "sort": domain, 
                  "axis": {"labelAngle": 0, "labelFontSize": 12}, "title": "감정"},
            "y": {
                "field": "count", "type": "quantitative", 
                "axis": {"values": y_values, "format": "d", "titleAngle": 0, "titleAlign": "right", "titleY": -10}, 
                "scale": {"domainMin": 0}, "title": "횟수"
            },
            "color": {"field": "emotion", "scale": {"domain": domain, "range": range_}, "legend": None},
            "tooltip": [{"field": "emotion"}, {"field": "count"}]
        }
    }, use_container_width=True)

    st.divider()
    if st.button("📂 행복 저장소 보러가기", use_container_width=True):
        st.session_state.page = "happy"
        st.rerun()

def page_happy_storage(sh):
    st.markdown("## 📂 행복 저장소")
    st.markdown("내가 **'기쁨'**을 느꼈던 순간들만 모아봤어요. 🥰")
    my_diaries = get_user_diaries(sh, st.session_state.username)
    happy_moments = {date: data for date, data in my_diaries.items() if data['emotion'] == '기쁨'}
    
    if not happy_moments:
        st.info("아직 기록된 기쁨의 순간이 없어요.")
    else:
        dates = sorted(happy_moments.keys(), reverse=True)
        for i in range(0, len(dates), 2):
            cols = st.columns(2)
            date1 = dates[i]
            data1 = happy_moments[date1]
            with cols[0]:
                st.markdown(f"""
                <div class="happy-card">
                    <div class="happy-date">{date1} {EMOTION_META['기쁨']['emoji']}</div>
                    <div class="happy-text">{data1['text']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            if i + 1 < len(dates):
                date2 = dates[i+1]
                data2 = happy_moments[date2]
                with cols[1]:
                    st.markdown(f"""
                    <div class="happy-card">
                        <div class="happy-date">{date2} {EMOTION_META['기쁨']['emoji']}</div>
                        <div class="happy-text">{data2['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.divider()
    if st.button("📊 통계 보러가기", use_container_width=True):
        st.session_state.page = "stats"
        st.rerun()

if st.session_state.logged_in: main_app()
elif st.session_state.page == "intro": intro_page()
else: login_page()
