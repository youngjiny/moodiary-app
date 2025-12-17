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

# --- 3) 커스텀 CSS (1번 디자인 + 2번 사이드바) ---
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    
    if is_dark:
        bg_start = "#121212"
        bg_mid = "#2c2c2c"
        bg_end = "#403A4E"
        main_bg = "rgba(40, 40, 40, 0.9)"
        main_text = "#f0f0f0"       
        secondary_text = "#bbbbbb"  
        sidebar_bg = "#1e1e1e"
        menu_checked = "#A29BFE"
        card_bg = "#3a3a3a"          
        card_text_happy = "#ffffff" 
        stat_card_line = "1px solid #444444" 
    else:
        bg_start = "#ee7752"
        bg_mid = "#e73c7e"
        bg_end = "#23d5ab"
        main_bg = "rgba(255, 255, 255, 0.85)"
        main_text = "#333333"
        secondary_text = "#666666"
        sidebar_bg = "#f8f9fa"
        menu_checked = "#6C5CE7"
        card_bg = "#fff9c4"
        card_text_happy = "#2c3e50"
        stat_card_line = "none"

    css = f"""
        <style>
        /* 1. 폰트 설정 (Noto Sans KR 통일) */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {main_text}; font-weight: 700; }}

        /* 2. 배경 애니메이션 */
        @keyframes gradient {{
            0% {{background-position: 0% 50%;}}
            50% {{background-position: 100% 50%;}}
            100% {{background-position: 0% 50%;}}
        }}
        .stApp {{
            background: linear-gradient(-45deg, {bg_start}, {bg_mid}, {bg_end});
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }}

        /* 3. 메인 컨테이너 (글래스모피즘) */
        .block-container {{
            background: {main_bg};
            backdrop-filter: blur(15px);
            border-radius: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
            padding: 3rem !important;
            margin-top: 2rem;
            max-width: 1000px;
        }}
        
        /* 4. 텍스트 가시성 확보 */
        p, label, .stMarkdown, .stTextarea, .stTextInput, .stCheckbox, [data-testid^="stBlock"] {{ color: {main_text} !important; }}
        section[data-testid="stSidebar"] * {{ color: {main_text} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}
        
        /* 감정 설명 문구 */
        .stMarkdown h4 {{ color: {secondary_text} !important; }} 
        .stTextInput, .stTextarea {{ color: {secondary_text} !important; }}

        /* 5. 버튼 스타일 */
        .stButton > button {{
            width: 100%; border-radius: 20px; border: none;
            background: linear-gradient(90deg, #6C5CE7 0%, #a29bfe 100%);
            color: white; font-weight: 700; padding: 0.6rem 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;
        }}
        .stButton > button:hover {{ transform: translateY(-2px); filter: brightness(1.1); }}

        /* 6. 사이드바 메뉴 버튼 (2번 스타일 - 심플하게) */
        section[data-testid="stSidebar"] .stButton > button {{
            color: {main_text}; 
            background: none !important; 
            font-weight: 600;
            box-shadow: none !important;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            color: {menu_checked}; 
            background: none !important; 
            transform: none;
        }}

        /* 7. 행복 저장소 카드 */
        .happy-card {{
            background: {card_bg}; border-left: 6px solid #FFD700;
            padding: 25px; border-radius: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-bottom: 15px;
        }}
        .happy-date {{ color: {main_text}; font-weight: 700; margin-bottom: 12px; }}
        .happy-text {{ font-size: 1.4em; font-weight: 600; line-height: 1.5; color: {card_text_happy}; }}

        /* 8. 통계 요약 카드 */
        .stat-card {{
            background: transparent;
            box-shadow: none;
            padding: 10px 0; 
            border: none; 
            text-align: center;
        }}
        .stat-card:first-child {{ border-right: {stat_card_line}; }} 
        
        /* 9. MOODIARY 텍스트 색상 애니메이션 */
        @keyframes color-shift {{
            0% {{ color: #6C5CE7; }}
            33% {{ color: #FF7675; }}
            66% {{ color: #23a6d5; }}
            100% {{ color: #6C5CE7; }}
        }}
        .animated-title {{ font-size: 3.5rem !important; font-weight: 800; animation: color-shift 5s ease-in-out infinite alternate; }}

        /* 10. 사이드바 숨김 처리 (인트로/로그인 시) */
        { 'section[data-testid="stSidebar"] { display: none !important; }' if st.session_state.page in ["intro", "login"] else '' }

        header {{visibility: hidden;}} footer {{visibility: hidden;}}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 4) 구글 시트 DB ---
@st.cache_resource
def get_gsheets_client():
    try:
        creds = st.secrets["connections"]["gsheets"]
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds, scopes=scope)
        return gspread.authorize(credentials)
    except:
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

# --- 5) AI 모델 ---
@st.cache_resource
def load_emotion_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        cfg_id2label = getattr(model.config, "id2label", None)
        if isinstance(cfg_id2label, dict) and cfg_id2label: 
            id2label = {int(k): v for k, v in cfg_id2label.items()}
        else: 
            id2label = {0: "기쁨", 1: "분노", 2: "불안", 3: "슬픔", 4: "중립", 5: "힘듦"}
        return model, tokenizer, device, id2label
    except:
        return None, None, None, None

def analyze_diary(text, model, tokenizer, device, id2label):
    if not text or model is None: return None, 0.0
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(device)
    with torch.no_grad(): logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    score = float(probs[pred_id].cpu().item())
    return id2label.get(pred_id, "중립"), score

# --- 6) 추천 시스템 ---
@st.cache_resource
def get_spotify_client():
    if not SPOTIPY_AVAILABLE: return None
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        sp = spotipy.Spotify(client_credentials_manager=manager, retries=3, backoff_factor=0.3)
        sp.search(q="test", limit=1)
        return sp
    except:
        return None

def recommend_music(emotion):
    sp = get_spotify_client()
    if not sp: return [{"error": "Spotify 연결 실패"}]
    
    SEARCH_KEYWORDS = {
        "기쁨": ["신나는 K-Pop", "Upbeat", "Happy Hits"], 
        "슬픔": ["Ballad", "Sad Songs", "새벽 감성"],
        "분노": ["Rock", "Hip Hop", "Workout"], 
        "불안": ["Lofi", "Piano", "Calm"],
        "힘듦": ["Healing", "Acoustic", "Comfort"], 
        "중립": ["Chill", "K-Pop", "Daily"]
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
                    if t and t.get("id"): 
                        valid_tracks.append({"id": t["id"], "title": t["name"]})
                if len(valid_tracks) >= 10: break
            except: continue
        
        if not valid_tracks: return [{"error": "곡 없음"}]
        seen = set(); unique = []
        for v in valid_tracks:
            if v["id"] not in seen: unique.append(v); seen.add(v["id"])
        return random.sample(unique, k=min(3, len(unique)))
    except Exception as e: 
        return [{"error": f"오류: {e}"}]

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or EMERGENCY_TMDB_KEY
    if not key: return [{"error": "API 키 없음"}]
    
    GENRES = {"기쁨": "35|10749", "분노": "28|12", "불안": "16|10751", "슬픔": "18", "힘듦": "18|10402", "중립": "35|18"}
    
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={
            "api_key": key, "language": "ko-KR", "sort_by": "popularity.desc",
            "with_genres": GENRES.get(emotion, "18"), "without_genres": "16",
            "page": random.randint(1, 5), "vote_count.gte": 500, "primary_release_date.gte": "2000-01-01"
        }, timeout=10)
        
        results = r.json().get("results", [])
        filtered_results = [m for m in results if m.get("vote_average", 0.0) >= 7.5 and m.get("vote_count", 0) >= 500]
        
        if not filtered_results: return [{"error": "조건에 맞는 영화가 없습니다"}]
        picks = random.sample(filtered_results, min(3, len(filtered_results)))
        
        return [{
            "title": m["title"], 
            "year": (m.get("release_date") or "")[:4], 
            "rating": m["vote_average"], 
            "overview": m["overview"], 
            "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None
        } for m in picks]
    except Exception as e: 
        return [{"error": f"오류: {e}"}]

# --- 7) 세션 상태 초기화 ---
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

apply_custom_css()

# --- 8) 표지 페이지 ---
def intro_page():
    st.write("")
    st.write("")
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("""
            <div style='text-align: center; padding: 40px; border-radius: 20px;'>
                <h1 class='animated-title'>MOODIARY</h1>
                <h3 style='color: #888; font-weight: normal; font-size: 2rem;'>당신의 감정은?</h3>
                <br>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("✨ 내 마음 기록하러 가기", use_container_width=True, key="intro_start"):
            st.session_state.page = "login"
            st.rerun()

# --- 9) 로그인 페이지 ---
def login_page():
    sh = init_db()
    c1, c2 = st.columns([0.6, 0.4])

    with c1:
        st.markdown("""
            <div style='padding-top: 5rem;'>
                <h1 class='animated-title'>MOODIARY</h1>
                <p style='font-size: 1.5rem; color:#555;'>오늘의 감정을 기록하고<br>나를 위한 처방을 받아보세요.</p>
            </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
        
        if sh is None:
            st.warning("⚠️ DB 연결 중입니다...")
            if st.button("🔄 새로고침"): st.rerun()
            return

        with tab1:
            lid = st.text_input("아이디", key="lid")
            lpw = st.text_input("비밀번호", type="password", key="lpw")
            if st.button("로그인", use_container_width=True, key="login_btn"):
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
            if st.button("가입하기", use_container_width=True, key="signup_btn"):
                users = get_all_users(sh)
                if str(nid) in users: st.error("이미 존재함")
                elif len(nid)<1 or len(npw)!=4: st.error("형식 확인 (비번 4자리)")
                else:
                    if add_user(sh, nid, npw): st.success("가입 성공! 로그인하세요.")
                    else: st.error("가입 실패")
                st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# --- 10) 메인 앱 (2번 스타일 사이드바) ---
def main_app():
    sh = init_db()
    if sh is None:
        st.error("데이터베이스 연결 끊김. 새로고침 해주세요.")
        if st.button("🔄 새로고침"): st.rerun()
        return

    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        st.divider()
        
        if st.button("📝 일기 작성", use_container_width=True, key="sb_write"): 
            st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력", use_container_width=True, key="sb_calendar"): 
            st.session_state.page = "dashboard"; st.rerun()
        if st.button("🎵 음악/영화 추천", use_container_width=True, key="sb_recommend"): 
            st.session_state.page = "result"; st.rerun()
        if st.button("📊 통계 보기", use_container_width=True, key="sb_stats"): 
            st.session_state.page = "stats"; st.rerun()
        if st.button("📂 행복 저장소", use_container_width=True, key="sb_happy"): 
            st.session_state.page = "happy"; st.rerun()

        st.divider()
        if st.button("🚪 로그아웃", use_container_width=True, key="sb_logout"):
            st.session_state.logged_in = False
            st.session_state.page = "intro"
            st.rerun()

    if st.session_state.page == "write": page_write(sh)
    elif st.session_state.page == "dashboard": page_dashboard(sh)
    elif st.session_state.page == "result": page_recommend(sh)
    elif st.session_state.page == "stats": page_stats(sh)
    elif st.session_state.page == "happy": page_happy_storage(sh)

# --- 11) 일기 작성 ---
def page_write(sh):
    st.markdown("## 📝 오늘의 이야기")
    model, tokenizer, device, id2label = load_emotion_model()
    if not model: st.error("AI 로드 실패"); return

    if "diary_input" not in st.session_state: st.session_state.diary_input = ""
    txt = st.text_area("오늘 하루는 어땠나요?", value=st.session_state.diary_input, height=300, 
                       placeholder="오늘 있었던 일과 감정을 자유롭게 적어주세요...", key="diary_text_input")
    
    if st.button("🔍 감정 분석하고 저장하기", type="primary", use_container_width=True, key="write_save"):
        if not txt.strip(): 
            st.warning("내용을 입력해주세요.")
            st.session_state.diary_input = txt
            st.rerun()
            return
            
        with st.spinner("분석 중..."):
            emo, sc = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            
            today = datetime.now(KST).strftime("%Y-%m-%d")
            add_diary(sh, st.session_state.username, today, emo, txt)
            
            st.session_state.page = "result"
            st.rerun()

# --- 12) 대시보드 ---
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
        text_color = "#f0f0f0" if st.session_state.get("dark_mode", False) else "#000000"
        events.append({"start": date_str, "display": "background", "backgroundColor": meta["color"]})
        events.append({"title": meta["emoji"], "start": date_str, "allDay": True, 
                      "backgroundColor": "transparent", "borderColor": "transparent", "textColor": text_color})
    
    calendar(events=events, options={"headerToolbar": {"left": "prev,next today", "center": "title", "right": ""}, "initialView": "dayGridMonth"},
              custom_css="""
              .fc-event-title { font-size: 3em !important; display: flex; justify-content: center; align-items: center; height: 100%; transform: translateY(-25px); text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
              .fc-daygrid-event { border: none !important; background-color: transparent !important; }
              .fc-daygrid-day-number { z-index: 10 !important; color: var(--main-text-color, black); font-weight: bold; }
              .fc-bg-event { opacity: 1.0 !important; }
              """)
    
    st.write("")
    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    if today_str in my_diaries:
        st.success(f"오늘의 기록 완료! ({my_diaries[today_str]['emotion']})")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✏️ 일기 수정하기", use_container_width=True, key="dash_edit"):
                st.session_state.diary_input = my_diaries[today_str]["text"]
                st.session_state.page = "write"; st.rerun()
        with c2:
            if st.button("🎵 오늘의 추천 보기", type="primary", use_container_width=True, key="dash_rec"):
                emo = my_diaries[today_str]["emotion"]
                st.session_state.final_emotion = emo
                st.session_state.music_recs = recommend_music(emo)
                st.session_state.movie_recs = recommend_movies(emo)
                st.session_state.page = "result"; st.rerun()
    else:
        if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", use_container_width=True, key="dash_write"):
            st.session_state.diary_input = ""
            st.session_state.page = "write"; st.rerun()

# --- 13) 추천 페이지 ---
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
            if st.button("일기 쓰러 가기", type="primary", key="rec_gtn"):
                st.session_state.page = "write"; st.rerun()
            return

    emo = st
