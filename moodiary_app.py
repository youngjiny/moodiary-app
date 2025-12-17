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

# ⭐️ 커스텀 CSS (야간 모드 CSS 조건부 렌더링 및 사이드바 수정)
def apply_custom_css():
    
    is_dark = st.session_state.get("dark_mode", False)
    
    if is_dark:
        # 야간 모드 색상
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
        # 주간 모드 색상
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
        
        /* 4. ⭐️ 텍스트 가시성 확보 */
        p, label, .stMarkdown, .stTextarea, .stTextInput, .stCheckbox, [data-testid^="stBlock"] {{ color: {main_text} !important; }}
        section[data-testid="stSidebar"] * {{ color: {main_text} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}
        
        /* 감정 설명 문구 (조마조마해요 등) */
        .stMarkdown h4 {{ color: {secondary_text} !important; }} 
        /* 입력창 힌트 텍스트 가시성 보장 */
        .stTextInput, .stTextarea {{ color: {secondary_text} !important; }}


        /* 5. 버튼 스타일 */
        .stButton > button {{
            width: 100%; border-radius: 20px; border: none;
            background: linear-gradient(90deg, #6C5CE7 0%, #a29bfe 100%);
            color: white; font-weight: 700; padding: 0.6rem 1rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: all 0.3s ease;
        }}
        .stButton > button:hover {{ transform: translateY(-2px); filter: brightness(1.1); }}

        /* 6. 사이드바 메뉴 버튼 (안정화) */
        section[data-testid="stSidebar"] .stButton > button {{
            color: {main_text}; background: none; font-weight: 600;
        }}
        section[data-testid="stSidebar"] .stButton > button:hover {{
            color: {menu_checked}; background: none; transform: none;
        }}

        /* 7. ⭐️ 행복 저장소 카드 (디자인 개선 및 가시성) */
        .happy-card {{
            background: {card_bg}; border-left: 6px solid #FFD700;
            padding: 25px; border-radius: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            width: 100%;
        }}
        .happy-date {{ color: {main_text}; font-weight: 700; margin-bottom: 12px; }}
        .happy-text {{ font-size: 1.4em; font-weight: 600; line-height: 1.5; color: {card_text_happy}; }}

        /* 8. ⭐️ 통계 요약 카드 (선/배경 제거 및 가시성) */
        .stat-card {{
            background: transparent;
            box-shadow: none;
            padding: 10px 0; 
            border: none; 
            text-align: center;
        }}
        /* 통계 요약 카드 간 구분선 (수직선) */
        .stat-card:first-child {{ border-right: {stat_card_line}; }} 
        
        /* 9. MOODIARY 텍스트 색상 애니메이션 */
        @keyframes color-shift {{
            0% {{ color: #6C5CE7; }}
            33% {{ color: #FF7675; }}
            66% {{ color: #23a6d5; }}
            100% {{ color: #6C5CE7; }}
        }}
        .animated-title {{ font-size: 3.5rem !important; font-weight: 800; animation: color-shift 5s ease-in-out infinite alternate; }}

        header {{visibility: visible;}} 
        footer {{visibility: hidden;}}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

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
    except Exception as e:
        st.error(f"❌ DB 연결 실패: 시트 이름/공유 권한 확인 필요. (에러 유형: {type(e).__name__})")
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
# 🧠 4) AI & 추천 로직 (생략)
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
        
        filtered_results = [m for m in results if m.get("vote_average", 0.0) >= 7.5 and m.get("vote_count", 0) >= 500]
        
        if not filtered_results: return [{"text": "조건에 맞는 영화가 없습니다.", "poster": None}]
        picks = random.sample(filtered_results, min(3, len(filtered_results)))
        return [{"title": m["title"], "year": (m.get("release_date") or "")[:4], "rating": m["vote_average"], "overview": m["overview"], "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in picks]
    except Exception as e: return [{"text": f"오류: {e}", "poster": None}]

# =========================================
# 🖥️ 화면 및 네비게이션 로직
# =========================================
apply_custom_css()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

# 0. 표지 (Intro) 페이지
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
        # ⭐️ 버튼 클릭 시 상태 변경 및 rerun
        if st.button("✨ 내 마음 기록하러 가기", use_container_width=True, key="intro_start"):
            st.session_state.page = "login"
            st.rerun()

# 1. 로그인 페이지
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
            # ⭐️ 로그인 버튼: 상태 변경 및 rerun 명시
            if st.button("로그인", use_container_width=True, key="login_btn"):
                users = get_all_users(sh)
                if str(lid) in users and str(users[str(lid)]) == str(lpw):
                    st.session_state.logged_in = True
                    st.session_state.username = lid
                    
                    today_str = datetime.now(KST).strftime("%Y-%m-%d")
                    diaries = get_user_diaries(sh, lid)
                    if today_str in diaries: st.session_state.page = "dashboard"
                    else: st.session_state.page = "write"
                    st.rerun() # ⭐️ 로그인 성공 시 reruN
                else: 
                    st.error("아이디/비밀번호 오류")
            
        with tab2:
            nid = st.text_input("새 아이디", key="nid")
            npw = st.text_input("새 비밀번호 (4자리)", type="password", key="npw", max_chars=4)
            # ⭐️ 가입 버튼: 상태 변경 및 rerun 명시
            if st.button("가입하기", use_container_width=True, key="signup_btn"):
                users = get_all_users(sh)
                if str(nid) in users: st.error("이미 존재함")
                elif len(nid)<1 or len(npw)!=4: st.error("형식 확인 (비번 4자리)")
                else:
                    if add_user(sh, nid, npw): st.success("가입 성공! 로그인하세요.")
                    else: st.error("가입 실패")
                st.rerun() # ⭐️ 가입 시도 후 reruN
        st.markdown("</div>", unsafe_allow_html=True)

# 2. 메인 앱
def main_app():
    sh = init_db()
    if sh is None:
        st.error("데이터베이스 연결 끊김. 새로고침 해주세요.")
        if st.button("🔄 새로고침"): st.rerun()
        return

    # --- 사이드바 (목차 + 토글) ---
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        st.write("")
        
        # ⭐️ [토글 버튼] 야간 모드 버튼
        is_dark_mode = st.checkbox(
            "🌙 야간 모드", 
            value=st.session_state.dark_mode,
            key="toggle_dark_mode",
            help="클릭하여 앱의 테마를 밝은 모드와 어두운 모드로 전환합니다."
        )
        
        # 야간 모드 상태 변경 시 CSS 갱신을 위해 rerun
        if is_dark_mode != st.session_state.dark_mode:
            st.session_state.dark_mode = is_dark_mode
            st.rerun()

        st.divider()
        
        # ⭐️ [목차] st.button 사용 및 st.rerun() 명시 (작동 보장)
        if st.button("📝 일기 작성", use_container_width=True, key="sb_write"): st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력", use_container_width=True, key="sb_calendar"): st.session_state.page = "dashboard"; st.rerun()
        if st.button("🎵 음악/영화 추천", use_container_width=True, key="sb_recommend"): st.session_state.page = "result"; st.rerun()
        if st.button("📊 통계 보기", use_container_width=True, key="sb_stats"): st.session_state.page = "stats"; st.rerun()
        if st.button("📂 행복 저장소", use_container_width=True, key="sb_happy"): st.session_state.page = "happy"; st.rerun()

        st.divider()
        if st.button("🚪 로그아웃", use_container_width=True, key="sb_logout"):
            st.session_state.logged_in = False
            st.session_state.page = "intro"
            st.rerun() # ⭐️ 로그아웃 시 reruN

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
    # st.text_area는 폼 외부에 두어 상태를 유지
    txt = st.text_area("오늘 하루는 어땠나요?", value=st.session_state.diary_input, height=300, placeholder="오늘 있었던 일과 감정을 자유롭게 적어주세요...", key="diary_text_input")
    
    # ⭐️ 감정 분석 및 저장 버튼: 상태 변경 및 rerun 명시
    if st.button("🔍 감정 분석하고 저장하기", type="primary", use_container_width=True, key="write_save"):
        if not txt.strip(): 
            st.warning("내용을 입력해주세요."); 
            st.session_state.diary_input = txt # 입력값 유지
            st.rerun()
            return
            
        # 폼 제출 성공 및 분석 시작
        with st.spinner("분석 중..."):
            emo, sc = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            # 추천 데이터 생성
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            
            today = datetime.now(KST).strftime("%Y-%m-%d")
            add_diary(sh, st.session_state.username, today, emo, txt)
            
            st.session_state.page = "result"
            st.rerun() # ⭐️ 페이지 이동을 위해 명시적 리런

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
        # 달력 텍스트 색상 조건부 설정 (야간 모드 가시성 확보)
        text_color = "#f0f0f0" if st.session_state.get("dark_mode", False) else "#000000"
        events.append({"start": date_str, "display": "background", "backgroundColor": meta["color"]})
        events.append({"title": meta["emoji"], "start": date_str, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": text_color})
    
    calendar(events=events, options={"headerToolbar": {"left": "prev,next today", "center": "title", "right": ""}, "initialView": "dayGridMonth"},
              custom_css="""
              .fc-event-title { font-size: 3em !important; display: flex; justify-content: center; align-items: center; height: 100%; transform: translateY(-25px); text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
              .fc-daygrid-event { border: none !important; background-color: transparent !important; }
              .fc-daygrid-day-number { z-index: 10 !important; color: var(--main-text-color, black); font-weight: bold; }
              .fc-bg-event { opacity: 1.0 !important; }
              .fc-col-header-cell-cushion { color: var(--main-text-color, #333); font-weight: bold; }
              """
              )
    
    st.write("")
    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    if today_str in my_diaries:
        st.success(f"오늘의 기록 완료! ({my_diaries[today_str]['emotion']})")
        c1, c2 = st.columns(2)
        with c1:
             # ⭐️ 일기 수정하기 버튼: 상태 변경 및 rerun 명시
            if st.button("✏️ 일기 수정하기", use_container_width=True, key="dash_edit"):
                st.session_state.diary_input = my_diaries[today_str]["text"]
                st.session_state.page = "write"
                st.rerun()
        with c2:
             # ⭐️ 오늘의 추천 보기 버튼: 상태 변경 및 rerun 명시
            if st.button("🎵 오늘의 추천 보기", type="primary", use_container_width=True, key="dash_rec"):
                emo = my_diaries[today_str]["emotion"]
                st.session_state.final_emotion = emo
                st.session_state.music_recs = recommend_music(emo)
                st.session_state.movie_recs = recommend_movies(emo)
                st.session_state.page = "result"
                st.rerun()
    else:
        # ⭐️ 오늘의 일기 쓰러 가기 버튼: 상태 변경 및 rerun 명시
        if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", use_container_width=True, key="dash_write"):
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
            # ⭐️ 일기 쓰러 가기 버튼: 상태 변경 및 rerun 명시
            if st.button("일기 쓰러 가기", type="primary", key="rec_gtn"):
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
        # ⭐️ 음악 새로고침 버튼: 추천 재생성 및 rerun 명시
        if st.button("🔄 음악 새로고침", use_container_width=True, key="music_refresh"):
            st.session_state.music_recs = recommend_music(emo)
            st.rerun()
        for item in st.session_state.get("music_recs", []):
            if item.get('id'):
                # ⭐️ Spotify iframe 높이 500으로 수정
                components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=500, width="100%")
    with c2:
        st.markdown("#### 🎬 추천 영화")
        # ⭐️ 영화 새로고침 버튼: 추천 재생성 및 rerun 명시
        if st.button("🔄 영화 새로고침", use_container_width=True, key="movie_refresh"):
            st.session_state.movie_recs = recommend_movies(emo)
            st.rerun()
        for item in st.session_state.get('movie_recs', []):
            if item.get('poster'):
                # ⭐️ 영화 추천 카드 디자인 유지
                ic, tc = st.columns([1, 2])
                ic.image(item['poster'], use_container_width=True)
                tc.markdown(f"**{item['title']} ({item['year']})**\n⭐ {item['rating']}\n\n*{item.get('overview','')}*")

    st.divider()
    b1, b2, b3 = st.columns(3)
    with b1:
        # ⭐️ 달력 보기 버튼: 상태 변경 및 rerun 명시
        if st.button("📅 달력 보기", use_container_width=True, key="rec_cal"): st.session_state.page = "dashboard"; st.rerun()
    with b2:
        # ⭐️ 통계 보기 버튼: 상태 변경 및 rerun 명시
        if st.button("📊 통계 보기", use_container_width=True, key="rec_stat"): st.session_state.page = "stats"; st.rerun()
    with b3:
        # ⭐️ 행복 저장소 버튼: 상태 변경 및 rerun 명시
        if st.button("📂 행복 저장소", use_container_width=True, key="rec_happy"): st.session_state.page = "happy"; st.rerun()

def page_stats(sh):
    st.markdown("## 📊 나의 감정 통계")
    
    if "stats_year" not in st.session_state:
        now = datetime.now(KST)
        st.session_state.stats_year = now.year
        st.session_state.stats_month = now.month

    c1, c2, c3 = st.columns([0.2, 0.6, 0.2])
    with c1:
        # ⭐️ 월 이동 버튼 (이전): 상태 변경 및 rerun 명시
        if st.button("◀️", use_container_width=True, key="prev_stats"):
            if st.session_state.stats_month == 1:
                st.session_state.stats_year -= 1
                st.session_state.stats_month = 12
            else: st.session_state.stats_month -= 1
            st.rerun()
    with c2:
        # 월/연도 텍스트 색상 직접 지정 (가시성 확보)
        text_color = "#f0f0f0" if st.session_state.get("dark_mode", False) else "#333"
        st.markdown(f"<h3 style='text-align: center; margin:0; color: {text_color};'>{st.session_state.stats_year}년 {st.session_state.stats_month}월</h3>", unsafe_allow_html=True)
    with c3:
        # ⭐️ 월 이동 버튼 (다음): 상태 변경 및 rerun 명시
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
    
    if month_data:
        max_val = int(chart_data['count'].max()) if not chart_data.empty else 5
        y_values = list(range(0, max_val + 2))
        most_common_emo = max(set(month_data), key=month_data.count)
        total_count = len(month_data)

        # ⭐️ 통계 요약 마크다운
        stat_label_color = "#555" if not st.session_state.dark_mode else "#bbbbbb"
        stat_divider_color = "rgba(128,128,128,0.3)" if not st.session_state.dark_mode else "#444444"

        st.markdown(f"""
            <div style='display:flex; justify-content:space-around; text-align:center; margin-bottom: 20px;'>
                <div style='flex:1; padding: 10px 0; border-right: 1px solid {stat_divider_color};'>
                    <div style='font-size:1.8em; font-weight:700; color:#6C5CE7;'>{total_count}개</div>
                    <div style='font-size:0.9em; color:{stat_label_color};'>총 기록 수</div>
                </div>
                <div style='flex:1; padding: 10px 0; margin-left: 10px;'>
                    <div style='font-size:1.8em; font-weight:700; color:{EMOTION_META[most_common_emo]['color'].replace('0.6', '1.0')}'>{EMOTION_META[most_common_emo]['emoji']} {most_common_emo}</div>
                    <div style='font-size:0.9em; color:{stat_label_color};'>가장 많이 느낀 감정</div>
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.vega_lite_chart(chart_data, {
            "mark": {"type": "bar", "cornerRadius": 10},
            "encoding": {
                "x": {
                    "field": "emotion", "type": "nominal", "sort": domain, 
                    "axis": {"labelAngle": 0, "labelFontSize": 12}, "title": "감정"
                },
                "y": {
                    "field": "count", "type": "quantitative", 
                    "axis": {"values": y_values, "format": "d", "titleAngle": 0, "titleAlign": "right", "titleY": -10}, 
                    "scale": {"domainMin": 0}, "title": "횟수"
                },
                "color": {"field": "emotion", "scale": {"domain": domain, "range": range_}, "legend": None},
                "tooltip": [{"field": "emotion"}, {"field": "count"}]
            }
        }, use_container_width=True)
    else:
        st.info("이 달에는 작성된 일기가 없습니다.")

    st.divider()
    b1, b2 = st.columns(2)
    with b1:
        # ⭐️ 달력 보기 버튼: 상태 변경 및 rerun 명시
        if st.button("📅 달력 보기", use_container_width=True, key="stats_cal"): st.session_state.page = "dashboard"; st.rerun()
    with b2:
        # ⭐️ 행복 저장소 버튼: 상태 변경 및 rerun 명시
        if st.button("📂 행복 저장소 보러가기", use_container_width=True, key="stats_happy"): st.session_state.page = "happy"; st.rerun()

def page_happy_storage(sh):
    st.markdown("## 📂 행복 저장소")
    
    # 상단 설명
    text_color = "#555" if not st.session_state.dark_mode else "#bbbbbb"
    st.markdown(f"<p style='color:{text_color}; font-size:1.1rem;'>내가 '기쁨'을 느꼈던 순간들만 모아봤어요. 🥰</p>", unsafe_allow_html=True)
    
    # 데이터 가져오기 및 '기쁨' 필터링
    my_diaries = get_user_diaries(sh, st.session_state.username)
    happy_moments = {date: data for date, data in my_diaries.items() if data['emotion'] == '기쁨'}
    
    if not happy_moments:
        st.info("아직 기록된 기쁨의 순간이 없어요.")
    else:
        # 월별 필터를 위한 선택창
        dates_all = sorted(happy_moments.keys(), reverse=True)
        years = sorted(list(set([d.split("-")[0] for d in dates_all])), reverse=True)
        
        c1, c2 = st.columns([0.3, 0.7])
        with c1:
            sel_year = st.selectbox("연도 선택", years, key="happy_sel_year")
            months = sorted(list(set([d.split("-")[1] for d in dates_all if d.startswith(sel_year)])), reverse=True)
            sel_month = st.selectbox("월 선택", months, key="happy_sel_month")
            
        target_prefix = f"{sel_year}-{sel_month}"
        filtered_dates = [d for d in dates_all if d.startswith(target_prefix)]
        
        st.write("") # 간격
        
        if not filtered_dates:
            st.warning(f"{sel_year}년 {sel_month}월에는 기쁨의 기록이 없네요.")
        else:
            # 한 줄에 하나씩(Full Width) 출력
            for date in filtered_dates:
                data = happy_moments[date]
                st.markdown(f"""
                <div class="happy-card">
                    <div class="happy-date">{date} {EMOTION_META['기쁨']['emoji']}</div>
                    <div class="happy-text">{data['text']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    b1, b2 = st.columns(2)
    with b1:
        if st.button("📅 달력 보기", use_container_width=True, key="happy_cal"): 
            st.session_state.page = "dashboard"
            st.rerun()
    with b2:
        if st.button("📊 통계 보러가기", use_container_width=True, key="happy_stats"): 
            st.session_state.page = "stats"
            st.rerun()

# --- 메인 실행 로직 ---
if st.session_state.logged_in: main_app()
elif st.session_state.page == "intro": intro_page()
else: login_page()
