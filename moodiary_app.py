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

# ⭐️ [색상 변경] 요청하신 색상으로 업데이트 (가독성을 위해 투명도 0.5 적용)
EMOTION_META = {
    "기쁨": {"color": "rgba(255, 215, 0, 0.5)", "emoji": "😆", "desc": "웃음이 끊이지 않는 하루!"},   # 밝은 노랑
    "분노": {"color": "rgba(255, 50, 50, 0.5)", "emoji": "🤬", "desc": "워워, 진정이 필요해요."},     # 빨강
    "불안": {"color": "rgba(255, 140, 0, 0.5)", "emoji": "😰", "desc": "마음이 조마조마해요."},     # 주황
    "슬픔": {"color": "rgba(65, 105, 225, 0.5)", "emoji": "😭", "desc": "마음의 위로가 필요해요."},   # 파랑
    "힘듦": {"color": "rgba(128, 128, 128, 0.5)", "emoji": "🤯", "desc": "휴식이 절실한 하루."},     # 회색
    "중립": {"color": "rgba(60, 179, 113, 0.5)", "emoji": "😐", "desc": "평온하고 무난한 하루."}      # 초록
}

KST = timezone(timedelta(hours=9))

st.set_page_config(layout="wide", page_title="MOODIARY", page_icon="💖")

# ⭐️ 커스텀 CSS
def apply_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        html, body, [class*="css"] { font-family: 'Noto Sans KR', sans-serif; }
        .stApp { background: linear-gradient(to bottom right, #FDFBF7, #E6E9F0); }
        .block-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 3rem !important;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
            max-width: 1000px;
        }
        .stButton > button {
            border-radius: 12px; border: none; background-color: #6C5CE7;
            color: white; font-weight: 700; transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #5b4bc4; transform: translateY(-2px); color: white;
        }
        .happy-card {
            background-color: #FFF9C4; padding: 20px; border-radius: 15px;
            margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #FFD700;
        }
        .happy-date { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .happy-text { font-size: 1.1em; color: #333; }
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

def init_db():
    client = get_gsheets_client()
    if not client: return None
    try:
        sh = client.open(GSHEET_DB_NAME)
    except:
        return None 

    try:
        sh.worksheet("users")
        sh.worksheet("diaries")
    except:
        return None 
    return sh

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

def get_user_diaries(sh, username):
    if not sh: return {}
    try:
        rows = sh.worksheet("diaries").get_all_records()
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
if "menu" not in st.session_state: st.session_state.menu = "일기 작성"

# 1. 로그인 페이지
def login_page():
    st.markdown("<h1 style='text-align: center;'>MOODIARY 💖</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
    
    sh = init_db()
    if sh is None: st.error("DB 연결 실패"); return

    with tab1:
        lid = st.text_input("아이디", key="lid")
        lpw = st.text_input("비밀번호", type="password", key="lpw")
        if st.button("로그인", use_container_width=True):
            users = get_all_users(sh)
            if str(lid) in users and str(users[str(lid)]) == str(lpw):
                st.session_state.logged_in = True
                st.session_state.username = lid
                
                # ⭐️ [순서 변경] 오늘 일기 유무에 따라 첫 화면 결정
                today_str = datetime.now(KST).strftime("%Y-%m-%d")
                diaries = get_user_diaries(sh, lid)
                
                if today_str in diaries:
                    st.session_state.menu = "달력 보기" # 일기 있으면 달력으로
                else:
                    st.session_state.menu = "일기 작성" # 일기 없으면 작성으로
                st.rerun()
            else: st.error("정보 불일치")
            
    with tab2:
        nid = st.text_input("새 아이디", key="nid")
        npw = st.text_input("새 비밀번호 (4자리)", type="password", key="npw", max_chars=4)
        if st.button("가입하기", use_container_width=True):
            users = get_all_users(sh)
            if str(nid) in users: st.error("이미 존재함")
            elif len(nid)<1 or len(npw)!=4: st.error("형식 확인")
            else:
                if add_user(sh, nid, npw): st.success("성공! 로그인하세요")
                else: st.error("실패")

# 2. 메인 앱
def main_app():
    sh = init_db()

    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        st.write("")
        
        menu_options = ["일기 작성", "달력 보기", "음악/영화 추천", "통계 보기", "행복 저장소"]
        if st.session_state.menu not in menu_options: st.session_state.menu = "일기 작성"
        idx = menu_options.index(st.session_state.menu)
        
        selected = st.radio("목록", menu_options, index=idx)
        if selected != st.session_state.menu:
            st.session_state.menu = selected
            st.rerun()
        
        st.divider()
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.rerun()

    if st.session_state.menu == "일기 작성": page_write(sh)
    elif st.session_state.menu == "달력 보기": page_calendar(sh)
    elif st.session_state.menu == "음악/영화 추천": page_recommend(sh)
    elif st.session_state.menu == "통계 보기": page_stats(sh)
    elif st.session_state.menu == "행복 저장소": page_happy_storage(sh)

# --- 페이지: 일기 작성 ---
def page_write(sh):
    st.title("오늘의 이야기 📝")
    model, tokenizer, device, id2label = load_emotion_model()
    if not model: st.error("AI 로드 실패"); return

    if "diary_input" not in st.session_state: st.session_state.diary_input = ""
    txt = st.text_area("오늘 하루는 어땠나요?", value=st.session_state.diary_input, height=300)
    
    if st.button("🔍 감정 분석하고 저장하기", type="primary", use_container_width=True):
        if not txt.strip(): st.warning("내용을 입력해주세요."); return
        
        with st.spinner("분석 중..."):
            emo, sc = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            
            today = datetime.now(KST).strftime("%Y-%m-%d")
            add_diary(sh, st.session_state.username, today, emo, txt)
            
            # ⭐️ [순서 변경] 일기 작성 -> 음악/영화 추천 페이지로 이동
            st.session_state.menu = "음악/영화 추천"
            st.rerun()

# --- 페이지: 달력 보기 ---
def page_calendar(sh):
    st.title("감정 달력 📅")
    cols = st.columns(6)
    for i, (k, v) in enumerate(EMOTION_META.items()):
        cols[i].markdown(f"<span style='color:{v['color']};'>●</span> {k}", unsafe_allow_html=True)
    
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
             .fc-daygrid-day-number { z-index: 10 !important; color: black; }
             .fc-bg-event { opacity: 1.0 !important; }
             """)
    
    st.write("")
    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    
    # ⭐️ 달력 페이지 하단 버튼 (일기 유무에 따라 분기)
    if today_str in my_diaries:
        st.success("오늘의 일기가 작성되었습니다!")
        c1, c2 = st.columns(2)
        with c1:
            # ⭐️ 일기 다시 작성 -> 작성 페이지로
            if st.button("✏️ 일기 다시 작성하기", use_container_width=True):
                st.session_state.diary_input = my_diaries[today_str]["text"]
                st.session_state.menu = "일기 작성"
                st.rerun()
        with c2:
            # ⭐️ 오늘의 추천 보기 -> 추천 페이지로
            if st.button("🎵 오늘의 추천 보기", type="primary", use_container_width=True):
                emo = my_diaries[today_str]["emotion"]
                st.session_state.final_emotion = emo
                st.session_state.music_recs = recommend_music(emo)
                st.session_state.movie_recs = recommend_movies(emo)
                st.session_state.menu = "음악/영화 추천"
                st.rerun()
    else:
        if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", use_container_width=True):
            st.session_state.diary_input = ""
            st.session_state.menu = "일기 작성"
            st.rerun()

# --- 페이지: 음악/영화 추천 ---
def page_recommend(sh):
    # 데이터 없으면 로드 시도
    if "final_emotion" not in st.session_state:
        today = datetime.now(KST).strftime("%Y-%m-%d")
        diaries = get_user_diaries(sh, st.session_state.username)
        if today in diaries:
            st.session_state.final_emotion = diaries[today]['emotion']
            st.session_state.music_recs = recommend_music(st.session_state.final_emotion)
            st.session_state.movie_recs = recommend_movies(st.session_state.final_emotion)
        else:
            st.info("오늘의 감정 기록이 없어요. 일기를 먼저 작성해주세요!")
            if st.button("일기 쓰러 가기", type="primary"):
                st.session_state.menu = "일기 작성"
                st.rerun()
            return

    emo = st.session_state.final_emotion
    if emo not in EMOTION_META: emo = "중립"
    meta = EMOTION_META[emo]
    
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem;'>
        <h2 style='color: {meta['color'].replace('0.5', '1.0').replace('0.4', '1.0')}; font-size: 3rem; margin-bottom: 0.5rem;'>
            {meta['emoji']} 오늘의 감정: {emo}
        </h2>
        <h4 style='color: #555;'>{meta['desc']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎵 추천 음악")
        if st.button("🔄 음악 새로고침", use_container_width=True):
            st.session_state.music_recs = recommend_music(emo)
            st.rerun()
        for item in st.session_state.get("music_recs", []):
            if item.get('id'):
                components.iframe(f"https://open.spotify.com/embed/track/{item['id']}", height=250, width="100%")
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

    # ⭐️ [추가됨] 추천 페이지 하단 3개의 버튼
    st.divider()
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📅 달력 보러가기", use_container_width=True):
            st.session_state.menu = "달력 보기"
            st.rerun()
    with b2:
        if st.button("📊 통계 보러가기", use_container_width=True):
            st.session_state.menu = "통계 보기"
            st.rerun()
    with b3:
        if st.button("📂 행복 저장소 가기", use_container_width=True):
            st.session_state.menu = "행복 저장소"
            st.rerun()

# --- 페이지: 통계 보기 ---
def page_stats(sh):
    st.title("나의 감정 통계 📊")
    my_diaries = get_user_diaries(sh, st.session_state.username)
    today = datetime.now(KST)
    cur_month = today.strftime("%Y-%m")
    
    st.subheader(f"{today.month}월의 감정 분포")
    month_data = []
    for date, d in my_diaries.items():
        if date.startswith(cur_month):
            e = d['emotion']
            if e in EMOTION_META: month_data.append(e)
    
    df = pd.DataFrame(month_data, columns=['emotion'])
    counts = df['emotion'].value_counts().reindex(EMOTION_META.keys(), fill_value=0)
    
    chart_data = counts.reset_index()
    chart_data.columns = ['emotion', 'count']
    domain = list(EMOTION_META.keys())
    range_ = [m['color'] for m in EMOTION_META.values()]
    
    # Y축 최대값 계산 (정수 눈금을 위해)
    max_val = int(chart_data['count'].max()) if not chart_data.empty else 5
    y_values = list(range(0, max_val + 2))

    st.vega_lite_chart(chart_data, {
        "mark": {"type": "bar", "cornerRadius": 5},
        "encoding": {
            "x": {
                "field": "emotion", "type": "nominal", "sort": domain, 
                "axis": {"labelAngle": 0}, "title": "감정"
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

# --- 페이지: 행복 저장소 ---
def page_happy_storage(sh):
    st.title("행복 저장소 📂")
    st.markdown("내가 **'기쁨'**을 느꼈던 순간들을 모아보세요.")
    my_diaries = get_user_diaries(sh, st.session_state.username)
    happy_moments = {date: data for date, data in my_diaries.items() if data['emotion'] == '기쁨'}
    
    if not happy_moments:
        st.info("아직 기록된 기쁨의 순간이 없어요.")
    else:
        for date in sorted(happy_moments.keys(), reverse=True):
            data = happy_moments[date]
            st.markdown(f"""<div class="happy-card"><div class="happy-date">{date} {EMOTION_META['기쁨']['emoji']}</div><div class="happy-text">{data['text']}</div></div>""", unsafe_allow_html=True)

if st.session_state.logged_in: main_app()
else: login_page()
