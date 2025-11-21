# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit.components.v1 as components
from datetime import datetime, timezone, timedelta  # KST
from streamlit_calendar import calendar
import sqlite3
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
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

# 감정별 메타 정보
EMOTION_META = {
    "기쁨": {"color": "rgba(255, 215, 0, 0.4)", "emoji": "😆", "desc": "기분 좋은 하루네요!"},
    "분노": {"color": "rgba(255, 69, 0, 0.4)", "emoji": "😡", "desc": "많이 답답했겠어요."},
    "불안": {"color": "rgba(138, 43, 226, 0.4)", "emoji": "😰", "desc": "불안한 마음이 느껴져요."},
    "슬픔": {"color": "rgba(65, 105, 225, 0.4)", "emoji": "😭", "desc": "토닥토닥, 수고 많았어요."},
    "힘듦": {"color": "rgba(128, 128, 128, 0.4)", "emoji": "🥺", "desc": "많이 지친 하루였겠네요."},
    "중립": {"color": "rgba(54, 54, 54, 0.2)", "emoji": "😐", "desc": "차분한 하루였어요."},
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
        /* 탭 스타일 */
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            height: 45px; border-radius: 10px; background-color: #F0F2F6;
            border: none; font-weight: 600; color: #666;
        }
        .stTabs [aria-selected="true"] { background-color: #6C5CE7 !important; color: white !important; }
        
        /* 행복 저장소 카드 스타일 */
        .happy-card {
            background-color: #FFF9C4; padding: 20px; border-radius: 15px;
            margin-bottom: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            border-left: 5px solid #FFD700;
        }
        .happy-date { font-size: 0.9em; color: #666; margin-bottom: 5px; }
        .happy-text { font-size: 1.1em; color: #333; }
        
        header {visibility: hidden;} footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# =========================================
# 🗂 DB & AI & 추천
# =========================================
@st.cache_resource
def get_db():
    conn = sqlite3.connect("moodiary.db", check_same_thread=False)
    cur = conn.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE, password TEXT)")
    cur.execute("CREATE TABLE IF NOT EXISTS diaries (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, date TEXT, emotion TEXT, text TEXT)")
    conn.commit()
    return conn
conn = get_db()

def get_all_users():
    cur = conn.cursor()
    cur.execute("SELECT username, password FROM users")
    return {u: p for (u, p) in cur.fetchall()}

def add_user(username, password):
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError: return False

def get_user_diaries(username):
    cur = conn.cursor()
    cur.execute("SELECT date, emotion, text FROM diaries WHERE username = ?", (username,))
    out = {}
    for d, e, t in cur.fetchall(): out[d] = {"emotion": e, "text": t}
    return out

def add_diary(username, date, emotion, text):
    cur = conn.cursor()
    cur.execute("SELECT id FROM diaries WHERE username = ? AND date = ?", (username, date))
    row = cur.fetchone()
    if row: cur.execute("UPDATE diaries SET emotion = ?, text = ? WHERE id = ?", (emotion, text, row[0]))
    else: cur.execute("INSERT INTO diaries (username, date, emotion, text) VALUES (?, ?, ?, ?)", (username, date, emotion, text))
    conn.commit()

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
# 🖥️ 화면 구성
# =========================================
apply_custom_css()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "login"

def login_page():
    st.markdown("<h1 style='text-align: center;'>MOODIARY 💖</h1>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
    with tab1:
        lid = st.text_input("아이디", key="lid")
        lpw = st.text_input("비밀번호", type="password", key="lpw")
        if st.button("로그인", use_container_width=True):
            users = get_all_users()
            if lid in users and str(users[lid]) == str(lpw):
                st.session_state.logged_in = True
                st.session_state.username = lid
                st.session_state.page = "dashboard"
                st.rerun()
            else: st.error("정보 불일치")
    with tab2:
        nid = st.text_input("새 아이디", key="nid")
        npw = st.text_input("새 비밀번호 (4자리)", type="password", key="npw", max_chars=4)
        if st.button("가입하기", use_container_width=True):
            if add_user(nid, npw): st.success("성공! 로그인하세요")
            else: st.error("실패/중복")

def dashboard_page():
    st.title(f"{st.session_state.username}님의 감정 달력 📅")
    
    cols = st.columns(6)
    for i, (k, v) in enumerate(EMOTION_META.items()):
        cols[i].markdown(f"<span style='color:{v['color']};'>●</span> {k}", unsafe_allow_html=True)
    st.divider()
    
    my_diaries = get_user_diaries(st.session_state.username)
    
    # ⭐️⭐️⭐️ [구조 변경] 탭 3개로 분리 ⭐️⭐️⭐️
    tab1, tab2, tab3 = st.tabs(["📅 감정 달력", "📊 이달의 통계", "📂 행복 저장소"])
    
    # 1. 달력 탭
    with tab1:
        events = []
        for date_str, data in my_diaries.items():
            emo = data.get("emotion", "중립")
            meta = EMOTION_META.get(emo, EMOTION_META["중립"])
            events.append({"start": date_str, "display": "background", "backgroundColor": meta["color"]})
            events.append({"title": meta["emoji"], "start": date_str, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": "#000000"})
        
        calendar(events=events, options={"headerToolbar": {"left": "prev,next today", "center": "title", "right": ""}, "initialView": "dayGridMonth"},
                 custom_css="""
                 .fc-event-title { font-size: 3em !important; display: flex; justify-content: center; align-items: center; height: 100%; transform: translateY(-25px); text-shadow: 1px 1px 2px rgba(0,0,0,0.2); }
                 .fc-daygrid-event { border: none !important; background-color: transparent !important; }
                 .fc-daygrid-day-number { z-index: 10 !important; color: black; }
                 .fc-bg-event { opacity: 1.0 !important; }
                 """)
    
    # 2. 통계 탭
    with tab2:
        today = datetime.now(KST)
        st.subheader(f"{today.month}월의 감정 분포")
        cur_month = today.strftime("%Y-%m")
        
        month_data = [d['emotion'] for date, d in my_diaries.items() if date.startswith(cur_month)]
        df = pd.DataFrame(month_data, columns=['emotion'])
        counts = df['emotion'].value_counts().reindex(EMOTION_META.keys(), fill_value=0)
        
        chart_data = counts.reset_index()
        chart_data.columns = ['emotion', 'count']
        
        # ⭐️ 차트 수정 (정수 단위, 가로 글자, 0부터 시작)
        st.vega_lite_chart(chart_data, {
            "mark": {"type": "bar", "cornerRadius": 5},
            "encoding": {
                "x": {"field": "emotion", "type": "nominal", "sort": list(EMOTION_META.keys()), "axis": {"labelAngle": 0}, "title": "감정"},
                "y": {"field": "count", "type": "quantitative", 
                      "axis": {"tickMinStep": 1, "format": "d"}, # 정수 단위
                      "scale": {"zero": True}, # 0부터 시작
                      "title": "횟수"},
                "color": {"field": "emotion", "scale": {"domain": list(EMOTION_META.keys()), "range": [m['color'] for m in EMOTION_META.values()]}, "legend": None},
                "tooltip": [{"field": "emotion"}, {"field": "count"}]
            }
        }, use_container_width=True)
        
        st.write("---")
        cols = st.columns(6)
        for idx, (k, v) in enumerate(counts.items()):
            cols[idx].metric(label=k, value=f"{v}회")

    # 3. 행복 저장소 탭 (분리됨)
    with tab3:
        st.subheader("🥰 행복 저장소")
        st.info("내가 **'기쁨'**을 느꼈던 순간들을 모아보세요.")
        
        happy_moments = {date: data for date, data in my_diaries.items() if data['emotion'] == '기쁨'}
        
        if not happy_moments:
            st.write("아직 기록된 기쁨의 순간이 없어요.")
        else:
            for date in sorted(happy_moments.keys(), reverse=True):
                data = happy_moments[date]
                st.markdown(f"""
                <div class="happy-card">
                    <div class="happy-date">{date} {EMOTION_META['기쁨']['emoji']}</div>
                    <div class="happy-text">{data['text']}</div>
                </div>
                """, unsafe_allow_html=True)

    st.divider()
    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    
    # 오늘 일기 확인/작성 버튼
    if today_str in my_diaries:
        st.info("오늘의 일기가 작성되었습니다.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ 수정하기", use_container_width=True):
                st.session_state.diary_input = my_diaries[today_str]["text"]
                st.session_state.page = "write"
                st.rerun()
        with col2:
            if st.button("🎵 추천 다시보기", type="primary", use_container_width=True):
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

def result_page():
    emo = st.session_state.final_emotion
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])
    
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem;'>
        <h2 style='color: {meta['color'].replace('0.4', '1.0').replace('0.2', '1.0')}; font-size: 3rem; margin-bottom: 0.5rem;'>
            {meta['emoji']} 오늘의 감정: {emo}
        </h2>
        <h4 style='color: #555;'>{meta['desc']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("⬅️ 달력으로 돌아가기"):
        st.session_state.page = "dashboard"
        st.rerun()
    st.divider()
    
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

def write_page():
    st.title("오늘의 이야기 📝")
    if st.button("⬅️ 취소"):
        st.session_state.page = "dashboard"
        st.rerun()

    model, tokenizer, device, id2label = load_emotion_model()
    if not model: st.error("AI 로드 실패"); return

    txt = st.text_area("오늘 하루는 어땠나요?", value=st.session_state.diary_input, height=300)
    
    if st.button("🔍 감정 분석하고 저장하기", type="primary", use_container_width=True):
        if not txt.strip(): st.warning("내용을 입력해주세요."); return
        
        with st.spinner("분석 중..."):
            emo, sc = analyze_diary(txt, model, tokenizer, device, id2label)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            
            today = datetime.now(KST).strftime("%Y-%m-%d")
            add_diary(st.session_state.username, today, emo, txt)
            
            st.session_state.page = "result"
            st.rerun()

# =========================================
# 🚀 메인 컨트롤러
# =========================================
if st.session_state.logged_in:
    with st.sidebar:
        st.write(f"**{st.session_state.username}**님 환영합니다!")
        if st.button("로그아웃", width='stretch'):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

if not st.session_state.logged_in: login_page()
elif st.session_state.page == "dashboard": dashboard_page()
elif st.session_state.page == "write": write_page()
elif st.session_state.page == "result": result_page()
