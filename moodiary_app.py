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
EMOTION_MODEL_ID = "JUDONGHYEOK/6-emotion-bert-korean-v2"  # 6감정 모델
TMDB_BASE_URL = "https://api.themoviedb.org/3"
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

# 감정별 메타 정보 (모델 라벨 그대로: 기쁨, 분노, 불안, 슬픔, 중립, 힘듦)
# ⭐️ 디자인을 위해 색상을 rgba로 설정
EMOTION_META = {
    "기쁨": {"color": "rgba(255, 215, 0, 0.4)", "emoji": "😆", "desc": "기분 좋은 하루네요!"},
    "분노": {"color": "rgba(255, 69, 0, 0.4)", "emoji": "😡", "desc": "많이 답답했겠어요."},
    "불안": {"color": "rgba(138, 43, 226, 0.4)", "emoji": "😰", "desc": "불안한 마음이 느껴져요."}, # 보라색 계열
    "슬픔": {"color": "rgba(65, 105, 225, 0.4)", "emoji": "😭", "desc": "토닥토닥, 수고 많았어요."},
    "힘듦": {"color": "rgba(128, 128, 128, 0.4)", "emoji": "🥺", "desc": "많이 지친 하루였겠네요."},
    "중립": {"color": "rgba(54, 54, 54, 0.2)", "emoji": "😐", "desc": "차분한 하루였어요."},
}

# 대한민국 표준시(KST)
KST = timezone(timedelta(hours=9))

st.set_page_config(layout="wide", page_title="MOODIARY", page_icon="💖")

# ⭐️⭐️⭐️ [디자인] 예쁜 UI를 위한 커스텀 CSS 적용 ⭐️⭐️⭐️
def apply_custom_css():
    st.markdown("""
        <style>
        /* 1. 폰트 설정 (Noto Sans KR) */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans KR', sans-serif;
        }

        /* 2. 전체 배경 (은은한 그라데이션) */
        .stApp {
            background: linear-gradient(to bottom right, #FDFBF7, #E6E9F0);
        }

        /* 3. 메인 컨텐츠 카드 UI (흰색 박스 + 그림자) */
        .block-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 3rem !important;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
            max-width: 1000px;
        }

        /* 4. 버튼 스타일링 (둥글고 부드러운 색상) */
        .stButton > button {
            width: 100%;
            border-radius: 15px;
            border: none;
            background-color: #6C5CE7; /* 시그니처 퍼플 */
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #5b4bc4;
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(108, 92, 231, 0.3);
            color: white;
        }

        /* 5. 탭 스타일링 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            border-radius: 10px;
            background-color: #F0F2F6;
            border: none;
            font-weight: 600;
            color: #666;
        }
        .stTabs [aria-selected="true"] {
            background-color: #6C5CE7 !important;
            color: white !important;
        }

        /* 6. 상단 헤더/푸터 숨기기 */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

# =========================================
# 🗂 3) SQLite 데이터베이스 (users + diaries)
# =========================================
@st.cache_resource
def get_db():
    conn = sqlite3.connect("moodiary.db", check_same_thread=False)
    cur = conn.cursor()
    # 사용자 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT
        )
    """)
    # 일기 테이블
    cur.execute("""
        CREATE TABLE IF NOT EXISTS diaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            date TEXT,
            emotion TEXT,
            text TEXT
        )
    """)
    conn.commit()
    return conn

conn = get_db()

def get_all_users():
    cur = conn.cursor()
    cur.execute("SELECT username, password FROM users")
    rows = cur.fetchall()
    return {u: p for (u, p) in rows}

def add_user(username, password):
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, password),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        # 이미 존재
        return False

def get_user_diaries(username):
    cur = conn.cursor()
    cur.execute(
        "SELECT date, emotion, text FROM diaries WHERE username = ?",
        (username,),
    )
    rows = cur.fetchall()
    out = {}
    for d, e, t in rows:
        out[d] = {"emotion": e, "text": t}
    return out

def add_diary(username, date, emotion, text):
    cur = conn.cursor()
    # 같은 날짜 있으면 업데이트, 없으면 INSERT
    cur.execute(
        "SELECT id FROM diaries WHERE username = ? AND date = ?",
        (username, date),
    )
    row = cur.fetchone()
    if row:
        cur.execute(
            "UPDATE diaries SET emotion = ?, text = ? WHERE id = ?",
            (emotion, text, row[0]),
        )
    else:
        cur.execute(
            "INSERT INTO diaries (username, date, emotion, text) VALUES (?, ?, ?, ?)",
            (username, date, emotion, text),
        )
    conn.commit()

# =========================================
# 🧠 4) 감정 분석 (라벨 후처리 없이 그대로)
# =========================================
@st.cache_resource
def load_emotion_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # id2label 처리
        cfg_id2label = getattr(model.config, "id2label", None)
        if isinstance(cfg_id2label, dict) and cfg_id2label:
            id2label = {}
            for k, v in cfg_id2label.items():
                try:
                    id2label[int(k)] = v
                except Exception:
                    pass
        else:
            id2label = {
                0: "기쁨",
                1: "분노",
                2: "불안",
                3: "슬픔",
                4: "중립",
                5: "힘듦",
            }

        return model, tokenizer, device, id2label
    except Exception as e:
        st.error(f"감정 분석 모델 로드 실패: {e}")
        return None, None, None, None

def analyze_diary(text, model, tokenizer, device, id2label):
    if not text or model is None:
        return None, 0.0

    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt",
    )
    for k in enc:
        enc[k] = enc[k].to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    score = float(probs[pred_id].cpu().item())

    label = id2label.get(pred_id, "중립")  # 기쁨/분노/불안/슬픔/중립/힘듦 그대로
    return label, score

# =========================================
# 🎧 5) 음악 / 🎬 영화 추천
# =========================================
@st.cache_resource
def get_spotify_client():
    if not SPOTIPY_AVAILABLE:
        return "Spotipy 라이브러리 설치 실패. (requirements.txt 확인)"

    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(
            client_id=creds["client_id"],
            client_secret=creds["client_secret"],
        )
        sp = spotipy.Spotify(
            client_credentials_manager=manager,
            retries=3,
            backoff_factor=0.3,
        )
        sp.search(q="test", limit=1)
        return sp
    except KeyError:
        return "Spotify Secrets 설정이 없습니다."
    except Exception as e:
        return f"Spotify 로그인 실패: {e}"

def recommend_music(emotion):
    sp = get_spotify_client()
    if not isinstance(sp, spotipy.Spotify):
        return [{"error": sp}]

    SEARCH_KEYWORDS_MAP = {
        "기쁨": ["신나는 K-Pop", "Upbeat K-Pop", "K-Pop Hits"],
        "슬픔": ["위로가 되는 발라드", "이별 발라드", "새벽 감성"],
        "분노": ["스트레스 해소 락", "신나는 힙합", "Workout K-Pop"],
        "불안": ["Lofi Hip Hop", "Chill K-Pop", "잔잔한 음악"],
        "힘듦": ["힐링 발라드", "위로 K-Pop", "감성 플리"],
        "중립": ["K-Pop Mix", "국힙 Top 100", "Chill Mix"],
    }

    keyword_list = SEARCH_KEYWORDS_MAP.get(emotion, SEARCH_KEYWORDS_MAP["중립"])
    query = random.choice(keyword_list)

    try:
        results = sp.search(q=query, type="playlist", limit=10, market="KR")
        playlists = results.get("playlists", {}).get("items", [])
        if not playlists:
            return [{"error": f"'{query}' 검색 결과 플레이리스트 없음"}]

        valid_tracks = []
        random.shuffle(playlists)
        for pl in playlists:
            try:
                pid = pl["id"]
                tracks_results = sp.playlist_items(pid, limit=30)
                items = tracks_results.get("items", []) if tracks_results else []
                for it in items:
                    t = it.get("track")
                    if t and t.get("id") and t.get("name"):
                        valid_tracks.append({"id": t["id"], "title": t["name"]})
                if len(valid_tracks) >= 10:
                    break
            except Exception:
                continue

        if not valid_tracks:
            return [{"error": "추천 곡을 찾지 못했습니다."}]

        seen = set()
        unique = []
        for v in valid_tracks:
            if v["id"] not in seen:
                unique.append(v)
                seen.add(v["id"])

        return random.sample(unique, k=min(3, len(unique)))
    except Exception as e:
        return [{"error": f"Spotify 검색 오류: {e}"}]

def recommend_movies(emotion):
    key = (
        st.secrets.get("tmdb", {}).get("api_key")
        or st.secrets.get("TMDB_API_KEY")
        or EMERGENCY_TMDB_KEY
    )
    if not key:
        return [{"text": "TMDB 연결 실패", "poster": None}]

    GENRES = {
        "기쁨": "35|10749|10751",
        "분노": "28|12|53",
        "불안": "53|9648",
        "슬픔": "18|10749",
        "힘듦": "18|10751",
        "중립": "35|18|10751",
    }

    try:
        r = requests.get(
            f"{TMDB_BASE_URL}/discover/movie",
            params={
                "api_key": key,
                "language": "ko-KR",
                "sort_by": "popularity.desc",
                "with_genres": GENRES.get(emotion, GENRES["중립"]),
                "without_genres": "16",
                "page": random.randint(1, 5),
                "vote_count.gte": 500,
                "primary_release_date.gte": "2000-01-01",
            },
            timeout=5,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return [{"text": "조건에 맞는 영화가 없습니다.", "poster": None}]

        picks = random.sample(results, min(3, len(results)))
        out = []
        for m in picks:
            out.append(
                {
                    "title": m.get("title"),
                    "year": (m.get("release_date") or "")[:4],
                    "rating": m.get("vote_average", 0.0),
                    "overview": m.get("overview", ""),
                    "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}"
                    if m.get("poster_path")
                    else None,
                }
            )
        return out
    except Exception as e:
        return [{"text": f"TMDb 오류: {e}", "poster": None}]

# =========================================
# 🖥️ 6) 화면 구성 (로그인 / 대시보드 / 작성 / 결과)
# =========================================
# ⭐️ 디자인 적용
apply_custom_css()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "login"

def login_page():
    st.markdown("<h1 style='text-align: center;'>MOODIARY 💖</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>당신의 하루를 기록하고, 감정에 맞는 처방을 받아보세요.</p>", unsafe_allow_html=True)
    st.write("")

    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])

    with tab1:
        lid = st.text_input("아이디", key="lid")
        lpw = st.text_input("비밀번호", type="password", key="lpw")
        st.write("")
        if st.button("로그인", use_container_width=True):
            users = get_all_users()
            if lid in users and str(users[lid]) == str(lpw):
                st.session_state.logged_in = True
                st.session_state.username = lid
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("정보가 일치하지 않습니다.")

    with tab2:
        nid = st.text_input("새 아이디", key="nid")
        npw = st.text_input("새 비밀번호 (4자리)", type="password", key="npw", max_chars=4)
        st.write("")
        if st.button("가입하기", use_container_width=True):
            users = get_all_users()
            if nid in users:
                st.error("이미 존재하는 아이디입니다.")
            elif len(nid) < 1 or len(npw) != 4:
                st.error("아이디/비밀번호 형식을 확인해주세요.")
            else:
                if add_user(nid, npw):
                    st.success("가입 성공! 로그인 해주세요.")
                else:
                    st.error("가입 실패 (DB 오류)")

def dashboard_page():
    st.markdown(f"### {st.session_state.username}님의 감정 달력 📅")

    # 범례
    legend_cols = st.columns(len(EMOTION_META))
    for i, (emo, meta) in enumerate(EMOTION_META.items()):
        legend_cols[i].markdown(
            f"<span style='color:{meta['color']}; font-size: 1.2em;'>●</span> {emo}",
            unsafe_allow_html=True,
        )
    st.divider()

    my_diaries = get_user_diaries(st.session_state.username)

    tab1, tab2 = st.tabs(["📅 감정 달력", "📊 이달의 통계"])

    # 📅 감정 달력
    with tab1:
        events = []
        for date_str, data in my_diaries.items():
            emo = data.get("emotion", "중립")
            meta = EMOTION_META.get(emo, EMOTION_META["중립"])
            # 1. 배경색 이벤트 (칸 채우기)
            events.append({
                "start": date_str,
                "display": "background",
                "backgroundColor": meta["color"],
            })
            # 2. 이모티콘 이벤트 (위로 띄우기)
            events.append({
                "title": meta["emoji"],
                "start": date_str,
                "allDay": True,
                "backgroundColor": "transparent",
                "borderColor": "transparent",
                "textColor": "#000000",
            })

        calendar(
            events=events,
            options={
                "headerToolbar": {
                    "left": "prev,next today",
                    "center": "title",
                    "right": "",
                },
                "initialView": "dayGridMonth",
            },
            custom_css="""
                .fc-event-title {
                    font-size: 3em !important;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100%;
                    line-height: 1;
                    transform: translateY(-25px); 
                    text-shadow: 1px 1px 2px rgba(0,0,0,0.2); 
                }
                .fc-daygrid-event {
                    padding: 0 !important;
                    margin: 0 !important;
                    border: none !important;
                    color: black !important;
                    background-color: transparent !important; 
                }
                .fc-daygrid-day-frame {
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    position: relative;
                }
                .fc-daygrid-day-number {
                    position: absolute !important;
                    top: 5px;
                    right: 5px;
                    font-size: 0.8em;
                    color: black;
                    z-index: 10 !important; 
                    text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
                }
                .fc-daygrid-day-top {
                   flex-grow: 1;
                   display: flex;
                   flex-direction: column;
                   justify-content: center;
                   align-items: center;
                   width: 100%;
                }
                .fc-bg-event {
                    opacity: 1.0 !important; 
                }
            """,
        )
        st.write("")

    # 📊 이달의 통계 (Vega-Lite 차트로 교체)
    with tab2:
        today = datetime.now(KST)
        st.subheader(f"{today.month}월의 감정 통계")

        current_month_str = today.strftime("%Y-%m")
        month_emotions = []
        for date_str, data in my_diaries.items():
            if date_str.startswith(current_month_str):
                month_emotions.append(data.get("emotion", "중립"))

        # 데이터프레임 생성
        df = pd.DataFrame(month_emotions, columns=["emotion"])
        # 빈 데이터라도 0으로 채워넣기
        emotion_counts = df["emotion"].value_counts().reindex(EMOTION_META.keys(), fill_value=0)
        
        chart_data = emotion_counts.reset_index()
        chart_data.columns = ["emotion", "count"]

        # 차트에 사용할 메타 정보 (도메인, 레인지)
        domain = list(EMOTION_META.keys())
        range_ = [meta["color"] for meta in EMOTION_META.values()]

        # Streamlit Vega-Lite Chart 사용 (안정적)
        st.vega_lite_chart(
            chart_data,
            {
                "title": f"{today.month}월의 감정 분포",
                "width": "container",
                "mark": {"type": "bar", "cornerRadius": 5, "opacity": 1.0}, # 옅은 색상 그대로 사용
                "encoding": {
                    "x": {
                        "field": "emotion",
                        "type": "nominal",
                        "sort": domain,
                        "title": "감정",
                        "axis": {"labelAngle": 0}, # 글자 가로 정렬
                    },
                    "y": {
                        "field": "count",
                        "type": "quantitative",
                        "title": "횟수",
                        "scale": {"zero": True}, # 0부터 시작
                        "axis": {"format": "d", "tickMinStep": 1},
                    },
                    "color": {
                        "field": "emotion",
                        "type": "nominal",
                        "scale": {"domain": domain, "range": range_},
                        "legend": None,
                    },
                    "tooltip": [
                        {"field": "emotion", "title": "감정"},
                        {"field": "count", "title": "횟수"},
                    ],
                },
            },
            use_container_width=True,
        )

        st.write("---")
        st.write("감정별 횟수:")
        cols = st.columns(len(EMOTION_META))
        for idx, (emo, count) in enumerate(emotion_counts.items()):
            cols[idx].metric(label=f"{EMOTION_META[emo]['emoji']} {emo}", value=f"{count}회")

    st.divider()

    today_str = datetime.now(KST).strftime("%Y-%m-%d")
    today_diary_exists = today_str in my_diaries

    if today_diary_exists:
        today_emo = my_diaries[today_str]["emotion"]
        st.info(
            f"오늘({today_str})의 일기({today_emo} {EMOTION_META.get(today_emo, EMOTION_META['중립'])['emoji']})가 이미 작성되었습니다."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ 오늘 일기 수정/확인하기", use_container_width=True):
                st.session_state.page = "write"
                st.session_state.diary_input = my_diaries[today_str]["text"]
                st.rerun()
        with col2:
            def handle_show_recs():
                st.session_state.final_emotion = today_emo
                st.session_state.music_recs = recommend_music(today_emo)
                st.session_state.movie_recs = recommend_movies(today_emo)
                st.session_state.page = "result"

            if st.button(
                "🎵🎬 오늘의 추천 바로 보기",
                type="primary",
                use_container_width=True,
            ):
                handle_show_recs()
                st.rerun()
    else:
        if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", use_container_width=True):
            st.session_state.page = "write"
            st.session_state.diary_input = ""
            st.rerun()

def result_page():
    emo = st.session_state.final_emotion
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])

    # 예쁜 타이틀 박스
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

    def refresh_music():
        st.session_state.music_recs = recommend_music(emo)

    def refresh_movies():
        st.session_state.movie_recs = recommend_movies(emo)

    c1, c2 = st.columns(2)

    # 🎵 음악 추천
    with c1:
        st.markdown("#### 🎵 추천 음악")
        st.button(
            "🔄 다른 음악", on_click=refresh_music, key="rm_btn", use_container_width=True
        )
        for item in st.session_state.music_recs:
            if item.get("id"):
                components.iframe(
                    f"https://open.spotify.com/embed/track/{item['id']}",
                    height=250,
                    width="100%",
                )
            else:
                st.error(item.get("error", "로딩 실패"))

    # 🎬 영화 추천
    with c2:
        st.markdown("#### 🎬 추천 영화")
        st.button(
            "🔄 다른 영화", on_click=refresh_movies, key="rv_btn", use_container_width=True
        )
        for item in st.session_state.movie_recs:
            if item.get("poster"):
                ic, tc = st.columns([1, 2])
                ic.image(item["poster"], use_container_width=True)
                tc.markdown(
                    f"**{item['title']} ({item['year']})**\n"
                    f"⭐ {item['rating']:.1f}\n\n"
                    f"*{item.get('overview','')}*" # 줄거리 전체 표시
                )
            else:
                st.error(item.get("text", "로딩 실패"))

def write_page():
    st.title("오늘의 이야기 📝")
    if st.button("⬅️ 뒤로 가기"):
        st.session_state.page = "dashboard"
        st.rerun()

    model, tokenizer, device, id2label = load_emotion_model()
    if not model:
        st.error("AI 모델 로드 실패")
        return

    if "diary_input" not in st.session_state:
        st.session_state.diary_input = ""

    txt = st.text_area(
        "오늘 하루는 어땠나요?",
        value=st.session_state.diary_input,
        height=300,
        key="diary_editor",
        placeholder="여기에 일기를 작성하세요..."
    )

    if st.button("🔍 감정 분석하고 저장하기", type="primary", use_container_width=True):
        if not txt.strip():
            st.warning("내용을 입력해주세요.")
            return

        with st.spinner("분석 및 저장 중..."):
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
        if st.button("로그아웃", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

if not st.session_state.logged_in:
    login_page()
elif st.session_state.page == "dashboard":
    dashboard_page()
elif st.session_state.page == "write":
    write_page()
elif st.session_state.page == "result":
    result_page()
