# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import time
import streamlit.components.v1 as components
import json
import os
from datetime import datetime
from streamlit_calendar import calendar

# --- 2) 기본 설정 ---
KOBERT_BASE_MODEL = "monologg/kobert"
KOBERT_SAVED_REPO = "Young-jin/kobert-moodiary-app"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
USER_DB_FILE = "users.json"
DIARY_DB_FILE = "diary_db.json"

# 비상용 TMDB 키
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

# 감정별 테마 색상/이모지
EMOTION_META = {
    "행복": {"color": "#FFD700", "emoji": "😆", "desc": "기분이 최고조인 하루였네요!"},
    "슬픔": {"color": "#1E90FF", "emoji": "😭", "desc": "마음이 조금 지친 하루였군요."},
    "분노": {"color": "#FF4500", "emoji": "🤬", "desc": "스트레스가 많았던 하루였네요."},
    "힘듦": {"color": "#808080", "emoji": "🤯", "desc": "정말 고생 많았어요. 휴식이 필요해요."},
    "놀람": {"color": "#8A2BE2", "emoji": "😱", "desc": "예상치 못한 일이 있었나 봐요!"},
    "중립": {"color": "#A9A9A9", "emoji": "😐", "desc": "평온한 하루였군요."}
}

st.set_page_config(layout="wide", page_title="MOODIARY")

# =========================================
# 🔐 3) 데이터 관리 함수
# =========================================
def load_json(filename):
    if not os.path.exists(filename): return {}
    try:
        with open(filename, "r", encoding='utf-8') as f: return json.load(f)
    except: return {}

def save_json(filename, data):
    with open(filename, "w", encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_diary_entry(username, date, emotion, text):
    db = load_json(DIARY_DB_FILE)
    if username not in db: db[username] = {}
    db[username][date] = {"emotion": emotion, "text": text}
    save_json(DIARY_DB_FILE, db)

def get_my_diaries(username):
    return load_json(DIARY_DB_FILE).get(username, {})

# =========================================
# 🧠 4) AI 및 추천 로직
# =========================================
@st.cache_resource
def load_kobert_model():
    try:
        CORRECT_ID_TO_LABEL = {0: '분노', 1: '기쁨', 2: '불안', 3: '당황', 4: '슬픔', 5: '상처'}
        config = AutoConfig.from_pretrained(KOBERT_BASE_MODEL, trust_remote_code=True, num_labels=6, id2label=CORRECT_ID_TO_LABEL, label2id={l: i for i, l in CORRECT_ID_TO_LABEL.items()})
        tokenizer = AutoTokenizer.from_pretrained(KOBERT_BASE_MODEL, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(KOBERT_SAVED_REPO, config=config, trust_remote_code=True, ignore_mismatched_sizes=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        post_processing_map = getattr(model.config, 'post_processing_map', None) or {'기쁨': '행복', '슬픔': '슬픔', '상처': '슬픔', '불안': '힘듦', '당황': '놀람', '분노': '분노'}
        return model, tokenizer, device, post_processing_map
    except: return None, None, None, None

def analyze_diary_kobert(text, model, tokenizer, device, post_processing_map):
    if not text: return None, 0.0
    enc = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(device)
    with torch.no_grad(): logits = model(**enc).logits
    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    score = float(probs[pred_id].cpu().item())
    id2label = getattr(model.config, "id2label", {})
    original = id2label.get(pred_id) or id2label.get(str(pred_id)) or "중립"
    return post_processing_map.get(original, original), score

@st.cache_resource
def get_spotify_client():
    creds = st.secrets.get("spotify", {})
    cid, secret = creds.get("client_id"), creds.get("client_secret")
    if not cid or not secret: return None
    try:
        manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        return spotipy.Spotify(client_credentials_manager=manager, retries=3, backoff_factor=0.3)
    except: return None

# ⭐️⭐️⭐️ Spotify 추천 로직 (강력한 안전장치 추가) ⭐️⭐️⭐️
def recommend_music(emotion):
    sp = get_spotify_client()
    if not sp: return [{"error": "Spotify 연결 실패"}]

    KR_KEYWORDS = {
        "행복": ["여행", "행복", "케이팝 최신", "여름 노래"],
        "슬픔": ["발라드 최신", "이별 노래", "감성 케이팝", "K-ballad"],
        "분노": ["인기 밴드", "팝송", "스트레스", "재즈"],
        "힘듦": ["위로 노래", "힐링 케이팝", "잔잔한 팝"],
        "놀람": ["파티 케이팝", "EDM 케이팝", "페스티벌 음악"],
    }
    query = random.choice(KR_KEYWORDS.get(emotion, ["케이팝"])) + " year:2010-2025 NOT children"

    try:
        # 1차 시도: 키워드 검색
        res = sp.search(q=query, type="track", limit=50, market="KR")
        tracks = (res.get("tracks") or {}).get("items") or []
        valid = []
        for t in tracks:
            if t.get('id') and t.get('name'):
                 valid.append({"id": t['id'], "title": t['name']})

        # 2차 시도 (만약 실패 시): 공식 차트에서 가져오기 (안전빵)
        if not valid:
            # Spotify 공식 K-Pop Top 50 차트 ID
            top_50_id = "37i9dQZEVXbNxXF4UeQlye" 
            res_pl = sp.playlist_items(top_50_id, limit=50, market="KR")
            items = res_pl.get('items', []) if res_pl else []
            for it in items:
                t = it.get('track')
                if t and t.get('id'):
                    valid.append({"id": t['id'], "title": t['name']})

        if not valid: return [{"error": "추천 곡을 찾지 못했습니다."}]
        
        # 중복 제거 및 3곡 선택
        seen = set(); unique = []
        for v in valid:
            if v['id'] not in seen: unique.append(v); seen.add(v['id'])
        return random.sample(unique, k=min(3, len(unique)))

    except Exception as e: return [{"error": f"Spotify 오류: {e}"}]

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or st.secrets.get("TMDB_API_KEY") or EMERGENCY_TMDB_KEY
    if not key: return [{"text": "TMDB 연결 실패", "poster": None}]
    GENRES = {"행복": "35|10749|10751|27", "분노": "28|12|35|878", "슬픔": "35|10751|14", "힘듦": "35|10751|14", "놀람": "35|10751|14"}
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={
            "api_key": key, "language": "ko-KR", "sort_by": "popularity.desc", "with_genres": GENRES.get(emotion), "without_genres": "16",
            "page": random.randint(1, 5), "vote_count.gte": 1000, "vote_average.gte": 7.5, "primary_release_date.gte": "2000-01-01"
        }, timeout=10)
        r.raise_for_status(); results = r.json().get("results", [])
        if not results: return [{"text": "조건에 맞는 영화가 없습니다.", "poster": None}]
        picks = random.sample(results, min(3, len(results)))
        return [{"title": m.get("title"), "year": (m.get("release_date") or "")[:4], "rating": m.get("vote_average", 0.0), "overview": m.get("overview", ""), "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in picks]
    except Exception as e: return [{"text": f"TMDb 오류: {e}", "poster": None}]

# =========================================
# 🖥️ 5) 화면 구성 (페이지 분리)
# =========================================

# 1. 로그인 페이지
def login_page():
    st.title("MOODIARY 💖")
    t1, t2 = st.tabs(["🔑 로그인", "📝 회원가입"])
    with t1:
        lid = st.text_input("아이디", key="lid")
        lpw = st.text_input("비밀번호 (4자리)", type="password", key="lpw")
        if st.button("로그인", use_container_width=True):
            users = load_json(USER_DB_FILE)
            if lid in users and users[lid] == lpw:
                st.session_state.logged_in = True
                st.session_state.username = lid
                st.session_state.page = "dashboard"
                st.rerun()
            else: st.error("정보가 일치하지 않습니다.")
    with t2:
        nid = st.text_input("새 아이디", key="nid")
        npw = st.text_input("새 비밀번호 (4자리)", type="password", key="npw", max_chars=4)
        if st.button("가입하기", use_container_width=True):
            users = load_json(USER_DB_FILE)
            if nid in users: st.error("이미 있는 아이디입니다.")
            elif len(nid)<1 or len(npw)!=4 or not npw.isdigit(): st.error("입력 형식을 확인해주세요.")
            else:
                users[nid] = npw
                save_json(USER_DB_FILE, users)
                st.success("가입 완료! 로그인해주세요.")

# 2. 대시보드 (달력) 페이지
def dashboard_page():
    st.title(f"{st.session_state.username}님의 MOODIARY 📅")
    if st.sidebar.button("로그아웃"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.rerun()

    my_diaries = get_my_diaries(st.session_state.username)
    events = []
    for date, data in my_diaries.items():
        emo = data.get("emotion", "중립")
        meta = EMOTION_META.get(emo, EMOTION_META["중립"])
        events.append({"title": meta["emoji"], "start": date, "backgroundColor": meta["color"], "borderColor": meta["color"], "allDay": True})

    calendar(events=events, options={"headerToolbar": {"left": "prev,next today", "center": "title", "right": ""}, "initialView": "dayGridMonth"}, custom_css=".fc-event-title { font-size: 1.5em !important; text-align: center; }")
    st.write("")
    if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", use_container_width=True):
        st.session_state.page = "write"
        st.rerun()

# 3. 일기 작성 페이지
def write_page(model, tokenizer, device, postmap):
    st.title("오늘의 이야기 📝")
    if st.button("⬅️ 달력으로 돌아가기"):
        st.session_state.page = "dashboard"
        st.rerun()

    txt = st.text_area("오늘 하루는 어땠나요?", height=300)
    
    if st.button("🔍 감정 분석하고 추천 받기", type="primary", use_container_width=True):
        if not txt.strip():
            st.warning("내용을 입력해주세요.")
            return
        
        with st.spinner("AI가 분석하고 추천을 찾고 있어요..."):
            # 분석 및 추천 실행
            emo, sc = analyze_diary_kobert(txt, model, tokenizer, device, postmap)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            
            # 일기 저장
            today = datetime.now().strftime("%Y-%m-%d")
            save_diary_entry(st.session_state.username, today, emo, txt)
            
            # 결과 페이지로 이동
            st.session_state.page = "result"
            st.rerun()

# 4. ⭐️ NEW 결과 페이지 (분리됨)
def result_page():
    emo = st.session_state.final_emotion
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])
    
    # 상단 헤더
    st.markdown(f"<h1 style='text-align: center; color: {meta['color']};'>{meta['emoji']} {emo}</h1>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='text-align: center;'>{meta['desc']}</h3>", unsafe_allow_html=True)
    st.divider()

    # 추천 컨텐츠 표시
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🎵 추천 음악")
        for item in st.session_state.music_recs:
            if item.get('id'):
                components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=80)
            else: st.error(item.get('error', '음악 로딩 실패'))
            
    with c2:
        st.subheader("🎬 추천 영화")
        for item in st.session_state.movie_recs:
            if item.get('poster'):
                ic, tc = st.columns([1, 2])
                ic.image(item['poster'], use_container_width=True)
                tc.markdown(f"**{item['title']} ({item['year']})**\n\n⭐ {item['rating']:.1f}\n\n*{item.get('overview','')[:100]}...*")
            else: st.error(item.get('text', '영화 로딩 실패'))
            st.write("") # 간격

    st.divider()
    if st.button("🏠 홈으로 돌아가기", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()

# =========================================
# 🚀 앱 메인 컨트롤러
# =========================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "login"

if not st.session_state.logged_in:
    login_page()
else:
    if st.session_state.page == "dashboard":
        dashboard_page()
    elif st.session_state.page == "write":
        model, tokenizer, device, postmap = load_kobert_model()
        if model: write_page(model, tokenizer, device, postmap)
        else: st.error("AI 모델 로드 중... 잠시 후 다시 시도해주세요.")
    elif st.session_state.page == "result":
        result_page()
