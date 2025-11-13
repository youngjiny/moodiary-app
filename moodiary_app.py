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
import gspread
from google.oauth2.service_account import Credentials

# (선택) Spotify SDK
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

# --- 2) 기본 설정 ---
KOBERT_BASE_MODEL = "monologg/kobert"
KOBERT_SAVED_REPO = "Young-jin/kobert-moodiary-app"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
GSHEET_DB_NAME = "moodiary_db" # ⭐️ 구글 시트 파일 이름

# 비상용 TMDB 키
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

# 감정별 테마 (색상, 이모지)
EMOTION_META = {
    "행복": {"color": "#FFD700", "emoji": "😆", "desc": "최고의 하루!"},
    "슬픔": {"color": "#1E90FF", "emoji": "😭", "desc": "토닥토닥, 힘내요."},
    "분노": {"color": "#FF4500", "emoji": "🤬", "desc": "워워, 진정해요."},
    "힘듦": {"color": "#808080", "emoji": "🤯", "desc": "휴식이 필요해."},
    "놀람": {"color": "#8A2BE2", "emoji": "😱", "desc": "깜짝 놀랐군요!"},
    "중립": {"color": "#A9A9A9", "emoji": "😐", "desc": "평온한 하루."}
}

st.set_page_config(layout="wide", page_title="MOODIARY")

# =========================================
# 🔐 3) 영구 데이터 관리 (Google Sheets)
# =========================================
@st.cache_resource
def get_gsheets_client():
    try:
        creds = st.secrets["connections"]["gsheets"]
        scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
        credentials = Credentials.from_service_account_info(creds, scopes=scope)
        return gspread.authorize(credentials)
    except Exception as e:
        # st.error(f"구글 시트 연결 실패: {e}")
        return None

def init_db():
    client = get_gsheets_client()
    if not client: return None
    try:
        sh = client.open(GSHEET_DB_NAME)
    except:
        return None # (시트가 없으면 None 반환)

    # 유저/일기 시트가 있는지 확인
    try:
        sh.worksheet("users")
        sh.worksheet("diaries")
    except:
        return None # (시트가 깨져있으면 None 반환)
    return sh

def get_all_users(sh):
    if not sh: return {}
    try:
        rows = sh.worksheet("users").get_all_records()
        return {row['username']: str(row['password']) for row in rows}
    except: return {}

def add_user(sh, username, password):
    if not sh: return False
    try:
        sh.worksheet("users").append_row([username, password])
        return True
    except: return False

def get_user_diaries(sh, username):
    if not sh: return {}
    try:
        rows = sh.worksheet("diaries").get_all_records()
        user_diaries = {}
        for row in rows:
            if row['username'] == username:
                user_diaries[row['date']] = {"emotion": row['emotion'], "text": row['text']}
        return user_diaries
    except: return {}

def add_diary(sh, username, date, emotion, text):
    if not sh: return False
    try:
        sh.worksheet("diaries").append_row([username, date, emotion, text])
        return True
    except: return False

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
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        return spotipy.Spotify(client_credentials_manager=manager, retries=3, backoff_factor=0.3)
    except: return None

# ⭐️ Spotify 로직 (강력한 안전장치)
def recommend_music(emotion):
    sp = get_spotify_client()
    if not sp: return [{"error": "Spotify 연결 실패"}]
    
    # 1. 감정별 공식/인기 큐레이션 ID (안정적)
    SAFE_PLAYLISTS = {
        "행복": ["37i9dQZEVXbJxxNsEk86S4", "37i9dQZF1DXcBWIGoYBM5M"], # K-Pop ON!, Today's Top Hits
        "슬픔": ["37i9dQZF1DXa29a0n9wGgC", "37i9dQZF1DX7qK8ma5wgG1"], # K-Pop Ballad, Sad Songs
        "분노": ["37i9dQZF1DXdfhOsjPtoaS", "37i9dQZF1DWWJOmJ7nRx0C"], # K-Rock, Rock Hard
        "힘듦": ["37i9dQZF1DXdls6m8FLMpo", "37i9dQZF1DWV7EzJMK2FUI"], # Healing K-Pop, Jazz in the Background
        "놀람": ["37i9dQZEVXbJxxNsEk86S4", "37i9dQZF1DX4dyzvuaRJ0n"], # K-Pop ON!, Mint
        "중립": ["37i9dQZF1DWT9uTRZAYj0c"] # Chill Tracks
    }
    
    try:
        candidates = SAFE_PLAYLISTS.get(emotion, SAFE_PLAYLISTS["중립"])
        random.shuffle(candidates)
        
        valid_tracks = []
        for pid in candidates:
            try:
                results = sp.playlist_items(pid, limit=30)
                items = results.get('items', []) if results else []
                for it in items:
                    t = it.get('track')
                    if t and t.get('id') and t.get('name'):
                         valid_tracks.append({"id": t['id'], "title": t['name']})
                if len(valid_tracks) >= 5: break
            except: continue

        # 2. 만약 1번이 (네트워크 등 이유로) 실패하면, 키워드 검색으로 재시도 (안전장치 2)
        if not valid_tracks:
            KR_KEYWORDS = {
                "행복": ["여행", "행복", "케이팝 최신", "여름 노래"],
                "슬픔": ["발라드 최신", "이별 노래", "감성 케이팝"],
                "분노": ["인기 밴드", "팝송", "스트레스", "재즈"],
                "힘듦": ["위로 노래", "힐링 케이팝", "잔잔한 팝"],
                "놀람": ["파티 케이팝", "EDM 케이팝", "페스티벌 음악"],
            }
            query = random.choice(KR_KEYWORDS.get(emotion, ["케이팝"])) + " year:2010-2025 NOT children"
            res = sp.search(q=query, type="track", limit=20)
            items = (res.get("tracks") or {}).get("items") or []
            for t in items:
                if t.get('id') and t.get('name'):
                    valid_tracks.append({"id": t['id'], "title": t['name']})

        if not valid_tracks: return [{"error": "추천 곡을 찾지 못했습니다."}]
        
        seen = set(); unique = []
        for v in valid_tracks:
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
        }, timeout=5)
        r.raise_for_status(); results = r.json().get("results", [])
        if not results: return [{"text": "조건에 맞는 영화가 없습니다.", "poster": None}]
        picks = random.sample(results, min(3, len(results)))
        return [{"title": m.get("title"), "year": (m.get("release_date") or "")[:4], "rating": m.get("vote_average", 0.0), "overview": m.get("overview", ""), "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None} for m in picks]
    except Exception as e: return [{"text": f"TMDb 오류: {e}", "poster": None}]

# =========================================
# 🖥️ 5) 화면 구성
# =========================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "login"

def login_page():
    st.title("MOODIARY 💖")
    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
    sh = init_db()
    if sh is None: st.error("데이터베이스 연결 실패. Secrets 설정을 확인하세요."); return

    with tab1:
        lid = st.text_input("아이디", key="lid")
        lpw = st.text_input("비밀번호", type="password", key="lpw")
        if st.button("로그인", width='stretch'):
            users = get_all_users(sh)
            if lid in users and str(users[lid]) == str(lpw):
                st.session_state.logged_in = True
                st.session_state.username = lid
                st.session_state.page = "dashboard"
                st.rerun()
            else: st.error("정보가 일치하지 않습니다.")
    with tab2:
        nid = st.text_input("새 아이디", key="nid")
        npw = st.text_input("새 비밀번호 (4자리)", type="password", key="npw", max_chars=4)
        if st.button("가입하기", width='stretch'):
            users = get_all_users(sh)
            if nid in users: st.error("이미 있는 아이디입니다.")
            elif len(nid)<1 or len(npw)!=4: st.error("입력을 확인해주세요.")
            else:
                if add_user(sh, nid, npw): st.success("가입 성공! 로그인해주세요.")
                else: st.error("가입 실패 (DB 오류)")

def dashboard_page():
    st.title(f"{st.session_state.username}님의 감정 달력 📅")
    
    # ⭐️ 감정 색상 범례 (Legend)
    legend_cols = st.columns(6)
    for i, (emo, meta) in enumerate(EMOTION_META.items()):
        legend_cols[i].markdown(f"<span style='color:{meta['color']}; font-size: 1.2em;'>●</span> {emo}", unsafe_allow_html=True)
    st.divider()

    # 달력 데이터 로드
    sh = init_db()
    my_diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    for date_str, data in my_diaries.items():
        emo = data.get("emotion", "중립")
        meta = EMOTION_META.get(emo, EMOTION_META["중립"])
        events.append({"title": meta["emoji"], "start": date_str, "display": "background", "backgroundColor": meta["color"], "borderColor": meta["color"]})
        events.append({"title": meta["emoji"], "start": date_str, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": "#000000"})

    calendar(events=events, options={"headerToolbar": {"left": "prev,next today", "center": "title", "right": ""}, "initialView": "dayGridMonth"}, 
             custom_css=".fc-event-title { font-size: 2em !important; text-align: center; } .fc-bg-event { opacity: 0.6; }")
    st.write("")

    # ⭐️⭐️⭐️ 신규 기능: 오늘 일기 유무에 따른 버튼 분리 ⭐️⭐️⭐️
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_diary_exists = today_str in my_diaries

    if today_diary_exists:
        st.info(f"오늘({today_str})의 일기({my_diaries[today_str]['emotion']} {EMOTION_META[my_diaries[today_str]['emotion']]['emoji']})가 이미 작성되었습니다.")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ 오늘 일기 수정/확인하기", width='stretch'):
                st.session_state.page = "write"
                st.session_state.diary_input = my_diaries[today_str]['text']
                st.rerun()
        with col2:
            def handle_show_recs():
                today_emo = my_diaries[today_str]['emotion']
                st.session_state.final_emotion = today_emo
                st.session_state.music_recs = recommend_music(today_emo)
                st.session_state.movie_recs = recommend_movies(today_emo)
                st.session_state.page = "result"
            if st.button("🎵🎬 오늘의 추천 바로 보기", type="primary", width='stretch'):
                handle_show_recs()
                st.rerun()
    else:
        # 오늘 일기가 없을 때
        if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", width='stretch'):
            st.session_state.page = "write"
            st.session_state.diary_input = "" # ⭐️ 새 일기
            st.rerun()

def result_page():
    emo = st.session_state.final_emotion
    meta = EMOTION_META.get(emo, EMOTION_META["중립"])
    st.markdown(f"<h2 style='text-align: center; color: {meta['color']};'>{meta['emoji']} 오늘의 감정: {emo}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>{meta['desc']}</h4>", unsafe_allow_html=True)
    
    if st.button("⬅️ 달력으로 돌아가기"):
        st.session_state.page = "dashboard"
        st.rerun()
    st.divider()

    def refresh_music(): st.session_state.music_recs = recommend_music(emo)
    def refresh_movies(): st.session_state.movie_recs = recommend_movies(emo)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎵 추천 음악")
        st.button("🔄 다른 음악", on_click=refresh_music, key="rm_btn", width='stretch')
        for item in st.session_state.music_recs:
            if item.get('id'):
                components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=80)
            else: st.error(item.get("error", "로딩 실패"))
    with c2:
        st.markdown("#### 🎬 추천 영화")
        st.button("🔄 다른 영화", on_click=refresh_movies, key="rv_btn", width='stretch')
        for item in st.session_state.movie_recs:
            if item.get('poster'):
                ic, tc = st.columns([1, 2])
                ic.image(item['poster'], use_container_width=True)
                tc.markdown(f"**{item['title']} ({item['year']})**\n⭐ {item['rating']:.1f}\n\n*{item.get('overview','')[:100]}...*")
            else: st.error(item.get("text", "로딩 실패"))

def write_page():
    st.title("오늘의 이야기 📝")
    if st.button("⬅️ 뒤로 가기"):
        st.session_state.page = "dashboard"
        st.rerun()

    model, tokenizer, device, postmap = load_kobert_model()
    if not model: st.error("AI 모델 로드 중..."); return

    # ⭐️ 수정 시 기존 일기 불러오기
    if "diary_input" not in st.session_state: st.session_state.diary_input = ""
    txt = st.text_area("오늘 하루는 어땠나요?", value=st.session_state.diary_input, height=300, key="diary_editor")
    
    if st.button("🔍 감정 분석하고 저장하기", type="primary", width='stretch'):
        if not txt.strip(): st.warning("내용을 입력해주세요."); return
        
        with st.spinner("분석 및 저장 중..."):
            emo, sc = analyze_diary_kobert(txt, model, tokenizer, device, postmap)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            
            sh = init_db()
            today = datetime.now().strftime("%Y-%m-%d")
            add_diary(sh, st.session_state.username, today, emo, txt) # ⭐️ 덮어쓰기 (저장)
            
            st.session_state.page = "result"
            st.rerun()

# =========================================
# 🚀 앱 메인 컨트롤러
# =========================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "login"

# ⭐️ use_container_width 경고 수정
if st.session_state.logged_in:
    with st.sidebar:
        st.write(f"**{st.session_state.username}**님")
        if st.button("로그아웃", width='stretch'):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

# 페이지 라우팅
if not st.session_state.logged_in: login_page()
elif st.session_state.page == "dashboard": dashboard_page()
elif st.session_state.page == "write": write_page()
elif st.session_state.page == "result": result_page()
