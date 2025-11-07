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
USER_DB_FILE = "users.json" # ⭐️ 회원 정보를 저장할 파일 이름

# 비상용 TMDB 키
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

st.set_page_config(layout="wide", page_title="MOODIARY")

# =========================================
# 🔐 3) 로그인/회원가입 관리 함수
# =========================================
def load_users():
    """users.json 파일에서 회원 정보를 읽어옵니다."""
    if not os.path.exists(USER_DB_FILE):
        return {}
    try:
        with open(USER_DB_FILE, "r") as f:
            return json.load(f)
    except:
        return {}

def save_user(username, password):
    """새로운 회원 정보를 users.json 파일에 저장합니다."""
    users = load_users()
    users[username] = password
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)

def login_page():
    """로그인 및 회원가입 화면을 그립니다."""
    st.title("MOODIARY 💖 에 오신 것을 환영합니다")
    
    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])

    # --- 로그인 탭 ---
    with tab1:
        st.subheader("로그인")
        login_id = st.text_input("아이디", key="login_id")
        login_pw = st.text_input("비밀번호 (숫자 4자리)", type="password", key="login_pw")
        
        if st.button("로그인 하기"):
            users = load_users()
            if login_id in users and users[login_id] == login_pw:
                st.session_state.logged_in = True
                st.session_state.username = login_id
                st.success(f"{login_id}님 환영합니다! 잠시 후 이동합니다...")
                time.sleep(1)
                st.rerun()
            else:
                st.error("아이디 또는 비밀번호가 잘못되었습니다.")

    # --- 회원가입 탭 ---
    with tab2:
        st.subheader("회원가입")
        new_id = st.text_input("새 아이디", key="new_id")
        new_pw = st.text_input("새 비밀번호 (숫자 4자리)", type="password", key="new_pw", max_chars=4)
        
        if st.button("가입하기"):
            users = load_users()
            if new_id in users:
                st.error("이미 존재하는 아이디입니다.")
            elif len(new_id) < 1:
                 st.error("아이디를 입력해주세요.")
            elif len(new_pw) != 4 or not new_pw.isdigit():
                st.error("비밀번호는 반드시 '숫자 4자리'여야 합니다.")
            else:
                save_user(new_id, new_pw)
                st.success("가입 성공! 로그인 탭에서 로그인해주세요.")

# =========================================
# 🧠 4) AI 및 추천 로직 (기존 코드 유지)
# =========================================
@st.cache_resource
def load_kobert_model():
    try:
        CORRECT_ID_TO_LABEL = {
            0: '분노', 1: '기쁨', 2: '불안', 3: '당황', 4: '슬픔', 5: '상처'
        }
        config = AutoConfig.from_pretrained(KOBERT_BASE_MODEL, trust_remote_code=True, num_labels=6, id2label=CORRECT_ID_TO_LABEL, label2id={label: idx for idx, label in CORRECT_ID_TO_LABEL.items()})
        tokenizer = AutoTokenizer.from_pretrained(KOBERT_BASE_MODEL, trust_remote_code=True)
        model = AutoModelForSequenceClassification.from_pretrained(KOBERT_SAVED_REPO, config=config, trust_remote_code=True, ignore_mismatched_sizes=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        post_processing_map = getattr(model.config, 'post_processing_map', None)
        if post_processing_map is None:
            post_processing_map = {'기쁨': '행복', '슬픔': '슬픔', '상처': '슬픔', '불안': '힘듦', '당황': '놀람', '분노': '분노'}
        return model, tokenizer, device, post_processing_map
    except Exception:
        return None, None, None, None

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

def recommend_music(emotion):
    sp = get_spotify_client()
    if not sp: return ["Spotify 연결 실패"]
    
    def is_korean(txt): return isinstance(txt, str) and any('가' <= ch <= '힣' for ch in txt)
    KR_KEYWORDS = {
        "행복": ["여행", "행복", "케이팝 최신", "여름 노래"],
        "슬픔": ["발라드 최신", "이별 노래", "감성 케이팝", "K-ballad"],
        "분노": ["인기 밴드", "팝송", "스트레스", "재즈"],
        "힘듦": ["위로 노래", "힐링 케이팝", "잔잔한 팝"],
        "놀람": ["파티 케이팝", "EDM 케이팝", "페스티벌 음악"],
    }
    query = random.choice(KR_KEYWORDS.get(emotion, ["케이팝"])) + " year:2010-2025"
    try:
        res = sp.search(q=query, type="track", limit=50, market="KR")
        tracks = (res.get("tracks") or {}).get("items") or []
        valid = []
        for t in tracks:
            if t['id'] and t['name'] and (is_korean(t['name']) or is_korean(t['artists'][0]['name'])):
                valid.append({"title": t['name'], "artist": t['artists'][0]['name'], "id": t['id']})
        
        if len(valid) < 10:
            pls = (sp.search(q=query, type="playlist", limit=10, market="KR").get("playlists") or {}).get("items") or []
            for pl in pls:
                if not pl or not pl.get('id'): continue
                try: items = (sp.playlist_items(pl['id'], limit=50, market="KR") or {}).get("items") or []
                except: continue
                for it in items:
                    tr = it.get("track")
                    if tr and tr.get('id') and tr.get('name'):
                        valid.append({"title": tr['name'], "artist": tr['artists'][0]['name'], "id": tr['id']})
                if len(valid) >= 10: break
                
        if not valid: return ["추천 곡을 찾지 못했습니다."]
        unique = {t['id']: t for t in valid}.values()
        return random.sample(list(unique), k=min(3, len(unique)))
    except Exception as e: return [f"Spotify 오류: {e}"]

def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key") or st.secrets.get("TMDB_API_KEY") or EMERGENCY_TMDB_KEY
    if not key: return [{"text": "TMDB 연결 실패", "poster": None, "overview": ""}]
    GENRES = {
        "행복": "35|10749|10751|27", "분노": "28|12|35|878",
        "슬픔": "35|10751|14", "힘듦": "35|10751|14", "놀람": "35|10751|14"
    }
    try:
        r = requests.get(f"{TMDB_BASE_URL}/discover/movie", params={
            "api_key": key, "language": "ko-KR", "sort_by": "popularity.desc",
            "with_genres": GENRES.get(emotion), "without_genres": "16",
            "page": random.randint(1, 5), "vote_count.gte": 1000, "vote_average.gte": 7.5,
            "primary_release_date.gte": "2000-01-01"
        }, timeout=10)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results: return [{"text": "조건에 맞는 영화가 없습니다.", "poster": None, "overview": ""}]
        picks = random.sample(results, min(3, len(results)))
        return [{"text": f"##### **{m['title']} ({m['release_date'][:4]})**\n⭐ {m['vote_average']:.1f}\n\n*{m.get('overview','')[:150]}...*", 
                 "poster": f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get('poster_path') else None} for m in picks]
    except Exception as e: return [{"text": f"TMDb 오류: {e}", "poster": None, "overview": ""}]

def recommend(emotion):
    return {"음악": recommend_music(emotion), "영화": recommend_movies(emotion)}

# =========================================
# 🖥️ 5) 메인 앱 화면 (로그인 성공 시 보임)
# =========================================
def main_app():
    st.title("MOODIARY 💖")
    
    # 사이드바: 로그인 정보 및 로그아웃
    with st.sidebar:
        st.write(f"환영합니다, **{st.session_state.username}**님! 👋")
        if st.button("로그아웃", type="primary"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.rerun()

    # 모델 로드 (사용자에게 안 보이게)
    model, tokenizer, device, postmap = load_kobert_model()

    # 세션 상태 초기화
    if "diary_text" not in st.session_state: st.session_state.diary_text = ""
    if "final_emotion" not in st.session_state: st.session_state.final_emotion = None
    if "confidence" not in st.session_state: st.session_state.confidence = 0.0
    if "music_recs" not in st.session_state: st.session_state.music_recs = []
    if "movie_recs" not in st.session_state: st.session_state.movie_recs = []

    # 콜백 함수
    def handle_analyze():
        if not st.session_state.diary_text.strip():
            st.warning("일기를 작성해주세요!")
            return
        if not model:
            st.error("AI 모델 로드 실패. 새로고침 해주세요.")
            return
        with st.spinner("AI가 감정을 분석 중입니다..."):
            emo, sc = analyze_diary_kobert(st.session_state.diary_text, model, tokenizer, device, postmap)
            st.session_state.final_emotion = emo
            st.session_state.confidence = sc
        with st.spinner("맞춤 컨텐츠를 찾고 있습니다..."):
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)

    def refresh_music_recs():
        if st.session_state.final_emotion:
            with st.spinner("새로운 음악을 찾는 중..."):
                st.session_state.music_recs = recommend_music(st.session_state.final_emotion)
    
    def refresh_movie_recs():
        if st.session_state.final_emotion:
            with st.spinner("새로운 영화를 찾는 중..."):
                st.session_state.movie_recs = recommend_movies(st.session_state.final_emotion)

    # 메인 UI
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 오늘의 일기를 작성해주세요:")
        st.text_area(" ", key="diary_text", height=230, label_visibility="collapsed")
    with col2:
        st.write("\n\n\n\n") # 간격 조정
        st.button("🔍 감정 분석하기", type="primary", on_click=handle_analyze, use_container_width=True)

    # 결과 표시
    if st.session_state.final_emotion:
        emo = st.session_state.final_emotion
        st.subheader(f"오늘의 핵심 감정: **{emo}**")
        st.divider()
        
        # 추천 섹션
        m_items = st.session_state.music_recs
        v_items = st.session_state.movie_recs
        
        for i in range(3):
            c1, c2 = st.columns(2)
            with c1:
                if i == 0:
                    st.markdown("#### 🎵 추천 음악")
                    st.button("🔄 다른 음악 보기", on_click=refresh_music_recs, key="rm_btn")
                if i < len(m_items):
                    item = m_items[i]
                    if isinstance(item, dict) and item.get('id'):
                        components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=80)
                    else: st.write(f"- {item}")
            with c2:
                if i == 0:
                    st.markdown("#### 🎬 추천 영화")
                    st.button("🔄 다른 영화 보기", on_click=refresh_movie_recs, key="rv_btn")
                if i < len(v_items):
                    item = v_items[i]
                    if item.get('poster'):
                        ic, tc = st.columns([1, 2])
                        ic.image(item['poster'], use_container_width=True)
                        tc.markdown(item['text'])
                    else: st.write(f"- {item.get('text')}")
            st.markdown("---")

# =========================================
# 🚀 앱 실행 진입점 (Entry Point)
# =========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if not st.session_state.logged_in:
    login_page()
else:
    main_app()
