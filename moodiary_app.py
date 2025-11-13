# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import time 
import streamlit.components.v1 as components 

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

# 비상용 TMDB 키
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

st.set_page_config(layout="wide", page_title="MOODIARY")

# --- 3) KoBERT 모델 로드 ---
@st.cache_resource
def load_kobert_model():
    try:
        CORRECT_ID_TO_LABEL = {
            0: '분노', 1: '기쁨', 2: '불안',
            3: '당황', 4: '슬픔', 5: '상처'
        }
        config = AutoConfig.from_pretrained(
            KOBERT_BASE_MODEL,
            trust_remote_code=True,
            num_labels=6,
            id2label=CORRECT_ID_TO_LABEL,
            label2id={label: idx for idx, label in CORRECT_ID_TO_LABEL.items()}
        )
        tokenizer = AutoTokenizer.from_pretrained(
            KOBERT_BASE_MODEL,
            trust_remote_code=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            KOBERT_SAVED_REPO,
            config=config,
            trust_remote_code=True,
            ignore_mismatched_sizes=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        post_processing_map = getattr(model.config, 'post_processing_map', None)
        if post_processing_map is None:
            post_processing_map = {
                '기쁨': '행복', '슬픔': '슬픔', '상처': '슬픔',
                '불안': '힘듦', '당황': '놀람', '분노': '분노'
            }

        return model, tokenizer, device, post_processing_map
    except Exception as e:
        st.error("🚨 AI 모델을 불러오는 데 실패했습니다.")
        return None, None, None, None

# --- 4) 감정 분석 ---
def analyze_diary_kobert(text, model, tokenizer, device, post_processing_map):
    if not text:
        return None, 0.0

    enc = tokenizer(text, truncation=True, padding=True, max_length=128, return_tensors="pt")
    for k in enc:
        enc[k] = enc[k].to(device)

    with torch.no_grad():
        logits = model(**enc).logits

    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    score = float(probs[pred_id].cpu().item())

    id2label = getattr(model.config, "id2label", {})
    original = id2label.get(pred_id) or id2label.get(str(pred_id)) or "중립"
    final_emotion = post_processing_map.get(original, original)
    return final_emotion, score

# --- 5) Spotify 클라이언트 ---
@st.cache_resource
def get_spotify_client():
    if spotipy is None or SpotifyClientCredentials is None:
        return None
    creds = st.secrets.get("spotify", {})
    cid = creds.get("client_id")
    secret = creds.get("client_secret")
    if not cid or not secret:
        return None
    try:
        manager = SpotifyClientCredentials(client_id=cid, client_secret=secret)
        return spotipy.Spotify(client_credentials_manager=manager, retries=3, status_retries=3, backoff_factor=0.3)
    except Exception:
        return None

# --- 6) Spotify 추천 (market="KR" 복구) ---
def recommend_music(emotion):
    sp = get_spotify_client()
    if not sp:
        return ["Spotify 연결 실패 (Secrets 확인 필요)"]

    # 1. 한/미/일 키워드
    SEARCH_KEYWORDS = {
        "행복": ["K-Pop Dance", "J-Pop Happy Hits", "American Pop Upbeat"],
        "슬픔": ["K-Pop Ballad", "J-Pop Sad", "US Pop Sad Songs"],
        "분노": ["K-Rock", "J-Rock Anthems", "American Hard Rock"],
        "힘듦": ["K-Indie Healing", "J-Pop Chill", "US Acoustic Pop"],
        "놀람": ["K-Pop EDM", "J-EDM Party", "US Festival EDM"],
    }
    
    base_query = random.choice(SEARCH_KEYWORDS.get(emotion, ["K-Pop"]))
    # ⭐️ 노키즈 필터
    query = f"{base_query} year:2010-2025 NOT children NOT nursery NOT 동요 NOT 키즈 NOT 어린이"

    try:
        # ⭐️⭐️⭐️ market="KR" 복구 ⭐️⭐️⭐️
        res = sp.search(q=query, type="track", limit=50, market="KR")
        tracks = (res.get("tracks") or {}).get("items") or []
        
        valid_candidates = []
        for t in tracks:
            tid = t.get("id")
            # ⭐️ 중복 방지 확인
            if tid and tid not in st.session_state.recent_music_ids:
                 # ⭐️ ID와 함께 제목/아티스트도 같이 저장 (UI 표시용)
                name = t.get("name")
                artists = t.get("artists") or []
                artist = artists[0].get("name") if artists else "Unknown"
                if name:
                    valid_candidates.append({"title": name, "artist": artist, "id": tid})

        if not valid_candidates:
             # 검색 결과가 없으면 기록 초기화 후 한 번 더 기회 제공
             st.session_state.recent_music_ids = []
             return ["새로운 곡을 찾지 못했습니다. 다시 눌러주세요."]

        # 최대 3개 선택
        final_picks = random.sample(valid_candidates, k=min(3, len(valid_candidates)))
        
        # 기록 업데이트
        for pick in final_picks:
            st.session_state.recent_music_ids.append(pick["id"])
        if len(st.session_state.recent_music_ids) > 60:
             st.session_state.recent_music_ids = st.session_state.recent_music_ids[-60:]

        return final_picks # ⭐️ ID, Title, Artist가 담긴 딕셔너리 리스트 반환

    except Exception as e:
        return [f"Spotify 오류: {e}"]


# --- 7) TMDB 추천 (2000년+, 평점 7.5+, 투표 1000+, 중복 방지) ---
def recommend_movies(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key", "")
    if not key:
        key = st.secrets.get("TMDB_API_KEY", "")
    if not key:
        key = EMERGENCY_TMDB_KEY

    if not key:
        return [{"text": "TMDB 연결 실패", "poster": None, "overview": ""}]

    GENRES = {
        "행복": "35|10749|10751|27",
        "분노": "28|12|35|878",
        "슬픔": "35|10751|14",
        "힘듦": "35|10751|14",
        "놀람": "35|10751|14",
    }
    g = GENRES.get(emotion)
    if not g:
        return [{"text": f"[{emotion}] 장르 매핑 오류", "poster": None, "overview": ""}]

    try:
        # 페이지 랜덤화 (중복 방지)
        random_page = random.randint(1, 5)
        
        r = requests.get(
            f"{TMDB_BASE_URL}/discover/movie",
            params={
                "api_key": key,
                "language": "ko-KR",
                "sort_by": "popularity.desc",
                "with_genres": g,
                "without_genres": "16",      
                "page": random_page,
                "vote_count.gte": 1000,      
                "vote_average.gte": 7.5,     
                "primary_release_date.gte": "2000-01-01" 
            },
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
             r = requests.get(
                f"{TMDB_BASE_URL}/discover/movie",
                params={
                    "api_key": key, "language": "ko-KR", "sort_by": "popularity.desc",
                    "with_genres": g, "without_genres": "16", "page": 1,
                    "vote_count.gte": 1000, "vote_average.gte": 7.5,
                    "primary_release_date.gte": "2000-01-01"
                },
                timeout=10,
             )
             r.raise_for_status()
             results = r.json().get("results", [])
             if not results:
                 return [{"text": f"조건에 맞는 명작 영화가 부족합니다.", "poster": None, "overview": ""}]

        valid_candidates = []
        for m in results:
            mid = m.get("id")
            if mid and mid not in st.session_state.recent_movie_ids:
                title = m.get("title", "제목없음")
                year = (m.get("release_date") or "")[:4] or "N/A"
                rating = m.get("vote_average", 0.0)
                poster = f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None
                overview = m.get("overview", "") or "줄거리 정보가 없습니다."
                
                valid_candidates.append({
                    "id": mid,
                    "poster": poster,
                    "title": title,
                    "year": year,
                    "rating": rating,
                    "overview": overview 
                })

        if not valid_candidates:
             st.session_state.recent_movie_ids = [] 
             return [{"text": "새로운 영화를 찾을 수 없습니다. 다시 시도해주세요.", "poster": None, "overview": ""}]

        final_picks = random.sample(valid_candidates, k=min(3, len(valid_candidates)))

        for pick in final_picks:
            st.session_state.recent_movie_ids.append(pick["id"])
        if len(st.session_state.recent_movie_ids) > 60:
            st.session_state.recent_movie_ids = st.session_state.recent_movie_ids[-60:]

        return final_picks

    except Exception as e:
        return [{"text": f"TMDb 오류: {type(e).__name__}: {e}", "poster": None, "overview": ""}]


# --- 8) 통합 추천 ---
def recommend(emotion):
    return {
        "음악": recommend_music(emotion),
        "영화": recommend_movies(emotion),
    }

# --- 9) 상태/입력/실행 ---
model, tokenizer, device, postmap = load_kobert_model()

if "diary_text" not in st.session_state:
    st.session_state.diary_text = ""
if "final_emotion" not in st.session_state:
    st.session_state.final_emotion = None
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0
if "music_recs" not in st.session_state:
    st.session_state.music_recs = []
if "movie_recs" not in st.session_state:
    st.session_state.movie_recs = []
if "recent_music_ids" not in st.session_state:
    st.session_state.recent_music_ids = []
if "recent_movie_ids" not in st.session_state:
    st.session_state.recent_movie_ids = []

# --- 10) 버튼 콜백 ---
def handle_analyze_click():
    txt = st.session_state.diary_text
    if not txt.strip():
        st.warning("일기를 입력해주세요.")
        return
    if model is None:
        st.error("AI 모델 로드에 실패했습니다. 잠시 후 다시 시도해주세요.")
        return
    with st.spinner("AI가 분석 중입니다..."):
        emo, sc = analyze_diary_kobert(txt, model, tokenizer, device, postmap)
        st.session_state.final_emotion = emo
        st.session_state.confidence = sc
        
        with st.spinner("추천을 불러오는 중..."):
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)

def refresh_music():
    if st.session_state.final_emotion:
        with st.spinner("새로운 음악을 찾고 있어요..."):
            st.session_state.music_recs = recommend_music(st.session_state.final_emotion)

def refresh_movies():
    if st.session_state.final_emotion:
        with st.spinner("새로운 영화를 찾고 있어요..."):
            st.session_state.movie_recs = recommend_movies(st.session_state.final_emotion)

# --- 11) 입력 UI ---
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 오늘의 일기를 작성해주세요:")
    st.text_area(" ", key="diary_text", height=230, label_visibility="collapsed")

with col2:
    st.write(" "); st.write(" ")
    st.write(" "); st.write(" ")
    st.button("🔍 내 하루 감정 분석하기", type="primary", on_click=handle_analyze_click, use_container_width=True)

# --- 12) 결과/추천 출력 ---
if st.session_state.final_emotion:
    emo = st.session_state.final_emotion
    st.subheader(f"오늘 하루의 핵심 감정은 '{emo}' 입니다.")
    st.divider()
    st.subheader(f"'{emo}' 감정을 위한 오늘의 Moodiary 추천")

    music_items = st.session_state.music_recs
    movie_items = st.session_state.movie_recs

    for i in range(3):
        col_music, col_movie = st.columns(2)

        with col_music:
            if i == 0: 
                st.markdown("#### 🎵 이런 음악도 들어보세요?")
                st.button("🔄 다른 음악 추천", on_click=refresh_music, use_container_width=True)
            
            if i < len(music_items):
                it = music_items[i]
                # ⭐️⭐️⭐️ 딕셔너리(정보 꾸러미)인지 확인 ⭐️⭐️⭐️
                if isinstance(it, dict) and it.get("id"):
                    track_id = it.get("id")
                    embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
                    components.iframe(embed_url, height=152)
                elif isinstance(it, str):
                    st.error(it) # "Spotify 오류: ..."
                else:
                    st.write(f"- {it}")
            
        with col_movie:
            if i == 0: 
                st.markdown("#### 🎬 이런 영화도 추천해요?")
                st.button("🔄 다른 영화 추천", on_click=refresh_movies, use_container_width=True)
                
            if i < len(movie_items):
                it = movie_items[i]
                if isinstance(it, dict) and it.get("title"):
                    poster = it.get("poster")
                    if poster:
                        st.image(poster, width=160)
                    title = it.get("title", "제목없음")
                    year = it.get("year", "N/A")
                    rating = float(it.get("rating", 0.0))
                    overview = it.get("overview", "") 
                    line = f"##### **{title} ({year})**\n⭐ {rating:.1f}\n\n*{overview}*"
                    st.markdown(line)
                elif isinstance(it, dict):
                    st.error(it.get("text", "영화 오류"))
                else:
                    st.error(f"- {it}")

        st.markdown("---")
