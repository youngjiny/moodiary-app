# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import time 

# (선택) Spotify SDK
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

# --- 2) 기본 설정 ---
KOBERT_BASE_MODEL = "monologg/kobert"
KOBERT_SAVED_REPO = "Young-jin/kobert-moodiary-app" # 학습 가중치(HF)
TMDB_BASE_URL = "https://api.themoviedb.org/3"

st.set_page_config(layout="wide")
st.title("MOODIARY 💖")

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
        st.exception(e)
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

# --- 6) ⭐️ Spotify 추천 (1곡만 나오는 오류 수정) ---
def get_spotify_ai_recommendations(emotion):
    sp = get_spotify_client()
    if not sp:
        return ["Spotify 연결 실패 (Secrets 누락 또는 클라이언트 초기화 실패)"]

    def is_korean(txt):
        return isinstance(txt, str) and any('가' <= ch <= '힣' for ch in txt)

    KR_KEYWORDS = {
        "행복": ["케이팝 최신", "국내 신나는 노래", "여름 노래", "K-pop happy"],
        "슬픔": ["발라드 최신", "이별 노래", "감성 케이팝", "K-ballad"],
        "분노": ["운동 음악", "락", "파워 송", "K-rock"],
        "힘듦": ["위로 노래", "힐링 케이팝", "잔잔한 팝"],
        "놀람": ["파티 케이팝", "EDM 케이팝", "페스티벌 음악"],
    }

    query = random.choice(KR_KEYWORDS.get(emotion, ["케이팝 최신"])) + " year:2015-2025"
    last_exception = None 

    try:
        # 1️⃣ 트랙 직접 검색
        res = sp.search(q=query, type="track", limit=50, market="KR")
        tracks = (res.get("tracks") or {}).get("items") or []
        valid = []
        for t in tracks:
            track_id = t.get("id")
            name = t.get("name")
            artists = t.get("artists") or []
            artist = artists[0].get("name") if artists else "Unknown"
            if track_id and name and (is_korean(name) or is_korean(artist)):
                valid.append({"title": name, "artist": artist, "id": track_id})

        # 2️⃣ 만약 트랙 검색 결과가 10곡 미만이면, "플레이리스트" 검색으로 추가
        if len(valid) < 10:
            fallback = sp.search(q=query, type="playlist", limit=10, market="KR") # 쿼리 통일
            pls = (fallback.get("playlists") or {}).get("items") or []
            for pl in pls:
                pid = pl.get("id")
                if not pid: continue 
                
                try:
                    items = (sp.playlist_items(pid, limit=50, market="KR") or {}).get("items") or []
                except spotipy.exceptions.SpotifyException as se:
                    if se.http_status == 404:
                        continue 
                    else:
                        last_exception = se 
                        continue 
                
                for it in items:
                    tr = (it or {}).get("track") or {}
                    if not tr:
                        continue
                    track_id = tr.get("id")
                    name = tr.get("name")
                    artists = tr.get("artists") or []
                    artist = artists[0].get("name") if artists else "Unknown"
                    if track_id and name:
                        valid.append({"title": name, "artist": artist, "id": track_id})
                
                # ⭐️⭐️⭐️ 1곡만 나오는 오류 수정 ⭐️⭐️⭐️
                # 'if valid: break' (X) -> 'if len(valid) >= 10: break' (O)
                # 최소 10곡은 모아야 멈춘다
                if len(valid) >= 10:
                    break 
                # ⭐️⭐️⭐️ 수정 끝 ⭐️⭐️⭐️

        # 3️⃣ 그래도 10곡 미만이면, 최신 TOP 트랙으로 추가
        if len(valid) < 10:
            top = sp.search(q="Top Hits Korea", type="track", limit=50, market="KR")
            titems = (top.get("tracks") or {}).get("items") or []
            for t in titems:
                track_id = t.get("id")
                name = t.get("name")
                artists = t.get("artists") or []
                artist = artists[0].get("name") if artists else "Unknown"
                if track_id and name:
                    valid.append({"title": name, "artist": artist, "id": track_id})

        if not valid:
            return [{"title": "추천 없음", "artist": "Spotify API 문제", "id": None}]
        
        # ⭐️ 중복 제거 (트랙 검색과 플레이리스트 검색에서 겹칠 수 있음)
        unique_tracks = {t['id']: t for t in valid}.values()
        
        return random.sample(list(unique_tracks), k=min(3, len(unique_tracks)))

    except Exception as e:
        last_exception = e
        return [f"Spotify AI 검색 오류: {type(last_exception).__name__}: {last_exception}"]


# --- 7) TMDB 추천 (포스터 + 줄거리 포함) ---
def get_tmdb_recommendations(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key", "")
    if not key:
        return [{"text": "TMDB 연결에 실패했습니다. API 키를 확인해주세요.", "poster": None, "overview": ""}]

    GENRES = {
        "행복": "35|10749|10751|10402|16",
        "분노": "28|12|35|878",
        "슬픔": "35|10751|16|14",
        "힘듦": "35|10751|16|14",
        "놀람": "35|10751|16|14",
    }
    g = GENRES.get(emotion)
    if not g:
        return [{"text": f"[{emotion}]에 대한 장르 맵핑이 없습니다.", "poster": None, "overview": ""}]

    try:
        r = requests.get(
            f"{TMDB_BASE_URL}/discover/movie",
            params={
                "api_key": key,
                "language": "ko-KR",
                "sort_by": "popularity.desc",
                "with_genres": g,
                "page": 1,
                "vote_count.gte": 100,
            },
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])

        if not results:
            return [{"text": f"[{emotion}] 관련 영화를 찾지 못했습니다.", "poster": None, "overview": ""}]

        picks = results if len(results) <= 3 else random.sample(results, 3)
        out = []
        for m in picks:
            title = m.get("title", "제목없음")
            year = (m.get("release_date") or "")[:4] or "N/A"
            rating = m.get("vote_average", 0.0)
            poster = f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None
            overview = m.get("overview", "줄거리 정보가 없습니다.")
            if not overview: 
                overview = "줄거리 정보가 없습니다."
                
            out.append({
                "poster": poster,
                "title": title,
                "year": year,
                "rating": rating,
                "overview": overview 
            })
        return out
    except Exception as e:
        return [{"text": f"TMDb 오류: {type(e).__name__}: {e}", "poster": None, "overview": ""}]

# --- 8) 통합 추천 ---
def recommend(emotion):
    return {
        "음악": get_spotify_ai_recommendations(emotion),
        "영화": get_tmdb_recommendations(emotion),
    }

# --- 9) 상태/입력/실행 ---
with st.expander("⚙️ 시스템 상태 확인"):
    with st.spinner("모델 로드 중..."):
        model, tokenizer, device, postmap = load_kobert_model()
    st.write("✅ 모델 로드 완료" if model else "❌ 모델 로드 실패")

if "diary_text" not in st.session_state:
    st.session_state.diary_text = ""
if "final_emotion" not in st.session_state:
    st.session_state.final_emotion = None
if "confidence" not in st.session_state:
    st.session_state.confidence = 0.0

st.text_area("오늘의 일기를 작성해주세요:", key="diary_text", height=230)

def handle_analyze_click():
    txt = st.session_state.diary_text
    if not txt.strip():
        st.warning("일기를 입력해주세요.")
        return
    if model is None:
        st.error("AI 모델이 로드되지 않았습니다.")
        return
    with st.spinner("AI가 분석 중입니다..."):
        emo, sc = analyze_diary_kobert(txt, model, tokenizer, device, postmap)
        st.session_state.final_emotion = emo
        st.session_state.confidence = sc

st.button("🔍 내 하루 감정 분석하기", type="primary", on_click=handle_analyze_click)

# --- 10) ⭐️ 결과/추천 출력 (UI 레이아웃 최종 수정) ---
if st.session_state.final_emotion:
    emo = st.session_state.final_emotion
    sc = st.session_state.confidence

    st.subheader(f"오늘 하루의 핵심 감정은 '{emo}' 입니다.")
    st.progress(sc, text=f"감정 신뢰도: {sc:.2%}")

    st.divider()
    st.subheader(f"'{emo}' 감정을 위한 오늘의 Moodiary 추천")

    with st.spinner("추천을 불러오는 중..."):
        recs = recommend(emo)

    music_items = recs.get("음악", [])
    movie_items = recs.get("영화", [])

    # ⭐️⭐️⭐️ UI 정렬을 위한 새 로직 ⭐️⭐️⭐️
    # (항목 3개에 맞춰 3번 반복)
    for i in range(3):
        col_music, col_movie = st.columns(2)

        # --- 음악 컬럼 (⭐️ 재생 버튼으로 변경) ---
        with col_music:
            if i == 0: 
                st.markdown("#### 🎵 이런 음악도 들어보세요?")
            
            if i < len(music_items):
                it = music_items[i]
                if isinstance(it, dict):
                    track_id = it.get("id")
                    if track_id:
                        # ⭐️ Spotify 임베드 플레이어 사용 (높이 152px)
                        embed_url = f"https://open.spotify.com/embed/track/{track_id}?utm_source=generator&theme=0"
                        components.iframe(embed_url, height=152)
                    else:
                        st.write(f"- {it.get('title', '오류')}")
                else:
                    st.write(f"- {it}")
            
        # --- 영화 컬럼 (⭐️ 정렬 맞춤) ---
        with col_movie:
            if i == 0: 
                st.markdown("#### 🎬 이런 영화도 추천해요?")
                
            if i < len(movie_items):
                it = movie_items[i]
                if isinstance(it, dict):
                    poster = it.get("poster")
                    if poster:
                        st.image(poster, width=160)
                    
                    title = it.get("title", "제목없음")
                    year = it.get("year", "N/A")
                    rating = float(it.get("rating", 0.0))
                    
                    overview = it.get("overview", "") 
                    
                    line = f"##### **{title} ({year})**\n⭐ {rating:.1f}\n\n*{overview}*"
                    st.markdown(line)
                else:
                    st.write(f"- {it}")

        # ⭐️⭐️⭐️ "실선"을 컬럼 밖, 루프 안에 둬서 라인을 맞춤 ⭐️⭐️⭐️
        st.markdown("---")
