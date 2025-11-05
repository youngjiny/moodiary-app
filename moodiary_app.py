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

# --- 6) ⭐️ Spotify 추천 (로직 변경: 공식 차트 TOP 50) ---
def get_spotify_ai_recommendations(emotion):
    sp = get_spotify_client()
    if not sp:
        return ["Spotify 연결 실패 (Secrets 누락 또는 클라이언트 초기화 실패)"]

    # ⭐️ "센스 있는" 추천을 위해, 감정 키워드 검색 대신 "공식 차트"를 사용합니다.
    # (감정별로 다른 차트를 매핑할 수도 있습니다)
    CHART_PLAYLISTS = {
        "행복": "37i9dQZEVXbNxXF4UeQlye", # Top 50 - South Korea
        "슬픔": "37i9dQZEVXbNxXF4UeQlye", # Top 50 - South Korea
        "분노": "37i9dQZEVXbJxxNsEk86S4", # K-Pop ON!
        "힘듦": "37i9dQZEVXbNxXF4UeQlye", # Top 50 - South Korea
        "놀람": "37i9dQZEVXbJxxNsEk86S4", # K-Pop ON!
    }
    
    # 해당 감정의 차트를 가져오되, 없으면 한국 Top 50을 기본값으로
    playlist_id = CHART_PLAYLISTS.get(emotion, "37i9dQZEVXbNxXF4UeQlye")

    try:
        # 1️⃣ 플레이리스트 트랙 가져오기 (50곡)
        tracks_results = sp.playlist_items(playlist_id, limit=50, market="KR")
        if not tracks_results or 'items' not in tracks_results:
             return ["Spotify 차트를 불러오지 못했습니다."]

        valid = []
        for item in tracks_results['items']:
            track = item.get('track')
            if track and track.get('artists') and track.get('name'):
                artists = track.get("artists") or []
                artist = artists[0].get("name") if artists else "Unknown"
                album = track.get("album") or {}
                images = album.get("images") or []
                cover = images[0]["url"] if images else None
                if track['artists'] and track['artists'][0].get('name'):
                    valid.append({"title": track['name'], "artist": artist, "cover": cover})
        
        # 2️⃣ 유효한 트랙이 없으면 (거의 불가능하지만)
        if not valid:
            return ["추천할 만한 노래를 찾지 못했습니다. (차트 로딩 문제)"]
        
        # 3️⃣ 50곡 중 3곡을 랜덤으로 뽑아 반환
        return random.sample(valid, k=min(3, len(valid)))

    except Exception as e:
        return [f"Spotify 추천 오류: {type(e).__name__}: {e}"]

# --- 7) ⭐️ TMDB 추천 (줄거리 추가) ---
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
                "api_key": key, "language": "ko-KR", "sort_by": "popularity.desc",
                "with_genres": g, "page": 1, "vote_count.gte": 100,
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
            
            # ⭐️⭐️⭐️ 2. 영화 줄거리 추가 (요청사항 반영) ⭐️⭐️⭐️
            overview = m.get("overview", "줄거리 정보가 없습니다.")
            if not overview: 
                overview = "줄거리 정보가 없습니다."
                
            out.append({
                "text": f"{title} ({year}) (평점: {rating:.1f})", # (이전 text는 이제 사용 안 함)
                "poster": poster,
                "title": title,
                "year": year,
                "rating": rating,
                "overview": overview # ⭐️ 줄거리 정보 추가
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

# --- 10) ⭐️ 결과/추천 출력 (UI 수정) ---
if st.session_state.final_emotion:
    emo = st.session_state.final_emotion
    sc = st.session_state.confidence

    st.subheader(f"오늘 하루의 핵심 감정은 '{emo}' 입니다.")
    st.progress(sc, text=f"감정 신뢰도: {sc:.2%}")

    st.divider()
    st.subheader(f"'{emo}' 감정을 위한 오늘의 Moodiary 추천")

    with st.spinner("추천을 불러오는 중..."):
        recs = recommend(emo)

    col_music, col_movie = st.columns(2)

    # ⭐️ 음악 (표지 크기 + 글씨 크기 수정)
    with col_music:
        st.markdown("#### 🎵 이런 음악도 들어보세요?")
        items = recs.get("음악", [])
        if items:
            for it in items:
                if isinstance(it, dict):
                    img_c, txt_c = st.columns([1, 4])
                    cover = it.get("cover")
                    if cover:
                        # ⭐️⭐️⭐️ 1. 음악 표지 크기 키우기 (80 -> 160) ⭐️⭐️⭐️
                        img_c.image(cover, width=160) 
                    else:
                        img_c.empty()
                    title = it.get("title", "제목없음")
                    artist = it.get("artist", "Unknown")
                    # ⭐️⭐️⭐️ 3. 글씨 크기 키우기 (H5 마크다운) ⭐️⭐️⭐️
                    txt_c.markdown(f"##### **{title}**\n{artist}")
                    st.markdown("---")
                else:
                    st.write(f"- {it}")
        else:
            st.write("- 추천을 찾지 못했어요.")

    # ⭐️ 영화 (줄거리 추가 + 글씨 크기 수정)
    with col_movie:
        st.markdown("#### 🎬 이런 영화도 추천해요?")
        items = recs.get("영화", [])
        if items:
            for it in items:
                if isinstance(it, dict):
                    img_c, txt_c = st.columns([1, 4])
                    poster = it.get("poster")
                    if poster:
                        img_c.image(poster, width=160) # (크기는 이미 160)
                    else:
                        img_c.empty()
                    
                    # ⭐️⭐️⭐️ 2 & 3. 줄거리 길게 + 글씨 크게 ⭐️⭐️⭐️
                    title = it.get("title", "제목없음")
                    year = it.get("year", "N/A")
                    rating = float(it.get("rating", 0.0))
                    overview = it.get("overview", "")
                    
                    # 줄거리 150자로 자르기 (요청사항 반영)
                    if len(overview) > 150:
                        overview = overview[:150] + "..."
                    
                    # 텍스트 조합 (H5 마크다운 + 줄거리)
                    line = f"##### **{title} ({year})**\n⭐ {rating:.1f}\n\n*{overview}*"
                    
                    txt_c.markdown(line)
                    st.markdown("---")
                else:
                    st.write(f"- {it}")
        else:
            st.write("- 추천을 찾지 못했어요.")
