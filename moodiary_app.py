# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# (선택) Spotify SDK
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
except Exception:
    spotipy = None
    SpotifyClientCredentials = None

# --- 2) 기본 설정 ---
KOBERT_BASE_MODEL = "monologg/kobert"                # 토크나이저/베이스
KOBERT_SAVED_REPO = "Young-jin/kobert-moodiary-app"  # 학습 가중치(HF)
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

        # 라벨 후처리 매핑 (모델 config에 없으면 하드코딩)
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
        return spotipy.Spotify(client_credentials_manager=manager)
    except Exception:
        return None

# --- 6) Spotify 추천 (앨범 커버 포함) ---
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

    try:
        # 1️⃣ 트랙 직접 검색 (최신 & 한국어 필터)
        res = sp.search(q=query, type="track", limit=50, market="KR")
        tracks = (res.get("tracks") or {}).get("items") or []
        valid = []
        for t in tracks:
            name = t.get("name")
            artists = t.get("artists") or []
            artist = artists[0].get("name") if artists else "Unknown"
            album = t.get("album") or {}
            images = album.get("images") or []
            cover = images[0]["url"] if images else None
            year = (album.get("release_date") or "2005")[:4]

            # 한국어 포함 & 2015년 이후 곡만
            if int(year) >= 2015 and (is_korean(name) or is_korean(artist)):
                valid.append({"title": name, "artist": artist, "cover": cover})

        # 2️⃣ 만약 없으면 그냥 최신 케이팝 플레이리스트에서 가져오기
        if not valid:
            fallback = sp.search(q="K-pop Hits Korea 2020-2025", type="playlist", limit=10, market="KR")
            pls = (fallback.get("playlists") or {}).get("items") or []
            for pl in pls:
                pid = pl.get("id")
                items = (sp.playlist_items(pid, limit=50, market="KR") or {}).get("items") or []
                for it in items:
                    tr = (it or {}).get("track") or {}
                    if not tr:
                        continue
                    name = tr.get("name")
                    artists = tr.get("artists") or []
                    artist = artists[0].get("name") if artists else "Unknown"
                    album = tr.get("album") or {}
                    images = album.get("images") or []
                    cover = images[0]["url"] if images else None
                    if name:
                        valid.append({"title": name, "artist": artist, "cover": cover})
                if valid:
                    break

        # 3️⃣ 그래도 없으면 전세계 최신 TOP 트랙 fallback (절대 비어있지 않음)
        if not valid:
            top = sp.search(q="top hits 2024", type="track", limit=50, market="KR")
            titems = (top.get("tracks") or {}).get("items") or []
            for t in titems:
                name = t.get("name")
                artists = t.get("artists") or []
                artist = artists[0].get("name") if artists else "Unknown"
                album = t.get("album") or {}
                images = album.get("images") or []
                cover = images[0]["url"] if images else None
                if name:
                    valid.append({"title": name, "artist": artist, "cover": cover})

        # 항상 최소 1곡 이상 보장
        if not valid:
            return [{"title": "그날들", "artist": "이문세", "cover": None}]
        return random.sample(valid, k=min(3, len(valid)))

    except Exception as e:
        return [f"Spotify AI 검색 오류: {type(e).__name__}: {e}"]



# --- 7) TMDB 추천 (포스터 포함) ---
def get_tmdb_recommendations(emotion):
    key = st.secrets.get("tmdb", {}).get("api_key", "")
    if not key:
        return ["TMDB 연결에 실패했습니다. API 키를 확인해주세요."]

    GENRES = {
        "행복": "35|10749|10751|10402|16",
        "분노": "28|12|35|878",
        "슬픔": "35|10751|16|14",
        "힘듦": "35|10751|16|14",
        "놀람": "35|10751|16|14",
    }
    g = GENRES.get(emotion)
    if not g:
        return [f"[{emotion}]에 대한 장르 맵핑이 없습니다."]

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
            return [f"[{emotion}] 관련 영화를 찾지 못했습니다."]

        picks = results if len(results) <= 3 else random.sample(results, 3)
        out = []
        for m in picks:
            title = m.get("title", "제목없음")
            year = (m.get("release_date") or "")[:4] or "N/A"
            rating = m.get("vote_average", 0.0)
            poster = f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get("poster_path") else None
            out.append({
                "text": f"{title} ({year}) (평점: {rating:.1f})",
                "poster": poster,
                "title": title,
                "year": year,
                "rating": rating
            })
        return out
    except Exception as e:
        return [f"TMDb 오류: {type(e).__name__}: {e}"]

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

# --- 10) 결과/추천 출력 (정렬 + 이미지) ---
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

    # 음악 (앨범 커버 + 텍스트)
    with col_music:
        st.markdown("#### 🎵 이런 음악도 들어보세요?")
        items = recs.get("음악", [])
        if items:
            for it in items:
                if isinstance(it, dict):
                    img_c, txt_c = st.columns([1, 4])
                    cover = it.get("cover")
                    if cover:
                        img_c.image(cover, width=80)
                    else:
                        img_c.empty()
                    title = it.get("title", "제목없음")
                    artist = it.get("artist", "Unknown")
                    txt_c.markdown(f"**{title}**  \n{artist}")
                    st.markdown("---")
                else:
                    st.write(f"- {it}")
        else:
            st.write("- 추천을 찾지 못했어요.")

    # 영화 (포스터 + 텍스트)
    with col_movie:
        st.markdown("#### 🎬 이런 영화도 추천해요?")
        items = recs.get("영화", [])
        if items:
            for it in items:
                if isinstance(it, dict):
                    img_c, txt_c = st.columns([1, 4])
                    poster = it.get("poster")
                    if poster:
                        img_c.image(poster, width=80)
                    else:
                        img_c.empty()
                    line = it.get("text")
                    if not line:
                        title = it.get("title", "제목없음")
                        year = it.get("year", "N/A")
                        rating = float(it.get("rating", 0.0))
                        line = f"**{title} ({year})**  \n⭐ {rating:.1f}"
                    txt_c.markdown(line)
                    st.markdown("---")
                else:
                    st.write(f"- {it}")
        else:
            st.write("- 추천을 찾지 못했어요.")
