# --- 1. 필수 라이브러리 임포트 ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import random
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

# --- 2. 기본 설정 및 경로 ---
KOBERT_BASE_MODEL = "monologg/kobert"
KOBERT_SAVED_REPO = "Young-jin/kobert-moodiary-app" 
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# 폰트 설정 (에러가 나도 무시하고 계속 진행)
try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    pass 

FINAL_EMOTIONS = ["행복", "슬픔", "분노", "힘듦", "놀람"]

# --- 3. KoBERT 모델 로드 (num_labels=6 강제) ---
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
            label2id={label: id for id, label in CORRECT_ID_TO_LABEL.items()}
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
        st.error(f"🚨 AI 모델을 불러오는 데 실패했습니다. 잠시 후 다시 시도해주세요.")
        return None, None, None, None

# --- 4. 핵심 분석 함수 (변경 없음) ---
def analyze_diary_kobert(text, model, tokenizer, device, post_processing_map):
    if not text:
        return None, 0.0
    encodings = tokenizer(
        text, truncation=True, padding=True, max_length=128, return_tensors="pt"
    )
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    predicted_class_id = torch.argmax(probabilities, dim=1).cpu().numpy()[0]
    score = probabilities[0, predicted_class_id].item()
    id_to_label = model.config.id2label
    original_label = id_to_label[predicted_class_id]
    final_emotion = post_processing_map.get(original_label, original_label)
    return final_emotion, score

# --- 5. API 연결 함수 (Spotify - 변경 없음) ---
@st.cache_resource
def get_spotify_client():
    spotify_creds = st.secrets.get("spotify", {})
    client_id = spotify_creds.get("client_id")
    client_secret = spotify_creds.get("client_secret")
    if not client_id or not client_secret:
        return None
    try:
        client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
        sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
        return sp
    except Exception as e:
        return None 

# --- 6. 추천 함수 (Spotify 오류 수정, TMDB 랜덤 추천) ---
def get_spotify_ai_recommendations(emotion):
    sp_client = get_spotify_client()
    if not sp_client: return ["Spotify 연결 실패 (클라이언트 초기화 실패)"]
    emotion_keywords = { 
        "행복": ["K-Pop Happy", "신나는"], 
        "슬픔": ["K-Pop Ballad", "슬픈", "이별"], 
        "분노": ["K-Rock", "화날 때", "스트레스"], 
        "힘듦": ["K-Pop healing", "위로", "지칠 때"], 
        "놀람": ["K-Pop Party", "신나는"], 
    }
    query = random.choice(emotion_keywords.get(emotion, ["K-Pop"]))
    try:
        results = sp_client.search(q=query, type='playlist', limit=20, market="KR")
        if not results: return [f"'{query}'에 대한 검색 결과가 없습니다."]
        playlists = results.get('playlists', {}).get('items')
        if not playlists: return [f"'{query}' 관련 플레이리스트를 찾지 못했어요."]
        
        for _ in range(3): 
            random_playlist = random.choice(playlists)
            playlist_id = random_playlist['id']
            tracks_results = sp_client.playlist_items(playlist_id, limit=50)
            
            if not tracks_results or 'items' not in tracks_results:
                continue 

            tracks = []
            for item in tracks_results['items']:
                 if item and item.get('track') and item['track'].get('artists') and item['track'].get('name'):
                     if item['track']['artists'] and item['track']['artists'][0].get('name'):
                         tracks.append(item['track'])
            
            if tracks: 
                random_tracks = random.sample(tracks, min(3, len(tracks)))
                return [f"{track['name']} - {track['artists'][0]['name']}" for track in random_tracks]

        return ["추천할 만한 노래를 찾지 못했습니다. (플레이리스트 문제)"]

    except Exception as e: 
        return [f"Spotify AI 검색 오류: {e}"]

def get_tmdb_recommendations(emotion):
    tmdb_creds = st.secrets.get("tmdb", {})
    current_tmdb_key = tmdb_creds.get("api_key", "")
    if not current_tmdb_key:
        return [{"text": "TMDB 연결에 실패했습니다. API 키를 확인해주세요.", "poster": None}]

    TMDB_GENRE_MAP = {
        "행복": "35|10749|10751|10402|16",
        "분노": "28|12|35|878",
        "슬픔": "35|10751|16|14",
        "힘듦": "35|10751|16|14",
        "놀람": "35|10751|16|14"
    }
    genre_ids_string = TMDB_GENRE_MAP.get(emotion)
    if not genre_ids_string:
        return [{"text": f"[{emotion}]에 대한 장르 맵핑이 없습니다.", "poster": None}]

    endpoint = "https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": current_tmdb_key,
        "language": "ko-KR",
        "sort_by": "popularity.desc",
        "with_genres": genre_ids_string,
        "page": 1,
        "vote_count.gte": 100
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()

        if not data.get('results'):
            return [{"text": f"[{emotion} 장르]의 인기 영화를 찾지 못했습니다.", "poster": None}]

        popular_movies = data['results']
        selected_movies = popular_movies if len(popular_movies) <= 3 else random.sample(popular_movies, 3)

        recs = []
        for m in selected_movies:
            title = m.get('title', '제목없음')
            year = (m.get('release_date') or '')[:4] or "N/A"
            rating = m.get('vote_average', 0.0)
            poster = f"https://image.tmdb.org/t/p/w500{m['poster_path']}" if m.get('poster_path') else None
            recs.append({
                "text": f"{title} ({year}) (평점: {rating:.1f})",
                "poster": poster
            })
        return recs

    except requests.exceptions.RequestException as e:
        return [{"text": f"TMDb API 호출 실패: {e}", "poster": None}]

def recommend(final_emotion):
    music_recs = get_spotify_ai_recommendations(final_emotion)
    movie_recs = get_tmdb_recommendations(final_emotion)
    return {'음악': music_recs, '영화': movie_recs}

# --- 7. Streamlit UI 구성 (최종 클린 버전) ---
st.set_page_config(layout="wide")
st.title("MOODIARY 💖")

model, tokenizer, device, post_processing_map = load_kobert_model()

if 'diary_text' not in st.session_state: st.session_state.diary_text = ""
if 'final_emotion' not in st.session_state: st.session_state.final_emotion = None

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("### 오늘의 일기를 작성해주세요:")
    st.text_area(
        "오늘의 일기를 작성해주세요:",
        key='diary_text', 
        height=250, 
        label_visibility="hidden"
    )
    
with col2:
    st.write(" "); st.write(" ")
    
    def handle_analyze_click():
        diary_content = st.session_state.diary_text
        if not diary_content.strip(): 
            st.warning("일기를 입력해주세요!")
            st.session_state.final_emotion = None
        elif model is None: 
            st.error("AI 모델이 로드되지 않았습니다. 잠시 후 새로고침 해주세요.")
            st.session_state.final_emotion = None
        else:
            with st.spinner('AI가 일기를 분석하고 있습니다... (KoBERT)'):
                emotion, score = analyze_diary_kobert(
                    diary_content, model, tokenizer, device, post_processing_map
                )
                st.session_state.final_emotion = emotion
                
    st.button("🔍 내 하루 감정 분석하기", type="primary", on_click=handle_analyze_click)

if st.session_state.final_emotion:
    final_emotion = st.session_state.final_emotion
    st.subheader(f"오늘 하루의 핵심 감정은 '{final_emotion}' 입니다.")
    
    st.divider()
    st.subheader(f"'{final_emotion}' 감정을 위한 오늘의 Moodiary 추천")
    with st.spinner(f"'{final_emotion}'에 맞는 추천 항목을 찾고 있습니다..."):
        recs = recommend(final_emotion)
        
    rec_col1, rec_col2 = st.columns(2)
    
    with rec_col1:
        st.markdown("#### 🎵 이런 음악도 들어보세요?")
        if recs['음악']:
            for item in recs['음악']: st.write(f"- {item}")
        else: st.write("- 추천을 찾지 못했어요.")
        
    # ⭐️⭐️⭐️ 여기가 수정된 부분입니다 ⭐️⭐️⭐️
    # (들여쓰기를 `with rec_col1`과 동일하게 맞췄습니다)
    with rec_col2:
        st.markdown("#### 🎬 이런 영화도 추천해요?")
        if recs['영화']:
            for item in recs['영화']:
                if isinstance(item, dict):
                    # 포스터가 있으면 이미지를 먼저 보여줍니다.
                    if item.get("poster"):
                        st.image(item["poster"], width=160)
                    # 텍스트를 나중에 보여줍니다.
                    st.write(f"- {item.get('text','')}")
                else:
                    # (혹시라도 딕셔너리가 아닌 텍스트가 반환될 경우 대비)
                    st.write(f"- {item}")
        else:
            st.write("- 추천을 찾지 못했어요.")
