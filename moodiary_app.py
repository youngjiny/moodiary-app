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

# 폰트 설정
try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    st.warning("Malgun Gothic 폰트를 찾을 수 없어 그래프의 한글이 깨질 수 있습니다.")

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
            st.warning("모델 config에서 post_processing_map을 찾지 못해 하드코딩합니다.")
            post_processing_map = {
                '기쁨': '행복', '슬픔': '슬픔', '상처': '슬픔',
                '불안': '힘듦', '당황': '놀람', '분노': '분노'
            }
        return model, tokenizer, device, post_processing_map
    except Exception as e:
        st.error(f"🚨 AI 모델 로드 실패: {e}")
        st.error("Hugging Face 저장소 또는 monologg/kobert 모델을 확인하세요.")
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
        st.error(f"Spotify 로그인 오류: {e}")
        return None

# --- 6. 추천 함수 (TMDB 장르 맵 "치유형"으로 수정) ---
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
        random_playlist = random.choice(playlists)
        playlist_id = random_playlist['id']
        tracks_results = sp_client.playlist_items(playlist_id, limit=50)
        if not tracks_results or 'items' not in tracks_results:
             return [f"'{random_playlist.get('name')}' 플레이리스트를 읽어오지 못했습니다."]
        tracks = [item['track'] for item in tracks_results['items'] if item and item['track']]
        if not tracks: return ["선택된 플레이리스트에 노래가 없어요."]
        random_tracks = random.sample(tracks, min(3, len(tracks)))
        return [f"{track['name']} - {track['artists'][0]['name']}" for track in random_tracks]
    except Exception as e: 
        return [f"Spotify AI 검색 오류: {e}"]

def get_tmdb_recommendations(emotion):
    tmdb_creds = st.secrets.get("tmdb", {})
    current_tmdb_key = tmdb_creds.get("api_key", "")
    
    if not current_tmdb_key:
        return ["TMDB API 키가 설정되지 않았습니다. (Secrets[tmdb][api_key] 읽기 실패)"]
        
    # ⭐️⭐️⭐️ 중요: 고객님 의견 반영, "치유" 및 "기분전환"용 장르로 수정 ⭐️⭐️⭐️
    TMDB_GENRE_MAP = {
        # 행복 (극대화): 코미디, 로맨스, 가족, 음악, 애니메이션 (기존 유지, 좋음)
        "행복": "35|10749|10751|10402|16",
        
        # 분노 (스트레스 해소): 액션, 모험, 코미디, SF
        "분노": "28|12|35|878",
        
        # 슬픔, 힘듦, 놀람 (위로/안정): 코미디, 가족, 애니메이션, 판타지 (따뜻한 장르)
        "슬픔": "35|10751|16|14",
        "힘듦": "35|10751|16|14",
        "놀람": "35|10751|16|14"
    }
    genre_ids_string = TMDB_GENRE_MAP.get(emotion)
    if not genre_ids_string:
        return [f"[{emotion}]에 대한 장르 맵핑이 없습니다."]
    
    endpoint = f"https://api.themoviedb.org/3/discover/movie"
    params = {
        "api_key": current_tmdb_key,
        "language": "ko-KR", "sort_by": "popularity.desc",
        "with_genres": genre_ids_string, "page": 1, "vote_count.gte": 100
    }
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        data = response.json()
        if data.get('results'):
            top_movies = data['results'][:3]
            recommendations = []
            for movie in top_movies:
                title = movie['title']
                date = movie['release_date'][:4] if movie.get('release_date') else "N/A"
                rating = movie['vote_average']
                recommendations.append(f"{title} ({date}) (평점: {rating:.1f})")
            return recommendations
        else:
            return [f"[{emotion} 장르]의 인기 영화를 찾지 못했습니다."]
    except requests.exceptions.RequestException as e:
        return [f"TMDb API 호출 실패: {e}"]

def recommend(final_emotion, method):
    music_recs = get_spotify_ai_recommendations(final_emotion)
    movie_recs = get_tmdb_recommendations(final_emotion)
    book_recommendations = {
        "행복": ["기분을 관리하면 인생이 관리된다"], "슬픔": ["아몬드"], 
        "분노": ["분노의 심리학"], "힘듦": ["죽고 싶지만 떡볶이는 먹고 싶어"], 
        "놀람": ["데미안"],
    }
    book_recs = book_recommendations.get(final_emotion, [])
    return {'책': book_recs, '음악': music_recs, '영화': movie_recs}

# --- 7. Streamlit UI 구성 (변경 없음) ---
st.set_page_config(layout="wide")
st.title("Moodiary 📝 감정 일기 (KoBERT Ver.)")

with st.expander("⚙️ 시스템 상태 확인"):
    with st.spinner("Hugging Face Hub에서 AI 모델을 불러오는 중입니다..."):
        model, tokenizer, device, post_processing_map = load_kobert_model()
    
    if model and tokenizer and device and post_processing_map:
        st.success("✅ AI 감정 분석 모델(KoBERT)이 성공적으로 로드되었습니다.")
    else:
        st.error("❗️ AI 모델 로드를 실패했습니다.")

    if st.secrets.get("spotify", {}).get("client_id"): st.success("✅ Spotify 인증 정보가 확인되었습니다.")
    else: st.error("❗️ Spotify 인증 정보('[spotify]' 섹션)를 찾을 수 없습니다.")
        
    if st.secrets.get("tmdb", {}).get("api_key"):
        st.success("✅ TMDB API 키가 Secrets에 존재합니다. ([tmdb][api_key])")
    else:
        st.error("❗️ TMDB API 키('tmdb.api_key')를 Secrets에서 찾을 수 없습니다.")

st.divider()

if 'diary_text' not in st.session_state: st.session_state.diary_text = ""
if 'final_emotion' not in st.session_state: st.session_state.final_emotion = None
if 'confidence_score' not in st.session_state: st.session_state.confidence_score = 0.0
if 'rec_method' not in st.session_state: st.session_state.rec_method = '내 플레이리스트'

col1, col2 = st.columns([3, 1])
with col1:
    st.text_area("오늘의 일기를 작성해주세요:", key='diary_text', height=250)
with col2:
    st.write(" "); st.write(" ")
    st.radio("음악 추천 방식 선택", ('내 플레이리스트', 'AI 자동 추천'), key='rec_method', horizontal=True)
    
    def handle_random_click():
        sample_diaries = [
            "남자친구랑 재밌는 데이트를 했어. 날씨도 좋아서 기분이 좋다. 맛있는 것도 먹고 선물도 받았다. 정말 행복한 하루다.",
            "오늘 팀 프로젝트 발표가 있었는데, 준비한 만큼 잘 안돼서 너무 속상하다. 팀원들에게 미안하고 내 자신이 원망스럽다.",
            "직장 상사가 또 말도 안 되는 걸로 트집을 잡았다. 정말 화가 머리 끝까지 났지만 꾹 참았다. 퇴근하고 매운 떡볶이를 먹어야겠다.",
            "내일 중요한 면접이 있어서 너무 불안하고 떨린다. 잠이 올 것 같지 않다. 잘 할 수 있겠지?",
            "길을 가다가 갑자기 친구를 만났다. 10년 만에 보는 거라 너무 놀랐고 반가웠다."
        ]
        st.session_state.diary_text = random.choice(sample_diaries)
        st.session_state.final_emotion = None
        
    st.button("🔄 랜덤 일기 생성", on_click=handle_random_click)
    
    def handle_analyze_click():
        diary_content = st.session_state.diary_text
        if not diary_content.strip(): 
            st.warning("일기를 입력해주세요!")
            st.session_state.final_emotion = None
        elif model is None: 
            st.error("AI 모델이 로드되지 않았습니다.")
            st.session_state.final_emotion = None
        else:
            with st.spinner('AI가 일기를 분석하고 있습니다... (KoBERT)'):
                emotion, score = analyze_diary_kobert(
                    diary_content, model, tokenizer, device, post_processing_map
                )
                st.session_state.final_emotion = emotion
                st.session_state.confidence_score = score
    st.button("🔍 내 하루 감정 분석하기", type="primary", on_click=handle_analyze_click)

if st.session_state.final_emotion:
    final_emotion = st.session_state.final_emotion
    score = st.session_state.confidence_score
    st.subheader(f"오늘 하루의 핵심 감정은 '{final_emotion}' 입니다.")
    st.progress(score, text=f"감정 신뢰도: {score:.2%}")
    st.success(f"오늘 하루를 종합해 보면, **'{final_emotion}'**의 감정이 가장 컸네요!")
    st.divider()
    st.subheader(f"'{final_emotion}' 감정을 위한 오늘의 Moodiary 추천")
    with st.spinner(f"'{final_emotion}'에 맞는 추천 항목을 찾고 있습니다..."):
        recs = recommend(final_emotion, st.session_state.rec_method)
    rec_col1, rec_col2, rec_col3 = st.columns(3)
    with rec_col1:
        st.write("📚 **이런 책은 어때요?**")
        if recs['책']:
            for item in recs['책']: st.write(f"- {item}")
        else: st.write("- 추천을 찾지 못했어요.")
    with rec_col2:
        st.write("🎵 **이런 음악도 들어보세요?**")
        if recs['음악']:
            for item in recs['음악']: st.write(f"- {item}")
        else: st.write("- 추천을 찾지 못했어요.")
    with rec_col3:
        st.write("🎬 **이런 영화도 추천해요?**")
        if recs['영화']:
            for item in recs['영화']: st.write(f"- {item}")
        else: st.write("- 추천을 찾지 못했어요.")
