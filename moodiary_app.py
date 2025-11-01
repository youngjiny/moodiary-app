# --- 1. 필수 라이브러리 임포트 ---
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials
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

TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# 폰트 설정
try:
    font_path = "c:/Windows/Fonts/malgun.ttf"
    font_name = font_manager.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
except FileNotFoundError:
    st.warning("Malgun Gothic 폰트를 찾을 수 없어 그래프의 한글이 깨질 수 있습니다.")

FINAL_EMOTIONS = ["행복", "슬픔", "분노", "힘듦", "놀람"]


# --- 3. KoBERT 모델 로드 (trust_remote_code=True 추가) ---
@st.cache_resource
def load_kobert_model():
    """
    원본 KoBERT 아키텍처를 로드한 뒤,
    Hugging Face Hub에 저장된 고객님의 가중치(weights)를 덮어씌웁니다.
    """
    try:
        # 1. ⭐️ 원본(monologg/kobert)에서 올바른 Config와 Tokenizer를 불러옵니다.
        #    trust_remote_code=True 플래그를 추가합니다.
        config = AutoConfig.from_pretrained(
            KOBERT_BASE_MODEL, 
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            KOBERT_BASE_MODEL, 
            trust_remote_code=True
        )
        
        # 2. ⭐️ 고객님의 저장소(Young-jin/...)에서 모델을 로드하되,
        #    원본 config와 trust_remote_code=True 플래그를 사용합니다.
        model = AutoModelForSequenceClassification.from_pretrained(
            KOBERT_SAVED_REPO, 
            config=config, 
            trust_remote_code=True
        )
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # 3. 모델 객체에서 후처리 맵핑을 불러옵니다.
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

# --- 5. API 연결 함수 (변경 없음) ---
@st.cache_resource
def get_gsheets_connection():
    try:
        creds_dict = st.secrets.get("connections", {}).get("gsheets")
        if creds_dict:
            scope = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
            credentials = Credentials.from_service_account_info(creds_dict, scopes=scope)
            client = gspread.authorize(credentials)
            return client
        return None
    except Exception:
        return None

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
    except Exception:
        return None

@st.cache_data(ttl=60)
def fetch_all_data_from_gsheets(_client):
    try:
        spreadsheet = _client.open("diary_app_feedback")
        worksheet = spreadsheet.worksheet("Sheet1")
        df = pd.DataFrame(worksheet.get_all_records())
        return df
    except Exception as e:
        st.error(f"Google Sheets 데이터 로딩 오류: {e}")
        return pd.DataFrame()

# --- 6. 추천 함수 (변경 없음) ---
def get_spotify_playlist_recommendations(emotion):
    sp_client = get_spotify_client()
    if not sp_client: return ["Spotify 연결 실패"]
    try:
        playlist_ids = { 
            "행복": "1kaEr7seXIYcPflw2M60eA", "슬픔": "3tAeVAtMWHzaGOXMGoRhTb", 
            "분노": "22O1tfJ7fSjIo2FdxtJU1", "힘듦": "68HSylU5xKtDVYiago9RDw", 
            "놀람": "3sHzse5FGtcafd8dY0mO8h", 
        }
        playlist_id = playlist_ids.get(emotion)
        if not playlist_id: return ["추천할 플레이리스트가 없어요."]
        results = sp_client.playlist_items(playlist_id, limit=50)
        tracks = [item['track'] for item in results['items'] if item and item['track']]
        if not tracks: return ["플레이리스트에 노래가 없어요."]
        random_tracks = random.sample(tracks, min(3, len(tracks)))
        return [f"{track['name']} - {track['artists'][0]['name']}" for track in random_tracks]
    except Exception as e: return [f"Spotify 추천 오류: {e}"]

def get_spotify_ai_recommendations(emotion):
    sp_client = get_spotify_client()
    if not sp_client: return ["Spotify 연결 실패"]
    try:
        emotion_keywords = { 
            "행복": ["행복", "신나는"], "슬픔": ["슬픈", "이별"], 
            "분노": ["화날 때", "스트레스"], "힘듦": ["위로", "지칠 때"], 
            "놀람": ["파티", "신나는"], 
        }
        query = emotion_keywords.get(emotion)
        if not query: return ["AI가 추천할 키워드를 찾지 못했어요."]
        results = sp_client.search(q=random.choice(query), type='playlist', limit=20, market="KR")
        if not results: return [f"'{query}'에 대한 검색 결과가 없습니다."]
        playlists = results.get('playlists', {}).get('items')
        if not playlists: return [f"'{query}' 관련 플레이리스트를 찾지 못했어요."]
        random_playlist = random.choice(playlists)
        playlist_id = random_playlist['id']
        results = sp_client.playlist_items(playlist_id, limit=50)
        tracks = [item['track'] for item in results['items'] if item and item['track']]
        if not tracks: return ["선택된 플레이리스트에 노래가 없어요."]
        random_tracks = random.sample(tracks, min(3, len(tracks)))
        return [f"{track['name']} - {track['artists'][0]['name']}" for track in random_tracks]
    except Exception as e: return [f"Spotify AI 추천 오류: {e}"]

@st.cache_data(ttl=86400)
def get_tmdb_recommendations(emotion):
    if not TMDB_API_KEY:
        return ["TMDB API 키가 설정되지 않았습니다."]
    TMDB_GENRE_MAP = {
        "행복": "35,10749,10751,10402,16", "슬픔": "18,10749,36,10402",
        "분노": "28,53,80,12,10752", "힘듦": "12,14,16",
        "놀람": "9648,53,27,878,80"
    }
    genre_ids_string = TMDB_GENRE_MAP.get(emotion)
    if not genre_ids_string:
        return [f"[{emotion}]에 대한 장르 맵핑이 없습니다."]
    endpoint = f"{TMDB_BASE_URL}/discover/movie"
    params = {
        "api_key": TMDB_API_KEY, "language": "ko-KR", "sort_by": "popularity.desc",
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
    if method == 'AI 자동 추천':
        music_recs = get_spotify_ai_recommendations(final_emotion)
    else:
        music_recs = get_spotify_playlist_recommendations(final_emotion)
    movie_recs = get_tmdb_recommendations(final_emotion)
    book_recommendations = {
        "행복": ["기분을 관리하면 인생이 관리된다"], "슬픔": ["아몬드"], 
        "분노": ["분노의 심리학"], "힘듦": ["죽고 싶지만 떡볶이는 먹고 싶어"], 
        "놀람": ["데미안"],
    }
    book_recs = book_recommendations.get(final_emotion, [])
    return {'책': book_recs, '음악': music_recs, '영화': movie_recs}

# --- 7. 피드백 저장 함수 (변경 없음) ---
def save_feedback_to_gsheets(client, diary_text, corrected_emotion):
    try:
        spreadsheet = client.open("diary_app_feedback")
        worksheet = spreadsheet.worksheet("Sheet1")
        worksheet.append_rows([[diary_text, corrected_emotion]], value_input_option='USER_ENTERED')
        st.success("소중한 피드백이 Google Sheets에 안전하게 저장되었습니다!")
        st.cache_data.clear()
    except Exception as e: 
        st.error(f"피드백 저장 중 오류 발생: {e}")

# --- 8. Streamlit UI 구성 (변경 없음) ---
st.set_page_config(layout="wide")
st.title("Moodiary 📝 감정 일기 (KoBERT Ver.)")

with st.expander("⚙️ 시스템 상태 확인"):
    with st.spinner("Hugging Face Hub에서 AI 모델을 불러오는 중입니다..."):
        model, tokenizer, device, post_processing_map = load_kobert_model()
    
    if model and tokenizer and device and post_processing_map:
        st.success("✅ AI 감정 분석 모델(KoBERT)이 성공적으로 로드되었습니다.")
    else:
        st.error("❗️ AI 모델 로드를 실패했습니다.")

    # ⭐️ 이 부분은 secrets.toml 파일이 올바르게 있어야 합니다.
    if st.secrets.get("connections", {}).get("gsheets"): st.success("✅ Google Sheets 인증 정보가 확인되었습니다.")
    else: st.error("❗️ Google Sheets 인증 정보('connections.gsheets')를 찾을 수 없습니다.")
    if st.secrets.get("spotify", {}).get("client_id"): st.success("✅ Spotify 인증 정보가 확인되었습니다.")
    else: st.error("❗️ Spotify 인증 정보('[spotify]' 섹션)를 찾을 수 없습니다.")
    if st.secrets.get("TMDB_API_KEY"): st.success("✅ TMDB API 키가 확인되었습니다.")
    else: st.error("❗️ TMDB API 키('TMDB_API_KEY')를 찾을 수 없습니다.")

st.divider()

if 'diary_text' not in st.session_state: st.session_state.diary_text = ""
if 'final_emotion' not in st.session_state: st.session_state.final_emotion = None
if 'confidence_score' not in st.session_state: st.session_state.confidence_score = 0.0
if 'rec_method' not in st.session_state: st.session_state.rec_method = '내 플레이리스트'

col1, col2 = st.columns([3, 1])
with col1:
    st.text_area("오늘의 일기를 작성해주세요:", key='diary_text', height=250,
                 value=st.session_state.diary_text)
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
    st.divider()
    st.subheader("🔍 분석 결과 피드백")
    st.write("AI의 분석 결과가 실제 감정과 다른가요? 피드백을 남겨주시면 모델 개선에 큰 도움이 됩니다.")
    feedback_options = FINAL_EMOTIONS + ["(감정을 선택해주세요)"]
    try:
        default_index = feedback_options.index(final_emotion)
    except ValueError:
        default_index = len(feedback_options) - 1
    corrected_emotion = st.selectbox(
        "이 일기의 진짜 감정은 무엇인가요?",
        options=feedback_options,
        index=default_index,
        key="feedback_emotion"
    )
    if st.button("피드백 제출하기"):
        if corrected_emotion == "(감정을 선택해주세요)":
            st.error("피드백할 감정을 선택해주세요.")
        elif corrected_emotion == st.session_state.final_emotion:
            st.info("AI의 분석과 동일한 감정이네요. 알려주셔서 감사합니다! 😄")
        else:
            client = get_gsheets_connection()
            if client:
                save_feedback_to_gsheets(client, st.session_state.diary_text, corrected_emotion)
            else:
                st.error("Google Sheets에 연결할 수 없습니다.")
st.divider()
with st.expander("피드백 저장 현황 보기 (Google Sheets)"):
    client = get_gsheets_connection()
    if client:
        df = fetch_all_data_from_gsheets(client)
        if not df.empty:
            st.dataframe(df.tail())
            st.info(f"현재 총 **{len(df)}개**의 데이터가 저장되어 있습니다. (1분마다 갱신)")
        else:
            st.write("아직 저장된 데이터가 없습니다.")
    else:
        st.error("Google Sheets에 연결할 수 없습니다. Secrets 설정을 다시 확인해주세요.")
