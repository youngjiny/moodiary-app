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
    SPOTIPY_AVAILABLE = True
except ImportError:
    spotipy = None
    SpotifyClientCredentials = None
    SPOTIPY_AVAILABLE = False # ⭐️ 라이브러리 설치 실패를 기억

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
        # ⭐️ 이미 해당 날짜에 일기가 있는지 확인
        ws = sh.worksheet("diaries")
        cell = ws.find(date, in_column=2)
        if cell and ws.cell(cell.row, 1).value == username:
            # 찾았으면 업데이트
            ws.update_cell(cell.row, 3, emotion)
            ws.update_cell(cell.row, 4, text)
        else:
            # 없으면 새로 추가
            ws.append_row([username, date, emotion, text])
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
    if not SPOTIPY_AVAILABLE:
        return "Spotipy 라이브러리 설치 실패. (requirements.txt 확인)"
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        sp = spotipy.Spotify(client_credentials_manager=manager, retries=3, backoff_factor=0.3)
        sp.search(q="test", limit=1) # ⭐️ 로그인 테스트
        return sp # ⭐️ 성공
    except KeyError:
        return "Spotify Secrets 설정이 없습니다."
    except Exception as e:
        return f"Spotify 로그인 실패: {e}"

# --- 6) ⭐️ Spotify 추천 (플레이리스트 검색 기반 + 404 방어) ---
def recommend_music(emotion):
    sp = get_spotify_client()
    if not sp:
        return ["Spotify 연결 실패 (Secrets 확인 필요)"]

    # ⭐️ 감정별 검색 키워드 (OR 조건 활용)
    GENRE_MAP = {
        '행복': ['happy k-pop', 'joyful pop', 'exciting k-pop', 'summer vibe'],
        '분노': ['angry rock', 'hard rock', 'stress relief', 'powerful k-pop'],
        '슬픔': ['sad k-pop ballad', 'gloomy pop', 'heartbreak song', 'emotional ost'],
        '힘듦': ['healing k-pop', 'chill pop', 'comforting song', 'acoustic'],
        '놀람': ['party music', 'edm k-pop', 'festival vibes', 'upbeat pop']
    }

    keywords = GENRE_MAP.get(emotion, ['k-pop'])
    query_base = " OR ".join(keywords)
    # 노키즈존 필터 추가
    query = f"({query_base}) NOT children NOT nursery"

    try:
        # 1️⃣ 플레이리스트 검색 (최대 20개)
        playlists_resp = sp.search(q=query, type='playlist', limit=20)
        playlists = playlists_resp.get('playlists', {}).get('items', [])
        
        if not playlists:
            return ["관련 플레이리스트를 찾지 못했습니다."]

        # 2️⃣ 플레이리스트 중 하나 랜덤 선택 & 트랙 가져오기 (최대 3번 재시도)
        for _ in range(3): # (안정성을 위해 3번 시도)
            chosen_playlist = random.choice(playlists)
            if not chosen_playlist or not chosen_playlist.get('id'): continue

            try:
                # 플레이리스트의 트랙 가져오기
                tracks_resp = sp.playlist_items(chosen_playlist['id'], limit=50)
                items = tracks_resp.get('items', []) if tracks_resp else []
                
                valid_tracks = []
                for item in items:
                    track = item.get('track')
                    # 트랙 유효성 검사 (ID가 있어야 재생 버튼 가능)
                    if track and track.get('id') and track.get('name'):
                        valid_tracks.append({
                            'id': track['id'],
                            'title': track['name'],
                            'artist': track['artists'][0]['name'] if track.get('artists') else 'Unknown'
                        })
                
                if valid_tracks:
                    # 3곡 랜덤 선택 후 반환
                    return random.sample(valid_tracks, k=min(3, len(valid_tracks)))
            
            except Exception:
                continue # 이 플레이리스트가 에러나면 다음 거 시도

        return ["추천할 만한 노래를 찾지 못했습니다. (플레이리스트 접근 불가)"]

    except Exception as e:
        return [f"Spotify 검색 오류: {e}"]

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
    
    legend_cols = st.columns(6)
    for i, (emo, meta) in enumerate(EMOTION_META.items()):
        legend_cols[i].markdown(f"<span style='color:{meta['color']}; font-size: 1.2em;'>●</span> {emo}", unsafe_allow_html=True)
    st.divider()

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
        if st.button("✏️ 오늘의 일기 쓰러 가기", type="primary", width='stretch'):
            st.session_state.page = "write"
            st.session_state.diary_input = "" 
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
            add_diary(sh, st.session_state.username, today, emo, txt)
            
            st.session_state.page = "result"
            st.rerun()

# =========================================
# 🚀 앱 메인 컨트롤러
# =========================================
if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "login"

if st.session_state.logged_in:
    with st.sidebar:
        st.write(f"**{st.session_state.username}**님")
        if st.button("로그아웃", width='stretch'):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

if not st.session_state.logged_in: login_page()
elif st.session_state.page == "dashboard": dashboard_page()
elif st.session_state.page == "write": write_page()
elif st.session_state.page == "result": result_page()

