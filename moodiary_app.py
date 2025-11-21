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
from datetime import datetime, timezone, timedelta # KST
from streamlit_calendar import calendar
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# (선택) Spotify SDK
try:
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials
    SPOTIPY_AVAILABLE = True
except ImportError:
    spotipy = None
    SpotifyClientCredentials = None
    SPOTIPY_AVAILABLE = False 

# --- 2) 기본 설정 ---
# ⭐️ [모델 변경] JUDONGHYEOK/6-emotion-bert-korean-v2
EMOTION_MODEL_ID = "JUDONGHYEOK/6-emotion-bert-korean-v2"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
GSHEET_DB_NAME = "moodiary_db" 

# 비상용 TMDB 키
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

# 감정별 테마 (색상 RGBA로 옅게 조정)
EMOTION_META = {
    "행복": {"color": "rgba(255, 215, 0, 0.4)", "emoji": "😆", "desc": "최고의 하루!"}, # 노랑
    "슬픔": {"color": "rgba(30, 144, 255, 0.4)", "emoji": "😭", "desc": "토닥토닥, 힘내요."}, # 파랑
    "분노": {"color": "rgba(255, 0, 0, 0.4)", "emoji": "🤬", "desc": "워워, 진정해요."},   # 빨강
    "힘듦": {"color": "rgba(128, 128, 128, 0.4)", "emoji": "🤯", "desc": "휴식이 필요해."}, # 회색
    "놀람": {"color": "rgba(138, 43, 226, 0.4)", "emoji": "😱", "desc": "깜짝 놀랐군요!"}, # 보라
    "중립": {"color": "rgba(54, 54, 54, 0.2)", "emoji": "😐", "desc": "평온한 하루."}    # 흑색
}

# 대한민국 표준시(KST) 정의 (UTC+9)
KST = timezone(timedelta(hours=9))

st.set_page_config(layout="wide", page_title="MOODIARY", page_icon="💖")

# ⭐️⭐️⭐️ [디자인] 예쁜 UI를 위한 커스텀 CSS ⭐️⭐️⭐️
def apply_custom_css():
    st.markdown("""
        <style>
        /* 1. 폰트 설정 (Noto Sans KR) */
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Noto Sans KR', sans-serif;
        }

        /* 2. 전체 배경 (은은한 그라데이션) */
        .stApp {
            background: linear-gradient(to bottom right, #FDFBF7, #E6E9F0);
        }

        /* 3. 메인 컨텐츠 카드 UI (흰색 박스 + 그림자) */
        .block-container {
            background-color: rgba(255, 255, 255, 0.95);
            padding: 3rem !important;
            border-radius: 20px;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.05);
            margin-top: 2rem;
            max-width: 1000px;
        }

        /* 4. 버튼 스타일링 (둥글고 부드러운 색상) */
        .stButton > button {
            width: 100%;
            border-radius: 15px;
            border: none;
            background-color: #6C5CE7; /* 시그니처 퍼플 */
            color: white;
            font-weight: 700;
            padding: 0.6rem 1rem;
            transition: all 0.3s ease;
        }
        .stButton > button:hover {
            background-color: #5b4bc4;
            transform: translateY(-2px);
            box-shadow: 0 5px 10px rgba(108, 92, 231, 0.3);
            color: white;
        }

        /* 5. 탭 스타일링 */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 45px;
            border-radius: 10px;
            background-color: #F0F2F6;
            border: none;
            font-weight: 600;
            color: #666;
        }
        .stTabs [aria-selected="true"] {
            background-color: #6C5CE7 !important;
            color: white !important;
        }

        /* 6. 상단 헤더/푸터 숨기기 */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)

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
        return None 

    try:
        sh.worksheet("users")
        sh.worksheet("diaries")
    except:
        return None 
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
        ws = sh.worksheet("diaries")
        cell = ws.find(date, in_column=2)
        if cell and ws.cell(cell.row, 1).value == username:
            ws.update_cell(cell.row, 3, emotion)
            ws.update_cell(cell.row, 4, text)
        else:
            ws.append_row([username, date, emotion, text])
        return True
    except: return False

# =========================================
# 🧠 4) AI 및 추천 로직 (새 모델 적용)
# =========================================
@st.cache_resource
def load_emotion_model():
    """
    JUDONGHYEOK/6-emotion-bert-korean-v2 모델 로드
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # 모델의 원본 라벨
        raw_idx2label = {
            0: "기쁨", 1: "분노", 2: "불안", 3: "슬픔", 4: "중립", 5: "힘듦"
        }

        # 우리 앱의 감정으로 매핑 (놀람은 예측되지 않음)
        post_processing_map = {
            "기쁨": "행복",
            "분노": "분노",
            "불안": "힘듦", # 불안 -> 힘듦으로 통합
            "슬픔": "슬픔",
            "중립": "중립",
            "힘듦": "힘듦"
        }

        return model, tokenizer, device, raw_idx2label, post_processing_map
    except Exception as e:
        st.error(f"감정 분석 모델 로드 실패: {e}")
        return None, None, None, None, None

def analyze_diary(text, model, tokenizer, device, raw_idx2label, post_processing_map):
    if not text or model is None: return None, 0.0

    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt")
    for k in enc: enc[k] = enc[k].to(device)

    with torch.no_grad(): logits = model(**enc).logits

    probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    score = float(probs[pred_id].cpu().item())

    raw_label = raw_idx2label.get(pred_id, "중립")
    final_label = post_processing_map.get(raw_label, "중립")

    return final_label, score

@st.cache_resource
def get_spotify_client():
    if not SPOTIPY_AVAILABLE:
        return "Spotipy 라이브러리 설치 실패. (requirements.txt 확인)"
    try:
        creds = st.secrets["spotify"]
        manager = SpotifyClientCredentials(client_id=creds["client_id"], client_secret=creds["client_secret"])
        sp = spotipy.Spotify(client_credentials_manager=manager, retries=3, backoff_factor=0.3)
        sp.search(q="test", limit=1)
        return sp
    except KeyError:
        return "Spotify Secrets 설정이 없습니다."
    except Exception as e:
        return f"Spotify 로그인 실패: {e}"

def recommend_music(emotion):
    sp = get_spotify_client()
    if not isinstance(sp, spotipy.Spotify):
        return [{"error": sp}]

    SEARCH_KEYWORDS_MAP = {
        "행복": ["신나는 K-Pop", "Upbeat Band", "K-Pop Hits", "Today's Top Hits"],
        "슬픔": ["위로가 되는 발라드", "새벽 감성 힙합", "Chill K-Pop", "K-Pop Ballad"],
        "분노": ["스트레스 해소 밴드", "신나는 힙합", "Driving K-Pop", "국내 힙합"],
        "힘듦": ["Lofi Hip Hop", "편안한 발라드", "Chill Band", "위로 K-Pop"],
        "놀람": ["K-Pop Party", "국내 밴드", "Upbeat Hip Hop"],
        "중립": ["K-Pop 발라드", "국힙 Top 100", "Chill", "Korean Band"]
    }
    
    keyword_list = SEARCH_KEYWORDS_MAP.get(emotion, SEARCH_KEYWORDS_MAP["중립"])
    query = random.choice(keyword_list)
    
    try:
        results = sp.search(q=query, type="playlist", limit=10, market="KR")
        playlists = results.get('playlists', {}).get('items', [])
        
        if not playlists:
            return [{"error": f"'{query}' 검색 결과 플레이리스트 없음"}]

        valid_tracks = []
        random.shuffle(playlists) 

        for pl in playlists:
            try:
                pid = pl['id']
                tracks_results = sp.playlist_items(pid, limit=30)
                items = tracks_results.get('items', []) if tracks_results else []
                for it in items:
                    t = it.get('track')
                    if t and t.get('id') and t.get('name'):
                         valid_tracks.append({"id": t['id'], "title": t['name']})
                
                if len(valid_tracks) >= 10: 
                    break
            except Exception as e:
                continue 

        if not valid_tracks: 
            return [{"error": "추천 곡을 찾지 못했습니다."}]
        
        seen = set(); unique = []
        for v in valid_tracks:
            if v['id'] not in seen: unique.append(v); seen.add(v['id'])
        
        return random.sample(unique, k=min(3, len(unique)))
    
    except Exception as e: 
        return [{"error": f"Spotify 검색 오류: {e}"}]

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
# ⭐️ 디자인 적용
apply_custom_css()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "login"

def login_page():
    st.markdown("<h1 style='text-align: center;'>MOODIARY 💖</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #666;'>당신의 하루를 기록하고, 감정에 맞는 처방을 받아보세요.</p>", unsafe_allow_html=True)
    st.write("") 

    tab1, tab2 = st.tabs(["🔑 로그인", "📝 회원가입"])
    sh = init_db()
    if sh is None: st.error("데이터베이스 연결 실패. Secrets 설정을 확인하세요."); return

    with tab1:
        lid = st.text_input("아이디", key="lid")
        lpw = st.text_input("비밀번호", type="password", key="lpw")
        st.write("")
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
        st.write("")
        if st.button("가입하기", width='stretch'):
            users = get_all_users(sh)
            if nid in users: st.error("이미 있는 아이디입니다.")
            elif len(nid)<1 or len(npw)!=4: st.error("입력을 확인해주세요.")
            else:
                if add_user(sh, nid, npw): st.success("가입 성공! 로그인해주세요.")
                else: st.error("가입 실패 (DB 오류)")

def dashboard_page():
    st.markdown(f"### {st.session_state.username}님의 감정 달력 📅")
    
    legend_cols = st.columns(6)
    for i, (emo, meta) in enumerate(EMOTION_META.items()):
        legend_cols[i].markdown(f"<span style='color:{meta['color']}; font-size: 1.2em;'>●</span> {emo}", unsafe_allow_html=True)
    st.divider()

    sh = init_db()
    my_diaries = get_user_diaries(sh, st.session_state.username)
    
    tab1, tab2 = st.tabs(["📅 감정 달력", "📊 이달의 통계"])

    with tab1:
        events = []
        for date_str, data in my_diaries.items():
            emo = data.get("emotion", "중립")
            meta = EMOTION_META.get(emo, EMOTION_META["중립"])
            events.append({"start": date_str, "display": "background", "backgroundColor": meta["color"]})
            events.append({"title": meta["emoji"], "start": date_str, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": "#000000"})

        calendar(events=events, options={"headerToolbar": {"left": "prev,next today", "center": "title", "right": ""}, "initialView": "dayGridMonth"}, 
                 custom_css="""
                 .fc-event-title {
                     font-size: 3em !important;
                     display: flex;
                     justify-content: center;
                     align-items: center;
                     height: 100%;
                     line-height: 1;
                     transform: translateY(-25px); 
                     text-shadow: 1px 1px 2px rgba(0,0,0,0.2); 
                 }
                 .fc-daygrid-event {
                     padding: 0 !important;
                     margin: 0 !important;
                     border: none !important;
                     color: black !important;
                     background-color: transparent !important; 
                 }
                 .fc-daygrid-day-frame {
                     height: 100%;
                     display: flex;
                     flex-direction: column;
                     justify-content: center;
                     align-items: center;
                     position: relative;
                 }
                 .fc-daygrid-day-number {
                      position: absolute !important;
                      top: 5px;
                      right: 5px;
                      font-size: 0.8em;
                      color: black;
                      z-index: 10 !important; 
                      text-shadow: 1px 1px 2px rgba(255,255,255,0.5);
                 }
                 .fc-daygrid-day-top {
                    flex-grow: 1;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    width: 100%;
                 }
                 .fc-bg-event {
                     opacity: 1.0 !important; 
                 }
                 """
                 )
        st.write("")

    with tab2:
        today = datetime.now(KST)
        st.subheader(f"{today.month}월의 감정 통계")
        
        current_month_str = today.strftime("%Y-%m")
        
        month_emotions = []
        for date_str, data in my_diaries.items():
            if date_str.startswith(current_month_str):
                month_emotions.append(data.get('emotion', '중립'))
        
        df = pd.DataFrame(month_emotions, columns=['emotion'])
        emotion_counts = df['emotion'].value_counts().reindex(EMOTION_META.keys(), fill_value=0)
            
        chart_data = emotion_counts.reset_index()
        chart_data.columns = ['emotion', 'count']

        # 옅은 색상 (RGBA) 매핑
        domain = list(EMOTION_META.keys())
        range_ = [meta['color'] for meta in EMOTION_META.values()]

        st.vega_lite_chart(chart_data, {
            "title": f"{today.month}월의 감정 분포",
            "width": "container",
            "mark": {"type": "bar", "cornerRadius": 5, "opacity": 1.0},
            "encoding": {
                "x": {"field": "emotion", "type": "nominal", "sort": domain, "title": "감정", "axis": {"labelAngle": 0}},
                "y": {"field": "count", "type": "quantitative", "title": "횟수", "scale": {"zero": True}, "axis": {"format": "d", "tickMinStep": 1}},
                "color": {"field": "emotion", "type": "nominal", "scale": {"domain": domain, "range": range_}, "legend": None},
                "tooltip": [{"field": "emotion", "title": "감정"}, {"field": "count", "title": "횟수"}]
            }
        }, use_container_width=True)
        
        st.write("---")
        st.write("**감정별 횟수**")
        cols = st.columns(6)
        for idx, (emo, count) in enumerate(emotion_counts.items()):
            cols[idx].metric(label=f"{EMOTION_META[emo]['emoji']} {emo}", value=f"{count}회")

    st.divider() 
    today_str = datetime.now(KST).strftime("%Y-%m-%d")
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
    
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem;'>
        <h2 style='color: {meta['color'].replace('0.4', '1.0').replace('0.2', '1.0')}; font-size: 3rem; margin-bottom: 0.5rem;'>
            {meta['emoji']} 오늘의 감정: {emo}
        </h2>
        <h4 style='color: #555;'>{meta['desc']}</h4>
    </div>
    """, unsafe_allow_html=True)
    
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
                components.iframe(f"https://open.spotify.com/embed/track/{item['id']}", height=250, width="100%") 
            else: 
                st.error(item.get("error", "로딩 실패"))
    with c2:
        st.markdown("#### 🎬 추천 영화")
        st.button("🔄 다른 영화", on_click=refresh_movies, key="rv_btn", width='stretch')
        for item in st.session_state.movie_recs:
            if item.get('poster'):
                ic, tc = st.columns([1, 2])
                ic.image(item['poster'], use_container_width=True)
                tc.markdown(f"**{item['title']} ({item['year']})**\n⭐ {item['rating']:.1f}\n\n*{item.get('overview','')}*")
            else: st.error(item.get("text", "로딩 실패"))

def write_page():
    st.title("오늘의 이야기 📝")
    if st.button("⬅️ 뒤로 가기"):
        st.session_state.page = "dashboard"
        st.rerun()

    # ⭐️ [변경] 새 모델 로드 함수 사용
    model, tokenizer, device, raw_idx2label, postmap = load_emotion_model()
    if not model: st.error("AI 모델 로드 중..."); return

    if "diary_input" not in st.session_state: st.session_state.diary_input = ""
    txt = st.text_area("오늘 하루는 어땠나요?", value=st.session_state.diary_input, height=300, key="diary_editor", placeholder="여기에 일기를 작성하세요...")
    
    if st.button("🔍 감정 분석하고 저장하기", type="primary", width='stretch'):
        if not txt.strip(): st.warning("내용을 입력해주세요."); return
        
        with st.spinner("분석 및 저장 중..."):
            # ⭐️ [변경] 새 분석 함수 사용
            emo, sc = analyze_diary(txt, model, tokenizer, device, raw_idx2label, postmap)
            st.session_state.final_emotion = emo
            st.session_state.music_recs = recommend_music(emo)
            st.session_state.movie_recs = recommend_movies(emo)
            
            sh = init_db()
            today = datetime.now(KST).strftime("%Y-%m-%d")
            add_diary(sh, st.session_state.username, today, emo, txt)
            
            st.session_state.page = "result"
            st.rerun()

# =========================================
# 🚀 앱 메인 컨트롤러
# =========================================
if st.session_state.logged_in:
    with st.sidebar:
        st.write(f"**{st.session_state.username}**님, 환영합니다!")
        if st.button("로그아웃", width='stretch'):
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.rerun()

if not st.session_state.logged_in: login_page()
elif st.session_state.page == "dashboard": dashboard_page()
elif st.session_state.page == "write": write_page()
elif st.session_state.page == "result": result_page()
