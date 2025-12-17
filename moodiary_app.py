# --- 1) 필수 라이브러리 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit.components.v1 as components
from datetime import datetime, timezone, timedelta  # KST
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
# ⭐️ v6-balanced 모델로 고정
EMOTION_MODEL_ID = "JUDONGHYEOK/6-emotion-bert-korean-v6-balanced"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
GSHEET_DB_NAME = "moodiary_db" 

# 비상용 TMDB 키
EMERGENCY_TMDB_KEY = "8587d6734fd278ecc05dcbe710c29f9c"

EMOTION_META = {
    "기쁨": {"color": "rgba(255, 215, 0, 0.6)", "emoji": "😆", "desc": "웃음이 끊이지 않는 하루!"},
    "분노": {"color": "rgba(255, 80, 80, 0.6)", "emoji": "🤬", "desc": "워워, 진정이 필요해요."},
    "불안": {"color": "rgba(255, 160, 50, 0.6)", "emoji": "😰", "desc": "마음이 조마조마해요."},
    "슬픔": {"color": "rgba(80, 120, 255, 0.6)", "emoji": "😭", "desc": "마음의 위로가 필요해요."},
    "힘듦": {"color": "rgba(150, 150, 150, 0.6)", "emoji": "🤯", "desc": "휴식이 절실한 하루."},
    "중립": {"color": "rgba(80, 180, 120, 0.6)", "emoji": "😐", "desc": "평온하고 무난한 하루."}
}

KST = timezone(timedelta(hours=9))

st.set_page_config(layout="wide", page_title="MOODIARY", page_icon="💖")

# ⭐️ 커스텀 CSS (야간 모드 및 행복 저장소 일렬 정렬 디자인)
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    if is_dark:
        bg_start, bg_mid, bg_end = "#121212", "#2c2c2c", "#403A4E"
        main_bg, main_text, secondary_text = "rgba(40, 40, 40, 0.9)", "#f0f0f0", "#bbbbbb"
        sidebar_bg, menu_checked = "#1e1e1e", "#A29BFE"
        card_bg, card_text_happy = "#3a3a3a", "#ffffff"
        stat_card_line = "1px solid #444444"
    else:
        bg_start, bg_mid, bg_end = "#ee7752", "#e73c7e", "#23d5ab"
        main_bg, main_text, secondary_text = "rgba(255, 255, 255, 0.85)", "#333333", "#666666"
        sidebar_bg, menu_checked = "#f8f9fa", "#6C5CE7"
        card_bg, card_text_happy = "#fff9c4", "#2c3e50"
        stat_card_line = "none"

    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700&display=swap');
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        h1, h2, h3, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{ color: {main_text}; font-weight: 700; }}
        
        .stApp {{
            background: linear-gradient(-45deg, {bg_start}, {bg_mid}, {bg_end});
            background-size: 400% 400%;
            animation: gradient 15s ease infinite;
        }}
        @keyframes gradient {{ 0% {{background-position: 0% 50%;}} 50% {{background-position: 100% 50%;}} 100% {{background-position: 0% 50%;}} }}

        .block-container {{
            background: {main_bg}; backdrop-filter: blur(15px); border-radius: 25px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15); padding: 3rem !important; max-width: 1000px;
        }}

        p, label, .stMarkdown, .stTextarea, .stTextInput, .stCheckbox {{ color: {main_text} !important; }}
        section[data-testid="stSidebar"] {{ background-color: {sidebar_bg} !important; }}
        
        .stButton > button {{
            width: 100%; border-radius: 20px; border: none; font-weight: 700;
            background: linear-gradient(90deg, #6C5CE7 0%, #a29bfe 100%); color: white;
        }}

        /* ⭐️ 행복 저장소 일렬 카드 디자인 */
        .happy-card {{
            background: {card_bg}; border-left: 8px solid #FFD700;
            padding: 22px; border-radius: 15px; margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.08); width: 100%;
        }}
        .happy-date {{ color: {main_text}; font-weight: 700; font-size: 1.1em; margin-bottom: 10px; }}
        .happy-text {{ font-size: 1.2em; font-weight: 500; line-height: 1.6; color: {card_text_happy}; }}
        .month-header {{ 
            margin: 30px 0 15px 0; padding-bottom: 10px; border-bottom: 2px solid #FFD700;
            font-size: 1.5rem; color: {main_text}; font-weight: 800;
        }}

        header {{visibility: hidden;}} footer {{visibility: hidden;}}
        </style>
    """, unsafe_allow_html=True)

# --- 3) DB 연결 (강력한 예외 처리 적용) ---
@st.cache_resource
def get_gsheets_client():
    try:
        creds_info = st.secrets["connections"]["gsheets"]
        credentials = Credentials.from_service_account_info(creds_info, scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])
        return gspread.authorize(credentials)
    except Exception as e:
        st.error(f"❌ 구글 시트 인증 오류: {e}")
        return None

@st.cache_resource
def init_db():
    client = get_gsheets_client()
    if client:
        try:
            return client.open(GSHEET_DB_NAME)
        except Exception as e:
            st.error(f"❌ 시트 파일을 찾을 수 없습니다: {e}")
            return None
    return None

def get_all_users(sh):
    try:
        return {str(row['username']): str(row['password']) for row in sh.worksheet("users").get_all_records()}
    except: return {}

@st.cache_data(ttl=5) # 데이터 갱신을 위해 TTL 단축
def get_user_diaries(_sh, username):
    try:
        rows = _sh.worksheet("diaries").get_all_records()
        # 공백이나 데이터 없음 오류 방지
        data = {}
        for row in rows:
            if str(row.get('username')) == str(username):
                date_val = str(row.get('date'))
                if date_val and date_val != "init":
                    data[date_val] = {"emotion": row.get('emotion'), "text": row.get('text')}
        return data
    except Exception as e:
        st.warning(f"데이터를 불러오는 중 문제가 발생했습니다: {e}")
        return {}

def add_diary(sh, username, date, emotion, text):
    try:
        ws = sh.worksheet("diaries")
        cell = ws.find(date, in_column=2)
        if cell and str(ws.cell(cell.row, 1).value) == str(username):
            ws.update_cell(cell.row, 3, emotion)
            ws.update_cell(cell.row, 4, text)
        else:
            ws.append_row([username, date, emotion, text])
        get_user_diaries.clear() # 캐시 초기화
        return True
    except: return False

# --- 4) AI 로직 (v6 고정) ---
@st.cache_resource
def load_emotion_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_ID)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        return model, tokenizer, device, {0: "기쁨", 1: "분노", 2: "불안", 3: "슬픔", 4: "중립", 5: "힘듦"}
    except: return None, None, None, None

def analyze_diary(text, model, tokenizer, device, id2label):
    if not text or model is None: return "중립", 0.0
    enc = tokenizer(text, truncation=True, padding=True, max_length=256, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]
    pred_id = int(probs.argmax().cpu().item())
    return id2label.get(pred_id, "중립"), float(probs[pred_id].cpu().item())

# --- 5) 메인 화면 로직 ---
apply_custom_css()

if "logged_in" not in st.session_state: st.session_state.logged_in = False
if "page" not in st.session_state: st.session_state.page = "intro" 
if "dark_mode" not in st.session_state: st.session_state.dark_mode = False

def main_app():
    sh = init_db()
    if not sh: return
    
    with st.sidebar:
        st.markdown(f"### 👋 **{st.session_state.username}**님")
        dark = st.checkbox("🌙 야간 모드", value=st.session_state.dark_mode)
        if dark != st.session_state.dark_mode:
            st.session_state.dark_mode = dark
            st.rerun()
        st.divider()
        if st.button("📝 일기 작성", use_container_width=True): st.session_state.page = "write"; st.rerun()
        if st.button("📅 감정 달력", use_container_width=True): st.session_state.page = "dashboard"; st.rerun()
        if st.button("📊 통계 보기", use_container_width=True): st.session_state.page = "stats"; st.rerun()
        if st.button("📂 행복 저장소", use_container_width=True): st.session_state.page = "happy"; st.rerun()
        st.divider()
        if st.button("🚪 로그아웃", use_container_width=True):
            st.session_state.logged_in = False; st.session_state.page = "intro"; st.rerun()

    # 페이지 렌더링
    if st.session_state.page == "write": page_write(sh)
    elif st.session_state.page == "dashboard": page_dashboard(sh)
    elif st.session_state.page == "stats": page_stats(sh)
    elif st.session_state.page == "happy": page_happy_storage(sh)
    elif st.session_state.page == "result": st.info("분석 완료! 왼쪽 메뉴를 선택하세요.")

def page_write(sh):
    st.markdown("## 📝 오늘의 이야기")
    model, tok, dev, labs = load_emotion_model()
    txt = st.text_area("오늘 하루는 어땠나요?", height=300, key="write_area")
    if st.button("🔍 분석 및 저장"):
        if not txt.strip(): st.warning("내용을 입력하세요."); return
        emo, _ = analyze_diary(txt, model, tok, dev, labs)
        add_diary(sh, st.session_state.username, datetime.now(KST).strftime("%Y-%m-%d"), emo, txt)
        st.success("기록되었습니다!"); st.session_state.page = "dashboard"; st.rerun()

def page_dashboard(sh):
    st.markdown("## 📅 감정 달력")
    diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    for d, v in diaries.items():
        meta = EMOTION_META.get(v['emotion'], EMOTION_META["중립"])
        events.append({"start": d, "display": "background", "backgroundColor": meta["color"]})
        events.append({"title": meta["emoji"], "start": d, "allDay": True, "backgroundColor": "transparent", "borderColor": "transparent", "textColor": "#000"})
    calendar(events=events, options={"initialView": "dayGridMonth"})

def page_stats(sh):
    st.markdown("## 📊 감정 통계")
    diaries = get_user_diaries(sh, st.session_state.username)
    if diaries:
        data = [v['emotion'] for v in diaries.values()]
        st.bar_chart(pd.Series(data).value_counts())
    else: st.info("데이터가 없습니다.")

# ⭐️ 행복 저장소 (일렬 배치 + 월별 구분)
def page_happy_storage(sh):
    st.markdown("## 📂 행복 저장소")
    st.markdown("내가 **'기쁨'**을 느꼈던 순간들만 월별로 모아봤어요. 🥰")
    
    diaries = get_user_diaries(sh, st.session_state.username)
    happy_list = [{"date": d, "text": v["text"]} for d, v in diaries.items() if v["emotion"] == "기쁨"]
    
    if not happy_list:
        st.info("아직 기록된 '기쁨'의 순간이 없어요.")
    else:
        happy_list.sort(key=lambda x: x["date"], reverse=True)
        current_month = ""
        for item in happy_list:
            month_str = item["date"][:7] 
            if month_str != current_month:
                y, m = month_str.split("-")
                st.markdown(f"<div class='month-header'>{y}년 {m}월</div>", unsafe_allow_html=True)
                current_month = month_str
            
            st.markdown(f"""
                <div class="happy-card">
                    <div class="happy-date">{item['date']} {EMOTION_META['기쁨']['emoji']}</div>
                    <div class="happy-text">{item['text']}</div>
                </div>
            """, unsafe_allow_html=True)

# 인트로 & 로그인
def intro_page():
    st.markdown("<div style='text-align: center; margin-top: 10rem;'><h1 class='animated-title'>MOODIARY</h1><h3>당신의 마음을 기록하세요</h3></div>", unsafe_allow_html=True)
    if st.button("시작하기"): st.session_state.page = "login"; st.rerun()

def login_page():
    sh = init_db()
    st.markdown("### 🔑 로그인")
    lid = st.text_input("ID")
    lpw = st.text_input("PW", type="password")
    if st.button("로그인"):
        users = get_all_users(sh)
        if lid in users and str(users[lid]) == str(lpw):
            st.session_state.logged_in, st.session_state.username = True, lid
            st.rerun()
        else: st.error("로그인 실패")

# 메인 실행부
if st.session_state.logged_in: main_app()
elif st.session_state.page == "intro": intro_page()
else: login_page()
