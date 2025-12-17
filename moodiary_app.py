# --- 수정된 전체 코드 ---
import streamlit as st
import random
import requests
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import streamlit.components.v1 as components
from datetime import datetime, timezone, timedelta
from streamlit_calendar import calendar
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd

# (이전 설정/라이브러리 부분 동일...)
EMOTION_META = {
    "기쁨": {"color": "#FFD700", "emoji": "😆"},
    "분노": {"color": "#FF5050", "emoji": "🤬"},
    "불안": {"color": "#FFA032", "emoji": "😰"},
    "슬픔": {"color": "#5078FF", "emoji": "😭"},
    "힘듦": {"color": "#969696", "emoji": "🤯"},
    "중립": {"color": "#50B478", "emoji": "😐"}
}

st.set_page_config(layout="wide", page_title="MOODIARY")

# --- CSS 수정 (영화 줄거리 가독성 향상) ---
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    main_bg = "rgba(255, 255, 255, 0.85)" if not is_dark else "rgba(40, 40, 40, 0.9)"
    main_text = "#333" if not is_dark else "#f0f0f0"
    
    st.markdown(f"""
        <style>
        .block-container {{ background: {main_bg}; border-radius: 25px; padding: 3rem !important; }}
        
        /* 영화 카드: 줄거리 전체 표시 */
        .movie-card {{
            background: white; border-radius: 15px; padding: 15px; margin-bottom: 20px;
            display: flex; gap: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .movie-card img {{ width: 120px; border-radius: 10px; }}
        .movie-overview {{ font-size: 0.9em; line-height: 1.5; color: #444; }}

        /* 행복 저장소 카드 */
        .happy-card {{ background: #fff9c4; border-left: 6px solid #FFD700; padding: 20px; border-radius: 20px; margin-bottom: 15px; }}
        .happy-date {{ font-weight: bold; color: #888; font-size: 0.85em; margin-bottom: 5px; }}
        .happy-text {{ font-size: 1.1em; color: #2c3e50; }}
        </style>
    """, unsafe_allow_html=True)

# (DB 연동 및 AI 분석 함수들은 기존과 동일하게 유지)

# --- 1. 달력 페이지 (겹침 해결 및 꽉 찬 배경) ---
def page_dashboard(sh):
    st.markdown("## 📅 감정 달력")
    my_diaries = get_user_diaries(sh, st.session_state.username)
    events = []
    
    for d, data in my_diaries.items():
        meta = EMOTION_META.get(data['emotion'], EMOTION_META["중립"])
        # 배경색과 이모지를 하나의 이벤트로 통합하여 겹침 방지
        events.append({
            "title": meta["emoji"],
            "start": d,
            "allDay": True,
            "display": "block", # 이벤트를 블록 형태로 표시
            "backgroundColor": meta["color"],
            "borderColor": meta["color"],
            "textColor": "white"
        })
    
    # 캘린더 커스텀 CSS: 칸 높이 및 이모지 중앙 정렬
    calendar_css = """
        .fc-event { border-radius: 0px !important; border: none !important; height: 100% !important; display: flex !important; align-items: center !important; justify-content: center !important; }
        .fc-event-title { font-size: 2.2em !important; }
        .fc-daygrid-day-frame { height: 120px !important; cursor: pointer; }
        .fc-daygrid-event-harness { height: 100% !important; margin: 0 !important; }
    """
    
    calendar(events=events, options={"initialView": "dayGridMonth", "height": "auto"}, custom_css=calendar_css)

# --- 2. 추천 페이지 (음악 크기 키움) ---
def page_recommend(sh):
    st.markdown("## 🎵 오늘을 위한 추천")
    emo = st.session_state.get("final_emotion", "중립")
    music_recs = st.session_state.get("music_recs", [])
    movie_recs = st.session_state.get("movie_recs", [])
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎵 음악 추천")
        for item in music_recs:
            # 높이를 160에서 200으로 늘려 크기를 키웠습니다.
            components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=200)
    with c2:
        st.markdown("#### 🎬 영화 추천")
        for item in movie_recs:
            st.markdown(f"""
            <div class="movie-card">
                <img src="{item['poster']}">
                <div>
                    <div style="font-weight:bold; font-size:1.1em;">{item['title']} ({item['year']})</div>
                    <div style="color:#f1c40f;">★ {item['rating']}</div>
                    <div class="movie-overview">{item['overview']}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- 3. 통계 페이지 (한글 똑바로 + 색상 일치) ---
def page_stats(sh):
    st.markdown("## 📊 감정 통계")
    diaries = get_user_diaries(sh, st.session_state.username)
    if not diaries: return
    
    df = pd.DataFrame([{"emotion": d['emotion']} for d in diaries.values()])
    counts = df['emotion'].value_counts().reindex(EMOTION_META.keys(), fill_value=0).reset_index()
    counts.columns = ['emotion', 'count']
    
    color_range = [m['color'] for m in EMOTION_META.values()]
    
    st.vega_lite_chart(counts, {
        "mark": {"type": "bar", "cornerRadius": 5},
        "encoding": {
            "x": {"field": "emotion", "type": "nominal", "axis": {"labelAngle": 0}, "sort": list(EMOTION_META.keys())},
            "y": {"field": "count", "type": "quantitative"},
            "color": {
                "field": "emotion", 
                "scale": {"domain": list(EMOTION_META.keys()), "range": color_range},
                "legend": None
            }
        }
    }, use_container_width=True)

# (나머지 페이지 함수 및 메인 로직 실행부 동일...)
