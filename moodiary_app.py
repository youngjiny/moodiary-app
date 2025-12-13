# ... (상단 코드 생략)

def page_recommend(sh):
    st.markdown("## 🎵 음악/영화 추천")

    if "final_emotion" not in st.session_state:
# ... (감정 확인 로직 생략)
            return

    emo = st.session_state.final_emotion
    if emo not in EMOTION_META: emo = "중립"
    meta = EMOTION_META[emo]
    st.markdown(f"""<div style='text-align: center; padding: 2rem;'><h2 style='color: {meta['color'].replace('0.6', '1.0').replace('0.5', '1.0')}; font-size: 3rem;'>{meta['emoji']} 오늘의 감정: {emo}</h2><h4 style='color: #555;'>{meta['desc']}</h4></div>""", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🎵 추천 음악")
        # ⭐️ 음악 새로고침 버튼: 추천 재생성 및 rerun 명시
        if st.button("🔄 음악 새로고침", use_container_width=True, key="music_refresh"):
            st.session_state.music_recs = recommend_music(emo)
            st.rerun()
        for item in st.session_state.get("music_recs", []):
            if item.get('id'):
                # ⭐️⭐️⭐️ Spotify iframe 높이 500으로 수정
                components.iframe(f"https://open.spotify.com/embed/track/{item['id']}?utm_source=generator", height=500, width="100%")
    with c2:
        st.markdown("#### 🎬 추천 영화")
        # ⭐️ 영화 새로고침 버튼: 추천 재생성 및 rerun 명시
        if st.button("🔄 영화 새로고침", use_container_width=True, key="movie_refresh"):
            st.session_state.movie_recs = recommend_movies(emo)
            st.rerun()
        for item in st.session_state.get('movie_recs', []):
            if item.get('poster'):
                # ⭐️ 영화 추천 카드 디자인 유지
                ic, tc = st.columns([1, 2])
                ic.image(item['poster'], use_container_width=True)
                tc.markdown(f"**{item['title']} ({item['year']})**\n⭐ {item['rating']}\n\n*{item.get('overview','')}*")

    st.divider()
# ... (하단 코드 생략)
