# --- (기타 라이브러리 및 설정 생략 - 2번 코드 기준 유지) ---

# --- 3) 커스텀 CSS (2번의 디자인 + 1번의 사이드바 제어 로직 통합) ---
def apply_custom_css():
    is_dark = st.session_state.get("dark_mode", False)
    primary_purple = "#7B61FF" 
    
    # 테마 색상 설정 (2번 코드 스타일)
    bg_color = "#121212" if is_dark else "#F8F9FA"
    main_bg = "rgba(40, 40, 40, 0.95)" if is_dark else "rgba(255, 255, 255, 1.0)"
    
    css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;700;900&display=swap');
        html, body, [class*="css"] {{ font-family: 'Noto Sans KR', sans-serif; }}
        
        .stApp {{ background-color: {bg_color}; }}
        
        /* 메인 컨테이너 디자인 (2번 기준) */
        .block-container {{ 
            background: {main_bg}; 
            border-radius: 30px; 
            padding: 4rem !important; 
            box-shadow: 0 10px 40px rgba(0,0,0,0.05);
            margin-top: 2rem;
            max-width: 900px;
        }}

        /* 제목 및 버튼 스타일 (2번의 보라색 테마) */
        .main-title {{ font-size: 4.5rem !important; font-weight: 900 !important; color: {primary_purple}; text-align: center; }}
        div.stButton > button {{
            background-color: {primary_purple} !important;
            color: white !important;
            border-radius: 50px !important;
            padding: 0.7rem 2.5rem !important;
        }}

        /* ⭐ 핵심: 사이드바 토글 제어 (1번의 로직 적용) */
        /* 로그인 전(intro, login 페이지)에는 사이드바와 왼쪽 상단 화살표 버튼을 완전히 숨김 */
        { 
            '''
            section[data-testid="stSidebar"] { display: none; }
            button[data-testid="stSidebarCollapseButton"] { display: none; }
            ''' 
            if st.session_state.page in ["intro", "login"] else 
            '''
            /* 로그인 후에는 사이드바가 정상적으로 보이게 함 */
            section[data-testid="stSidebar"] { display: block; }
            ''' 
        }
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- 6) 네비게이션 및 라우팅 로직 ---
# (상태 관리를 명확히 하여 사이드바 노출 여부를 결정합니다)

if "logged_in" not in st.session_state: 
    st.session_state.logged_in = False
if "page" not in st.session_state: 
    st.session_state.page = "intro" 

# CSS 적용 (현재 페이지 상태에 따라 사이드바 숨김 여부 결정)
apply_custom_css()

if st.session_state.logged_in:
    main_app()  # 로그인 된 상태에서만 사이드바가 있는 메인 앱 실행
else:
    if st.session_state.page == "intro":
        intro_page()  # 사이드바 없음
    elif st.session_state.page == "login":
        login_page()  # 사이드바 없음
