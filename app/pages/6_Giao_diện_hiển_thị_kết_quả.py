# app/pages/6_Gợi_ý_&_Giao_diện.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import json
from pathlib import Path

# ========== CẤU HÌNH TRANG ==========
st.set_page_config(
    page_title="MovieRec - Gợi Ý Phim Thông Minh",
    page_icon="🎬",
    layout="wide"
)

# ========== CUSTOM CSS ==========
st.markdown("""
<style>
    .main-container {
        background: white;
        border-radius: 20px;
        padding: 30px;
        margin: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        min-height: 85vh;
    }
    
    .main-header {
        text-align: center;
        margin-bottom: 30px;
    }
    
    .main-header h1 {
        color: #4f46e5;
        font-size: 2.5rem;
        margin-bottom: 10px;
    }
    
    .main-header p {
        color: #6b7280;
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #f3f4f6;
        padding: 8px;
        border-radius: 12px;
        margin-bottom: 25px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 8px;
        padding: 12px 24px;
        border: 2px solid transparent;
        font-weight: 500;
        color: #6b7280;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: #4f46e5 !important;
        color: white !important;
        border-color: #4f46e5 !important;
        box-shadow: 0 4px 6px -1px rgba(79, 70, 229, 0.1), 0 2px 4px -1px rgba(79, 70, 229, 0.06);
    }
    
    .movie-card {
        background: white;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        height: 100%;
    }
    
    .movie-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        border-color: #4f46e5;
    }
    
    .movie-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 5px;
        font-size: 1rem;
    }
    
    .movie-info {
        color: #6b7280;
        font-size: 0.85rem;
        margin-bottom: 10px;
    }
    
    .movie-rating {
        display: flex;
        align-items: center;
        color: #f59e0b;
        font-weight: 600;
        font-size: 0.9rem;
    }
    
    .selected-genre-tag {
        background: linear-gradient(135deg, #ec4899 0%, #d946ef 100%);
        color: white;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 5px;
    }
    
    .other-genre-tag {
        background: #e5e7eb;
        color: #6b7280;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin-right: 5px;
        display: inline-block;
        margin-bottom: 5px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(79, 70, 229, 0.4);
    }
    
    .sidebar-section {
        background: #f9fafb;
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .history-item {
        padding: 10px;
        background: white;
        border-radius: 8px;
        margin: 5px 0;
        border-left: 3px solid #4f46e5;
    }
    
    .genre-filter-tag {
        display: inline-block;
        background: linear-gradient(135deg, #ec4899 0%, #d946ef 100%);
        color: white;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 0.9rem;
        font-weight: 500;
        margin: 0 5px 5px 0;
    }
    
    .no-results-container {
        text-align: center;
        padding: 40px 20px;
        background: #f9fafb;
        border-radius: 12px;
        margin: 20px 0;
    }
    
    .search-movie-card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e2e8f0;
        transition: all 0.3s ease;
        margin-bottom: 15px;
    }
    
    .search-movie-card:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
        transform: translateY(-2px);
    }
    
    .found-movie-highlight {
        background: linear-gradient(135deg, #dbeafe 0%, #e0e7ff 100%);
        border: 2px solid #4f46e5;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ========== KHỞI TẠO & LƯU TRỮ DỮ LIỆU ==========
def init_session_state():
    """Khởi tạo và tải dữ liệu session state"""
    if Path("data/session_state.pkl").exists():
        try:
            with open("data/session_state.pkl", "rb") as f:
                saved_state = pickle.load(f)
                for key, value in saved_state.items():
                    if key not in st.session_state:
                        st.session_state[key] = value
        except:
            pass
    
    defaults = {
        'search_history': [],
        'user_history': {},
        'current_user': 1,
        'loaded_data': False
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def save_session_state():
    """Lưu session state vào file"""
    state_to_save = {
        k: v for k, v in st.session_state.items() 
        if k != 'loaded_data'
    }
    
    Path("data").mkdir(exist_ok=True)
    
    try:
        with open("data/session_state.pkl", "wb") as f:
            pickle.dump(state_to_save, f)
    except:
        pass

init_session_state()

# ========== LOAD DỮ LIỆU PHIM ==========
@st.cache_data
def load_movie_data():
    movies = pd.read_csv("data/movies_final.csv")
    return movies

movies = load_movie_data()

# ========== HÀM PHÂN TÍCH THỂ LOẠI ==========
def filter_movies_by_genres(selected_genres, movies_df, require_all=False):
    """
    Lọc phim theo thể loại đã chọn
    """
    if not selected_genres:
        return pd.DataFrame()
    
    if require_all:
        condition = pd.Series([True] * len(movies_df))
        for genre in selected_genres:
            condition = condition & movies_df['genres'].str.contains(genre, na=False)
        filtered_movies = movies_df[condition]
    else:
        search_pattern = '|'.join(selected_genres)
        filtered_movies = movies_df[movies_df['genres'].str.contains(search_pattern, na=False)]
    
    return filtered_movies

def get_genre_display(genres_str, selected_genres):
    """
    Tạo HTML để hiển thị thể loại, làm nổi bật thể loại đã chọn
    """
    if pd.isna(genres_str):
        return ""
    
    genres_list = genres_str.split('|')
    html_parts = []
    
    for genre in genres_list:
        if genre in selected_genres:
            html_parts.append(f'<span class="selected-genre-tag">{genre}</span>')
        else:
            html_parts.append(f'<span class="other-genre-tag">{genre}</span>')
    
    return ''.join(html_parts)

# ========== HÀM TÌM PHIM VÀ PHIM TƯƠNG TỰ ==========
@st.cache_data
def find_movie_and_similar(search_term, n_similar=10):
    """
    Tìm phim theo tên và hiển thị các phim tương tự cùng thể loại
    """
    if not search_term or len(search_term.strip()) < 2:
        return None, pd.DataFrame(), ""
    
    search_term = search_term.lower()
    
    # Tìm phim chính xác hoặc gần đúng
    found_movies = movies[
        movies['title'].str.lower().str.contains(search_term, na=False)
    ]
    
    if len(found_movies) == 0:
        return None, pd.DataFrame(), ""
    
    # Lấy phim đầu tiên tìm thấy (phổ biến nhất)
    found_movie = found_movies.iloc[0]
    
    # Lấy thể loại của phim tìm thấy
    if pd.isna(found_movie['genres']):
        return found_movie, pd.DataFrame(), ""
    
    movie_genres = found_movie['genres'].split('|')
    
    # Lấy thể loại chính
    main_genre = movie_genres[0]
    
    # Tìm phim cùng thể loại (trừ phim đã tìm thấy)
    similar_movies = movies[
        (movies['genres'].str.contains(main_genre, case=False, na=False)) &
        (movies['movieId'] != found_movie['movieId'])
    ].sort_values('rating_count', ascending=False).head(n_similar)
    
    return found_movie, similar_movies, main_genre

# ========== XỬ LÝ URL PARAMETERS ==========
def get_url_params():
    """Lấy và xử lý URL parameters"""
    params = st.query_params
    
    # Lấy search term từ URL
    search_from_url = params.get("search", [""])[0]
    
    # Nếu có search từ URL, lưu vào session
    if search_from_url and search_from_url != st.session_state.get('current_search_term', ''):
        st.session_state.current_search_term = search_from_url
        # Thêm vào lịch sử
        if len(search_from_url.strip()) > 2:
            st.session_state.search_history.append({
                'term': search_from_url,
                'timestamp': datetime.now().isoformat(),
                'user_id': st.session_state.current_user
            })
    
    return search_from_url

def set_url_params(search_term=""):
    """Cập nhật URL parameters"""
    params = {"search": search_term} if search_term else {}
    st.query_params.clear()
    if params:
        st.query_params.update(params)

# ========== TẠO KEY AN TOÀN ==========
def create_safe_key(base_name, identifier):
    """Tạo key an toàn cho widget"""
    # Loại bỏ ký tự đặc biệt và thay thế dấu cách
    safe_identifier = str(identifier).replace(' ', '_').replace('|', '_').replace('(', '').replace(')', '')
    return f"{base_name}_{safe_identifier}"

# ========== MAIN CONTAINER ==========
with st.container():
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎬 MovieRec</h1>
        <p>Hệ thống gợi ý phim thông minh - Tìm phim phù hợp với bạn</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== SIDEBAR ==========
    with st.sidebar:
        st.markdown("### 👤 Tài khoản")
        
        # Chọn user
        user_id = st.selectbox(
            "Chọn User ID:", 
            range(1, 11),
            index=st.session_state.current_user-1,
            format_func=lambda x: f"👤 User {x}",
            key="user_select"
        )
        
        # Cập nhật current user
        if user_id != st.session_state.current_user:
            st.session_state.current_user = user_id
            save_session_state()
        
        # Lịch sử xem gần đây
        st.markdown("### 📜 Lịch sử gần đây")
        
        if user_id in st.session_state.user_history:
            user_history = st.session_state.user_history[user_id]
            if user_history:
                for item in user_history[-5:]:
                    with st.container():
                        st.markdown(f"""
                        <div class="history-item">
                            <div style="font-weight: 500; color: #1f2937;">{item['title'][:25]}...</div>
                            <div style="color: #6b7280; font-size: 0.85rem;">
                                {datetime.fromisoformat(item['timestamp']).strftime("%H:%M %d/%m")}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            else:
                st.info("Chưa có lịch sử")
        else:
            st.info("Chưa có lịch sử")
        
        # Xóa lịch sử
        if st.button("🗑️ Xóa lịch sử", key="clear_history_btn", use_container_width=True):
            st.session_state.user_history[user_id] = []
            save_session_state()
            st.success("Đã xóa lịch sử!")
        
        # Thống kê đơn giản
        st.markdown("### 📊 Thống kê")
        
        if user_id in st.session_state.user_history:
            history_count = len(st.session_state.user_history[user_id])
        else:
            history_count = 0
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Phim đã xem", history_count)
        with col2:
            st.metric("Tổng phim", f"{len(movies):,}")
        
        # Nút lưu dữ liệu
        if st.button("💾 Lưu dữ liệu", key="save_data_btn", use_container_width=True):
            save_session_state()
            st.success("Đã lưu dữ liệu!")

    
    # ========== TABS CHÍNH ==========
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎭 Tìm theo thể loại", 
        "🔥 Xu hướng", 
        "⏱️ Gợi ý theo lịch sử", 
        "🔍 Tìm kiếm"
    ])
    
    # ========== TÌM THEO THỂ LOẠI ==========
    with tab1:
        st.markdown("### 🎭 Tìm phim theo thể loại yêu thích")
        
        # Lấy tất cả thể loại duy nhất
        all_genres = set()
        for genres in movies['genres'].dropna():
            if isinstance(genres, str):
                for genre in genres.split('|'):
                    all_genres.add(genre)
        
        # Chọn thể loại
        col_genre1, col_genre2 = st.columns([2, 1])
        
        with col_genre1:
            selected_genres = st.multiselect(
                "Chọn thể loại bạn thích:",
                sorted(list(all_genres)),
                placeholder="Chọn một hoặc nhiều thể loại...",
                max_selections=3,
                key="genre_multiselect"
            )
        
        with col_genre2:
            sort_option = st.selectbox(
                "Sắp xếp theo:",
                ["Phổ biến nhất", "Mới nhất", "Đánh giá cao"],
                key="genre_sort"
            )
        
        # Hiển thị thể loại đã chọn
        if selected_genres:
            st.markdown("#### 🎯 Đang tìm phim có thể loại:")
            genre_tags_html = ""
            for genre in selected_genres:
                genre_tags_html += f'<span class="genre-filter-tag">{genre}</span>'
            st.markdown(genre_tags_html, unsafe_allow_html=True)
            st.markdown("---")
        
        if selected_genres:
            # Tìm phim có ít nhất một thể loại đã chọn
            filtered_movies = filter_movies_by_genres(selected_genres, movies, require_all=False)
            
            if len(filtered_movies) == 0:
                st.markdown("""
                <div class="no-results-container">
                    <div style="font-size: 3rem; margin-bottom: 20px;">😔</div>
                    <h3 style="color: #6b7280; margin-bottom: 15px;">Không tìm thấy phim nào</h3>
                    <p style="color: #9ca3af;">Không có phim nào có thể loại bạn đã chọn. Hãy thử thể loại khác!</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Sắp xếp
                if sort_option == "Phổ biến nhất":
                    filtered_movies = filtered_movies.sort_values('rating_count', ascending=False)
                elif sort_option == "Mới nhất":
                    filtered_movies = filtered_movies.sort_values('year', ascending=False)
                else:
                    filtered_movies = filtered_movies.sort_values('rating_count', ascending=False)
                
                st.markdown(f"### 🎬 **{len(filtered_movies)}** phim có thể loại bạn chọn")
                
                # Tách phim có thể loại chính là thể loại đã chọn
                primary_genre_movies = []
                secondary_genre_movies = []
                
                for _, movie in filtered_movies.iterrows():
                    if pd.isna(movie['genres']):
                        continue
                    
                    movie_genres = movie['genres'].split('|')
                    # Kiểm tra nếu thể loại đầu tiên là thể loại đã chọn
                    if movie_genres and movie_genres[0] in selected_genres:
                        primary_genre_movies.append(movie)
                    else:
                        secondary_genre_movies.append(movie)
                
                # Ưu tiên hiển thị phim có thể loại chính là thể loại đã chọn
                display_movies = primary_genre_movies + secondary_genre_movies
                
                # Hiển thị phim dạng grid
                cols = st.columns(4)
                for idx, movie in enumerate(display_movies[:12]):
                    with cols[idx % 4]:
                        # Hiển thị thể loại với màu sắc khác nhau
                        genres_display = get_genre_display(movie['genres'], selected_genres)
                        
                        # Lấy năm
                        year = int(movie['year']) if not pd.isna(movie['year']) else "N/A"
                        
                        st.markdown(f"""
                        <div class="movie-card">
                            <div style="background: linear-gradient(135deg, #ec4899 0%, #d946ef 100%); 
                                        height: 150px; border-radius: 8px; display: flex; align-items: center; 
                                        justify-content: center; color: white; font-weight: bold; 
                                        margin-bottom: 10px;">
                                🎭
                            </div>
                            <div class="movie-title">{movie['title'][:25] + ("..." if len(movie['title']) > 25 else "")}</div>
                            <div class="movie-info">📅 {year}</div>
                            <div style="margin-bottom: 10px;">
                                {genres_display}
                            </div>
                            <div class="movie-rating">⭐ {movie['rating_count']:,} đánh giá</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Sử dụng key an toàn
                        if st.button("➕ Thêm vào lịch sử", key=f"tab1_btn_{idx}", 
                                   use_container_width=True):
                            if user_id not in st.session_state.user_history:
                                st.session_state.user_history[user_id] = []
                            
                            st.session_state.user_history[user_id].append({
                                'movieId': int(movie['movieId']),
                                'title': movie['title'],
                                'timestamp': datetime.now().isoformat(),
                                'type': 'genre_search',
                                'genres': selected_genres
                            })
                            save_session_state()
                            st.success(f"Đã thêm '{movie['title'][:30]}...' vào lịch sử!")
                
                # Hiển thị thông tin về kết quả
                if len(primary_genre_movies) > 0 and len(secondary_genre_movies) > 0:
                    st.info(f"""
                    **📊 Kết quả tìm kiếm:**
                    - **{len(primary_genre_movies)}** phim có thể loại chính là **{', '.join(selected_genres)}**
                    - **{len(secondary_genre_movies)}** phim có chứa thể loại **{', '.join(selected_genres)}** nhưng không phải là thể loại chính
                    """)
                elif len(secondary_genre_movies) > 0:
                    st.info(f"""
                    **ℹ️ Lưu ý:** Tất cả {len(secondary_genre_movies)} phim đều có chứa thể loại **{', '.join(selected_genres)}** 
                    nhưng đây không phải là thể loại chính của phim. Thể loại đầu tiên được liệt kê là thể loại chính của phim.
                    """)
        else:
            st.info("👈 **Vui lòng chọn ít nhất một thể loại để bắt đầu tìm kiếm**")
            

    # ========== XU HƯỚNG ==========
    with tab2:
        st.markdown("### 🔥 Phim đang xu hướng")
        
        # Top phim được đánh giá nhiều nhất
        trending_movies = movies.sort_values('rating_count', ascending=False).head(12)
        
        # Thêm các thể loại xu hướng
        st.markdown("#### 🎭 Thể loại phổ biến")
        
        # Tính thể loại phổ biến
        all_genres_list = []
        for genres in movies['genres'].dropna():
            if isinstance(genres, str):
                all_genres_list.extend(genres.split('|'))
        
        genre_counts = pd.Series(all_genres_list).value_counts().head(8)
        
        # Hiển thị các thể loại phổ biến
        genre_cols = st.columns(4)
        for idx, (genre, count) in enumerate(genre_counts.items()):
            with genre_cols[idx % 4]:
                # Sử dụng key an toàn
                safe_key = create_safe_key("trend_genre", genre)
                if st.button(f"🎬 {genre}", key=safe_key, use_container_width=True):
                    # Chuyển sang tab thể loại và chọn genre này
                    set_url_params(genre)
        
        st.markdown("---")
        st.markdown("#### 🎬 Top phim được đánh giá nhiều nhất")
        
        # Hiển thị phim xu hướng
        cols = st.columns(4)
        for idx, (_, movie) in enumerate(trending_movies.iterrows()):
            with cols[idx % 4]:
                st.markdown("""
                <div class="movie-card">
                    <div style="background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); 
                                height: 150px; border-radius: 8px; display: flex; align-items: center; 
                                justify-content: center; color: white; font-weight: bold; 
                                margin-bottom: 10px;">
                        🔥
                    </div>
                    <div class="movie-title">{}</div>
                    <div class="movie-info">📅 {} • 🎭 {}</div>
                    <div class="movie-rating">⭐ {} đánh giá</div>
                </div>
                """.format(
                    movie['title'][:20] + ("..." if len(movie['title']) > 20 else ""),
                    int(movie['year']),
                    movie['genres'].split('|')[0] if '|' in movie['genres'] else movie['genres'][:15],
                    f"{movie['rating_count']:,}"
                ), unsafe_allow_html=True)
                
                # Sử dụng key đơn giản
                if st.button("➕ Lưu", key=f"tab2_btn_{idx}", use_container_width=True):
                    if user_id not in st.session_state.user_history:
                        st.session_state.user_history[user_id] = []
                    
                    st.session_state.user_history[user_id].append({
                        'movieId': int(movie['movieId']),
                        'title': movie['title'],
                        'timestamp': datetime.now().isoformat(),
                        'type': 'trending'
                    })
                    save_session_state()
                    st.success(f"Đã lưu vào lịch sử!")
    
    # ========== GỢI Ý THEO LỊCH SỬ ==========
    with tab3:
        st.markdown("### ⏱️ Gợi ý dựa trên lịch sử xem phim")
        
        if user_id in st.session_state.user_history and st.session_state.user_history[user_id]:
            user_history = st.session_state.user_history[user_id]
            
            # Phân tích lịch sử
            st.markdown("#### 📊 Phân tích sở thích của bạn")
            
            # Tìm thể loại phổ biến trong lịch sử
            history_genres = {}
            for item in user_history:
                movie_id = item['movieId']
                movie_genres = movies[movies['movieId'] == movie_id]['genres']
                if len(movie_genres) > 0:
                    for genre in movie_genres.iloc[0].split('|'):
                        history_genres[genre] = history_genres.get(genre, 0) + 1
            
            if history_genres:
                # Hiển thị top 3 thể loại
                top_genres = sorted(history_genres.items(), key=lambda x: x[1], reverse=True)[:3]
                
                col_pref1, col_pref2, col_pref3 = st.columns(3)
                for i, (genre, count) in enumerate(top_genres):
                    with [col_pref1, col_pref2, col_pref3][i]:
                        st.metric(f"🎭 {genre}", count)
                
                st.markdown("---")
                st.markdown("#### 🎯 Phim đề xuất dành riêng cho bạn")
                
                # Gợi ý phim cùng thể loại chưa xem
                top_genre = top_genres[0][0]
                watched_movies = [item['movieId'] for item in user_history]
                
                # Phim cùng thể loại
                similar_movies = movies[
                    (movies['genres'].str.contains(top_genre)) & 
                    (~movies['movieId'].isin(watched_movies))
                ].sort_values('rating_count', ascending=False).head(6)
                
                if len(similar_movies) > 0:
                    rec_cols = st.columns(3)
                    for idx, (_, movie) in enumerate(similar_movies.iterrows()):
                        with rec_cols[idx % 3]:
                            st.markdown("""
                            <div class="movie-card">
                                <div style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%); 
                                            height: 150px; border-radius: 8px; display: flex; align-items: center; 
                                            justify-content: center; color: white; font-weight: bold; 
                                            margin-bottom: 10px;">
                                    🎯
                                </div>
                                <div class="movie-title">{}</div>
                                <div class="movie-info">📅 {} • 🎭 {}</div>
                                <div class="movie-rating">⭐ {} đánh giá</div>
                            </div>
                            """.format(
                                movie['title'][:20] + ("..." if len(movie['title']) > 20 else ""),
                                int(movie['year']),
                                top_genre,
                                f"{movie['rating_count']:,}"
                            ), unsafe_allow_html=True)
                            
                            # Sử dụng key đơn giản
                            if st.button("➕ Xem sau", key=f"tab3_btn_{idx}", 
                                       use_container_width=True):
                                st.session_state.user_history[user_id].append({
                                    'movieId': int(movie['movieId']),
                                    'title': movie['title'],
                                    'timestamp': datetime.now().isoformat(),
                                    'type': 'recommendation'
                                })
                                save_session_state()
                                st.success("Đã thêm vào danh sách!")
                else:
                    st.info("Bạn đã xem hết các phim cùng thể loại này!")
            else:
                st.info("Chưa có đủ thông tin để phân tích sở thích")
        else:
            st.markdown("""
            <div style="text-align: center; padding: 50px 20px;">
                <div style="font-size: 4rem; margin-bottom: 20px;">📝</div>
                <h3 style="color: #6b7280; margin-bottom: 15px;">Chưa có lịch sử xem phim</h3>
                <p style="color: #9ca3af;">Hãy bắt đầu bằng cách tìm và thêm phim vào lịch sử!</p>
                <div style="margin-top: 30px;">
                    <a href="#tab1" style="background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%); 
                            color: white; border: none; border-radius: 8px; padding: 12px 24px; 
                            font-weight: 500; cursor: pointer; margin: 0 10px; text-decoration: none;">
                        🎭 Tìm theo thể loại
                    </a>
                    <a href="#tab2" style="background: white; color: #4f46e5; 
                            border: 2px solid #4f46e5; border-radius: 8px; padding: 12px 24px; 
                            font-weight: 500; cursor: pointer; margin: 0 10px; text-decoration: none;">
                        🔥 Xem xu hướng
                    </a>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # ========== TÌM KIẾM ==========
    with tab4:
        st.markdown("### 🔍 Tìm kiếm phim & đề xuất phim tương tự")
        
        # Lấy search term từ URL
        search_from_url = get_url_params()
        
        # Search bar
        search_col1, search_col2 = st.columns([3, 1])
        
        with search_col1:
            # Lấy giá trị hiện tại
            current_value = search_from_url if search_from_url else ""
            search_term = st.text_input(
                "Nhập tên phim bạn muốn tìm:",
                placeholder="Ví dụ: Toy Story, Inception, Titanic...",
                key="search_input",
                value=current_value
            )
        
        with search_col2:
            num_recommendations = st.selectbox(
                "Số phim tương tự:",
                [5, 10, 15, 20],
                index=1,
                key="num_recs"
            )
        
        # Nếu có search term, thực hiện tìm kiếm
        if search_term:
            # Cập nhật URL nếu search term thay đổi
            if search_term != search_from_url:
                set_url_params(search_term)
            
            # Thêm vào lịch sử
            if len(search_term.strip()) > 2:
                # Kiểm tra xem đã lưu chưa
                if not any(h.get('term') == search_term and h.get('user_id') == user_id 
                          for h in st.session_state.search_history[-10:]):
                    st.session_state.search_history.append({
                        'term': search_term,
                        'timestamp': datetime.now().isoformat(),
                        'user_id': user_id
                    })
                    save_session_state()
            
            with st.spinner("🔍 Đang tìm phim và đề xuất phim tương tự..."):
                # Tìm phim chính và phim tương tự
                found_movie, similar_movies, main_genre = find_movie_and_similar(
                    search_term, n_similar=num_recommendations
                )
                
                if found_movie is not None:
                    # PHẦN 1: HIỂN THỊ PHIM ĐÃ TÌM THẤY
                    st.markdown("""
                    <div class="found-movie-highlight">
                        <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937; margin-bottom: 10px;">
                            🎬 Phim tìm thấy
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_found1, col_found2 = st.columns([4, 1])
                    
                    with col_found1:
                        # Hiển thị thông tin phim tìm thấy
                        genres_display = found_movie['genres']
                        if isinstance(genres_display, str) and len(genres_display) > 100:
                            genres_display = genres_display[:100] + "..."
                        
                        st.markdown(f"""
                        <div class="search-movie-card">
                            <div style="font-size: 1.3rem; font-weight: 600; color: #4f46e5; margin-bottom: 10px;">
                                {found_movie['title']}
                            </div>
                            <div style="color: #6b7280; font-size: 1rem; margin-bottom: 10px;">
                                <strong>🎭 Thể loại:</strong> {genres_display}
                            </div>
                            <div style="color: #6b7280; font-size: 0.95rem; display: flex; gap: 20px;">
                                <div><strong>📅 Năm:</strong> {int(found_movie['year'])}</div>
                                <div><strong>⭐ Đánh giá:</strong> {found_movie['rating_count']:,}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_found2:
                        if st.button("➕ Lưu phim này", key="save_found_main", 
                                   use_container_width=True):
                            if user_id not in st.session_state.user_history:
                                st.session_state.user_history[user_id] = []
                            
                            st.session_state.user_history[user_id].append({
                                'movieId': int(found_movie['movieId']),
                                'title': found_movie['title'],
                                'timestamp': datetime.now().isoformat(),
                                'type': 'search_found',
                                'search_term': search_term
                            })
                            save_session_state()
                            st.success(f"Đã lưu '{found_movie['title'][:30]}...' vào lịch sử!")
                    
                    st.markdown("---")
                    
                    # PHẦN 2: HIỂN THỊ PHIM TƯƠNG TỰ
                    if len(similar_movies) > 0:
                        st.markdown(f"""
                        <div style="background: #f3f4f6; border-radius: 12px; padding: 20px; margin-bottom: 20px;">
                            <div style="font-size: 1.4rem; font-weight: 600; color: #1f2937; margin-bottom: 10px;">
                                🎯 {len(similar_movies)} phim tương tự cùng thể loại "{main_genre}"
                            </div>
                            <div style="color: #6b7280; font-size: 1rem;">
                                Các phim đề xuất dựa trên phim bạn đã tìm
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Hiển thị phim tương tự dạng grid
                        cols = st.columns(3)
                        for idx, (_, movie) in enumerate(similar_movies.iterrows()):
                            with cols[idx % 3]:
                                genres_short = movie['genres']
                                if isinstance(genres_short, str) and '|' in genres_short:
                                    genres_short = genres_short.split('|')[0]
                                if len(genres_short) > 20:
                                    genres_short = genres_short[:20] + "..."
                                
                                st.markdown("""
                                <div class="movie-card">
                                    <div style="background: linear-gradient(135deg, #10b981 0%, #34d399 100%); 
                                                height: 120px; border-radius: 8px; display: flex; align-items: center; 
                                                justify-content: center; color: white; font-weight: bold; 
                                                margin-bottom: 10px;">
                                        🎯
                                    </div>
                                    <div class="movie-title">{}</div>
                                    <div class="movie-info">📅 {} • 🎭 {}</div>
                                    <div class="movie-rating">⭐ {} đánh giá</div>
                                </div>
                                """.format(
                                    movie['title'][:25] + ("..." if len(movie['title']) > 25 else ""),
                                    int(movie['year']),
                                    genres_short,
                                    f"{movie['rating_count']:,}"
                                ), unsafe_allow_html=True)
                                
                                # Sử dụng key đơn giản
                                if st.button("➕ Lưu đề xuất", key=f"save_similar_{idx}", 
                                           use_container_width=True):
                                    if user_id not in st.session_state.user_history:
                                        st.session_state.user_history[user_id] = []
                                    
                                    st.session_state.user_history[user_id].append({
                                        'movieId': int(movie['movieId']),
                                        'title': movie['title'],
                                        'timestamp': datetime.now().isoformat(),
                                        'type': 'similar_movie',
                                        'search_term': search_term,
                                        'main_genre': main_genre
                                    })
                                    save_session_state()
                                    st.success(f"Đã lưu đề xuất!")
                    else:
                        st.info(f"Không tìm thấy phim tương tự nào cùng thể loại '{main_genre}'")
                else:
                    st.warning("Không tìm thấy phim nào. Hãy thử tên phim khác!")
        
        
        # LỊCH SỬ TÌM KIẾM
        with st.expander("📜 Lịch sử tìm kiếm của bạn", expanded=False):
            if st.session_state.search_history:
                search_history_filtered = [h for h in st.session_state.search_history if h.get('user_id') == user_id]
                
                if search_history_filtered:
                    st.markdown("**Tìm lại nhanh:**")
                    
                    # Hiển thị tối đa 5 lịch sử gần nhất
                    for idx, item in enumerate(reversed(search_history_filtered[-5:])):
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            time_str = datetime.fromisoformat(item['timestamp']).strftime("%H:%M %d/%m")
                            st.markdown(f"`{item['term'][:30]}` - *{time_str}*")
                        with col2:
                            if st.button("🔍 Tìm lại", key=f"re_{idx}"):
                                set_url_params(item['term'])
                else:
                    st.info("Chưa có lịch sử tìm kiếm")
            else:
                st.info("Chưa có lịch sử tìm kiếm")
      
# ========== LƯU DỮ LIỆU KHI THOÁT ==========
save_session_state()

# ========== FOOTER ==========
st.markdown("""
<div style="text-align: center; color: white; margin-top: 20px; padding: 15px;">
    <p>🎬 <strong>MovieRec</strong> - Hệ thống gợi ý phim đơn giản & thông minh</p>
    <p style="font-size: 0.9rem; opacity: 0.8;">Tìm phim theo thể loại • Xu hướng • Gợi ý theo lịch sử • Tìm kiếm & đề xuất phim tương tự</p>
</div>
""", unsafe_allow_html=True)