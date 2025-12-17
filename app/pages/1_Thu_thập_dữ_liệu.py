# app/pages/1_Thu_thập_dữ_liệu.py – BẢN ĐẸP NHẤT, TỐI ƯU, XỬ LÝ LỖI TỐT
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.header("📊 1. THU THẬP DỮ LIỆU")

# ========== TIÊU ĐỀ ĐẸP VỚI ICON ==========
col_title = st.columns([1, 8, 1])
with col_title[1]:
    st.markdown("""
    <div style="background:linear-gradient(90deg, #1E3A8A, #3B82F6); padding:20px; border-radius:10px; color:white;">
        <h2 style="text-align:center; margin:0;">🎬 HỆ THỐNG GỢI Ý PHIM - DATASET MOVIELENS</h2>
    </div>
    """, unsafe_allow_html=True)

# ========== LOAD DỮ LIỆU VỚI XỬ LÝ LỖI ==========
@st.cache_data
def load_data():
    try:
        movies = pd.read_csv("data/movies_final.csv")
        ratings = pd.read_csv("data/ratings.csv")
        return movies, ratings
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {e}")
        return None, None

movies, ratings = load_data()

if movies is None or ratings is None:
    st.warning("Không thể tải dữ liệu. Kiểm tra đường dẫn file.")
    st.stop()

# ========== BỐ CỤC CHÍNH: 2 CỘT ==========
col1, col2 = st.columns([3, 7])

with col1:
    # PHẦN THÔNG TIN NGUỒN
    st.markdown("### 📋 THÔNG TIN DATASET")
    
    # Card thông tin
    st.markdown("""
    <div style="background:#f07; padding:15px; border-radius:10px; border-left:5px solid #3B82F6;">
        <h4 style="margin:0; color:#1E3A8A;">🎯 MovieLens 20M Dataset</h4>
        <p style="margin:5px 0;">Nguồn: <b>grouplens.org</b></p>
        <p style="margin:5px 0;">📅 Từ: 1995-2015</p>
        <p style="margin:5px 0;">👥 138,000+ người dùng</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== THỐNG KÊ NHANH - PHIÊN BẢN ĐẸP ==========
    st.markdown("### 📈 THỐNG KÊ NHANH")
    
    # Tính toán các thống kê
    total_movies = len(movies)
    total_ratings = len(ratings)
    total_users = ratings['userId'].nunique()
    min_year = int(movies['year'].min())
    max_year = int(movies['year'].max())
    avg_rating = ratings['rating'].mean()
    rating_std = ratings['rating'].std()
    
    # Tạo 4 cards đẹp với màu sắc khác nhau
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        # Card 1: Số phim
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 15px; 
                    border-radius: 10px; 
                    color: white;
                    margin-bottom: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 24px; margin-right: 15px;">🎬</div>
                <div>
                    <div style="font-size: 15px; opacity: 0.9;">SỐ PHIM</div>
                    <div style="font-size: 15px; font-weight: bold;">{total_movies:,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 2: Số rating
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
                    padding: 15px; 
                    border-radius: 10px; 
                    color: white;
                    margin-bottom: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 12px; margin-right: 15px;">⭐</div>
                <div>
                    <div style="font-size: 15px; opacity: 0.9;">SỐ RATING</div>
                    <div style="font-size: 15px; font-weight: bold;">{total_ratings:,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_stat2:
        # Card 3: Số người dùng
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                    padding: 15px; 
                    border-radius: 10px; 
                    color: white;
                    margin-bottom: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 24px; margin-right: 15px;">👤</div>
                <div>
                    <div style="font-size: 15px; opacity: 0.9;">NGƯỜI DÙNG</div>
                    <div style="font-size: 15px; font-weight: bold;">{total_users:,}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card 4: Năm phim
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); 
                    padding: 15px; 
                    border-radius: 10px; 
                    color: white;
                    margin-bottom: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            <div style="display: flex; align-items: center;">
                <div style="font-size: 24px; margin-right: 15px;">📅</div>
                <div>
                    <div style="font-size: 15px; opacity: 0.9;">NĂM PHIM</div>
                    <div style="font-size: 15px; font-weight: bold;">{min_year}-{max_year}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Thêm thống kê phụ dưới dạng text
    st.markdown(f"""
    <div style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px;">
        <div style="font-size: 14px; color: #666;">
            📊 <b>Rating trung bình:</b> {avg_rating:.2f} ⭐ (độ lệch: {rating_std:.2f})<br>
            🎯 <b>Mỗi phim có:</b> ~{total_ratings/total_movies:.0f} rating<br>
            👥 <b>Mỗi người dùng:</b> ~{total_ratings/total_users:.0f} rating
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== THÊM BIỂU ĐỒ NHỎ ==========
    with st.expander("📊 XEM BIỂU ĐỒ THỐNG KÊ"):
        # Phân phối năm phát hành
        year_counts = movies['year'].value_counts().sort_index()
        fig_year = go.Figure(data=[go.Bar(x=year_counts.index, y=year_counts.values)])
        fig_year.update_layout(
            title="Phân phối phim theo năm",
            xaxis_title="Năm",
            yaxis_title="Số phim",
            height=300
        )
        st.plotly_chart(fig_year, use_container_width=True)
        
        # Top thể loại
        all_genres = movies['genres'].str.split('|').explode()
        top_genres = all_genres.value_counts().head(10)
        fig_genre = go.Figure(data=[go.Bar(x=top_genres.values, y=top_genres.index, orientation='h')])
        fig_genre.update_layout(
            title="Top 10 thể loại phim",
            xaxis_title="Số phim",
            yaxis_title="Thể loại",
            height=300
        )
        st.plotly_chart(fig_genre, use_container_width=True)
    
    # TIẾN TRÌNH ĐẠT YÊU CẦU
    st.markdown("### ✅ KIỂM TRA YÊU CẦU")
    
    # Tạo checklist
    requirements = {
        "Dataset ≥ 2,000 items": len(movies) >= 2000,
        "Có ≥ 5 features": len(movies.columns) >= 5,
        "Có dữ liệu rating": len(ratings) > 0,
        "Có đa thể loại": movies['genres'].str.contains('|').any()
    }
    
    for req, status in requirements.items():
        if status:
            st.markdown(f"<div style='color: green;'>✓ {req}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color: red;'>✗ {req}</div>", unsafe_allow_html=True)
    
    # ĐÁNH GIÁ TỔNG QUAN
    score = sum(requirements.values()) / len(requirements) * 100
    st.progress(int(score)/100)
    st.markdown(f"<div style='text-align: center; font-weight: bold; color: {'green' if score == 100 else 'orange'};'>Đạt {score:.0f}% yêu cầu</div>", unsafe_allow_html=True)

with col2:
    # PHẦN HIỂN THỊ DỮ LIỆU CHI TIẾT
    tab1, tab2, tab3 = st.tabs(["🎬 BẢNG PHIM", "⭐ BẢNG RATING", "🔍 XEM CHI TIẾT"])
    
    with tab1:
        st.markdown(f"### Movies Dataset: **{len(movies):,}** phim")
        
        # Tùy chọn xem
        view_option = st.radio(
            "Hiển thị:",
            ["10 dòng đầu", "10 dòng cuối", "Mẫu ngẫu nhiên"],
            horizontal=True
        )
        
        if view_option == "10 dòng đầu":
            data_to_show = movies.head(10)
        elif view_option == "10 dòng cuối":
            data_to_show = movies.tail(10)
        else:
            data_to_show = movies.sample(10)
        
        # Hiển thị bảng với định dạng đẹp
        st.dataframe(
            data_to_show[['movieId', 'title', 'genres', 'year', 'rating_count']],
            use_container_width=True,
            hide_index=True,
            column_config={
                "movieId": "ID Phim",
                "title": "Tên Phim",
                "genres": "Thể Loại",
                "year": "Năm",
                "rating_count": "Số Rating"
            }
        )
        
        # Thông tin cột
        with st.expander("📋 THÔNG TIN CÁC CỘT TRONG MOVIES"):
            st.markdown("""
            | Cột | Mô tả | Ví dụ |
            |------|--------|-------|
            | **movieId** | ID duy nhất của phim | 1, 2, 3... |
            | **title** | Tên phim + năm | "Toy Story (1995)" |
            | **genres** | Các thể loại, phân cách bằng \| | "Adventure\|Animation\|Children" |
            | **year** | Năm phát hành | 1995 |
            | **rating_count** | Số lượng đánh giá | 541 |
            | **content** | Thông tin tổng hợp để xử lý | "toy story 1995 adventure animation children" |
            """)
    
    with tab2:
        st.markdown(f"### Ratings Dataset: **{len(ratings):,}** đánh giá")
        
        # Thống kê rating chi tiết
        rating_stats = ratings['rating'].describe()
        
        col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
        with col_stats1:
            st.metric("Rating trung bình", f"{rating_stats['mean']:.2f}", f"±{rating_stats['std']:.2f}")
        with col_stats2:
            st.metric("Rating nhỏ nhất", f"{rating_stats['min']:.1f}")
        with col_stats3:
            st.metric("Rating lớn nhất", f"{rating_stats['max']:.1f}")
        with col_stats4:
            st.metric("Median", f"{rating_stats['50%']:.1f}")
        
        # Hiển thị bảng rating
        st.dataframe(
            ratings[['userId', 'movieId', 'rating', 'timestamp']].head(10),
            use_container_width=True,
            hide_index=True,
            column_config={
                "userId": "ID Người dùng",
                "movieId": "ID Phim",
                "rating": "Điểm (0.5-5.0)",
                "timestamp": "Thời gian"
            }
        )
        
        # Phân phối rating với Plotly
        with st.expander("📊 PHÂN PHỐI RATING CHI TIẾT"):
            rating_dist = ratings['rating'].value_counts().sort_index()
            fig = px.bar(x=rating_dist.index, y=rating_dist.values,
                        labels={'x': 'Rating', 'y': 'Số lượng'},
                        title='Phân phối điểm rating')
            fig.update_traces(marker_color='#3B82F6')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔎 TÌM KIẾM PHIM THEO TIÊU CHÍ")
        
        col_search1, col_search2 = st.columns(2)
        with col_search1:
            search_term = st.text_input("Tìm theo tên phim:", placeholder="Nhập từ khóa...")
        with col_search2:
            min_year = int(movies['year'].min())
            max_year = int(movies['year'].max())
            year_range = st.slider("Năm phát hành:", min_year, max_year, (1990, 2010))
        
        # Lọc dữ liệu
        filtered_movies = movies.copy()
        
        if search_term:
            filtered_movies = filtered_movies[filtered_movies['title'].str.contains(search_term, case=False, na=False)]
        
        filtered_movies = filtered_movies[
            (filtered_movies['year'] >= year_range[0]) & 
            (filtered_movies['year'] <= year_range[1])
        ]
        
        st.markdown(f"**Tìm thấy:** <span style='color: #3B82F6; font-weight: bold;'>{len(filtered_movies):,}</span> phim", unsafe_allow_html=True)
        
        if len(filtered_movies) > 0:
            st.dataframe(
                filtered_movies[['title', 'genres', 'year', 'rating_count']].head(20),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "title": "Tên Phim",
                    "genres": "Thể Loại",
                    "year": "Năm",
                    "rating_count": "Số Rating"
                }
            )
        else:
            st.info("Không tìm thấy phim nào phù hợp với tiêu chí")

# ========== KẾT LUẬN ==========
st.markdown("---")
st.markdown("### 📋 KẾT LUẬN PHẦN THU THẬP DỮ LIỆU")

# Tạo layout 3 cột cho kết luận
col_con1, col_con2, col_con3 = st.columns(3)

with col_con1:
    st.success("✅ **Nguồn dữ liệu uy tín**")
    st.write("• Dataset từ **MovieLens** - trường Đại học Minnesota")
    st.write("• Được sử dụng rộng rãi trong nghiên cứu")

with col_con2:
    st.success("✅ **Quy mô đủ lớn**")
    st.write(f"• **{len(movies):,}** phim (vượt yêu cầu ≥2,000)")
    st.write(f"• **{len(ratings):,}** đánh giá")
    st.write(f"• **{ratings['userId'].nunique():,}** người dùng")

with col_con3:
    st.success("✅ **Đặc trưng đầy đủ**")
    st.write("• Có đủ **5+ features** mô tả item")
    st.write("• Có dữ liệu người dùng (ratings)")
    st.write("• Có metadata phim (genres, year)")

# ========== DOWNLOAD DATASET MẪU ==========
st.markdown("---")
st.markdown("### 📥 TẢI DATASET MẪU")

# Tạo sample dataset để download
@st.cache_data
def convert_df(df):
    return df.head(100).to_csv(index=False).encode('utf-8')

csv_movies = convert_df(movies)
csv_ratings = convert_df(ratings)

col_dl1, col_dl2, col_dl3 = st.columns([2, 1, 1])
with col_dl1:
    st.markdown("Tải dataset mẫu (100 dòng đầu) để kiểm tra:")
with col_dl2:
    st.download_button(
        label="📥 Movies.csv",
        data=csv_movies,
        file_name="movies_sample.csv",
        mime="text/csv",
    )
with col_dl3:
    st.download_button(
        label="📥 Ratings.csv",
        data=csv_ratings,
        file_name="ratings_sample.csv",
        mime="text/csv",
    )

# ========== FOOTER ==========
st.markdown("---")
st.caption("🎯 **Dữ liệu đã sẵn sàng cho bước tiếp theo: LÀM SẠCH & CHUẨN BỊ**")