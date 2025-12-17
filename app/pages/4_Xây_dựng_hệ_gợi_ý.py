# pages/4_Xây_dựng_hệ_gợi_ý.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.header("🎯 4. XÂY DỰNG MÔ HÌNH GỢI Ý (CONTENT-BASED FILTERING)")

# ========== TIÊU ĐỀ ==========
st.markdown("""
<div style="background:linear-gradient(90deg, #1E3A8A, #3B82F6); padding:25px; border-radius:10px; color:white; margin-bottom:20px;">
    <h2 style="text-align:center; margin:0;">🧠 MÔ HÌNH CONTENT-BASED FILTERING</h2>
    <p style="text-align:center; margin:10px 0 0 0; font-size:18px;">Sử dụng TF-IDF + Cosine Similarity</p>
</div>
""", unsafe_allow_html=True)

# ========== TẢI DỮ LIỆU ==========
@st.cache_data
def load_data():
    """Tải dữ liệu phim đã làm sạch"""
    try:
        movies = pd.read_csv("data/movies_final.csv")
        return movies
    except:
        # Tạo dữ liệu mẫu nếu file không tồn tại
        st.error("Không tìm thấy file dữ liệu. Tạo dữ liệu mẫu...")
        data = {
            'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'title': [
                'Toy Story (1995)', 
                'Jumanji (1995)', 
                'Grumpier Old Men (1995)', 
                'Waiting to Exhale (1995)', 
                'Father of the Bride Part II (1995)',
                'Heat (1995)',
                'Sabrina (1995)',
                'Tom and Huck (1995)',
                'Sudden Death (1995)',
                'GoldenEye (1995)'
            ],
            'genres': [
                'Adventure|Animation|Children|Comedy|Fantasy',
                'Adventure|Children|Fantasy',
                'Comedy|Romance',
                'Comedy|Drama|Romance',
                'Comedy',
                'Action|Crime|Thriller',
                'Comedy|Romance',
                'Adventure|Children',
                'Action',
                'Action|Adventure|Thriller'
            ],
            'year': [1995, 1995, 1995, 1995, 1995, 1995, 1995, 1995, 1995, 1995],
            'rating_count': [541, 471, 302, 251, 212, 204, 198, 195, 191, 185]
        }
        movies = pd.DataFrame(data)
        movies['content'] = movies['title'] + ' ' + movies['genres'].str.replace('|', ' ')
        return movies

movies = load_data()

# ========== HIỂN THỊ DỮ LIỆU ==========
st.markdown("### 📊 DỮ LIỆU ĐẦU VÀO")

col_data1, col_data2 = st.columns([1, 1])

with col_data1:
    st.metric("Tổng số phim", f"{len(movies):,}")
    st.metric("Số thể loại duy nhất", movies['genres'].str.split('|').explode().nunique())

with col_data2:
    st.metric("Năm đầu tiên", int(movies['year'].min()))
    st.metric("Năm cuối cùng", int(movies['year'].max()))

with st.expander("👀 Xem 10 phim đầu tiên"):
    st.dataframe(movies[['title', 'genres', 'year', 'rating_count']].head(10), 
                 use_container_width=True)

# ========== XÂY DỰNG MÔ HÌNH ==========
st.markdown("---")
st.markdown("## 🔨 XÂY DỰNG MÔ HÌNH")

# Tab cho từng bước
tab1, tab2, tab3, tab4 = st.tabs([
    "1. Chuẩn bị dữ liệu", 
    "2. TF-IDF Vectorizer", 
    "3. Cosine Similarity", 
    "4. Demo gợi Ý"
])

# ========== TAB 1: CHUẨN BỊ DỮ LIỆU ==========
with tab1:
    st.markdown("### 📝 BƯỚC 1: CHUẨN BỊ DỮ LIỆU")
    
    col_prep1, col_prep2 = st.columns(2)
    
    with col_prep1:
        st.markdown("""
        #### Vấn đề:
        - Dữ liệu văn bản (text) không thể tính toán trực tiếp
        - Cần chuyển đổi thành vector số
        
        #### Giải pháp:
        - Tạo cột `content` kết hợp:
          - Title của phim
          - Thể loại (genres)
        - Chuẩn hóa text (lowercase, xử lý đặc biệt)
        """)
        
        # Hiển thị ví dụ content
        if 'content' not in movies.columns:
            movies['content'] = movies['title'].fillna('') + ' ' + movies['genres'].str.replace('|', ' ', regex=False)
        
        st.markdown("#### Ví dụ cột `content`:")
        st.code(movies['content'].head(3).tolist())
    
    with col_prep2:
        st.markdown("#### Code xử lý:")
        st.code("""
# Chuẩn bị dữ liệu cho TF-IDF
def prepare_content(movies_df):
    # Tạo cột content: title + genres
    movies_df['content'] = (
        movies_df['title'].fillna('') + ' ' + 
        movies_df['genres'].str.replace('|', ' ', regex=False)
    )
    
    # Chuyển thành lowercase
    movies_df['content'] = movies_df['content'].str.lower()
    
    # Loại bỏ ký tự đặc biệt
    movies_df['content'] = movies_df['content'].str.replace(r'[^\w\s]', ' ', regex=True)
    
    return movies_df

# Áp dụng hàm
movies = prepare_content(movies)
print(f"Mẫu content: {movies['content'].iloc[0]}")
        """, language="python")
        
        if st.button("▶️ Áp dụng xử lý", key="prep_btn"):
            # Áp dụng xử lý
            movies['content'] = movies['title'].fillna('') + ' ' + movies['genres'].str.replace('|', ' ', regex=False)
            movies['content'] = movies['content'].str.lower()
            movies['content'] = movies['content'].str.replace(r'[^\w\s]', ' ', regex=True)
            
            st.success("✅ Đã chuẩn bị xong dữ liệu!")
            st.write("**Mẫu content sau xử lý:**")
            st.write(movies['content'].iloc[0][:100] + "...")

# ========== TAB 2: TF-IDF VECTORIZER ==========
with tab2:
    st.markdown("### 🔢 BƯỚC 2: TF-IDF VECTORIZER")
    
    col_tfidf1, col_tfidf2 = st.columns(2)
    
    with col_tfidf1:
        st.markdown("""
        #### TF-IDF là gì?
        
        **TF (Term Frequency):**
        - Tần suất từ xuất hiện trong văn bản
        - Từ càng xuất hiện nhiều càng quan trọng
        
        **IDF (Inverse Document Frequency):**
        - Độ hiếm của từ trong toàn bộ corpus
        - Từ xuất hiện ở nhiều văn bản sẽ có trọng số thấp
        
        #### Ưu điểm:
        - Đánh trọng số cho từ quan trọng
        - Giảm trọng số từ phổ biến (stopwords)
        - Chuyển text → vector số
        """)
        
        # Cấu hình TF-IDF
        st.markdown("#### ⚙️ Cấu hình:")
        max_features = st.slider("Số features tối đa:", 100, 10000, 5000, 100)
        ngram_range = st.selectbox("N-gram range:", ["(1,1) - Unigram", "(1,2) - Unigram+Bigram", "(1,3) - Unigram+Bigram+Trigram"])
        
        # Map selection
        ngram_map = {
            "(1,1) - Unigram": (1, 1),
            "(1,2) - Unigram+Bigram": (1, 2),
            "(1,3) - Unigram+Bigram+Trigram": (1, 3)
        }
    
    with col_tfidf2:
        st.markdown("#### Code TF-IDF:")
        st.code(f"""
from sklearn.feature_extraction.text import TfidfVectorizer

# Khởi tạo TF-IDF Vectorizer
tfidf = TfidfVectorizer(
    stop_words='english',      # Loại bỏ stopwords tiếng Anh
    max_features={max_features},     # Giới hạn số features
    ngram_range={ngram_map[ngram_range]},  # Xét 1-N từ liên tiếp
    min_df=2,                  # Từ phải xuất hiện ít nhất 2 lần
    max_df=0.95                # Tối đa 95% documents
)

# Áp dụng lên dữ liệu
tfidf_matrix = tfidf.fit_transform(movies['content'])

print(f"Shape của TF-IDF matrix: {{tfidf_matrix.shape}}")
print(f"Số từ vựng: {{len(tfidf.get_feature_names_out())}}")
        """, language="python")
        
        if st.button("▶️ Chạy TF-IDF", key="tfidf_btn"):
            with st.spinner("Đang chạy TF-IDF..."):
                try:
                    # Áp dụng TF-IDF
                    ngram = ngram_map[ngram_range]
                    tfidf = TfidfVectorizer(
                        stop_words='english',
                        max_features=max_features,
                        ngram_range=ngram,
                        min_df=2,
                        max_df=0.95
                    )
                    
                    # Tạo content nếu chưa có
                    if 'content' not in movies.columns:
                        movies['content'] = movies['title'].fillna('') + ' ' + movies['genres'].str.replace('|', ' ', regex=False)
                    
                    tfidf_matrix = tfidf.fit_transform(movies['content'])
                    
                    st.success("✅ TF-IDF hoàn thành!")
                    st.metric("Shape TF-IDF matrix", f"{tfidf_matrix.shape[0]} x {tfidf_matrix.shape[1]}")
                    st.metric("Số từ vựng", len(tfidf.get_feature_names_out()))
                    
                    # Hiển thị một số từ vựng
                    with st.expander("👁️ Xem một số từ vựng (features)"):
                        features = tfidf.get_feature_names_out()[:50]
                        st.write(", ".join(features))
                        
                    # Lưu vào session state để dùng sau
                    st.session_state['tfidf_matrix'] = tfidf_matrix
                    st.session_state['tfidf'] = tfidf
                    
                except Exception as e:
                    st.error(f"Lỗi: {e}")

# ========== TAB 3: COSINE SIMILARITY ==========
with tab3:
    st.markdown("### 📐 BƯỚC 3: COSINE SIMILARITY")
    
    col_cos1, col_cos2 = st.columns(2)
    
    with col_cos1:
        st.markdown("""
        #### Cosine Similarity là gì?
        
        **Công thức:**
        ```
        similarity = cos(θ) = (A·B) / (||A|| * ||B||)
        ```
        
        **Ý nghĩa:**
        - Đo góc giữa 2 vector
        - Range: [-1, 1]
        - 1: Hoàn toàn giống nhau
        - 0: Không liên quan
        - -1: Hoàn toàn ngược nhau
        
        **Ứng dụng:**
        - So sánh độ giống nhau giữa các phim
        - Tìm phim tương tự dựa trên content
        """)
        
        # Visualization
        st.markdown("#### 🎨 Minh họa:")
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Vẽ góc giữa 2 vector
        ax.arrow(0, 0, 0.8, 0.6, head_width=0.05, head_length=0.1, fc='blue', ec='blue', label='Vector A (Phim 1)')
        ax.arrow(0, 0, 0.4, 0.8, head_width=0.05, head_length=0.1, fc='red', ec='red', label='Vector B (Phim 2)')
        
        # Góc giữa 2 vector
        ax.text(0.3, 0.3, 'θ', fontsize=20)
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('Cosine Similarity = cos(θ)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
    
    with col_cos2:
        st.markdown("#### Code Cosine Similarity:")
        st.code("""
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Tính cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print(f"Shape của Cosine Similarity matrix: {cosine_sim.shape}")
print(f"Kiểu dữ liệu: {cosine_sim.dtype}")

# Ma trận đối xứng
print(f"Đối xứng: {np.allclose(cosine_sim, cosine_sim.T)}")

# Diagonal = 1 (tự so với chính nó)
print(f"Đường chéo toàn 1: {np.allclose(np.diag(cosine_sim), 1)}")

# Lấy similarity cho một phim cụ thể
movie_idx = 0  # Toy Story
similarities = cosine_sim[movie_idx]
print(f"Similarities với phim đầu tiên: {similarities[:5]}")
        """, language="python")
        
        if st.button("▶️ Tính Cosine Similarity", key="cos_btn"):
            if 'tfidf_matrix' not in st.session_state:
                st.warning("⚠️ Cần chạy TF-IDF trước!")
            else:
                with st.spinner("Đang tính Cosine Similarity..."):
                    try:
                        # Tính cosine similarity
                        from sklearn.metrics.pairwise import cosine_similarity
                        cosine_sim = cosine_similarity(st.session_state['tfidf_matrix'])
                        
                        st.success("✅ Cosine Similarity hoàn thành!")
                        st.metric("Shape matrix", f"{cosine_sim.shape[0]} x {cosine_sim.shape[1]}")
                        
                        # Hiển thị ví dụ
                        st.markdown("#### 🎯 Ví dụ: Similarity matrix (5x5 đầu tiên)")
                        st.dataframe(
                            pd.DataFrame(
                                cosine_sim[:5, :5],
                                index=movies['title'].head(5),
                                columns=movies['title'].head(5)
                            ).round(3),
                            use_container_width=True
                        )
                        
                        # Lưu vào session state
                        st.session_state['cosine_sim'] = cosine_sim
                        
                        # Lưu vào file
                        with open('data/cosine_sim.pkl', 'wb') as f:
                            pickle.dump(cosine_sim, f)
                        st.info("💾 Đã lưu cosine_sim vào data/cosine_sim.pkl")
                        
                    except Exception as e:
                        st.error(f"Lỗi: {e}")

# ========== TAB 4: DEMO GỢI Ý ==========
with tab4:
    st.markdown("### 🎬 BƯỚC 4: DEMO HỆ THỐNG GỢI Ý")
    
    col_demo1, col_demo2 = st.columns([3, 2])
    
    with col_demo1:
        st.markdown("#### 🔍 TÌM PHIM TƯƠNG TỰ")
        
        # Chọn phim gốc
        base_movie = st.selectbox(
            "Chọn phim bạn thích:",
            movies['title'].tolist(),
            index=0,
            help="Chọn một phim để tìm phim tương tự"
        )
        
        # Số phim tương tự
        num_recommendations = st.slider("Số phim tương tự:", 3, 20, 10)
        
        # Tùy chọn filter
        st.markdown("##### 🎛️ Tùy chọn lọc")
        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            min_similarity = st.slider("Độ tương đồng tối thiểu:", 0.0, 1.0, 0.3, 0.05)
        with col_filter2:
            same_year = st.checkbox("Chỉ phim cùng năm", value=False)
    
    with col_demo2:
        st.markdown("#### 📊 THÔNG TIN PHIM")
        
        if base_movie:
            movie_info = movies[movies['title'] == base_movie]
            if len(movie_info) > 0:
                movie = movie_info.iloc[0]
                
                st.write(f"**🎬 {movie['title']}**")
                st.write(f"🎭 **Thể loại:** {movie['genres']}")
                st.write(f"📅 **Năm:** {int(movie['year'])}")
                st.write(f"⭐ **Số rating:** {movie['rating_count']:,}")
                
                # Highlight genres
                genres_list = movie['genres'].split('|')
                st.write("**🏷️ Tags:**")
                for genre in genres_list:
                    st.markdown(f"`{genre}` ", unsafe_allow_html=True)
    
    # Nút tìm kiếm
    if st.button("🔍 TÌM PHIM TƯƠNG TỰ", type="primary", use_container_width=True):
        if 'cosine_sim' not in st.session_state:
            st.warning("⚠️ Cần tính Cosine Similarity trước!")
        else:
            try:
                # Tìm index của phim
                movie_idx = movies[movies['title'] == base_movie].index[0]
                
                # Lấy similarity scores
                sim_scores = list(enumerate(st.session_state['cosine_sim'][movie_idx]))
                
                # Sắp xếp theo similarity
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                
                # Lấy top N (bỏ qua chính nó)
                sim_scores = sim_scores[1:num_recommendations*2]
                
                # Lọc theo similarity
                filtered_scores = []
                for idx, score in sim_scores:
                    if score >= min_similarity:
                        if same_year:
                            if movies.iloc[idx]['year'] == movies.iloc[movie_idx]['year']:
                                filtered_scores.append((idx, score))
                        else:
                            filtered_scores.append((idx, score))
                    
                    if len(filtered_scores) >= num_recommendations:
                        break
                
                # Hiển thị kết quả
                if filtered_scores:
                    st.success(f"### 🎯 TÌM THẤY {len(filtered_scores)} PHIM TƯƠNG TỰ")
                    
                    # Tạo DataFrame kết quả
                    results = []
                    for i, (idx, score) in enumerate(filtered_scores, 1):
                        movie = movies.iloc[idx]
                        results.append({
                            'STT': i,
                            'Phim': movie['title'],
                            'Thể loại': movie['genres'],
                            'Năm': int(movie['year']),
                            'Độ tương đồng': f"{score:.3f}",
                            'Số rating': f"{movie['rating_count']:,}"
                        })
                    
                    results_df = pd.DataFrame(results)
                    
                    # Hiển thị dạng bảng
                    st.dataframe(
                        results_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Độ tương đồng": st.column_config.ProgressColumn(
                                format="%.3f",
                                min_value=0,
                                max_value=1
                            )
                        }
                    )
                    
                    # Hiển thị dạng visual
                    st.markdown("#### 📊 BIỂU ĐỒ ĐỘ TƯƠNG ĐỒNG")
                    
                    # Tạo biểu đồ
                    fig, ax = plt.subplots(figsize=(10, 6))
                    movies_list = [movies.iloc[idx]['title'][:20] + "..." for idx, _ in filtered_scores]
                    similarity_scores = [score for _, score in filtered_scores]
                    
                    colors = plt.cm.YlOrRd(similarity_scores)  # Màu theo độ tương đồng
                    
                    bars = ax.barh(movies_list, similarity_scores, color=colors)
                    ax.set_xlabel('Độ tương đồng (Cosine Similarity)')
                    ax.set_title('Top phim tương tự với "' + base_movie[:30] + '"')
                    ax.set_xlim(0, 1)
                    
                    # Thêm giá trị trên mỗi bar
                    for bar, score in zip(bars, similarity_scores):
                        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                               f'{score:.3f}', va='center')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Thống kê
                    avg_similarity = np.mean(similarity_scores)
                    st.metric("Độ tương đồng trung bình", f"{avg_similarity:.3f}")
                    
                else:
                    st.warning("Không tìm thấy phim tương tự nào đạt ngưỡng similarity.")
                    
            except Exception as e:
                st.error(f"Lỗi: {e}")

# ========== TỔNG KẾT MÔ HÌNH ==========
st.markdown("---")
st.markdown("## 🏆 TỔNG KẾT MÔ HÌNH")

col_summary1, col_summary2 = st.columns(2)

with col_summary1:
    st.markdown("""
    ### ✅ ƯU ĐIỂM MÔ HÌNH
    
    1. **Không cần dữ liệu người dùng:**
       - Chỉ cần metadata của phim
       - Không bị cold-start problem
       
    2. **Giải thích được:**
       - Dựa trên thể loại, nội dung
       - User hiểu tại sao được gợi Ý
       
    3. **Đơn giản, hiệu quả:**
       - Dễ triển khai
       - Tính toán nhanh
       - Phù hợp với hệ thống nhỏ
    """)

with col_summary2:
    st.markdown("""
    ### ⚠️ HẠN CHẾ & GIẢI PHÁP
    
    1. **Limited diversity:**
       - Gợi Ý phim quá giống nhau
       - **Giải pháp:** Thêm serendipity factor
       
    2. **Không cá nhân hóa sâu:**
       - Mọi user cùng xem 1 phim sẽ nhận gợi Ý giống nhau
       - **Giải pháp:** Kết hợp Collaborative Filtering
       
    3. **Phụ thuộc metadata:**
       - Cần metadata chất lượng
       - **Giải pháp:** Làm sạch dữ liệu kỹ
    """)

# ========== DOWNLOAD MÔ HÌNH ==========
st.markdown("---")
st.markdown("### 💾 LƯU VÀ TẢI MÔ HÌNH")

if st.button("💾 LƯU MÔ HÌNH HOÀN CHỈNH", type="primary"):
    try:
        # Lưu model và dữ liệu
        model_data = {
            'movies': movies,
            'cosine_sim': st.session_state.get('cosine_sim', None),
            'tfidf': st.session_state.get('tfidf', None)
        }
        
        with open('data/model_content_based.pkl', 'wb') as f:
            pickle.dump(model_data, f)
        
        st.success("✅ Đã lưu mô hình hoàn chỉnh vào data/model_content_based.pkl")
        st.balloons()
        
        # Hiển thị thông tin
        st.info("""
        **📦 Các file đã lưu:**
        1. `data/movies_final.csv` - Dữ liệu phim
        2. `data/cosine_sim.pkl` - Ma trận similarity
        3. `data/model_content_based.pkl` - Full model
        
        **🚀 Mô hình sẵn sàng cho:**
        - Triển khai hệ thống gợi Ý
        - Tích hợp vào ứng dụng
        - Demo cho người dùng
        """)
        
    except Exception as e:
        st.error(f"Lỗi khi lưu model: {e}")

# ========== NEXT STEPS ==========
st.markdown("---")
st.markdown("### 📈 BƯỚC TIẾP THEO")

col_next1, col_next2, col_next3 = st.columns(3)

with col_next1:
    st.markdown("""
    #### 🧪 ĐÁNH GIÁ MÔ HÌNH
    - Precision@K, Recall@K
    - A/B testing
    - User feedback
    """)

with col_next2:
    st.markdown("""
    #### 🔗 KẾT HỢP MÔ HÌNH
    - Hybrid với Collaborative
    - Thêm popularity factor
    - Time-based filtering
    """)

with col_next3:
    st.markdown("""
    #### 🚀 TRIỂN KHAI
    - API endpoints
    - Real-time recommendations
    - Scaling với Spark
    """)

st.success("""
### 🎉 HOÀN THÀNH XÂY DỰNG MÔ HÌNH CONTENT-BASED FILTERING!

**📊 THÀNH QUẢ:**
✅ Đã xây dựng pipeline hoàn chỉnh  
✅ Xử lý dữ liệu với TF-IDF  
✅ Tính toán Cosine Similarity  
✅ Demo hệ thống gợi Ý phim tương tự  
✅ Lưu model để tái sử dụng  

**🎯 SẴN SÀNG CHO:** Đánh giá mô hình và tích hợp hệ thống!
""")