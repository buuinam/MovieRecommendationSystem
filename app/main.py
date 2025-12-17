# app/main.py
import download_data
download_data.download_if_needed()
import streamlit as st

st.set_page_config(
    page_title="HỆ THỐNG GỢI Ý PHIM",
    page_icon="🎬",
    layout="wide"
)

# ====== TIÊU ĐỀ ======
st.title("🎬 HỆ THỐNG GỢI Ý XEM PHIM THÔNG MINH")
st.markdown("**Final Project – Movie Recommendation System**")

st.markdown("""
Hệ thống được xây dựng nhằm hỗ trợ người dùng **tìm kiếm và lựa chọn phim phù hợp**
dựa trên sở thích cá nhân và dữ liệu hành vi xem phim.
""")

st.divider()


# ====== HƯỚNG DẪN ======
st.info("👉 Vui lòng chọn chức năng ở **thanh menu bên trái** để khám phá hệ thống.")
