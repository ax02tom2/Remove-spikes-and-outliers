import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re

st.set_page_config(page_title="監測儀器濾波與異常值消除工具", page_icon="🎛️", layout="wide")
st.title("🎛️ 監測資料濾波與突波消除專用工具")
st.write("上傳監測儀器原始檔，透過不同演算法找出異常突波並自動修補，最後匯出乾淨的 CSV 檔案。")

# --- 讀取與清理資料模組 ---
@st.cache_data
def load_data(file):
    try:
        df = pd.read_csv(file, encoding='utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        try:
            df = pd.read_csv(file, encoding='big5')
        except UnicodeDecodeError:
            file.seek(0)
            df = pd.read_csv(file, encoding='cp950', errors='ignore')

    time_col = df.columns[0]
    
    # 強健的時間清理機制 (應付各種亂碼與中文時分秒)
    def clean_time(t):
        t = str(t)
        t = t.replace('®É', ':').replace('¤À', ':').replace('¬í', '')
        t = t.replace('時', ':').replace('分', ':').replace('秒', '')
        cleaned = re.sub(r'[^\d/\-\: ]', ' ', t)
        return re.sub(r'\s+', ' ', cleaned).strip()

    df[time_col] = df[time_col].apply(clean_time)
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', format='mixed')
    
    # 數值轉換
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    return df.dropna(subset=[time_col]).sort_values(by=time_col).reset_index(drop=True), time_col

# --- 左側面板：上傳與設定 ---
st.sidebar.header("📁 1. 資料上傳")
uploaded_file = st.sidebar.file_uploader("上傳原始 CSV 檔", type=["csv"], help="第一欄需為時間，其餘為數值")

if uploaded_file:
    with st.spinner("正在解析資料..."):
        df, time_col = load_data(uploaded_file)
    
    if df.empty:
        st.error("⚠️ 檔案解析失敗，請確認時間格式。")
        st.stop()

    # 選擇要濾波的欄位
    val_columns = [col for col in df.columns if col != time_col]
    target_col = st.sidebar.selectbox("選擇要濾波的數值欄位", val_columns)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ 2. 濾波演算法設定")
    filter_method = st.sidebar.radio("選擇濾波方式", ["移動標準差 (抓取瞬間突波)", "上下限絕對值過濾 (抓取超界值)"])
    
    # 複製一份 DataFrame 用來標記異常
    df_clean = df.copy()
    df_clean['is_outlier'] = False
    
    if filter_method == "上下限絕對值過濾 (抓取超界值)":
        st.sidebar.write("💡 將超出設定範圍的數值判定為異常。")
        min_val = st.sidebar.number_input("合理最小值", value=float(df[target_col].min() - 5))
        max_val = st.sidebar.number_input("合理最大值", value=float(df[target_col].max() + 5))
        
        # 標記異常
        df_clean.loc[(df_clean[target_col] < min_val) | (df_clean[target_col] > max_val), 'is_outlier'] = True
        
    elif filter_method == "移動標準差 (抓取瞬間突波)":
        st.sidebar.write("💡 透過計算前後時段的標準差，自動抓出突然飆高/驟降的雜訊。")
        window_size = st.sidebar.slider("移動視窗大小 (筆數)", min_value=5, max_value=200, value=24, step=1, 
                                        help="數值越大，代表參考前後越長的時間範圍。")
        z_score = st.sidebar.slider("標準差容忍倍數 (Z-Score)", min_value=0.5, max_value=5.0, value=3.0, step=0.1,
                                    help="數值越小，過濾越嚴格，越容易把波動當作突波。")
        
        # 計算移動平均與標準差
        rolling_mean = df_clean[target_col].rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = df_clean[target_col].rolling(window=window_size, center=True, min_periods=1).std()
        
        # 標記異常 (偏離平均值超過 Z 倍標準差)
        diff = (df_clean[target_col] - rolling_mean).abs()
        df_clean.loc[diff > (z_score * rolling_std), 'is_outlier'] = True

    st.sidebar.markdown("---")
    st.sidebar.header("🩹 3. 數值修復方式")
    fill_method = st.sidebar.selectbox("異常值被挖除後，如何填補？", ["線性內插 (Linear Interpolation)", "使用前一筆正常數值填補 (Forward Fill)", "不填補 (保留空白 NaN)"])
    
    # 進行資料修補
    # 1. 先把異常值變成 NaN
    df_clean.loc[df_clean['is_outlier'], 'Cleaned_Value'] = np.nan
    # 2. 把正常的數值放進去
    df_clean.loc[~df_clean['is_outlier'], 'Cleaned_Value'] = df_clean[target_col]
    
    # 3. 執行填補
    if fill_method == "線性內插 (Linear Interpolation)":
        df_clean['Cleaned_Value'] = df_clean['Cleaned_Value'].interpolate(method='linear')
    elif fill_method == "使用前一筆正常數值填補 (Forward Fill)":
        df_clean['Cleaned_Value'] = df_clean['Cleaned_Value'].ffill()

    # --- 主畫面：儀表板與圖表 ---
    outlier_count = df_clean['is_outlier'].sum()
    total_count = len(df_clean)
    outlier_pct = (outlier_count / total_count) * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("總資料筆數", f"{total_count:,} 筆")
    col2.metric("抓出的異常突波", f"{outlier_count:,} 筆", f"{outlier_pct:.2f}% 剔除率", delta_color="inverse")
    
    st.markdown("### 🔍 濾波效果比對圖")
    st.write("淺藍色實線為修復後的資料；🔴 紅色點為被系統判定為異常並剔除的原始突波。")
    
    # 繪製比對圖
    fig = go.Figure()
    
    # 1. 畫出乾淨/修復後的線
    fig.add_trace(
        go.Scatter(
            x=df_clean[time_col], y=df_clean['Cleaned_Value'], 
            mode='lines', name="修復後平滑資料", 
            line=dict(color="#1f77b4", width=2)
        )
    )
    
    # 2. 畫出被剔除的異常紅點
    df_outliers = df_clean[df_clean['is_outlier']]
    if not df_outliers.empty:
        fig.add_trace(
            go.Scatter(
                x=df_outliers[time_col], y=df_outliers[target_col], 
                mode='markers', name="被剔除的異常突波", 
                marker=dict(color="red", size=6, symbol="x")
            )
        )
        
    fig.update_layout(
        template="plotly_white",
        hovermode="x unified",
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="時間",
        yaxis_title="監測數值"
    )
    
    # 提供手動固定 Y 軸功能，方便檢視細節
    use_manual_y = st.checkbox("手動鎖定 Y 軸範圍 (方便放大檢視不跑版)")
    if use_manual_y:
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            y_min = st.number_input("Y 軸下限", value=float(df_clean['Cleaned_Value'].min() - 2))
        with col_y2:
            y_max = st.number_input("Y 軸上限", value=float(df_clean['Cleaned_Value'].max() + 2))
        fig.update_yaxes(range=[y_min, y_max])
    else:
        fig.update_yaxes(autorange=True)
        
    st.plotly_chart(fig, use_container_width=True)
    
    # --- 匯出資料 ---
    st.markdown("---")
    st.markdown("### 📥 匯出乾淨資料")
    
    # 整理準備匯出的格式 (用修補後的值取代原始值，並移除輔助運算欄位)
    df_export = df.copy()
    df_export[target_col] = df_clean['Cleaned_Value']
    
    st.dataframe(df_export.head(10), use_container_width=True)
    st.caption("▲ 上表為匯出資料的前 10 筆預覽")
    
    csv_output = df_export.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label=f"📥 下載濾波修正檔 (Cleaned_{target_col}.csv)",
        data=csv_output,
        file_name=f"Cleaned_{target_col}.csv",
        mime="text/csv"
    )

else:
    st.info("👈 請先由左側面板上傳要濾波的 CSV 檔案。")