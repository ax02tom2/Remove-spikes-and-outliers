import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import re

st.set_page_config(page_title="監測儀器濾波與異常值消除工具", page_icon="🎛️", layout="wide")
st.title("🎛️ 監測資料濾波與突波消除專用工具")
st.write("上傳監測儀器原始檔，支援多種濾波演算法、手動區間剔除與平滑化處理，適用於水位、GPS、伸縮計等各類訊號。")

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
    
    def clean_time(t):
        t = str(t)
        t = t.replace('®É', ':').replace('¤À', ':').replace('¬í', '')
        t = t.replace('時', ':').replace('分', ':').replace('秒', '')
        cleaned = re.sub(r'[^\d/\-\: ]', ' ', t)
        return re.sub(r'\s+', ' ', cleaned).strip()

    df[time_col] = df[time_col].apply(clean_time)
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce', format='mixed')
    
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

    val_columns = [col for col in df.columns if col != time_col]
    target_col = st.sidebar.selectbox("選擇要濾波的數值欄位", val_columns)
    
    st.sidebar.markdown("---")
    st.sidebar.header("🎛️ 2. 濾波演算法設定")
    
    filter_options = [
        "移動中位數檢定 (Hampel Filter) - 推薦！對付連續突波剋星",
        "變化率過濾法 (Rate of Change) - 強制消除異常跳水",
        "相鄰突變檢測 (僅抓單一獨立突波)",
        "上下限絕對值過濾 (抓取超界值)"
    ]
    filter_method = st.sidebar.radio("選擇濾波方式", filter_options)
    
    df_clean = df.copy()
    df_clean['is_outlier'] = False
    
    # 演算法 1：Hampel Filter
    if filter_method == "移動中位數檢定 (Hampel Filter) - 推薦！對付連續突波剋星":
        # 放大 window 上限至 2000，以應付大面積的區塊型當機或異常
        window_size = st.sidebar.slider("移動視窗大小 (筆數)", min_value=5, max_value=2000, value=24, help="遇到連續較長的異常區塊時，請把此數值拉大 (例如拉到 100~500 以上)。")
        z_score = st.sidebar.slider("嚴格程度 (Z-Score 倍數)", min_value=0.1, max_value=10.0, value=3.0, step=0.1, help="數值越小，過濾越嚴格。")
        
        rolling_median = df_clean[target_col].rolling(window=window_size, center=True, min_periods=1).median()
        mad = (df_clean[target_col] - rolling_median).abs().rolling(window=window_size, center=True, min_periods=1).median()
        
        # 使用極微小的容忍基準，適應精密儀器的微小數值變動
        base_tol = df[target_col].std() * 0.001 if pd.notnull(df[target_col].std()) else 1e-6
        threshold = np.maximum(z_score * 1.4826 * mad, base_tol)
        
        diff = (df_clean[target_col] - rolling_median).abs()
        df_clean.loc[diff > threshold, 'is_outlier'] = True
        
    # 演算法 2：變化率
    elif filter_method == "變化率過濾法 (Rate of Change) - 強制消除異常跳水":
        suggested_jump = float(df[target_col].std()) if pd.notnull(df[target_col].std()) else 1.0
        max_jump = st.sidebar.number_input("允許的最大單步跳動量", value=suggested_jump, min_value=0.0001, step=0.01)
        diff_prev = (df_clean[target_col] - df_clean[target_col].shift(1)).abs()
        df_clean.loc[diff_prev > max_jump, 'is_outlier'] = True

    # 演算法 3：相鄰突變
    elif filter_method == "相鄰突變檢測 (僅抓單一獨立突波)":
        suggested_jump = float(df[target_col].std() * 1.5) if pd.notnull(df[target_col].std()) else 1.0
        max_jump = st.sidebar.number_input("允許的最大瞬間跳動量", value=suggested_jump, min_value=0.0001, step=0.01)
        diff_prev = (df_clean[target_col] - df_clean[target_col].shift(1)).abs()
        diff_next = (df_clean[target_col] - df_clean[target_col].shift(-1)).abs()
        mask = (diff_prev > max_jump) & (diff_next > max_jump)
        df_clean.loc[mask, 'is_outlier'] = True

    # 演算法 4：上下限
    elif filter_method == "上下限絕對值過濾 (抓取超界值)":
        min_val = st.sidebar.number_input("合理最小值", value=float(df[target_col].min() - 5))
        max_val = st.sidebar.number_input("合理最大值", value=float(df[target_col].max() + 5))
        df_clean.loc[(df_clean[target_col] < min_val) | (df_clean[target_col] > max_val), 'is_outlier'] = True

    # --- 新增功能：手動區間剔除 ---
    st.sidebar.markdown("---")
    st.sidebar.header("✂️ 3. 手動指定異常區間 (選用)")
    st.sidebar.write("若有儀器故障導致的大段異常，可直接輸入時間區間強制剔除。")
    manual_remove = st.sidebar.text_area("輸入需剔除的時間區間 (格式: YYYY-MM-DD HH:MM ~ YYYY-MM-DD HH:MM)", 
                                         placeholder="例如:\n2023-01-01 12:00 ~ 2023-01-05 00:00")
    if manual_remove:
        for line in manual_remove.split("\n"):
            if "~" in line:
                try:
                    start_str, end_str = line.split("~")
                    start_dt = pd.to_datetime(start_str.strip())
                    end_dt = pd.to_datetime(end_str.strip())
                    df_clean.loc[(df_clean[time_col] >= start_dt) & (df_clean[time_col] <= end_dt), 'is_outlier'] = True
                except:
                    pass

    # --- 修復與平滑化 ---
    st.sidebar.markdown("---")
    st.sidebar.header("🩹 4. 數值修復與平滑")
    fill_method = st.sidebar.selectbox("異常值被挖除後，如何填補？", ["線性內插 (Linear Interpolation)", "使用前一筆正常數值填補 (Forward Fill)", "不填補 (保留空白 NaN)"])
    
    apply_smoothing = st.sidebar.checkbox("補值後再進行平滑化 (減少雜訊毛刺)", value=False)
    smooth_window = 5
    if apply_smoothing:
        smooth_window = st.sidebar.slider("平滑視窗大小 (筆數)", min_value=3, max_value=200, value=12)
    
    # 執行清理與修補
    df_clean.loc[df_clean['is_outlier'], 'Cleaned_Value'] = np.nan
    df_clean.loc[~df_clean['is_outlier'], 'Cleaned_Value'] = df_clean[target_col]
    
    if fill_method == "線性內插 (Linear Interpolation)":
        df_clean['Cleaned_Value'] = df_clean['Cleaned_Value'].interpolate(method='linear')
    elif fill_method == "使用前一筆正常數值填補 (Forward Fill)":
        df_clean['Cleaned_Value'] = df_clean['Cleaned_Value'].ffill()

    # 執行平滑化
    if apply_smoothing:
        df_clean['Cleaned_Value'] = df_clean['Cleaned_Value'].rolling(window=smooth_window, center=True, min_periods=1).mean()

    # --- 主畫面：儀表板與圖表 ---
    outlier_count = df_clean['is_outlier'].sum()
    total_count = len(df_clean)
    outlier_pct = (outlier_count / total_count) * 100 if total_count > 0 else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("總資料筆數", f"{total_count:,} 筆")
    col2.metric("抓出/剔除的異常點", f"{outlier_count:,} 筆", f"{outlier_pct:.2f}% 剔除率", delta_color="inverse")
    
    use_manual_y = st.checkbox("手動鎖定 Y 軸範圍 (放大檢視不跑版)")
    y_min, y_max = None, None
    if use_manual_y:
        col_y1, col_y2 = st.columns(2)
        suggest_min = float(df_clean['Cleaned_Value'].min() - 2) if not df_clean['Cleaned_Value'].isna().all() else 0.0
        suggest_max = float(df_clean['Cleaned_Value'].max() + 2) if not df_clean['Cleaned_Value'].isna().all() else 100.0
        with col_y1:
            y_min = st.number_input("Y 軸下限", value=suggest_min)
        with col_y2:
            y_max = st.number_input("Y 軸上限", value=suggest_max)

    tab1, tab2 = st.tabs(["🔍 濾波效果比對 (含異常標記)", "✨ 修復後最終成果 (純淨歷線)"])
    
    with tab1:
        st.write("淺藍色實線為修復後的資料；🔴 紅色點為被系統判定為異常並剔除的原始突波。")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_clean[time_col], y=df_clean['Cleaned_Value'], mode='lines', name="修復後平滑資料", line=dict(color="#1f77b4", width=2)))
        df_outliers = df_clean[df_clean['is_outlier']]
        if not df_outliers.empty:
            fig1.add_trace(go.Scatter(x=df_outliers[time_col], y=df_outliers[target_col], mode='markers', name="被剔除的異常點", marker=dict(color="red", size=6, symbol="x")))
            
        fig1.update_layout(template="plotly_white", hovermode="x unified", height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        if use_manual_y: fig1.update_yaxes(range=[y_min, y_max])
        else: fig1.update_yaxes(autorange=True)
        st.plotly_chart(fig1, use_container_width=True, config={"scrollZoom": True})

    with tab2:
        st.write("這是最終即將匯出的乾淨資料歷線，您可以清楚確認濾波與修補的效果是否符合預期。")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=df_clean[time_col], y=df_clean['Cleaned_Value'], mode='lines', name="最終修復資料", line=dict(color="#1f77b4", width=2)))
            
        fig2.update_layout(template="plotly_white", hovermode="x unified", height=550, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        if use_manual_y: fig2.update_yaxes(range=[y_min, y_max])
        else: fig2.update_yaxes(autorange=True)
        st.plotly_chart(fig2, use_container_width=True, config={"scrollZoom": True})
    
    # --- 匯出資料 ---
    st.markdown("---")
    st.markdown("### 📥 匯出乾淨資料")
    
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
