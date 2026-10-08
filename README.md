# 監測儀器濾波與異常值消除工具 - 程式結構與功能說明文件

> **📄 系統與 AI 讀取提示 (AI System Prompting Guide)**
> 本文件旨在提供跨 AI 協作時的快速語境（Context）建立。當其他 AI 模型需要接手修改、擴充或理解本 Streamlit 專案時，請優先讀取本文件的「資料流 (Data Flow)」、「核心變數 (Core Variables)」與「演算法邏輯 (Algorithm Logic)」。

---

## 1. 專案概述 (Project Overview)
本專案是一個基於 **Streamlit** 開發的互動式資料清洗與視覺化 Web 應用程式。專為地工、水文與結構監測儀器（如：地下水位計、GPS 位移計、伸縮計、傾度盤等）所設計。
核心目標是解決監測資料常見的**高頻毛刺（雜訊）**、**瞬間突波（Spikes）**、**區塊型當機斷層**等異常狀況，並支援疊加**多種時距（1hr, 3hr, 24hr...）的雨量資料**進行對照分析。處理後的純淨資料可即時預覽並匯出成 CSV 檔案。

### 1.1 技術棧 (Tech Stack)
*   **前端/框架**：`streamlit`
*   **資料處理**：`pandas`, `numpy`, `re` (正規表達式)
*   **視覺化繪圖**：`plotly.graph_objects`, `plotly.subplots`

---

## 2. 核心功能與模組 (Core Modules)

### 2.1 資料讀取與前處理 (`load_data` 函數)
*   **防呆編碼讀取**：自動以 `utf-8` 嘗試讀取，若失敗則降級嘗試 `big5` 與 `cp950`，避免中文 Windows 環境常見的亂碼報錯。
*   **強健的時間解析**：監測儀器常產出帶有中文字元（如 `時`, `分`, `秒`）或亂碼（如 `®É`, `¤À`）的時間字串。透過自訂的 `clean_time(t)` 正規表達式函數，將非標準字元替換為 `:` 與空白，再交由 `pd.to_datetime(..., format='mixed')` 解析。

### 2.2 降雨量多維度整合 (Rain Data Integration)
*   **多欄位支援**：針對類似 `2020-2026_時雨量.csv` 的檔案結構，程式能識別除了 `time` 以外的多個累計雨量欄位（如 `1小時`, `3小時`, `24小時` 等）。
*   **動態選擇**：允許使用者在側邊欄透過 Dropdown 選擇特定的雨量時距進行疊圖分析。

### 2.3 四大濾波與異常檢測演算法 (Filtering Algorithms)
儲存於 `df_clean['is_outlier']` 遮罩（Mask）中：
1.  **移動中位數檢定 (Hampel Filter)**：*（預設推薦）*
    *   利用 `rolling(window).median()` 結合絕對中位差 (MAD, Median Absolute Deviation)。
    *   **優勢**：對抗「連續多筆錯誤（群聚型突波）」極度有效，不會像平均數一樣被極端值拉偏。
2.  **變化率過濾法 (Rate of Change)**：
    *   計算 `abs(df - df.shift(1))`。
    *   **優勢**：嚴格限制單步跳水幅度，有效消除不合理的階梯狀突變。
3.  **相鄰突變檢測 (Adjacent Jump)**：
    *   同時比較 `shift(1)` 與 `shift(-1)`，僅當前值與前後皆產生巨大落差時才剔除。
    *   **優勢**：完美保留豪雨期間「階梯式爬升」的正常訊號，僅殺單點突波。
4.  **上下限絕對值過濾 (Absolute Bounds)**：
    *   基礎的 Min / Max 靜態閾值過濾。

### 2.4 手動區間剔除 (Manual Range Removal)
*   採用 `st.session_state.manual_ranges` 儲存使用者自訂的多個 datetime 區間（Tuple array）。
*   透過 Streamlit 的 `st.date_input` 與 `st.time_input` 讓使用者精確圈選儀器保養或已知失效的時段，強制將該區間標記為 `is_outlier = True`。

### 2.5 修復與平滑化 (Imputation & Smoothing)
*   **空值填補 (Imputation)**：針對被挖除的異常點，提供 `線性內插 (Linear Interpolation)`、`前值填補 (Forward Fill)` 或保留 `NaN`。
*   **平滑化 (Smoothing)**：填補完成後，可額外開啓 `rolling.mean()` 平滑功能，消除微小毛刺（針對 GPS 雜訊特別有效）。

### 2.6 雙分頁互動視覺化 (Interactive Plotly UI)
*   使用 `st.tabs` 分為兩個視角：
    1.  **濾波效果比對**：底圖為修復後的藍線，上方疊加被剔除的異常紅點（`markers`），供使用者確認演算法的「誤殺」或「漏抓」狀況。
    2.  **修復後最終成果**：純淨版歷線，供匯出前的最終確認。
*   具備 `hovermode="x unified"`（游標聯動）與手動鎖定 Y 軸功能（避免放大縮小跑版）。

---

## 3. 資料流與變數結構 (Data Pipeline)

供其他 AI 進行 Code Review 或修改時的重點變數追蹤：

1.  **`df` (Raw Data)**: 原始載入的 Pandas DataFrame。
2.  **`time_col`, `target_col`**: 使用者選擇的時間欄位名稱與要進行濾波的數值欄位名稱。
3.  **`df_clean` (Processing Data)**:
    *   **`df_clean['is_outlier']`** (Boolean Series): 核心遮罩。所有演算法與手動區間的操作，最終目的都是把異常點的 index 標記為 `True`。
    *   **`df_clean['Cleaned_Value']`** (Float Series): 存放最終結果的欄位。`is_outlier == True` 的點會先被轉為 `np.nan`，接著進行內插修補與平滑計算。
4.  **`df_export` (Output Data)**: 匯出用的 DataFrame，將原始的 `target_col` 替換為完成運算的 `Cleaned_Value`，確保輸出的 CSV 欄位名稱與使用者上傳時完全一致。

---

## 4. 跨 AI 擴充建議 (Future AI Extension Guidelines)
若未來的 AI 助手需要修改本程式，請留意以下設計原則：
1.  **雨量資料維度擴充**：目前的 Subplot 設計為 `rows=2`。若需將 `1小時`, `3小時`, `24小時` 雨量同時畫在同一張 Bar chart 內，請修改 `Plotly.add_trace` 區段，並將 Bar mode 設為 `overlay` 或 `group`，且務必加入 `autorange='reversed'` 使雨量圖朝下（或依照使用者習慣朝上）顯示。
2.  **演算法擴充**：若需加入如 `Savitzky-Golay filter` 或 `Kalman Filter`，請統一在「演算法設定區段」產生 boolean mask 並更新至 `df_clean['is_outlier']`。
3.  **效能考量**：若資料列數超過 100,000 筆，Plotly 繪製 Markers 會導致瀏覽器卡頓，程式中已有 `if total_count > 100000` 的降級渲染邏輯，擴充視覺化時請保留此防護機制。