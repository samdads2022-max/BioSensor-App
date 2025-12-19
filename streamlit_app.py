import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image, ImageOps

# ==========================================
# 1. 核心算法工具箱
# ==========================================

def sort_circles_robust(circles, rows, cols):
    """
    抗歪斜排序算法：
    不依赖绝对Y坐标切割，而是基于 Y 轴的'空隙'来自动分行。
    只要行与行之间没有重叠，图片歪了也能排对。
    """
    if len(circles) == 0: return []
    
    # 1. 先按 Y 坐标粗略排序
    circles = sorted(circles, key=lambda x: x[1])
    
    # 2. 寻找行的“断层” (Gap)
    # 计算相邻圆心的 Y 距离，如果距离大于半径，说明换行了
    rows_groups = []
    current_row = [circles[0]]
    
    # 获取平均半径作为阈值参考
    avg_r = np.median([c[2] for c in circles])
    gap_threshold = avg_r * 1.0 # 阈值：如果Y差值超过1倍半径，认为是下一行
    
    for i in range(1, len(circles)):
        y_diff = circles[i][1] - circles[i-1][1]
        
        if y_diff > gap_threshold:
            # 发现断层，保存当前行，开启新一行
            rows_groups.append(current_row)
            current_row = []
        
        current_row.append(circles[i])
    
    # 加入最后一行
    rows_groups.append(current_row)
    
    # 3. 对每一行内部，按 X 坐标排序 (从左到右)
    final_sorted = []
    for row_group in rows_groups:
        row_group = sorted(row_group, key=lambda x: x[0])
        final_sorted.extend(row_group)
        
    return final_sorted

def extract_signal(img, circles, mode="Saturation (S)"):
    """
    支持多种颜色分析模式
    """
    values = []
    # 预先转换颜色空间，避免重复计算
    if "Saturation" in mode:
        target_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1] # S通道
    elif "Value" in mode: # 亮度/灰度
        target_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2] # V通道
    elif "Red" in mode:
        target_img = img[:,:,2] # BGR中的R
    elif "Green" in mode:
        target_img = img[:,:,1] # BGR中的G
    elif "Blue" in mode:
        target_img = img[:,:,0] # BGR中的B
    else:
        target_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # 默认灰度

    for (x, y, r) in circles:
        # 缩小取样范围，只取圆心 50%
        roi_r = int(r * 0.5)
        mask = np.zeros(img.shape[:2], dtype="uint8")
        cv2.circle(mask, (x, y), roi_r, 255, -1)
        
        # 计算平均值
        mean_val = cv2.mean(target_img, mask=mask)[0]
        values.append(mean_val)
        
    return values

def process_image(img_file_buffer, rows, cols, required_count=None, analysis_mode="Saturation (S)"):
    # 1. 图像标准化
    image_pil = Image.open(img_file_buffer)
    image_pil = ImageOps.exif_transpose(image_pil)
    target_width = 1000
    w_percent = (target_width / float(image_pil.size[0]))
    h_size = int((float(image_pil.size[1]) * float(w_percent)))
    image_pil = image_pil.resize((target_width, h_size), Image.Resampling.LANCZOS)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    output_img = img.copy()
    
    # 2. 动态参数
    approx_diameter = target_width / (cols + 0.5)
    dynamic_min_r = int(approx_diameter / 2 * 0.7)
    dynamic_max_r = int(approx_diameter / 2 * 1.2)
    min_dist_param = int(approx_diameter * 0.8) # 严防重叠
    
    # 3. 霍夫检测
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(enhanced_gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=min_dist_param,
        param1=50, param2=25,
        minRadius=dynamic_min_r, 
        maxRadius=dynamic_max_r
    )

    s_values = []
    final_circles = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # --- 步骤 A: 颜色海选 (Saturation Filter) ---
        # 先不管位置，只管“谁最有颜色”。
        # 无论你选什么分析模式，筛选孔位时依然用“饱和度”最稳，因为孔肯定比背景艳
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        candidates = []
        for (x, y, r) in circles:
            if y < 0 or x < 0 or y >= img.shape[0] or x >= img.shape[1]: continue
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), int(r * 0.6), 255, -1)
            score = cv2.mean(hsv_img, mask=mask)[1] # 饱和度得分
            candidates.append({'data': (x, y, r), 'score': score})
        
        # 按得分从高到低排
        candidates.sort(key=lambda k: k['score'], reverse=True)
        
        # --- 步骤 B: 录取前 N 名 ---
        target_n = required_count if (required_count and required_count > 0) else (rows * cols)
        if len(candidates) > target_n:
            candidates = candidates[:target_n]
        
        accepted_circles = [c['data'] for c in candidates]
        
        # --- 步骤 C: 抗歪斜排序 (Gap-based Sorting) ---
        # 这里的排序不再依赖死板的切片，而是智能分行
        final_circles = sort_circles_robust(accepted_circles, rows, cols)

        # --- 步骤 D: 取值与画图 ---
        s_values = extract_signal(img, final_circles, analysis_mode)
        
        for i, (x, y, r) in enumerate(final_circles):
            # 视觉标记 (收缩圈)
            draw_r = int(r * 0.7)
            cv2.circle(output_img, (x, y), draw_r, (0, 255, 0), 3)
            # 标记序号
            cv2.putText(output_img, f"{i+1}", (x-15, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    return output_img, s_values, len(final_circles)

# ==========================================
# 2. 拟合引擎 (保持不变，略去以节省篇幅，请保留之前的代码)
# ==========================================
def linear_func(x, k, b): return k * x + b
def exp_decay_func(x, a, b, c): return a * np.exp(-b * x) + c
def inverse_linear(y, k, b): return (y - b) / k
def inverse_exp(y, a, b, c):
    try:
        val = (y - c) / a
        if val <= 0: return 0
        return -(1/b) * np.log(val)
    except: return 0

def auto_fit_engine(x_data, y_data):
    # ... (此处请保留原来的 auto_fit_engine 代码，完全不用变) ...
    # 为了完整性，我再贴一次核心部分，防止你复制漏了
    report = {}
    x_data = np.array(x_data); y_data = np.array(y_data)
    
    # 线性
    s, i, r, _, _ = stats.linregress(x_data, y_data)
    report['linear_global'] = {'params':(s,i), 'r2':r**2, 'func':linear_func, 'inv_func':inverse_linear, 'name':'全局线性'}
    
    # 指数
    try:
        p0 = [np.max(y_data)-np.min(y_data), 0.5, np.min(y_data)]
        popt, _ = curve_fit(exp_decay_func, x_data, y_data, p0=p0, maxfev=5000)
        res = y_data - exp_decay_func(x_data, *popt)
        r2 = 1 - (np.sum(res**2)/np.sum((y_data-np.mean(y_data))**2))
        report['exp_global'] = {'params':popt, 'r2':r2, 'func':exp_decay_func, 'inv_func':inverse_exp, 'name':'指数衰减'}
    except: report['exp_global'] = {'r2':-1}
    
    # 局部线性
    best_r2 = -1
    if len(x_data)>=4:
        for i in range(len(x_data)-3):
            for j in range(i+4, len(x_data)+1):
                sx = x_data[i:j]; sy = y_data[i:j]
                ts, ti, tr, _, _ = stats.linregress(sx, sy)
                if tr**2 > best_r2:
                    best_r2 = tr**2
                    report['best_linear_range'] = {'range_text':f"{sx[0]}-{sx[-1]}", 'indices':(i,j), 'params':(ts,ti), 'r2':best_r2, 'func':linear_func, 'inv_func':inverse_linear}
    else: report['best_linear_range'] = None

    if report['exp_global']['r2'] > report['linear_global']['r2'] + 0.02:
        report['recommended'] = report['exp_global']
    else:
        report['recommended'] = report['linear_global']
    return report

# ==========================================
# 3. Streamlit 界面
# ==========================================
st.set_page_config(page_title="BioSensor Pro Max", layout="wide")
st.title("🧬 生物传感器智能分析系统")

# --- 侧边栏 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 新增：颜色分析模式
    analysis_mode = st.selectbox(
        "📊 信号分析模式", 
        ["Saturation (S) - 通用推荐", "Value (V) - 亮度/黑白", "Red Channel (R)", "Green Channel (G)", "Blue Channel (B)"],
        help="通常比色法使用 Saturation (S) 即可。如果试纸是变黑，选 Value。如果是特定变红，选 Red。"
    )
    
    conc_input = st.text_area("标准品浓度 (mM)", "0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20")
    try: known_concs = [float(x.strip()) for x in conc_input.split(',')]
    except: known_concs = []
    
    st.markdown("---")
    rows = st.number_input("行数 (Rows)", 1, 10, 2)
    cols = st.number_input("列数 (Cols)", 1, 20, 7)

tab1, tab2 = st.tabs(["📏 建立标曲", "🧪 样品检测"])

if 'fit_report' not in st.session_state: st.session_state.fit_report = None

with tab1:
    uploaded_calib = st.file_uploader("上传标准品图片", type=['jpg', 'png', 'jpeg'])
    if uploaded_calib:
        col1, col2 = st.columns([1,1])
        with col1:
            target_count = len(known_concs)
            # 传入分析模式
            img, vals, count = process_image(uploaded_calib, rows, cols, target_count, analysis_mode)
            st.image(img, channels="BGR", use_container_width=True, caption=f"识别结果 ({count}/{target_count})")
        
        with col2:
            if count != target_count:
                st.error(f"⚠️ 数量不匹配！需要 {target_count}，找到 {count}。")
            else:
                report = auto_fit_engine(known_concs, vals)
                st.session_state.fit_report = report
                rec = report['recommended']
                
                st.success(f"✅ 推荐: {rec['name']}")
                st.metric("R²", f"{rec['r2']:.4f}")
                
                fig, ax = plt.subplots()
                xs = np.linspace(min(known_concs), max(known_concs), 100)
                ax.scatter(known_concs, vals, color='black', label='Data')
                ax.plot(xs, rec['func'](xs, *rec['params']), 'r-', label='Fit')
                
                # 局部线性
                br = report.get('best_linear_range')
                if br and br['r2'] > report['linear_global']['r2']:
                    i1, i2 = br['indices']
                    ax.scatter(known_concs[i1:i2], vals[i1:i2], s=150, facecolors='none', edgecolors='lime', lw=2, label='Best Range')
                    st.info(f"💡 最佳线性范围: {br['range_text']} (R²={br['r2']:.4f})")
                
                ax.legend()
                st.pyplot(fig)

with tab2:
    if not st.session_state.fit_report:
        st.info("👈 请先建立标曲")
    else:
        rep = st.session_state.fit_report
        opts = {"智能推荐": rep['recommended'], "全局线性": rep['linear_global'], "全局非线性": rep['exp_global']}
        if rep.get('best_linear_range'): opts[f"最佳线性 ({rep['best_linear_range']['range_text']})"] = rep['best_linear_range']
        
        sel = opts[st.selectbox("计算模型", list(opts.keys()))]
        
        limit = st.slider("样品数量", 1, rows*cols, rows*cols)
        up_test = st.file_uploader("上传样品", type=['jpg', 'png'], key='t')
        
        if up_test:
            # 传入分析模式
            t_img, t_vals, t_cnt = process_image(up_test, rows, cols, limit, analysis_mode)
            st.image(t_img, channels="BGR", caption=f"检测 {t_cnt} 个")
            if t_cnt > 0:
                res = []
                for v in t_vals: res.append(sel['inv_func'](v, *sel['params']))
                st.dataframe({"Sample": range(1, len(res)+1), "Signal": [f"{v:.1f}" for v in t_vals], "Conc": [f"{c:.4f}" for c in res]})





