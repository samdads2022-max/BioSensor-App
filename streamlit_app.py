import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image, ImageOps

# ==========================================
# 1. 辅助算法：一维 K-Means 聚类
# ==========================================
def robust_sort_circles(circles, rows):
    if not circles: return []
    y_coords = np.array([c[1] for c in circles]).reshape(-1, 1)
    k = min(rows, len(circles))
    if k <= 1: return sorted(circles, key=lambda x: x[0])
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(np.float32(y_coords), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    row_groups = {}
    for i, label in enumerate(labels.flatten()):
        if label not in row_groups: row_groups[label] = []
        row_groups[label].append(circles[i])
        
    row_stats = []
    for label, group in row_groups.items():
        avg_y = np.mean([c[1] for c in group])
        row_stats.append((label, avg_y))
    row_stats.sort(key=lambda x: x[1])
    
    final_sorted_circles = []
    for label, _ in row_stats:
        group = row_groups[label]
        group.sort(key=lambda x: x[0])
        final_sorted_circles.extend(group)
    return final_sorted_circles

# ==========================================
# 2. 核心图像处理 (包含 1x1 特判)
# ==========================================
def process_image(img_file_buffer, rows, cols, required_count=None, analysis_mode="Saturation (S)"):
    image_pil = Image.open(img_file_buffer)
    image_pil = ImageOps.exif_transpose(image_pil)
    target_width = 1000
    w_percent = (target_width / float(image_pil.size[0]))
    h_size = int((float(image_pil.size[1]) * float(w_percent)))
    image_pil = image_pil.resize((target_width, h_size), Image.Resampling.LANCZOS)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    output_img = img.copy()
    
    # --- 1x1 单孔模式特判 ---
    if rows == 1 and cols == 1:
        dynamic_min_r = int(target_width * 0.2)
        dynamic_max_r = int(target_width * 0.48)
        min_dist_param = int(target_width * 0.5)
    else:
        approx_diameter = target_width / (cols + 0.5)
        dynamic_min_r = int(approx_diameter / 2 * 0.7)
        dynamic_max_r = int(approx_diameter / 2 * 1.2)
        min_dist_param = int(approx_diameter * 0.8)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(enhanced_gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=min_dist_param, param1=50, param2=25,
        minRadius=dynamic_min_r, maxRadius=dynamic_max_r
    )

    s_values = []
    final_circles = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # 颜色打分
        if "Saturation" in analysis_mode: score_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1]
        elif "Value" in analysis_mode: score_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
        elif "Red" in analysis_mode: score_img = img[:,:,2]
        elif "Green" in analysis_mode: score_img = img[:,:,1]
        elif "Blue" in analysis_mode: score_img = img[:,:,0]
        else: score_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        candidates = []
        for (x, y, r) in circles:
            if y < 0 or x < 0 or y >= img.shape[0] or x >= img.shape[1]: continue
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), int(r * 0.4), 255, -1)
            score = cv2.mean(score_img, mask=mask)[0]
            candidates.append({'data': (x, y, r), 'score': score})
        
        candidates.sort(key=lambda k: k['score'], reverse=True)
        
        # 数量截取 (先填满网格)
        max_possible = rows * cols
        if len(candidates) > max_possible:
            candidates = candidates[:max_possible]
        
        accepted_circles = [c['data'] for c in candidates]
        
        # 排序
        if len(accepted_circles) <= 1:
            sorted_circles = accepted_circles
        else:
            sorted_circles = robust_sort_circles(accepted_circles, rows)

        # 二次截取 (用户滑块限制)
        if required_count is not None and required_count > 0:
            if len(sorted_circles) > required_count:
                final_circles = sorted_circles[:required_count]
            else:
                final_circles = sorted_circles
        else:
            final_circles = sorted_circles

        # 取值
        roi_scale = 0.7 
        for i, (x, y, r) in enumerate(final_circles):
            draw_r = int(r * roi_scale)
            cv2.circle(output_img, (x, y), draw_r, (0, 255, 0), 3)
            cv2.putText(output_img, f"{i+1}", (x-15, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), int(r * (roi_scale - 0.1)), 255, -1)
            mean_val = cv2.mean(score_img, mask=mask)[0]
            s_values.append(mean_val)

    return output_img, s_values, len(final_circles)

# ==========================================
# 3. 拟合引擎
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
    report = {}
    x_data = np.array(x_data); y_data = np.array(y_data)
    
    s, i, r, _, _ = stats.linregress(x_data, y_data)
    report['linear_global'] = {'params':(s,i), 'r2':r**2, 'func':linear_func, 'inv_func':inverse_linear, 'name':'全局线性'}
    
    try:
        p0 = [np.max(y_data)-np.min(y_data), 0.5, np.min(y_data)]
        popt, _ = curve_fit(exp_decay_func, x_data, y_data, p0=p0, maxfev=5000)
        res = y_data - exp_decay_func(x_data, *popt)
        r2 = 1 - (np.sum(res**2)/np.sum((y_data-np.mean(y_data))**2))
        report['exp_global'] = {'params':popt, 'r2':r2, 'func':exp_decay_func, 'inv_func':inverse_exp, 'name':'指数衰减'}
    except: report['exp_global'] = {'r2':-1}
    
    best_r2 = -1
    min_pts = 5 
    if len(x_data) >= min_pts:
        for i in range(len(x_data) - min_pts + 1):
            for j in range(i + min_pts, len(x_data) + 1):
                sx = x_data[i:j]; sy = y_data[i:j]
                ts, ti, tr, _, _ = stats.linregress(sx, sy)
                if tr**2 > best_r2:
                    best_r2 = tr**2
                    report['best_linear_range'] = {'range_text': f"{sx[0]} - {sx[-1]}", 'indices':(i,j), 'params':(ts,ti), 'r2':best_r2, 'func':linear_func, 'inv_func':inverse_linear, 'x_range': sx}
    else: report['best_linear_range'] = None

    if report['exp_global']['r2'] > report['linear_global']['r2'] + 0.02: report['recommended'] = report['exp_global']
    else: report['recommended'] = report['linear_global']
    return report

# ==========================================
# 4. Streamlit 界面 (布局解耦版)
# ==========================================
st.set_page_config(page_title="BioSensor Pro Max", layout="wide")
st.title("🧬 生物传感器智能分析系统")

# --- 侧边栏 (只放通用设置) ---
with st.sidebar:
    st.header("⚙️ 全局设置")
    analysis_mode = st.selectbox("📊 信号分析模式", ["Saturation (S)", "Green Channel (G)", "Red Channel (R)", "Blue Channel (B)", "Value (V)"])
    conc_input = st.text_area("标准品浓度 (mM)", "0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20")
    try: known_concs = [float(x.strip()) for x in conc_input.split(',')]
    except: known_concs = []

tab1, tab2 = st.tabs(["📏 建立标曲", "🧪 样品检测"])
if 'fit_report' not in st.session_state: st.session_state.fit_report = None

# --- Tab 1: 标曲建立 (有自己的布局设置) ---
with tab1:
    # 这里的设置只属于 Tab 1
    c1, c2 = st.columns(2)
    with c1: calib_rows = st.number_input("标曲板 - 行数", 1, 10, 2, key='c_rows')
    with c2: calib_cols = st.number_input("标曲板 - 列数", 1, 20, 7, key='c_cols')
    
    uploaded_calib = st.file_uploader("上传标准品图片", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_calib:
        max_points = len(known_concs)
        if max_points < 3:
            st.error("⚠️ 浓度数量太少，无法拟合")
        else:
            fit_count = st.slider("拟合孔数 (从头保留)", 3, max_points, max_points, key='calib_slider')
            col_img, col_res = st.columns([1, 1])
            
            with col_img:
                # 使用 Tab 1 自己的 rows/cols
                img, vals, count = process_image(uploaded_calib, calib_rows, calib_cols, fit_count, analysis_mode)
                st.image(img, channels="BGR", use_container_width=True, caption=f"识别结果")
            
            with col_res:
                if count != fit_count:
                    st.error(f"⚠️ 识别错误: 需要 {fit_count}，找到 {count}")
                else:
                    curr_x = np.array(known_concs[:count])
                    curr_y = np.array(vals)
                    report = auto_fit_engine(curr_x, curr_y)
                    st.session_state.fit_report = report
                    rec = report['recommended']
                    
                    st.success(f"✅ 推荐: {rec['name']}")
                    st.metric("R²", f"{rec['r2']:.4f}")
                    
                    fig, ax = plt.subplots()
                    xs = np.linspace(min(curr_x), max(curr_x), 100)
                    ax.scatter(curr_x, curr_y, c='k', label='Data', zorder=5)
                    ax.plot(xs, rec['func'](xs, *rec['params']), 'r-', label='Fit')
                    
                    br = report.get('best_linear_range')
                    if br and br['r2'] > report['linear_global']['r2'] + 0.01:
                        i1, i2 = br['indices']
                        ax.scatter(curr_x[i1:i2], curr_y[i1:i2], s=150, facecolors='none', edgecolors='lime', lw=2)
                        # 画虚线
                        lx = np.array(br['x_range'])
                        ly = br['func'](lx, *br['params'])
                        ax.plot(lx, ly, 'g--', lw=2, label='Local Linear')
                        st.info(f"💡 最佳局部线性 (5+点): {br['range_text']} (R²={br['r2']:.4f})")
                    
                    ax.legend()
                    st.pyplot(fig)

# --- Tab 2: 样品检测 (有自己的布局设置) ---
with tab2:
    if not st.session_state.fit_report:
        st.info("👈 请先建立标曲")
    else:
        # 这里的设置只属于 Tab 2
        c3, c4 = st.columns(2)
        with c3: test_rows = st.number_input("样品板 - 行数", 1, 10, 1, key='t_rows') # 默认1 (方便单孔)
        with c4: test_cols = st.number_input("样品板 - 列数", 1, 20, 1, key='t_cols') # 默认1
        
        rep = st.session_state.fit_report
        opts = {"智能推荐": rep['recommended'], "全局线性": rep['linear_global'], "全局非线性": rep['exp_global']}
        if rep.get('best_linear_range'): opts[f"最佳线性 ({rep['best_linear_range']['range_text']})"] = rep['best_linear_range']
        
        sel = opts[st.selectbox("计算模型", list(opts.keys()))]
        
        # 只有在多孔模式下才显示滑块
        max_samples = test_rows * test_cols
        if max_samples > 1:
            limit = st.slider("样品数量", 1, max_samples, max_samples, key='test_slider')
        else:
            limit = 1
            st.caption("当前模式：单孔检测 (1x1)")
            
        up_test = st.file_uploader("上传样品", type=['jpg', 'png'], key='test_up')
        if up_test:
            # 使用 Tab 2 自己的 rows/cols
            t_img, t_vals, t_cnt = process_image(up_test, test_rows, test_cols, limit, analysis_mode)
            st.image(t_img, channels="BGR", caption=f"检测 {t_cnt} 个")
            if t_cnt > 0:
                res = []
                for v in t_vals: res.append(sel['inv_func'](v, *sel['params']))
                st.dataframe({"Sample": range(1, len(res)+1), "Signal": [f"{v:.1f}" for v in t_vals], "Conc": [f"{c:.4f}" for c in res]})









