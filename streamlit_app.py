import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image, ImageOps

# ==========================================
# 1. 辅助算法：一维 K-Means 聚类 (用于抗倾斜分行)
# ==========================================
def simple_kmeans_1d(values, k, max_iter=100):
    """
    手动实现简单的一维 K-Means，用于将圆心的 Y 坐标分成 k 类（即 k 行）。
    这比简单的阈值切分更能抵抗图片倾斜。
    """
    if len(values) < k: return [0] * len(values)
    
    # 初始化中心点 (均匀分布)
    values = np.array(values)
    min_v, max_v = np.min(values), np.max(values)
    centroids = np.linspace(min_v, max_v, k)
    
    for _ in range(max_iter):
        # 1. 分配簇
        # 计算每个点到各个中心的距离，取最小的索引
        distances = np.abs(values[:, np.newaxis] - centroids)
        labels = np.argmin(distances, axis=1)
        
        # 2. 更新中心
        new_centroids = np.array([values[labels == i].mean() if np.sum(labels == i) > 0 else centroids[i] 
                                  for i in range(k)])
        
        # 收敛检测
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    # 对 centroids 排序，确保 label 0 是最上面一行，label 1 是下一行...
    sorted_indices = np.argsort(centroids)
    map_label = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    final_labels = [map_label[l] for l in labels]
    
    return final_labels

# ==========================================
# 2. 核心图像处理
# ==========================================
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
        
        # --- 步骤 A: 颜色打分 (优胜劣汰) ---
        # 准备颜色通道
        if "Saturation" in analysis_mode:
            score_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,1]
        elif "Value" in analysis_mode:
            score_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
        elif "Red" in analysis_mode: score_img = img[:,:,2]
        elif "Green" in analysis_mode: score_img = img[:,:,1]
        elif "Blue" in analysis_mode: score_img = img[:,:,0]
        else: score_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        candidates = []
        for (x, y, r) in circles:
            if y < 0 or x < 0 or y >= img.shape[0] or x >= img.shape[1]: continue
            mask = np.zeros(img.shape[:2], dtype="uint8")
            # 只取圆心最中间的 40% 计算分数，避开边缘反光
            cv2.circle(mask, (x, y), int(r * 0.4), 255, -1)
            score = cv2.mean(score_img, mask=mask)[0]
            candidates.append({'data': (x, y, r), 'score': score})
        
        # 按分数排序，取前 N 个
        # 即使图片里找到了 50 个圈，我们只取最像孔的 N 个
        candidates.sort(key=lambda k: k['score'], reverse=True)
        target_n = required_count if (required_count and required_count > 0) else (rows * cols)
        if len(candidates) > target_n:
            candidates = candidates[:target_n]
        
        accepted_circles = [c['data'] for c in candidates]
        
        # --- 步骤 B: 智能聚类分行 (K-Means Clustering) ---
        # 这是解决图片歪斜的核心！不按绝对Y切分，而是按聚类切分。
        if len(accepted_circles) > 0:
            y_coords = [c[1] for c in accepted_circles]
            # 调用自定义 K-Means，把 Y 坐标分成 'rows' 个簇
            # 注意：如果实际孔数很少（比如只有一行），强行聚成2类可能会有问题
            # 所以这里做一个保护：如果 target_n 很小，就只聚类成 1 行
            k_rows = rows if len(accepted_circles) >= rows else 1
            labels = simple_kmeans_1d(y_coords, k_rows)
            
            # 组装带行号的数据: (row_idx, x, y, r)
            circles_with_row = []
            for i, c in enumerate(accepted_circles):
                circles_with_row.append((labels[i], c[0], c[1], c[2]))
            
            # --- 步骤 C: 排序 (先按行号排，再按 X 排) ---
            # 1. 先按行号排序
            circles_with_row.sort(key=lambda x: x[0])
            
            # 2. 同一行内，按 X 排序
            final_circles = []
            current_row_idx = circles_with_row[0][0]
            current_row_circles = []
            
            for item in circles_with_row:
                r_idx, x, y, r = item
                if r_idx != current_row_idx:
                    # 结算上一行
                    current_row_circles.sort(key=lambda x: x[0])
                    final_circles.extend([(c[1], c[2], c[3]) for c in current_row_circles])
                    # 开启新一行
                    current_row_idx = r_idx
                    current_row_circles = []
                current_row_circles.append(item)
            
            # 结算最后一行
            current_row_circles.sort(key=lambda x: x[0]) # 按 X 坐标 (index 1) 排序
            final_circles.extend([(c[1], c[2], c[3]) for c in current_row_circles])
        
        # --- 步骤 D: 取值与画图 ---
        roi_scale = 0.7 
        for i, (x, y, r) in enumerate(final_circles):
            # 画图
            draw_r = int(r * roi_scale)
            cv2.circle(output_img, (x, y), draw_r, (0, 255, 0), 3)
            cv2.putText(output_img, f"{i+1}", (x-15, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 取值
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), int(r * (roi_scale - 0.1)), 255, -1)
            mean_val = cv2.mean(score_img, mask=mask)[0]
            s_values.append(mean_val)

    return output_img, s_values, len(final_circles)

# ==========================================
# 3. 拟合引擎 (新增虚线绘图逻辑)
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
    
    # 局部线性 (最少 5 个点)
    best_r2 = -1
    min_pts = 5 # <--- 修改：至少需要5个点
    
    if len(x_data) >= min_pts:
        for i in range(len(x_data) - min_pts + 1):
            for j in range(i + min_pts, len(x_data) + 1):
                sx = x_data[i:j]; sy = y_data[i:j]
                ts, ti, tr, _, _ = stats.linregress(sx, sy)
                if tr**2 > best_r2:
                    best_r2 = tr**2
                    report['best_linear_range'] = {
                        'range_text': f"{sx[0]} - {sx[-1]}", 
                        'indices':(i,j), 'params':(ts,ti), 'r2':best_r2, 
                        'func':linear_func, 'inv_func':inverse_linear,
                        'x_range': sx # 保存x数据用于画图
                    }
    else: report['best_linear_range'] = None

    if report['exp_global']['r2'] > report['linear_global']['r2'] + 0.02:
        report['recommended'] = report['exp_global']
    else:
        report['recommended'] = report['linear_global']
    return report

# ==========================================
# 4. Streamlit 界面
# ==========================================
st.set_page_config(page_title="BioSensor Pro Max", layout="wide")
st.title("🧬 生物传感器智能分析系统")

with st.sidebar:
    st.header("⚙️ 参数设置")
    
    analysis_mode = st.selectbox(
        "📊 信号分析模式", 
        ["Green Channel (G)", "Saturation (S)", "Red Channel (R)", "Blue Channel (B)", "Value (V)"],
        index=1 # 默认 Saturation
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
                
                # --- 绘图逻辑更新 ---
                fig, ax = plt.subplots()
                xs = np.linspace(min(known_concs), max(known_concs), 100)
                
                # 1. 原始数据点
                ax.scatter(known_concs, vals, color='black', label='Data', zorder=5)
                
                # 2. 全局推荐曲线 (实线)
                ax.plot(xs, rec['func'](xs, *rec['params']), 'r-', linewidth=2, label='Global Fit')
                
                # 3. 最佳局部线性 (虚线)
                br = report.get('best_linear_range')
                if br and br['r2'] > report['linear_global']['r2'] + 0.01: # 只有比全局线性好才画
                    i1, i2 = br['indices']
                    # 高亮选中的点
                    ax.scatter(known_concs[i1:i2], vals[i1:i2], s=150, facecolors='none', edgecolors='lime', lw=2, label='Best Range Pts')
                    
                    # 画局部虚线 (延长一点点以便看清趋势)
                    local_x = np.array(br['x_range'])
                    local_y_fit = br['func'](local_x, *br['params'])
                    ax.plot(local_x, local_y_fit, color='lime', linestyle='--', linewidth=2.5, label=f"Local Linear (R²={br['r2']:.4f})")
                    
                    st.info(f"💡 最佳局部线性范围 ({min_pts}+点): {br['range_text']} (R²={br['r2']:.4f})")
                
                ax.legend()
                ax.set_xlabel("Concentration")
                ax.set_ylabel(f"Signal ({analysis_mode})")
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
            t_img, t_vals, t_cnt = process_image(up_test, rows, cols, limit, analysis_mode)
            st.image(t_img, channels="BGR", caption=f"检测 {t_cnt} 个")
            if t_cnt > 0:
                res = []
                for v in t_vals: res.append(sel['inv_func'](v, *sel['params']))
                st.dataframe({"Sample": range(1, len(res)+1), "Signal": [f"{v:.1f}" for v in t_vals], "Conc": [f"{c:.4f}" for c in res]})






