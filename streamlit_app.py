import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image, ImageOps

# ==========================================
# 1. 智能图像处理模块 (动态尺度 + 数量锁定版)
# ==========================================
def process_image(img_file_buffer, rows, cols, required_count=None):
    """
    required_count: 如果指定了数量(比如11)，则只输出前11个孔，后面的忽略。
    """
    # 1. 标准化缩放
    image_pil = Image.open(img_file_buffer)
    image_pil = ImageOps.exif_transpose(image_pil)
    target_width = 1000
    w_percent = (target_width / float(image_pil.size[0]))
    h_size = int((float(image_pil.size[1]) * float(w_percent)))
    image_pil = image_pil.resize((target_width, h_size), Image.Resampling.LANCZOS)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    output_img = img.copy()
    
    # 2. 动态参数计算
    approx_diameter = target_width / (cols + 0.5)
    dynamic_min_r = int(approx_diameter / 2 * 0.75) 
    dynamic_max_r = int(approx_diameter / 2 * 1.1)
    min_dist_param = int(approx_diameter * 0.85)
    
    # 3. 图像增强
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    gray_blur = cv2.GaussianBlur(enhanced_gray, (9, 9), 2)

    # 4. 霍夫圆检测 (定位塑料外壁)
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=min_dist_param,
        param1=50, 
        param2=30,
        minRadius=dynamic_min_r, 
        maxRadius=dynamic_max_r
    )

    s_values = []
    sorted_circles = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # --- 排序与过滤 ---
        # 1. 过滤背景杂质
        circles = sorted(circles, key=lambda x: x[1])
        if len(circles) > 0:
            median_r = np.median([c[2] for c in circles])
            circles = [c for c in circles if abs(c[2] - median_r) < median_r * 0.4]
        
        # 2. 截取最大可能数量 (rows * cols)
        max_possible = rows * cols
        if len(circles) > max_possible:
             circles = circles[:max_possible]

        # 3. 严格的网格排序 (Row-Major)
        circles = sorted(circles, key=lambda x: x[1]) # 先按Y排
        temp_sorted = []
        for r in range(rows):
            start_idx = r * cols
            end_idx = min((r + 1) * cols, len(circles))
            if start_idx < len(circles):
                row_circles = circles[start_idx : end_idx]
                # 行内按X排
                row_circles = sorted(row_circles, key=lambda x: x[0])
                temp_sorted.extend(row_circles)
        
        # --- 关键修改：智能数量锁定 ---
        # 如果用户指定了 required_count (例如11个)，我们就只取排序后的前11个
        # 这样第12, 13, 14个空孔就会被直接丢弃，不画圈也不计算
        if required_count is not None and required_count > 0:
            if len(temp_sorted) > required_count:
                temp_sorted = temp_sorted[:required_count]
        
        sorted_circles = temp_sorted

        # 5. 提取 S 值 (带收缩系数)
        roi_scale = 0.7 
        
        for i, (x, y, r) in enumerate(sorted_circles):
            # 画图
            draw_r = int(r * roi_scale)
            cv2.circle(output_img, (x, y), draw_r, (0, 255, 0), 3)
            cv2.putText(output_img, f"{i+1}", (x-15, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # 取色
            sample_r = int(r * (roi_scale - 0.1)) 
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), sample_r, 255, -1)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_val = cv2.mean(hsv, mask=mask)
            s_values.append(mean_val[1])

    return output_img, s_values, len(sorted_circles)

# ==========================================
# 2. 智能拟合引擎
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
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # 全局线性
    slope, intercept, r_value_lin, _, _ = stats.linregress(x_data, y_data)
    report['linear_global'] = {
        'params': (slope, intercept),
        'r2': r_value_lin**2,
        'func': linear_func,
        'inv_func': inverse_linear,
        'name': '全局线性 (Global Linear)'
    }

    # 全局指数
    try:
        p0 = [np.max(y_data)-np.min(y_data), 0.5, np.min(y_data)]
        popt_exp, _ = curve_fit(exp_decay_func, x_data, y_data, p0=p0, maxfev=5000)
        residuals = y_data - exp_decay_func(x_data, *popt_exp)
        r2_exp = 1 - (np.sum(residuals**2) / np.sum((y_data - np.mean(y_data))**2))
        report['exp_global'] = {
            'params': popt_exp, 'r2': r2_exp, 'func': exp_decay_func,
            'inv_func': inverse_exp, 'name': '指数衰减 (Exp Decay)'
        }
    except:
        report['exp_global'] = {'r2': -1}

    # 最佳线性范围
    best_subset_r2 = -1
    min_points = 4
    if len(x_data) >= min_points:
        for i in range(len(x_data) - min_points + 1):
            for j in range(i + min_points, len(x_data) + 1):
                sub_x = x_data[i:j]; sub_y = y_data[i:j]
                s, i_cept, r, _, _ = stats.linregress(sub_x, sub_y)
                if r**2 > best_subset_r2:
                    best_subset_r2 = r**2
                    report['best_linear_range'] = {
                        'range_text': f"{sub_x[0]} - {sub_x[-1]} mM",
                        'indices': (i, j), 'params': (s, i_cept), 'r2': best_subset_r2,
                        'func': linear_func, 'inv_func': inverse_linear,
                        'name': f"最佳线性范围 ({sub_x[0]}-{sub_x[-1]})"
                    }
    else: report['best_linear_range'] = None

    # 推荐逻辑
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
    conc_input = st.text_area("标准品浓度 (mM)", "0, 0.1, 0.5, 1, 2, 4, 6, 8, 10, 15, 20")
    try:
        known_concs = [float(x.strip()) for x in conc_input.split(',')]
    except:
        st.error("浓度格式错误")
        known_concs = []
    
    st.markdown("---")
    st.subheader("阵列布局")
    rows = st.number_input("行数 (Rows)", 1, 10, 2)
    cols = st.number_input("列数 (Cols)", 1, 20, 7)
    
    # 增加一个开关，方便调试
    st.markdown("---")
    st.caption(f"当前模式：检测前 {len(known_concs)} 个孔 (与浓度数量一致)")

tab1, tab2 = st.tabs(["📏 建立标曲 (Calibration)", "🧪 样品检测 (Test)"])

if 'fit_report' not in st.session_state:
    st.session_state.fit_report = None

with tab1:
    uploaded_calib = st.file_uploader("上传标准品图片", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_calib:
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            # === 核心改动 ===
            # 直接把 len(known_concs) 传进去
            # 代码会自动只识别前 N 个孔，把后面多余的空孔全部丢弃！
            target_count = len(known_concs)
            img_show, s_vals, count = process_image(uploaded_calib, rows, cols, required_count=target_count)
            st.image(img_show, channels="BGR", use_container_width=True, caption=f"自动锁定前 {count} 个有效孔")
        
        with col_res:
            if count != target_count:
                # 只有当识别到的孔比浓度还少时才报错
                st.error(f"⚠️ 识别数量不足！需要 {target_count} 个，只找到 {count} 个。请检查图片清晰度。")
            else:
                report = auto_fit_engine(known_concs, s_vals)
                st.session_state.fit_report = report
                
                rec = report['recommended']
                st.success(f"✅ 推荐模型：{rec['name']}")
                st.metric("拟合优度 (R²)", f"{rec['r2']:.4f}")
                
                fig, ax = plt.subplots()
                x_smooth = np.linspace(min(known_concs), max(known_concs), 100)
                ax.scatter(known_concs, s_vals, color='black', label='Raw Data', zorder=5)
                ax.plot(x_smooth, rec['func'](x_smooth, *rec['params']), 'r-', label='Fit Curve')
                
                best_r = report.get('best_linear_range')
                if best_r and best_r['r2'] > report['linear_global']['r2']:
                     i1, i2 = best_r['indices']
                     ax.scatter(known_concs[i1:i2], s_vals[i1:i2], s=100, facecolors='none', edgecolors='lime', linewidth=2)
                     st.info(f"💡 发现更优线性范围：{best_r['range_text']} (R²={best_r['r2']:.4f})")
                
                ax.legend(); ax.set_xlabel("Conc"); ax.set_ylabel("Signal")
                st.pyplot(fig)

with tab2:
    if st.session_state.fit_report is None:
        st.info("👈 请先建立标曲")
    else:
        report = st.session_state.fit_report
        opts = {"智能推荐": report['recommended'], "全局线性": report['linear_global'], "全局非线性": report['exp_global']}
        if report.get('best_linear_range'): opts[f"最佳线性 ({report['best_linear_range']['range_text']})"] = report['best_linear_range']
        
        sel_model = opts[st.selectbox("计算模型：", list(opts.keys()))]
        
        # === 样品检测部分的智能改动 ===
        # 增加一个滑块，让用户决定测几个样品，默认全测
        st.markdown("---")
        test_limit = st.slider("预计样品数量 (自动忽略后续空孔)", 1, rows*cols, rows*cols)
        
        uploaded_test = st.file_uploader("上传待测样品", type=['jpg', 'png'], key='test')
        if uploaded_test:
            # 传入用户的限制数量
            img_test, s_test, count_test = process_image(uploaded_test, rows, cols, required_count=test_limit)
            st.image(img_test, channels="BGR", caption=f"检测前 {count_test} 个样品")
            
            if count_test > 0:
                results = []
                for s in s_test:
                    conc = sel_model['inv_func'](s, *sel_model['params'])
                    results.append(conc)
                
                st.dataframe({
                    "Sample": [f"#{i+1}" for i in range(len(results))],
                    "S-Value": [f"{v:.1f}" for v in s_test],
                    "Conc (mM)": [f"{c:.4f}" for c in results]
                }, use_container_width=True)



