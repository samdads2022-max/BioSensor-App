import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image, ImageOps

# ==========================================
# 1. 智能图像处理模块 (动态尺度版)
# ==========================================
def process_image(img_file_buffer, rows, cols):
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

    # 4. 霍夫圆检测 (依然检测最稳定的外圈)
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
        
        # 排序与过滤逻辑
        circles = sorted(circles, key=lambda x: x[1])
        if len(circles) > 0:
            median_r = np.median([c[2] for c in circles])
            circles = [c for c in circles if abs(c[2] - median_r) < median_r * 0.4]
        
        expected_total = rows * cols
        if len(circles) > expected_total:
             circles = circles[:expected_total]

        circles = sorted(circles, key=lambda x: x[1])
        for r in range(rows):
            start_idx = r * cols
            end_idx = min((r + 1) * cols, len(circles))
            if start_idx < len(circles):
                row_circles = sorted(circles[start_idx:end_idx], key=lambda x: x[0])
                sorted_circles.extend(row_circles)

        # 5. 提取 S 值 (关键修改在这里！)
        # 定义收缩系数：0.7 表示只取中间 70% 的区域
        # 你可以根据实际情况微调这个数 (0.6 - 0.8)
        roi_scale = 0.7 
        
        for i, (x, y, r) in enumerate(sorted_circles):
            # --- 视觉修正 ---
            # r 是外圈半径，draw_r 是我们画在图上的半径
            draw_r = int(r * roi_scale)
            
            # 画圆 (使用收缩后的半径)
            cv2.circle(output_img, (x, y), draw_r, (0, 255, 0), 3)
            cv2.putText(output_img, f"{i+1}", (x-15, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # --- 数据提取修正 ---
            # 采样半径：为了数据纯净，通常比画的圈再小一点点 (比如 0.5)
            # 这样能避开纸片边缘可能存在的阴影
            sample_r = int(r * (roi_scale - 0.1)) 
            
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), sample_r, 255, -1)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_val = cv2.mean(hsv, mask=mask)
            s_values.append(mean_val[1])

    return output_img, s_values, len(sorted_circles)

# ==========================================
# 2. 智能拟合引擎 (新增模块)
# ==========================================

# 定义各种模型及其反函数
def linear_func(x, k, b): 
    return k * x + b

def exp_decay_func(x, a, b, c): 
    return a * np.exp(-b * x) + c

def inverse_linear(y, k, b):
    return (y - b) / k

def inverse_exp(y, a, b, c):
    try:
        val = (y - c) / a
        if val <= 0: return 0
        return -(1/b) * np.log(val)
    except: return 0

def auto_fit_engine(x_data, y_data):
    """
    全自动拟合引擎：比较线性vs非线性，并寻找最佳线性范围
    """
    report = {}
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # --- A. 全局线性拟合 ---
    slope, intercept, r_value_lin, _, _ = stats.linregress(x_data, y_data)
    report['linear_global'] = {
        'params': (slope, intercept),
        'r2': r_value_lin**2,
        'func': linear_func,
        'inv_func': inverse_linear,
        'name': '全局线性 (Global Linear)'
    }

    # --- B. 全局非线性拟合 (指数衰减) ---
    try:
        p0 = [np.max(y_data)-np.min(y_data), 0.5, np.min(y_data)]
        popt_exp, _ = curve_fit(exp_decay_func, x_data, y_data, p0=p0, maxfev=5000)
        residuals = y_data - exp_decay_func(x_data, *popt_exp)
        r2_exp = 1 - (np.sum(residuals**2) / np.sum((y_data - np.mean(y_data))**2))
        
        report['exp_global'] = {
            'params': popt_exp,
            'r2': r2_exp,
            'func': exp_decay_func,
            'inv_func': inverse_exp,
            'name': '指数衰减 (Exp Decay)'
        }
    except:
        report['exp_global'] = {'r2': -1} # 拟合失败标记

    # --- C. 寻找最佳线性范围 (Sliding Window) ---
    best_subset_r2 = -1
    min_points = 4 # 至少需要4个点
    
    if len(x_data) >= min_points:
        for i in range(len(x_data) - min_points + 1):
            for j in range(i + min_points, len(x_data) + 1):
                sub_x = x_data[i:j]
                sub_y = y_data[i:j]
                s, i_cept, r, _, _ = stats.linregress(sub_x, sub_y)
                if r**2 > best_subset_r2:
                    best_subset_r2 = r**2
                    report['best_linear_range'] = {
                        'range_text': f"{sub_x[0]} - {sub_x[-1]} mM",
                        'indices': (i, j),
                        'params': (s, i_cept),
                        'r2': best_subset_r2,
                        'func': linear_func,
                        'inv_func': inverse_linear,
                        'name': f"最佳线性范围 ({sub_x[0]}-{sub_x[-1]})"
                    }
    else:
        report['best_linear_range'] = None

    # --- D. 最终推荐 ---
    # 如果指数R2比线性高出0.02以上，推荐指数，否则推荐线性
    if report['exp_global']['r2'] > report['linear_global']['r2'] + 0.02:
        report['recommended'] = report['exp_global']
    else:
        report['recommended'] = report['linear_global']
        
    return report

# ==========================================
# 3. Streamlit 界面
# ==========================================
st.set_page_config(page_title="BioSensor Pro Max", layout="wide")
st.title("🧬 生物传感器智能分析系统 (Auto-Fit版)")

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
    # 这里的设置将直接影响 process_image 里的动态半径计算
    rows = st.number_input("行数 (Rows)", 1, 10, 2)
    cols = st.number_input("列数 (Cols)", 1, 20, 7)

# --- 页面逻辑 ---
tab1, tab2 = st.tabs(["📏 建立标曲 (Calibration)", "🧪 样品检测 (Test)"])

if 'fit_report' not in st.session_state:
    st.session_state.fit_report = None

with tab1:
    uploaded_calib = st.file_uploader("上传标准品图片", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_calib:
        col_img, col_res = st.columns([1, 1])
        
        with col_img:
            # 调用新版 process_image
            img_show, s_vals, count = process_image(uploaded_calib, rows, cols)
            st.image(img_show, channels="BGR", use_container_width=True, caption=f"识别到 {count} 个孔")
        
        with col_res:
            if count != len(known_concs):
                st.warning(f"⚠️ 数量不匹配：浓度有 {len(known_concs)} 个，但识别到 {count} 个孔。")
                st.info("提示：请检查侧边栏的‘阵列布局’是否正确，这会影响孔径识别。")
            else:
                # 运行拟合引擎
                report = auto_fit_engine(known_concs, s_vals)
                st.session_state.fit_report = report
                
                rec_model = report['recommended']
                st.success(f"✅ 推荐模型：{rec_model['name']}")
                st.metric("拟合优度 (R²)", f"{rec_model['r2']:.4f}")
                
                # 绘图
                fig, ax = plt.subplots()
                x_smooth = np.linspace(min(known_concs), max(known_concs), 100)
                
                # 原始点
                ax.scatter(known_concs, s_vals, color='black', label='Raw Data', zorder=5)
                
                # 绘制推荐曲线
                y_fit = rec_model['func'](x_smooth, *rec_model['params'])
                ax.plot(x_smooth, y_fit, 'r-', label='Recommended Fit')
                
                # 如果有最佳线性范围，额外画绿线
                best_range = report.get('best_linear_range')
                if best_range and best_range['r2'] > report['linear_global']['r2']:
                    idx1, idx2 = best_range['indices']
                    ax.scatter(known_concs[idx1:idx2], s_vals[idx1:idx2], 
                               s=100, facecolors='none', edgecolors='lime', linewidth=2, label='Best Linear Range')
                    st.info(f"💡 发现更优的局部线性范围：{best_range['range_text']} (R²={best_range['r2']:.4f})")

                ax.legend()
                ax.set_xlabel("Concentration")
                ax.set_ylabel("Signal S")
                st.pyplot(fig)

with tab2:
    if st.session_state.fit_report is None:
        st.info("👈 请先在‘建立标曲’页面完成分析")
    else:
        report = st.session_state.fit_report
        
        # 让用户选择用哪个模型
        options = {
            "智能推荐": report['recommended'],
            "全局线性": report['linear_global'],
            "全局非线性": report['exp_global']
        }
        if report.get('best_linear_range'):
            options[f"最佳线性范围 ({report['best_linear_range']['range_text']})"] = report['best_linear_range']
            
        choice = st.selectbox("选择计算模型：", list(options.keys()))
        selected_model = options[choice]
        
        uploaded_test = st.file_uploader("上传待测样品", type=['jpg', 'png', 'jpeg'], key='test')
        if uploaded_test:
            img_test, s_test, count_test = process_image(uploaded_test, rows, cols)
            st.image(img_test, channels="BGR", caption=f"检测到 {count_test} 个样品")
            
            if count_test > 0:
                results = []
                for s in s_test:
                    # 使用选中模型的反函数
                    conc = selected_model['inv_func'](s, *selected_model['params'])
                    results.append(conc)
                
                st.dataframe({
                    "Sample": [f"#{i+1}" for i in range(len(results))],
                    "S-Value": [f"{v:.1f}" for v in s_test],
                    "Conc (mM)": [f"{c:.4f}" for c in results]
                }, use_container_width=True)


