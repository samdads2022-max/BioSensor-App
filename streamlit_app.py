import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import stats
from PIL import Image, ImageOps

# ==========================================
# 1. 辅助算法：一维 K-Means 聚类 (抗歪斜核心)
# ==========================================
def robust_sort_circles(circles, rows):
    """
    最稳健的排序策略：
    1. 用 K-Means 把 Y 坐标聚类成 N 行。
    2. 计算每一行的平均 Y 值，确定行的上下顺序。
    3. 在每一行内部，按 X 坐标排序。
    """
    if not circles: return []
    
    # 提取 Y 坐标
    y_coords = np.array([c[1] for c in circles]).reshape(-1, 1)
    
    # 1. K-Means 聚类 (这里用 OpenCV 自带的，更稳)
    # 如果检测到的圆少于行数，就设 K = 圆的数量
    k = min(rows, len(circles))
    if k <= 1:
        # 只有一行，直接按 X 排序
        return sorted(circles, key=lambda x: x[0])
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(np.float32(y_coords), k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    
    # 2. 将圆按 Label 分组
    row_groups = {}
    for i, label in enumerate(labels.flatten()):
        if label not in row_groups: row_groups[label] = []
        row_groups[label].append(circles[i])
        
    # 3. 确定行的上下顺序 (按每组的平均 Y 值排序)
    # row_order 存储的是 [(label, avg_y), (label, avg_y)...]
    row_stats = []
    for label, group in row_groups.items():
        avg_y = np.mean([c[1] for c in group])
        row_stats.append((label, avg_y))
    
    # 按 avg_y 从小到大排序 (Y小的是上面)
    row_stats.sort(key=lambda x: x[1])
    
    # 4. 生成最终有序列表 (行内按 X 排序)
    final_sorted_circles = []
    for label, _ in row_stats:
        group = row_groups[label]
        # 行内按 X 从小到大排序
        group.sort(key=lambda x: x[0])
        final_sorted_circles.extend(group)
        
    return final_sorted_circles

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
    min_dist_param = int(approx_diameter * 0.8)
    
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
        
        # --- 步骤 A: 颜色打分 ---
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
            cv2.circle(mask, (x, y), int(r * 0.4), 255, -1)
            score = cv2.mean(score_img, mask=mask)[0]
            candidates.append({'data': (x, y, r), 'score': score})
        
        # --- 步骤 B: 筛选逻辑重构 (关键修改) ---
        
        # 1. 按分数排序
        candidates.sort(key=lambda k: k['score'], reverse=True)
        
        # 2. 先保留“满板”数量的圆 (Rows * Cols)
        #    注意：这里先不应用用户的 required_count，而是先填满网格。
        #    这样可以保证颜色浅的孔（只要比背景强）也能入选。
        max_possible_slots = rows * cols
        if len(candidates) > max_possible_slots:
            candidates = candidates[:max_possible_slots]
        
        accepted_circles = [c['data'] for c in candidates]
        
        # --- 步骤 C: 空间排序 (从上到下，从左到右) ---
        # 此时我们手里的圆是乱序的，但数量是对的（或者包含了所有有效孔）
        spatial_sorted_circles = robust_sort_circles(accepted_circles, rows)

        # --- 步骤 D: 用户截断 (Apply Limit) ---
        # 排序完成后，再根据用户滑块的数值，从尾部切断
        # 这样就能保证减去的一定是最后几个孔
        if required_count is not None and required_count > 0:
            if len(spatial_sorted_circles) > required_count:
                final_circles = spatial_sorted_circles[:required_count]
            else:
                final_circles = spatial_sorted_circles
        else:
            final_circles = spatial_sorted_circles

        # --- 步骤 E: 取值与画图 ---
        roi_scale = 0.7 
        for i, (x, y, r) in enumerate(final_circles):
            draw_r = int(r * roi_scale)
            cv2.circle(output_img, (x, y), draw_r, (0, 255, 0), 3)
            cv2.putText(output_img, f"{i+1}", (x-15, y+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            mask = np.zeros(img.shape[:2], dtype="uint8")
            cv2.circle(mask, (x, y), int(r * (roi_scale - 0.1)), 255, -1)
            mean_val = cv2.mean(score_img, mask=mask)[0]
            s_values.append(mean_val)

    return output_img, s_values, len(final_circles)
# ==========================================
# 3. 拟合引擎 (修复 min_pts 作用域问题)
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
    
    # 局部线性
    best_r2 = -1
    min_pts = 5 # 定义在这里
    
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
                        'x_range': sx
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
        index=1
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
        
        # --- 新增功能：标曲数量控制器 ---
        # 默认最大值是用户输入的浓度个数
        max_points = len(known_concs)
        
        if max_points < 3:
            st.error("⚠️ 请至少输入 3 个浓度值以进行拟合。")
        else:
            # 1. 在这里加一个滑块，允许用户减少拟合点的数量
            # 默认值设为 max_points (全选)
            st.markdown("##### 🎚️ 有效标孔选择")
            fit_count = st.slider(
                "拟合孔数 (从第1个孔开始保留)", 
                min_value=3, 
                max_value=max_points, 
                value=max_points,
                help="如果最后几个孔（高浓度）出现异常，可以减小此数值将其剔除。"
            )

            with col1:
                # 2. 将滑块的值传给图像处理函数
                # process_image 会自动按照空间排序，只保留前 fit_count 个孔
                img, vals, count = process_image(uploaded_calib, rows, cols, required_count=fit_count, analysis_mode=analysis_mode)
                
                st.image(img, channels="BGR", use_container_width=True, caption=f"当前拟合前 {count} 个点")
            
            with col2:
                if count != fit_count:
                    # 这种情况通常是图片太模糊，连要求的数量都没找齐
                    st.error(f"⚠️ 识别数量不足！计划拟合 {fit_count} 个，但只找到 {count} 个有效圆。")
                else:
                    # 3. 关键步骤：数据对齐
                    # 用户可能输入了 14 个浓度，但滑块只选了 11 个
                    # 我们必须截取前 11 个浓度，才能和 11 个 S值 对应
                    current_x = np.array(known_concs[:count])
                    current_y = np.array(vals)
                    
                    # 4. 运行拟合引擎
                    report = auto_fit_engine(current_x, current_y)
                    st.session_state.fit_report = report
                    rec = report['recommended']
                    
                    st.success(f"✅ 推荐模型: {rec['name']}")
                    st.metric("R²", f"{rec['r2']:.4f}")
                    
                    # 5. 绘图
                    fig, ax = plt.subplots()
                    xs = np.linspace(min(current_x), max(current_x), 100)
                    
                    # 画原始数据点
                    ax.scatter(current_x, current_y, color='black', label='Data', zorder=5)
                    
                    # 画拟合线
                    ax.plot(xs, rec['func'](xs, *rec['params']), 'r-', linewidth=2, label='Fit Curve')
                    
                    # 画被剔除的点 (如果有的话，用灰色叉叉表示)
                    if len(known_concs) > count:
                        ignored_x = known_concs[count:]
                        # 这里没法画对应的Y值，因为没取出来，但在逻辑上提示用户即可
                        st.caption(f"已忽略末尾 {len(known_concs) - count} 个高浓度点")
                    
                    # 局部线性
                    br = report.get('best_linear_range')
                    if br and br['r2'] > report['linear_global']['r2']:
                        i1, i2 = br['indices']
                        ax.scatter(current_x[i1:i2], current_y[i1:i2], s=150, facecolors='none', edgecolors='lime', lw=2, label='Best Range')
                        st.info(f"💡 最佳局部线性范围: {br['range_text']} (R²={br['r2']:.4f})")
                    
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








