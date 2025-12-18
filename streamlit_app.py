import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from PIL import Image, ImageOps

# ==========================================
# 核心算法区 (更智能的排序)
# ==========================================

def process_image(img_file_buffer, rows, cols, is_standard=True):
    # 1. 标准化缩放 (保持图片宽度为 1000px，统一所有参数的基准)
    image_pil = Image.open(img_file_buffer)
    image_pil = ImageOps.exif_transpose(image_pil)
    target_width = 1000
    w_percent = (target_width / float(image_pil.size[0]))
    h_size = int((float(image_pil.size[1]) * float(w_percent)))
    image_pil = image_pil.resize((target_width, h_size), Image.Resampling.LANCZOS)
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    
    output_img = img.copy()
    
    # 2. 图像增强 (核心步骤：让透明孔显形)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 使用 CLAHE (对比度受限自适应直方图均衡化)
    # 这步操作能极大增强透明孔边缘与背景的对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # 稍微模糊一下，去除噪点
    gray_blur = cv2.GaussianBlur(enhanced_gray, (9, 9), 2)

    # 3. 霍夫圆检测
    # 因为宽度固定1000了，这里的参数我们可以调教得非常通用
    circles = cv2.HoughCircles(
        gray_blur, cv2.HOUGH_GRADIENT, dp=1, 
        minDist=80,         # 两个圆心至少相距80像素 (对于1000宽的图，这能有效防止重叠)
        param1=50,          # 边缘检测阈值 (调低点，为了识别透明孔微弱的边缘)
        param2=25,          # 圆心检测阈值 (越小越灵敏，为了不漏掉透明孔)
        minRadius=35,       # 1000宽图下的经验半径
        maxRadius=60        # 1000宽图下的经验半径
    )

    s_values = []
    sorted_circles = []

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # --- 智能网格筛选 (防止找多了) ---
        # 霍夫变换经常会找到背景里的杂物，我们需要利用 "Grid" 特性来过滤
        
        # 1. 先按 Y 轴排序
        circles = sorted(circles, key=lambda x: x[1])
        
        final_candidates = []
        expected_total = rows * cols
        
        # 简单的聚类逻辑：
        # 如果我们找到了 20 个圆，但只要 14 个。
        # 我们优先保留那些 "半径大小正常" 且 "位置比较整齐" 的。
        # 这里使用一个简单的逻辑：优先取 "Y轴最接近中间区域" 的圆 (假设板子在图中间)
        
        if len(circles) > expected_total:
            # 这种简单粗暴的截取通常有效，因为霍夫通常给予强边缘更高的权重
            # 但为了保险，我们可以根据半径过滤一下
            # 计算中位半径
            median_r = np.median([c[2] for c in circles])
            # 只保留半径差异在 20% 以内的圆
            circles = [c for c in circles if abs(c[2] - median_r) < median_r * 0.3]
            
            # 再次排序并截取
            circles = sorted(circles, key=lambda x: x[1])
            if len(circles) > expected_total:
                 circles = circles[:expected_total]

        # 2. 网格排序 (Row-Major)
        for r in range(rows):
            start_idx = r * cols
            end_idx = min((r + 1) * cols, len(circles))
            if start_idx < len(circles):
                row_circles = circles[start_idx : end_idx]
                # 按 X 轴排序
                row_circles = sorted(row_circles, key=lambda x: x[0])
                sorted_circles.extend(row_circles)

        # 4. 提取 S 值
        for i, (x, y, r) in enumerate(sorted_circles):
            # 视觉标记
            cv2.circle(output_img, (x, y), r, (0, 255, 0), 4)
            # 在圆心画个十字，方便确认是否对准
            cv2.line(output_img, (x-10, y), (x+10, y), (0, 0, 255), 2)
            cv2.line(output_img, (x, y-10), (x, y+10), (0, 0, 255), 2)
            
            # 文字
            cv2.putText(output_img, f"{i+1}", (x-20, y-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # 提取颜色：只取圆心中间 60% 的区域，避开边缘
            mask = np.zeros(img.shape[:2], dtype="uint8")
            roi_r = int(r * 0.6) 
            cv2.circle(mask, (x, y), roi_r, 255, -1)
            
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mean_val = cv2.mean(hsv, mask=mask)
            s_values.append(mean_val[1])

    return output_img, s_values, len(sorted_circles)

def exponential_decay(x, a, b, c):
    return a * np.exp(-b * x) + c

def inverse_exponential(y, a, b, c):
    try:
        val = (y - c) / a
        if val <= 0: return 0 
        return -(1/b) * np.log(val)
    except:
        return 0

# ==========================================
# 界面显示区
# ==========================================
st.set_page_config(page_title="BioSensor Pro", layout="wide")
st.title("🧬 生物传感器智能分析系统 ")

# --- 侧边栏：全局设置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    st.subheader("1. 标准品浓度")
    conc_input = st.text_area(
        "输入浓度 (逗号分隔)", 
        "0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0"
    )
    try:
        known_concs = [float(x.strip()) for x in conc_input.split(',')]
        std_count = len(known_concs)
    except:
        st.error("浓度格式错误")
        std_count = 14

    st.subheader("2. 布局模式")
    # 这里的布局仅用于排序，帮助程序知道什么时候换行
    layout_mode = st.radio("选择板孔排列方式:", ["固定 2行 x 7列 (标准)", "自定义行列"])
    
    if layout_mode == "自定义行列":
        user_rows = st.number_input("行数 (Rows)", min_value=1, value=2)
        user_cols = st.number_input("列数 (Cols)", min_value=1, value=7)
    else:
        user_rows, user_cols = 2, 7

# --- 主页面 ---
col1, col2 = st.columns(2)

curve_params = None 

# --- 左边：标准曲线 ---
with col1:
    st.markdown("### 步骤 1: 建立标曲")
    uploaded_calib = st.file_uploader("上传标准品图片", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_calib:
        # 智能推断：如果是2x7布局，但浓度只有5个，怎么排？
        # 这里为了简单，标准品尽量按满排或者用户指定的行列排
        img_show, s_vals, count = process_image(uploaded_calib, user_rows, user_cols)
        
        st.image(img_show, channels="BGR", use_column_width=True)
        
        if count != std_count:
            st.warning(f"⚠️ 数量警告: 输入了 {std_count} 个浓度，但检测到 {count} 个孔。")
            st.info("提示：请检查左侧'布局模式'是否设置正确，或调整图片拍摄角度。")
        else:
            st.success(f"✅ 成功匹配 {count} 个点")
            
            # 拟合
            x_data = np.array(known_concs)
            y_data = np.array(s_vals)
            p0 = [np.max(y_data)-np.min(y_data), 0.5, np.min(y_data)]
            try:
                popt, pcov = curve_fit(exponential_decay, x_data, y_data, p0=p0, maxfev=5000)
                curve_params = popt
                
                # R2计算
                residuals = y_data - exponential_decay(x_data, *popt)
                ss_res = np.sum(residuals**2)
                ss_tot = np.sum((y_data - np.mean(y_data))**2)
                r2 = 1 - (ss_res / ss_tot)
                
                # 画图
                fig, ax = plt.subplots(figsize=(5, 3)) # 图小一点
                ax.scatter(x_data, y_data, color='blue', alpha=0.6)
                x_smooth = np.linspace(min(x_data), max(x_data), 100)
                ax.plot(x_smooth, exponential_decay(x_smooth, *popt), 'r--')
                ax.set_title(f"Fit: R²={r2:.4f}")
                st.pyplot(fig)
                
            except Exception as e:
                st.error(f"拟合失败: {e}")

# --- 右边：未知样品 ---
with col2:
    st.markdown("### 步骤 2: 检测样品")
    
    if curve_params is None:
        st.info("👈 等待标曲建立...")
    else:
        uploaded_test = st.file_uploader("上传样品图片", type=['jpg', 'png', 'jpeg'], key="test")
        
        if uploaded_test:
            # 这里的巧妙之处：我们传入用户设定的 rows/cols
            # 这样即使用户传了一张只有 1行 3个孔 的图，只要设成 1行 3列，就能正确识别
            img_test_show, s_vals_test, count_test = process_image(uploaded_test, user_rows, user_cols)
            
            st.image(img_test_show, channels="BGR", use_column_width=True)
            st.write(f"检测到 {count_test} 个样品")
            
            if count_test > 0:
                results = []
                for s in s_vals_test:
                    conc = inverse_exponential(s, *curve_params)
                    results.append(conc)
                
                # 结果展示优化
                st.dataframe({
                    "孔号": [f"#{i+1}" for i in range(len(results))],
                    "S值": [f"{v:.1f}" for v in s_vals_test],
                    "预测浓度 (mM)": [f"{c:.4f}" for c in results]
                }, use_container_width=True)