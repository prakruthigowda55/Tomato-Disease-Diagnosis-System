import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
import os
from PIL import Image
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph
)

from reportlab.lib.styles import getSampleStyleSheet

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="Tomato Disease Diagnosis System",
    page_icon="🍅",
    layout="wide"
)


# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "models/best_efficientnet.keras"
    )

    return model

model = load_model()

# ==========================================
# SIDEBAR
# ==========================================

with st.sidebar:

    st.title("AI System")

    st.success(
        "System Ready"
    )

    st.write(
        " Model: EfficientNetB0"
    )

    st.write(
        " Accuracy: 96.88%"
    )

    st.write(
        " Classes: 4"
    )

    st.write(
        " Grad-CAM Enabled"
    )

    st.write(
        " PDF Reports Enabled"
    )

    st.write(
        " Dashboard Enabled"
    )

    st.markdown("---")

    st.info(
        "Upload a tomato leaf image and click Predict Disease."
    )

# ==========================================
# CLASS NAMES
# ==========================================

class_names = [

    "Tomato_Early_blight",

    "Tomato_Late_blight",

    "Tomato_Leaf_Mold",

    "Tomato_healthy"

]

# ==========================================
# DISEASE DATABASE
# ==========================================

disease_data = {

    "Tomato_Early_blight": {

        "cause":
        "Alternaria solani fungal infection",

        "organic":
        "Neem Oil Spray",

        "chemical":
        "Mancozeb",

        "prevention":
        "Remove infected leaves"
    },

    "Tomato_Late_blight": {

        "cause":
        "Phytophthora infestans",

        "organic":
        "Copper-based fungicide",

        "chemical":
        "Chlorothalonil",

        "prevention":
        "Avoid excess moisture"
    },

    "Tomato_Leaf_Mold": {

        "cause":
        "Passalora fulva fungus",

        "organic":
        "Improve ventilation",

        "chemical":
        "Copper Fungicide",

        "prevention":
        "Reduce humidity"
    },

    "Tomato_healthy": {

        "cause":
        "No Disease",

        "organic":
        "Maintain good farming practices",

        "chemical":
        "Not Required",

        "prevention":
        "Regular monitoring"
    }
}



# ==========================================
# GRAD-CAM FUNCTION
# ==========================================

def generate_gradcam(model, img_array):

    base_model = model.get_layer("efficientnetb0")

    last_conv_layer = base_model.get_layer("top_conv")

    classifier = tf.keras.Sequential([

        model.layers[3],
        model.layers[4],
        model.layers[5],
        model.layers[6],
        model.layers[7]

    ])

    grad_model = tf.keras.models.Model(

        inputs=base_model.input,

        outputs=[

            last_conv_layer.output,

            base_model.output

        ]
    )

    with tf.GradientTape() as tape:

        conv_outputs, features = grad_model(
            img_array
        )

        tape.watch(conv_outputs)

        predictions = classifier(
            features
        )

        pred_index = tf.argmax(
            predictions[0]
        )

        class_channel = predictions[
            :,
            pred_index
        ]

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(
        conv_outputs * pooled_grads,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    heatmap /= (
        tf.reduce_max(heatmap)
        + 1e-8
    )

    return heatmap.numpy()

# ==========================================
# PDF Report FUNCTION
# ==========================================

def create_pdf_report(

    disease,

    confidence,

    severity,

    info

):

    pdf_file = "report.pdf"

    doc = SimpleDocTemplate(
        pdf_file
    )

    styles = getSampleStyleSheet()

    elements = []

    elements.append(
        Paragraph(
            "Tomato Health Report",
            styles["Title"]
        )
    )

    elements.append(
        Paragraph(
            f"Disease: {disease}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Confidence: {confidence:.2f}%",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Severity: {severity:.2f}%",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Cause: {info['cause']}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Organic: {info['organic']}",
            styles["BodyText"]
        )
    )

    elements.append(
        Paragraph(
            f"Chemical: {info['chemical']}",
            styles["BodyText"]
        )
    )

    doc.build(elements)

    return pdf_file



# ==========================================
# TITLE
# ==========================================

st.markdown("""
# 🍅 Tomato Disease Diagnosis System

### Explainable Tomato Disease Detection, Severity Assessment and Crop Advisory System
---
""")
# ==========================================
# FILE UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "Upload Tomato Leaf Image",
    type=["jpg", "jpeg", "png"]
)

# ==========================================
# MAIN LOGIC
# ==========================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="📷 Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.info("""
        Model: EfficientNetB0

        Accuracy: 96.88%

        Classes: 4

        Grad-CAM Enabled
        """)

    predict = st.button("Predict Disease")
    if predict:
        st.session_state["predict_done"] = True

    if st.session_state.get("predict_done", False):

    # prediction calculations

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Prediction",
            "🔥 Grad-CAM",
            "⚠ Severity",
            "🌿 Nutrient Advisor",
            "📊 Dashboard"
        ])

        with tab1:
             # ==========================
            # PREPROCESS IMAGE
            # ==========================

            img = image.resize((224,224))

            img_array = np.array(img)

            img_normalized = img_array.astype(np.float32)

            input_image = np.expand_dims(
                img_normalized,
                axis=0
            )

            # ==========================
            # PREDICTION
            # ==========================

            prediction = model.predict(input_image)

            # ==========================
            # GENERATE GRAD-CAM
            # ==========================

            heatmap = generate_gradcam(
                model,
                input_image
            )

            st.write("Raw Prediction:", prediction)

            predicted_index = np.argmax(prediction)

            confidence = np.max(prediction) * 100

            disease = class_names[predicted_index]


            # ==========================
            # SAVE PREDICTION HISTORY
            # ==========================

            history_file = "prediction_history.csv"

            new_record = pd.DataFrame({

                "Disease": [disease],

                "Confidence": [confidence]

            })

            if os.path.exists(history_file):

                old_df = pd.read_csv(
                    history_file
                )

                history_df = pd.concat(
                    [old_df, new_record],
                    ignore_index=True
                )

            else:

                history_df = new_record

            history_df.to_csv(

                history_file,

                index=False
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Disease",
                    disease.replace("Tomato_", "")
                )

            with col2:
                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

            with col3:

                st.metric(
                    "Crop Health",
                    "Pending"
        )


        with tab2:
            # ==========================
            # GRAD-CAM VISUALIZATION
            # ==========================

            heatmap = cv2.resize(

                heatmap,

                (
                    img_array.shape[1],
                    img_array.shape[0]
                )
            )

            heatmap = np.uint8(
                255 * heatmap
            )

            heatmap_color = cv2.applyColorMap(

                heatmap,

                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(

                cv2.cvtColor(
                    img_array.astype(np.uint8),
                    cv2.COLOR_RGB2BGR
                ),

                0.6,

                heatmap_color,

                0.4,

                0
            )

            st.header(
                "🔥 Explainable AI Analysis"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.image(
                    image,
                    caption="Original Image",
                    use_container_width=True
                )

            with col2:

                st.image(
                    cv2.cvtColor(
                        overlay,
                        cv2.COLOR_BGR2RGB
                    ),
                    caption="Grad-CAM Heatmap",
                    use_container_width=True
                )


        with tab3:
                # ==========================
            # SEVERITY ANALYSIS
            # ==========================

            img_rgb = np.array(image)

            hsv = cv2.cvtColor(
                img_rgb,
                cv2.COLOR_RGB2HSV
            )

            lower = np.array([5,50,20])

            upper = np.array([35,255,255])

            mask = cv2.inRange(
                hsv,
                lower,
                upper
            )

            affected_pixels = np.sum(
                mask > 0
            )

            total_pixels = (
                mask.shape[0]
                *
                mask.shape[1]
            )

            severity = (
                affected_pixels
                /
                total_pixels
            ) * 100

            if severity < 20:

                status = "Mild"

            elif severity < 50:

                status = "Moderate"

            else:

                status = "Severe"

            st.header("Severity Analysis")

            st.subheader(
                "⚠ Disease Severity"
            )

            st.progress(
                min(int(severity), 100)
            )

            if status == "Mild":

                st.success(
                    f"Severity: {severity:.2f}% ({status})"
                )

            elif status == "Moderate":

                st.warning(
                    f"Severity: {severity:.2f}% ({status})"
                )

            else:

                st.error(
                    f"Severity: {severity:.2f}% ({status})"
                )

            # ==========================
            # CROP HEALTH INDEX
            # ==========================
            if disease == "Tomato_healthy":

                health_score = 100

            else:
                health_score = max(
                    0,
                    100 - severity
            )

            st.subheader(
                "🌱 Crop Health Index"
            )

            st.progress(
                int(health_score)
            )

            st.metric(
                "Health Score",
                f"{health_score:.0f}/100"
            )
            # ==========================
            # TREATMENT
            # ==========================

            info = disease_data[disease]

            st.header(
                "Treatment Recommendation"
            )

            st.write(
                "Cause:",
                info["cause"]
            )

            st.write(
                "Organic Treatment:",
                info["organic"]
            )

            st.write(
                "Chemical Treatment:",
                info["chemical"]
            )

            st.write(
                "Prevention:",
                info["prevention"]
            )

        with tab4:
            # ==========================
            # NUTRIENT ADVISOR
            # ==========================

            if disease == "Tomato_healthy":

                st.header(
                    "Nutrient Advisor"
                )

                symptom = st.selectbox(

                    "Select Symptom",

                    [

                        "Older leaves yellow",

                        "Brown leaf edges",

                        "Green veins with yellow tissue"

                    ]
                )

                if symptom == "Older leaves yellow":

                    st.success(
                        "Nitrogen Deficiency"
                    )

                    st.write(
                        "Apply Urea or Compost"
                    )

                elif symptom == "Brown leaf edges":

                    st.success(
                        "Potassium Deficiency"
                    )

                    st.write(
                        "Apply Potassium Sulphate"
                    )

                elif symptom == "Green veins with yellow tissue":

                    st.success(
                        "Magnesium Deficiency"
                    )

                    st.write(
                        "Apply Epsom Salt"
                    )


            # ==========================
            # PDF download button
            # ==========================


            pdf_path = create_pdf_report(

                disease,

                confidence,

                severity,

                info

            )

            with open(
                pdf_path,
                "rb"
            ) as pdf_file:

                st.download_button(

                    label="📄 Download Report",

                    data=pdf_file,

                    file_name=
                    "Tomato_Report.pdf",

                    mime=
                    "application/pdf"
                )
            


        with tab5:
            # ==========================================
            # DASHBOARD
            # ==========================================

            if os.path.exists("prediction_history.csv"):

                st.header("📊 Dashboard")

                history_df = pd.read_csv(
                    "prediction_history.csv"
                )

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "📊 Total Scans",
                        len(history_df)
                        )

                with col2:

                    st.metric(
                        "🎯 Avg Confidence",
                        round(
                            history_df["Confidence"].mean(),
                            2
                            )
                        )

                st.subheader(
                    "Disease Distribution"
                )

                disease_counts = history_df[
                    "Disease"
                ].value_counts()

                st.bar_chart(
                    disease_counts
                )

                st.subheader(
                    "Recent Predictions"
                )

                st.dataframe(
                    history_df.tail(10)
                )

        