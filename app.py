import pandas as pd
import streamlit as st
from model import DATA_PATH, predict, train

st.set_page_config(page_title="ReviewGuard | Fake Review Detection", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #fff4fb 0%, #f4fbff 45%, #eaf7ff 100%);
        color: #3f4962;
    }
    .block-container {
        max-width: 1100px;
        padding-top: 2.2rem;
    }
    .hero {
        background: linear-gradient(115deg, #ffd9ec, #d9f2ff);
        padding: 2.2rem;
        border: 1px solid #c7e9ff;
        border-radius: 22px;
        margin-bottom: 1.2rem;
        box-shadow: 0 8px 24px rgba(184, 220, 245, 0.35);
    }
    .hero h1 {
        font-size: 2.8rem;
        margin: 0 0 .35rem;
        color: #5d496d;
    }
    .hero p {
        color: #50657b;
        font-size: 1.08rem;
        margin: 0;
    }
    .note {
        color: #71839a;
        font-size: .87rem;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffe6f2 0%, #e8f7ff 100%);
        border-right: 1px solid #d6ecfb;
    }
    [data-testid="stMetricValue"] {
        color: #5c5a87;
    }
    [data-testid="stInfo"] {
        background-color: #edf8ff;
        border: 1px solid #cfeeff;
        color: #4e6578;
    }
    div[data-baseweb="tab-list"] {
        gap: 0.4rem;
    }
    button[kind="primary"], .stButton > button {
        border-radius: 12px;
        border: 1px solid #d9b3d2;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='hero'>
        <h1>🛡️ ReviewGuard</h1>
        <p>Spot suspicious product reviews with machine learning, TF-IDF, and review metadata.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.info("Educational demo: the included reviews are synthetic. Predictions are not verified fraud findings.")


@st.cache_resource(show_spinner="Training demonstration models...")
def demo_model():
    return train(pd.read_csv(DATA_PATH))


if "custom_model" not in st.session_state:
    st.session_state.custom_model = None

with st.sidebar:
    st.header("⚙️ Model settings")
    upload = st.file_uploader("Train on your labeled CSV", type=["csv"])
    st.caption(
        "Required columns: review, label (fake/genuine or 1/0). "
        "Optional: rating, verified_purchase, reviewer_review_count."
    )
    if upload is not None and st.button("Train uploaded dataset", use_container_width=True):
        try:
            data = pd.read_csv(upload)
            with st.spinner("Fitting and evaluating models..."):
                st.session_state.custom_model = train(data)
            st.success("Your dataset is loaded for this session.")
        except Exception as exc:
            st.error(f"Unable to train: {exc}")
    if st.button("Use demo model", use_container_width=True):
        st.session_state.custom_model = None
    st.divider()
    st.caption("Data stays in the current Streamlit session when using the upload option. No database required.")

try:
    payload = st.session_state.custom_model or demo_model()
except Exception as exc:
    st.error(f"Model initialization failed: {exc}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["🔎 Check a review", "📊 Model performance", "📘 How it works"])
with tab1:
    st.subheader("Analyze a product review")
    review = st.text_area(
        "Review text",
        height=160,
        placeholder="Example: The product arrived on time and works well...",
    )
    left, mid, right = st.columns(3)
    with left:
        rating = st.selectbox("Rating", ["Not provided", 1, 2, 3, 4, 5], index=0)
    with mid:
        verified = st.selectbox("Verified purchase", ["Unknown", "Yes", "No"])
    with right:
        count = st.number_input("Reviewer's past reviews (0 = unknown)", min_value=0, value=0, step=1)
    if st.button("Analyze review", type="primary", use_container_width=True):
        if len(review.strip()) < 10:
            st.warning("Enter a review with at least 10 characters.")
        else:
            score, label = predict(
                payload,
                review.strip(),
                None if rating == "Not provided" else rating,
                None if verified == "Unknown" else verified == "Yes",
                None if count == 0 else count,
            )
            st.subheader(label)
            st.metric("Estimated probability of fake label", f"{score:.1%}")
            st.progress(score)
            st.caption("This is a model output, not proof of deception. Confidence can be poorly calibrated.")
with tab2:
    st.subheader("Held-out test performance")
    st.caption(
        "Results use a stratified 75/25 split; two models are compared on the same test set for demo simplicity."
    )
    st.write(f"**Selected model:** {payload['model_name']}")
    a, b, c = st.columns(3)
    rep = payload["report"]
    a.metric("Fake precision", f"{rep['Fake']['precision']:.2f}")
    b.metric("Fake recall", f"{rep['Fake']['recall']:.2f}")
    c.metric("Fake F1-score", f"{rep['Fake']['f1-score']:.2f}")
    st.bar_chart(pd.Series(payload["scores"], name="Fake F1-score"))
    st.write("**Confusion matrix** (rows = actual, columns = predicted)")
    cm = pd.DataFrame(payload["confusion_matrix"], index=["Genuine", "Fake"], columns=["Genuine", "Fake"])
    st.dataframe(cm, use_container_width=True)
    st.caption(f"Training samples: {payload['train_size']} · Test samples: {payload['test_size']}")
with tab3:
    st.markdown(
        """**Pipeline:** CSV → cleaning and feature extraction → stratified train/test split →
TF-IDF (unigrams + bigrams) plus numeric review signals → class-balanced Logistic Regression / Random Forest →
F1 comparison → prediction. The TF-IDF vocabulary and scaling parameters are fitted only on the training subset.

**Important limits:** Classifying deception from text alone is intrinsically uncertain. A synthetic demo cannot establish real-world performance. For a credible project, replace it with an independently labeled, real review dataset; keep product/reviewer groups separated across splits where IDs exist; and tune/select models with validation or cross-validation rather than the final test set.
"""
    )
