import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; padding: 20px; border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    # Suppress all warnings
    warnings.filterwarnings('ignore')
    
    # Load XGBoost silently
    xgb_model = None
    try:
        xgb_model = joblib.load("xgb_model.pkl")
    except:
        xgb_model = None
    
    # Load MLP and Scaler
    mlp_model = None
    scaler = None
    try:
        mlp_model = joblib.load("mlp_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except:
        mlp_model = None
        scaler = None
    
    return xgb_model, mlp_model, scaler

# Initialize session state
if 'predictions' not in st.session_state:
    st.session_state.predictions = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = pd.DataFrame()

# Sidebar navigation
st.sidebar.title("🔒 Fraud Detection System")
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🔍 Predict Fraud", "📊 Results & Analysis", "ℹ️ About"],
    label_visibility="collapsed"
)

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    st.title("💳 Financial Transaction Fraud Detection System")
    st.markdown("---")

    # Welcome section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
### Welcome to the Fraud Detection System
This advanced machine learning system helps identify potentially fraudulent financial transactions in real-time using state-of-the-art algorithms.

#### 🎯 Key Features:
- **Real-time Prediction**: Instant fraud detection for individual transactions
- **Multiple ML Models**: XGBoost and Neural Network models for accurate predictions
- **Comprehensive Analysis**: Detailed insights and risk scoring
- **Interactive Dashboard**: Visualize patterns and trends
- **High Accuracy**: Achieved 98% accuracy with 97% precision on test data

#### 📈 System Performance:
- **AUC-ROC Score**: 0.99+
- **Precision**: 95-100%
- **Recall**: 97-100%
- **Dataset**: Trained on 5 million transactions
        """)

    with col2:
        st.markdown("### Quick Stats")

    # Display metrics
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Total Predictions", len(st.session_state.predictions))
    with metric_col2:
        if st.session_state.predictions:
            fraud_count = sum(st.session_state.predictions)
            st.metric("Fraud Detected", fraud_count)

    st.info("📊 Navigate to **Predict Fraud** to start analyzing transactions")
    st.markdown("---")

    # Dataset overview
    st.subheader("📋 Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
**Transaction Types:**
- Withdrawal
- Deposit
- Transfer
- Payment
        """)
    with col2:
        st.markdown("""
**Key Features:**
- Amount & Velocity
- Spending Deviation
- Geographic Anomaly
- Device & Location
        """)
    with col3:
        st.markdown("""
**Fraud Rate:**
- Overall: 3.59%
- Peak Month: July (3.63%)
- Transfer Type: Highest risk
        """)

    # Model information
    st.markdown("---")
    st.subheader("🤖 Available Models")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
#### XGBoost Model
- **Type**: Gradient Boosting
- **Accuracy**: 98%
- **AUC-ROC**: 0.9912
- **Best for**: Balanced detection
        """)
    with col2:
        st.markdown("""
#### Neural Network (MLP)
- **Type**: Multi-Layer Perceptron
- **Accuracy**: 98%
- **AUC-ROC**: 0.9913
- **Best for**: Pattern recognition
        """)

# ==================== PREDICT PAGE ====================
elif page == "🔍 Predict Fraud":
    st.title("🔍 Transaction Fraud Prediction")
    st.markdown("---")

    # Load models
    xgb_model, mlp_model, scaler = load_models()
    
    if mlp_model is None or scaler is None:
        st.error("❌ MLP Model files not found. Please ensure mlp_model.pkl and scaler.pkl exist.")
        st.stop()
    st.markdown("---")
    
    # Model selection
    available_models = ["Neural Network (MLP)"]
    if xgb_model is not None:
        available_models.insert(0, "XGBoost")
        available_models.append("Ensemble (XGBoost + MLP)")
    
    model_choice = st.selectbox(
        "Select Prediction Model:",
        available_models,
        key="model_select"
    )
    
    st.markdown("---")
    st.subheader("📝 Enter Transaction Details")

    # TRANSACTION INPUTS - SINGLE SECTION (Fixed: All unique keys)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Basic Information")
        transaction_id = st.number_input("Transaction ID", min_value=0, value=100000, key="tid")
        sender_account = st.number_input("Sender Account ID", min_value=0, value=500000, key="sender_acc")
        receiver_account = st.number_input("Receiver Account ID", min_value=0, value=300000, key="receiver_acc")
        amount = st.number_input("Transaction Amount ($)", min_value=0.01, value=100.0, step=0.01, key="amt")

    with col2:
        st.markdown("#### Transaction Details")
        transaction_type = st.selectbox(
            "Transaction Type",
            ["withdrawal", "deposit", "transfer", "payment"],
            key="trans_type"
        )
        transaction_type_encoded = {"withdrawal": 3, "deposit": 0, "transfer": 2, "payment": 1}[transaction_type]

        merchant_category = st.selectbox(
            "Merchant Category",
            ["utilities", "online", "other", "grocery", "travel", "restaurant", "retail", "entertainment"],
            key="merchant_cat"
        )
        merchant_category_encoded = {"utilities": 7, "online": 2, "other": 3, "grocery": 0, "travel": 6, "restaurant": 5, "retail": 4, "entertainment": 1}[merchant_category]

        location = st.selectbox(
            "Location",
            ["Tokyo", "Toronto", "London", "Sydney", "Berlin", "Singapore", "Dubai", "New York"],
            key="loc"
        )
        location_encoded = {"Tokyo": 6, "Toronto": 7, "London": 2, "Sydney": 5, "Berlin": 0, "Singapore": 4, "Dubai": 1, "New York": 3}[location]

        device_used = st.selectbox("Device Used", ["mobile", "atm", "pos", "web"], key="device")
        device_used_encoded = {"mobile": 1, "atm": 0, "pos": 2, "web": 3}[device_used]

        payment_channel = st.selectbox("Payment Channel", ["card", "ACH", "wire_transfer", "UPI"], key="payment")
        payment_channel_encoded = {"card": 2, "ACH": 0, "wire_transfer": 3, "UPI": 1}[payment_channel]

    with col3:
        st.markdown("#### Risk Indicators")
        time_since_last_transaction = st.number_input(
            "Time Since Last Transaction (sec)",
            value=1000.0,
            key="time_gap"
        )
        spending_deviation_score = st.slider(
            "Spending Deviation Score",
            -5.0, 5.0, 0.0,
            step=0.01,
            key="spend_dev"
        )
        velocity_score = st.slider(
            "Velocity Score",
            1, 20, 10,
            key="velocity"
        )
        geo_anomaly_score = st.slider(
            "Geographic Anomaly Score",
            0.0, 1.0, 0.5,
            step=0.01,
            key="geo"
        )

    st.markdown("---")
    st.subheader("⏰ Temporal Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hour = st.slider("Hour of Day", 0, 23, 12, key="hour")
    with col2:
        day = st.slider("Day of Month", 1, 31, 15, key="day")
    with col3:
        day_of_week = st.slider("Day of Week", 0, 6, 3, key="dow")
    with col4:
        month = st.slider("Month", 1, 12, 6, key="month")

    st.markdown("---")
    st.subheader("🌐 Network Features")
    col1, col2 = st.columns(2)
    with col1:
        sender_degree = st.number_input("Sender Degree (unique receivers)", min_value=1, value=5, key="sd")
        sender_total_transaction = st.number_input("Sender Total Transactions", min_value=1, value=10, key="stt")
        sender_avg_amount = st.number_input("Sender Avg Amount ($)", min_value=0.01, value=200.0, key="savg")
        sender_std_amount = st.number_input("Sender Std Dev Amount ($)", min_value=0.0, value=100.0, key="sstd")
        sender_fraud_transaction = st.number_input("Sender Past Frauds", min_value=0, value=0, key="sfraud")

    with col2:
        receiver_degree = st.number_input("Receiver Degree (unique senders)", min_value=1, value=5, key="rd")
        receiver_total_transaction = st.number_input("Receiver Total Transactions", min_value=1, value=10, key="rtt")
        receiver_fraud_transaction = st.number_input("Receiver Past Frauds", min_value=0, value=0, key="rfraud")

    # Calculate derived features
    amount_per_velocity = amount / (velocity_score + 1)
    amount_log = np.log1p(amount)
    amount_to_avg_ratio = amount / sender_avg_amount if sender_avg_amount > 0 else 1.0
    transaction_per_day = sender_total_transaction / 30
    transaction_gap = time_since_last_transaction
    is_night_transaction = 1 if 18 <= hour <= 24 else 0
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_self_transfer = 1 if sender_account == receiver_account else 0
    sender_fraud_percentage = (sender_fraud_transaction * 100 / sender_total_transaction) if sender_total_transaction > 0 else 0
    receiver_fraud_percentage = (receiver_fraud_transaction * 100 / receiver_total_transaction) if receiver_total_transaction > 0 else 0
    deviation_squared = spending_deviation_score ** 2
    device_hash = np.random.randint(0, 1000000)
    ip_address = np.random.randint(0, 2000000)

    # Prepare feature vector (38 features)
    features = np.array([[
        transaction_id, sender_account, receiver_account, amount,
        transaction_type_encoded, merchant_category_encoded, location_encoded, device_used_encoded,
        time_since_last_transaction, spending_deviation_score, velocity_score, geo_anomaly_score,
        payment_channel_encoded, ip_address, device_hash,
        hour, day, day_of_week, month,
        amount_per_velocity, amount_log, amount_to_avg_ratio, transaction_per_day, transaction_gap,
        is_night_transaction, is_weekend, is_self_transfer,
        sender_degree, receiver_degree, sender_total_transaction, receiver_total_transaction,
        sender_avg_amount, sender_std_amount, sender_fraud_transaction, receiver_fraud_transaction,
        sender_fraud_percentage, receiver_fraud_percentage, deviation_squared
    ]])

    st.markdown("---")
    # Prediction button
    if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction..."):
            try:
                # Make predictions based on selected model
                if model_choice == "XGBoost" and xgb_model is not None:
                    prediction_proba = xgb_model.predict_proba(features)[0]
                    prediction = 1 if prediction_proba[1] > 0.5 else 0
                    model_used = "XGBoost"
                    
                elif model_choice == "Neural Network (MLP)":
                    features_scaled = scaler.transform(features)
                    prediction_proba = mlp_model.predict_proba(features_scaled)[0]
                    prediction = 1 if prediction_proba[1] > 0.5 else 0
                    model_used = "Neural Network (MLP)"
                    
                elif model_choice == "Ensemble (XGBoost + MLP)" and xgb_model is not None:
                    # XGBoost prediction
                    xgb_proba = xgb_model.predict_proba(features)[0][1]
                    # MLP prediction
                    features_scaled = scaler.transform(features)
                    mlp_proba = mlp_model.predict_proba(features_scaled)[0][1]
                    # Average probabilities
                    ensemble_proba = (xgb_proba + mlp_proba) / 2
                    prediction_proba = [1 - ensemble_proba, ensemble_proba]
                    prediction = 1 if ensemble_proba > 0.5 else 0
                    model_used = "Ensemble (XGBoost + MLP)"
                
                else:
                    # Default to MLP if something goes wrong
                    features_scaled = scaler.transform(features)
                    prediction_proba = mlp_model.predict_proba(features_scaled)[0]
                    prediction = 1 if prediction_proba[1] > 0.5 else 0
                    model_used = "Neural Network (MLP)"

                fraud_probability = prediction_proba[1] * 100

                # Store prediction
                st.session_state.predictions.append(prediction)

                # Create result dataframe
                result_df = pd.DataFrame({
                    'Transaction ID': [transaction_id],
                    'Timestamp': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
                    'Amount': [amount],
                    'Type': [transaction_type],
                    'Location': [location],
                    'Fraud Probability': [fraud_probability],
                    'Prediction': ['FRAUD' if prediction == 1 else 'LEGITIMATE'],
                    'Model Used': [model_used]
                })

                # Add to history
                if st.session_state.prediction_history.empty:
                    st.session_state.prediction_history = result_df
                else:
                    st.session_state.prediction_history = pd.concat([
                        st.session_state.prediction_history,
                        result_df
                    ], ignore_index=True)

                st.markdown("---")
                st.subheader("🎯 Prediction Results")

                # Display result
                col1, col2, col3 = st.columns(3)
                with col1:
                    if prediction == 1:
                        st.error("⚠️ **FRAUD DETECTED**")
                    else:
                        st.success("✅ **LEGITIMATE TRANSACTION**")
                with col2:
                    st.metric("Fraud Probability", f"{fraud_probability:.2f}%")
                with col3:
                    st.info(f"**Model**: {model_used}")

                # Risk breakdown
                st.markdown("---")
                st.subheader("📊 Risk Factor Breakdown")

                risk_factors = {
                    'Amount Risk': min(100, (amount / 1000) * 50),
                    'Velocity Risk': (velocity_score / 20) * 100,
                    'Deviation Risk': min(100, abs(spending_deviation_score) * 20),
                    'Geographic Risk': geo_anomaly_score * 100,
                    'Time Risk': 100 if is_night_transaction else 30,
                    'Account History Risk': sender_fraud_percentage
                }

                # Create bar chart
                fig = go.Figure()
                for factor, value in risk_factors.items():
                    fig.add_trace(go.Bar(
                        x=[value],
                        y=[factor],
                        orientation='h',
                        marker_color='red' if value > 70 else 'orange' if value > 40 else 'green',
                        text=f'{value:.1f}%',
                        textposition='outside'
                    ))

                fig.update_layout(
                    title="Risk Factor Analysis",
                    xaxis_title="Risk Level (%)",
                    showlegend=False,
                    height=400,
                    xaxis=dict(range=[0, 100])
                )
                st.plotly_chart(fig, use_container_width=True)

                # Transaction details
                with st.expander("📋 View Detailed Transaction Information"):
                    st.json({
                        "Transaction ID": int(transaction_id),
                        "Amount": f"${amount:.2f}",
                        "Type": transaction_type,
                        "Merchant Category": merchant_category,
                        "Location": location,
                        "Device": device_used,
                        "Payment Channel": payment_channel,
                        "Velocity Score": int(velocity_score),
                        "Spending Deviation": float(spending_deviation_score),
                        "Geographic Anomaly": float(geo_anomaly_score),
                        "Sender Account": int(sender_account),
                        "Receiver Account": int(receiver_account),
                        "Time": f"{hour:02d}:00",
                        "Date": f"2024-{month:02d}-{day:02d}",
                        "Is Night Transaction": bool(is_night_transaction),
                        "Is Weekend": bool(is_weekend),
                        "Model Used": model_used,
                        "Fraud Probability": f"{fraud_probability:.2f}%"
                    })

            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")
                st.exception(e)

# ==================== RESULTS & ANALYSIS PAGE ====================
elif page == "📊 Results & Analysis":
    st.title("📊 Results & Analysis Dashboard")
    st.markdown("---")

    if st.session_state.prediction_history.empty:
        st.info("ℹ️ No predictions yet. Go to the **Predict Fraud** page to start analyzing transactions.")
    else:
        # Summary metrics
        st.subheader("📈 Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        total_predictions = len(st.session_state.prediction_history)
        fraud_count = (st.session_state.prediction_history['Prediction'] == 'FRAUD').sum()
        legit_count = total_predictions - fraud_count
        avg_fraud_prob = st.session_state.prediction_history['Fraud Probability'].mean()

        with col1:
            st.metric("Total Transactions", total_predictions)
        with col2:
            st.metric("Fraud Detected", fraud_count, delta=f"{(fraud_count/total_predictions*100):.1f}%")
        with col3:
            st.metric("Legitimate", legit_count)
        with col4:
            st.metric("Avg Fraud Probability", f"{avg_fraud_prob:.2f}%")

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            # Prediction distribution pie chart
            fig_pie = px.pie(
                st.session_state.prediction_history,
                names='Prediction',
                title='Fraud vs Legitimate Transactions',
                color='Prediction',
                color_discrete_map={'FRAUD': '#ff4b4b', 'LEGITIMATE': '#00cc66'}
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Fraud probability distribution
            fig_hist = px.histogram(
                st.session_state.prediction_history,
                x='Fraud Probability',
                nbins=20,
                title='Fraud Probability Distribution',
                color='Prediction',
                color_discrete_map={'FRAUD': '#ff4b4b', 'LEGITIMATE': '#00cc66'}
            )
            fig_hist.update_layout(xaxis_title="Fraud Probability (%)", yaxis_title="Count")
            st.plotly_chart(fig_hist, use_container_width=True)

        # Amount analysis
        if 'Amount' in st.session_state.prediction_history.columns:
            st.markdown("---")
            st.subheader("💰 Transaction Amount Analysis")
            col1, col2 = st.columns(2)

            with col1:
                # Box plot
                fig_box = px.box(
                    st.session_state.prediction_history,
                    x='Prediction',
                    y='Amount',
                    title='Transaction Amount by Prediction',
                    color='Prediction',
                    color_discrete_map={'FRAUD': '#ff4b4b', 'LEGITIMATE': '#00cc66'}
                )
                st.plotly_chart(fig_box, use_container_width=True)

            with col2:
                # Scatter plot
                fig_scatter = px.scatter(
                    st.session_state.prediction_history,
                    x='Amount',
                    y='Fraud Probability',
                    color='Prediction',
                    title='Amount vs Fraud Probability',
                    color_discrete_map={'FRAUD': '#ff4b4b', 'LEGITIMATE': '#00cc66'},
                    size='Amount',
                    hover_data=['Transaction ID', 'Type']
                )
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Transaction timeline
        if 'Timestamp' in st.session_state.prediction_history.columns:
            st.markdown("---")
            st.subheader("⏱️ Transaction Timeline")
            fig_timeline = px.scatter(
                st.session_state.prediction_history,
                x='Timestamp',
                y='Fraud Probability',
                color='Prediction',
                title='Fraud Detection Timeline',
                color_discrete_map={'FRAUD': '#ff4b4b', 'LEGITIMATE': '#00cc66'},
                hover_data=['Transaction ID', 'Amount']
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        # Model performance
        if 'Model Used' in st.session_state.prediction_history.columns:
            st.markdown("---")
            st.subheader("🤖 Model Usage Statistics")
            model_counts = st.session_state.prediction_history['Model Used'].value_counts()
            fig_model = px.bar(
                x=model_counts.index,
                y=model_counts.values,
                title='Predictions by Model',
                labels={'x': 'Model', 'y': 'Count'},
                color=model_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_model, use_container_width=True)

        # Transaction details table
        st.markdown("---")
        st.subheader("📋 Transaction History")

        # Add filtering options
        col1, col2 = st.columns(2)
        with col1:
            filter_prediction = st.multiselect(
                "Filter by Prediction:",
                options=['FRAUD', 'LEGITIMATE'],
                default=['FRAUD', 'LEGITIMATE'],
                key="filter_pred"
            )
        with col2:
            if 'Model Used' in st.session_state.prediction_history.columns:
                filter_model = st.multiselect(
                    "Filter by Model:",
                    options=st.session_state.prediction_history['Model Used'].unique(),
                    default=st.session_state.prediction_history['Model Used'].unique(),
                    key="filter_model"
                )

        # Apply filters
        filtered_df = st.session_state.prediction_history[
            st.session_state.prediction_history['Prediction'].isin(filter_prediction)
        ]
        if 'Model Used' in st.session_state.prediction_history.columns:
            filtered_df = filtered_df[filtered_df['Model Used'].isin(filter_model)]

        # Display table
        st.dataframe(
            filtered_df.sort_values('Timestamp', ascending=False),
            use_container_width=True,
            hide_index=True
        )

        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        # Clear history button
        st.markdown("---")
        if st.button("🗑️ Clear Prediction History", type="secondary"):
            st.session_state.prediction_history = pd.DataFrame()
            st.session_state.predictions = []
            st.rerun()

# ==================== ABOUT PAGE ====================
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.markdown("---")

    st.markdown("""
## Financial Transaction Fraud Detection System
This application uses advanced machine learning techniques to detect fraudulent financial transactions in real-time with high accuracy.

### 🎯 Project Overview
Financial fraud is a critical concern in the digital age. This system analyzes various transaction features to identify potentially fraudulent activities, helping financial institutions protect their customers and reduce losses.

### 🔬 Methodology

#### Data Processing
- **Dataset Size**: 5 million transactions
- **Fraud Rate**: 3.59% (179,553 fraudulent transactions)
- **Features**: 38 engineered features including:
  - Transaction amount and velocity
  - Spending deviation scores
  - Geographic anomaly indicators
  - Network-based features
  - Temporal patterns
  - Account history metrics

#### Machine Learning Models

**1. XGBoost Classifier**
- Gradient boosting algorithm
- Performance metrics:
  - Accuracy: 98%
  - Precision: 95-100%
  - Recall: 97-100%
  - AUC-ROC: 0.9912

**2. Neural Network (MLP)**
- Multi-layer perceptron with 3 hidden layers (36, 72, 36 neurons)
- Adaptive learning rate
- Performance metrics:
  - Accuracy: 98%
  - Precision: 94%
  - Recall: 100%
  - AUC-ROC: 0.9913

**3. Ensemble Method**
- Combines predictions from both models
- Averages probability scores for robust predictions

### 📊 Key Features Analyzed
1. **Amount Features**: Transaction amount, ratios, log-transformed amounts
2. **Temporal Features**: Hour, day, month, time gaps
3. **Risk Indicators**: Velocity, deviation, geographic anomalies
4. **Network Features**: Account connectivity and history

### 🎓 Model Training
- **Data Split**: 70% training, 20% testing, 10% validation
- **Balancing**: Downsampling majority class (2:1 ratio)
- **Scaling**: Standard scaling for MLP
- **Cross-validation**: Stratified sampling

### 📈 Performance Highlights
- **Average Precision**: 0.9734-0.9743
- **F1-Score**: 0.97-0.99
- **False Positive Rate**: < 3%
- **False Negative Rate**: < 1%

### 💡 Technical Stack
- **Frontend**: Streamlit
- **ML Libraries**: scikit-learn, XGBoost
- **Data Processing**: pandas, numpy
- **Visualization**: plotly
- **Model Persistence**: joblib

---

**Version**: 1.0.0
**Last Updated**: February 2026
**Developed by**: Financial ML Team
    """)

    st.markdown("---")
    st.subheader("🔧 System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
**Session Statistics**
- Total Predictions: {len(st.session_state.predictions)}
- History Records: {len(st.session_state.prediction_history)}
- Cache Status: Active
        """)
    with col2:
        st.markdown("""
**System Status**
- Models: ✅ Loading...
- Predictions: ✅ Active
- Storage: ✅ Available
        """)