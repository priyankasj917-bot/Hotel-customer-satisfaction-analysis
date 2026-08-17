import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Set page configuration
st.set_page_config(
    page_title="HotelPulse AI",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    /* Dark Navy Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a2238;
        color: #ffffff;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] span, [data-testid="stSidebar"] p {
        color: #ffffff !important;
    }
    
    /* Clean white content area - default Streamlit is fine, but ensure card backgrounds */
    .kpi-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        text-align: center;
        margin-bottom: 20px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
    }
    .kpi-label {
        font-size: 13px;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 5px;
    }
    .kpi-caption {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 5px;
    }
    
    /* Semantic Colors */
    .color-teal { color: #0d9488; }
    .color-amber { color: #d97706; }
    .color-red { color: #dc2626; }
    .color-navy { color: #1e3a8a; }

    /* Alert Boxes */
    .alert-card {
        padding: 12px 15px;
        border-radius: 6px;
        margin-bottom: 10px;
        font-size: 14px;
        font-weight: 500;
    }
    .alert-high { background-color: #fef2f2; color: #991b1b; border-left: 4px solid #dc2626; }
    .alert-med { background-color: #fffbeb; color: #92400e; border-left: 4px solid #f59e0b; }
    .alert-low { background-color: #f0fdf4; color: #166534; border-left: 4px solid #10b981; }
</style>
""", unsafe_allow_html=True)

# --- Initialize Data and Models ---
@st.cache_data
def load_data():
    data_path = 'sample_hotel_reviews.csv'
    if not os.path.exists(data_path):
        import data_generator
        data_generator.generate_mock_data()
    df = pd.read_csv(data_path)
    df['Stay_Date'] = pd.to_datetime(df['Stay_Date'])
    return df

@st.cache_resource
def get_vader_analyzer():
    return SentimentIntensityAnalyzer()

def analyze_sentiment(text, analyzer):
    if pd.isna(text):
        return 'Neutral'
    score = analyzer.polarity_scores(text)['compound']
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Complaint keywords mapping
COMPLAINT_KEYWORDS = {
    'Wi-Fi': ['wi-fi', 'wifi', 'internet', 'connection', 'disconnect'],
    'Cleanliness': ['clean', 'dirty', 'dust', 'smell', 'hair', 'stain', 'housekeeping'],
    'Food': ['food', 'breakfast', 'restaurant', 'meal', 'taste', 'poisoning'],
    'Check-in': ['check-in', 'reception', 'wait', 'booking', 'front desk'],
    'Staff': ['staff', 'rude', 'unhelpful', 'attitude', 'management'],
    'Room Quality': ['bed', 'ac', 'noisy', 'wall', 'broken', 'small', 'uncomfortable'],
    'Price/Billing': ['price', 'expensive', 'fee', 'overpriced', 'money', 'bill']
}

def detect_complaint(text):
    if pd.isna(text):
        return 'None'
    text_lower = str(text).lower()
    for category, keywords in COMPLAINT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return category
    return 'None'

# Load data
try:
    df_raw = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# Apply text analysis
analyzer = get_vader_analyzer()
if 'Sentiment' not in df_raw.columns:
    df_raw['Sentiment'] = df_raw['Review'].apply(lambda x: analyze_sentiment(x, analyzer))
if 'Complaint_Category' not in df_raw.columns:
    df_raw['Complaint_Category'] = df_raw['Review'].apply(detect_complaint)

# --- Sidebar ---
st.sidebar.title("🏨 HotelPulse AI")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Satisfaction Predictor"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Filters")

# Date Filter
min_date = df_raw['Stay_Date'].min().date()
max_date = df_raw['Stay_Date'].max().date()
start_date = st.sidebar.date_input("Start Date", min_date)
end_date = st.sidebar.date_input("End Date", max_date)

# Guest Type Filter
guest_types = ['All'] + list(df_raw['Guest_Type'].unique())
selected_guest = st.sidebar.selectbox("Guest Type", guest_types)

# Room Type Filter
room_types = ['All'] + list(df_raw['Room_Type'].unique())
selected_room = st.sidebar.selectbox("Room Type", room_types)

# Sentiment Filter
sentiments = ['All', 'Positive', 'Neutral', 'Negative']
selected_sentiment = st.sidebar.selectbox("Sentiment", sentiments)

# Rating Filter
rating_range = st.sidebar.slider("Overall Rating Range", 1, 5, (1, 5))

# Apply filters
df = df_raw[
    (df_raw['Stay_Date'].dt.date >= start_date) &
    (df_raw['Stay_Date'].dt.date <= end_date) &
    (df_raw['Overall_Rating'] >= rating_range[0]) &
    (df_raw['Overall_Rating'] <= rating_range[1])
]
if selected_guest != 'All':
    df = df[df['Guest_Type'] == selected_guest]
if selected_room != 'All':
    df = df[df['Room_Type'] == selected_room]
if selected_sentiment != 'All':
    df = df[df['Sentiment'] == selected_sentiment]

# --- Main Page: Dashboard ---
if page == "Dashboard":
    # Branded Header
    st.title("HotelPulse AI — Guest Satisfaction Command Center")
    st.markdown("#### Turn guest feedback into better hotel decisions.")
    st.markdown("---")

    if df.empty:
        st.warning("No data matches the selected filters. Please adjust your criteria.")
    else:
        # 1. KPI Cards
        total_reviews = len(df)
        total_raw = len(df_raw)
        avg_satisfaction = df['Overall_Rating'].mean()
        
        # Color coding satisfaction
        sat_color = "color-teal" if avg_satisfaction >= 4.0 else "color-amber" if avg_satisfaction >= 3.0 else "color-red"
        
        positive_pct = (df['Sentiment'] == 'Positive').mean() * 100 if total_reviews > 0 else 0
        negative_pct = (df['Sentiment'] == 'Negative').mean() * 100 if total_reviews > 0 else 0
        urgent_issues = len(df[(df['Sentiment'] == 'Negative') & (df['Overall_Rating'] <= 2)])

        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value {sat_color}">{avg_satisfaction:.1f} / 5</div>
                <div class="kpi-label">Overall Satisfaction</div>
                <div class="kpi-caption">Based on {total_reviews} ratings</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            pct_of_total = (total_reviews / total_raw) * 100
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value color-navy">{total_reviews}</div>
                <div class="kpi-label">Reviews Analyzed</div>
                <div class="kpi-caption">{pct_of_total:.0f}% of total dataset</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value color-teal">{positive_pct:.1f}%</div>
                <div class="kpi-label">Positive Sentiment</div>
                <div class="kpi-caption">via VADER NLP</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value color-red">{negative_pct:.1f}%</div>
                <div class="kpi-label">Negative Sentiment</div>
                <div class="kpi-caption">Needs attention</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col5:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value color-amber">{urgent_issues}</div>
                <div class="kpi-label">Urgent Issues</div>
                <div class="kpi-caption">Rating ≤ 2 & Negative</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Layout for Charts Row 1
        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Monthly Satisfaction Trend")
            df_trend = df.copy()
            df_trend['Month'] = df_trend['Stay_Date'].dt.to_period('M').astype(str)
            trend_data = df_trend.groupby('Month')['Overall_Rating'].mean().reset_index()
            
            if len(trend_data) > 1:
                first_month_val = trend_data.iloc[0]['Overall_Rating']
                last_month_val = trend_data.iloc[-1]['Overall_Rating']
                diff = last_month_val - first_month_val
                
                if diff > 0.1:
                    trend_msg = f"**Status:** Satisfaction is **improving** (up by {diff:.2f} points since {trend_data.iloc[0]['Month']})."
                elif diff < -0.1:
                    trend_msg = f"**Status:** Satisfaction is **declining** (down by {abs(diff):.2f} points since {trend_data.iloc[0]['Month']})."
                else:
                    trend_msg = "**Status:** Satisfaction is **stable**."
            else:
                trend_msg = "**Status:** Not enough data to determine a trend."

            fig_trend = px.line(trend_data, x='Month', y='Overall_Rating', markers=True, 
                                line_shape='spline', color_discrete_sequence=['#1e3a8a'])
            fig_trend.update_layout(yaxis=dict(range=[1, 5]), margin=dict(t=10, b=10))
            st.plotly_chart(fig_trend, use_container_width=True)
            st.markdown(trend_msg)

        with c2:
            st.subheader("Service Performance")
            categories = ['Cleanliness_Rating', 'Staff_Rating', 'Food_Rating', 'WiFi_Rating', 'CheckIn_Rating']
            avg_ratings = [df[cat].mean() for cat in categories]
            
            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=avg_ratings + [avg_ratings[0]],
                theta=[c.replace('_Rating', '') for c in categories] + [categories[0].replace('_Rating', '')],
                fill='toself',
                line_color='#0d9488'
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[1, 5])),
                showlegend=False, margin=dict(t=10, b=10)
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            worst_service_idx = np.argmin(avg_ratings)
            worst_service = categories[worst_service_idx].replace('_Rating', '')
            st.markdown(f"⚠️ **Lowest-rated service:** {worst_service} ({avg_ratings[worst_service_idx]:.1f}/5)")

        st.markdown("---")

        # Layout for Charts Row 2
        c3, c4 = st.columns(2)
        
        # Determine Priority Data
        complaints_df = df[df['Complaint_Category'] != 'None']
        priority_data = pd.DataFrame()
        
        if not complaints_df.empty:
            priority_data = complaints_df.groupby('Complaint_Category').agg(
                Count=('Review', 'count'),
                Avg_Rating=('Overall_Rating', 'mean')
            ).reset_index()
            
            avg_count = priority_data['Count'].mean()
            
            def calc_priority(row):
                if row['Count'] >= avg_count and row['Avg_Rating'] <= 2.5:
                    return 'High'
                elif row['Count'] >= avg_count or row['Avg_Rating'] <= 3.5:
                    return 'Medium'
                else:
                    return 'Low'
                    
            priority_data['Priority'] = priority_data.apply(calc_priority, axis=1)
            priority_data = priority_data.sort_values(by='Count', ascending=False)

        with c3:
            st.subheader("Guest Segment Analysis")
            segment_data = df.groupby('Guest_Type')['Overall_Rating'].mean().reset_index().sort_values('Overall_Rating')
            fig_segment = px.bar(segment_data, x='Overall_Rating', y='Guest_Type', orientation='h',
                                 color='Overall_Rating', color_continuous_scale='Teal')
            fig_segment.update_layout(xaxis=dict(range=[1, 5]), margin=dict(t=10, b=10))
            st.plotly_chart(fig_segment, use_container_width=True)
            
            least_satisfied = segment_data.iloc[0]
            st.markdown(f"The **least satisfied** guest segment is **{least_satisfied['Guest_Type']}** travelers, with an average rating of {least_satisfied['Overall_Rating']:.1f}.")

        with c4:
            st.subheader("Issue Priority Score")
            if not priority_data.empty:
                color_map = {'High': '#dc2626', 'Medium': '#d97706', 'Low': '#0d9488'}
                fig_complaints = px.bar(priority_data, x='Complaint_Category', y='Count', color='Priority',
                                        color_discrete_map=color_map,
                                        text='Avg_Rating')
                fig_complaints.update_traces(texttemplate='%{text:.1f}★', textposition='outside')
                fig_complaints.update_layout(margin=dict(t=10, b=10))
                st.plotly_chart(fig_complaints, use_container_width=True)
            else:
                st.info("No explicit complaints detected in the current filter.")

        st.markdown("---")

        # Smart Recommendations & Alerts
        st.subheader("💡 Smart Recommendations & Alerts")
        
        rec_col, alert_col = st.columns(2)
        
        with rec_col:
            st.markdown("#### Smart Recommendations")
            if not priority_data.empty:
                top_complaint = priority_data.iloc[0]['Complaint_Category']
                
                recs = []
                # 1. Recommendation for worst service
                if worst_service == 'Wi-Fi':
                    recs.append("Upgrade router hardware and audit ISP bandwidth to resolve connectivity issues.")
                elif worst_service == 'Cleanliness':
                    recs.append("Implement a dual-signature checklist for housekeeping to ensure thorough room cleaning.")
                elif worst_service == 'Staff':
                    recs.append("Schedule a customer service retraining session focusing on guest empathy.")
                elif worst_service == 'Food':
                    recs.append("Review the breakfast menu and audit food temperature controls during buffet hours.")
                elif worst_service == 'Check-in':
                    recs.append("Add a dedicated express check-in lane or kiosk to reduce wait times.")
                
                # 2. Recommendation for top complaint if different from worst service
                if top_complaint != worst_service:
                    recs.append(f"Investigate recent {top_complaint} complaints, as they are the most frequently mentioned issue by guests.")
                
                # 3. Recommendation for least satisfied segment
                recs.append(f"Create a targeted amenity package or discount for {least_satisfied['Guest_Type']} travelers to boost their satisfaction.")
                
                for r in recs:
                    st.markdown(f"- {r}")
            else:
                st.success("- Keep up the great work! No major action items identified.")
                
        with alert_col:
            st.markdown("#### System Alerts")
            
            # Check for specific wording requirements
            if not priority_data.empty:
                top_pri_item = priority_data.iloc[0]
                if top_pri_item['Complaint_Category'] == 'Wi-Fi' and top_pri_item['Priority'] == 'High':
                    st.markdown('<div class="alert-card alert-high">⚠️ Wi-Fi is the highest-priority issue.</div>', unsafe_allow_html=True)
                elif top_pri_item['Priority'] == 'High':
                    st.markdown(f'<div class="alert-card alert-high">⚠️ {top_pri_item["Complaint_Category"]} is the highest-priority issue.</div>', unsafe_allow_html=True)
            
            if negative_pct > 15:
                st.markdown('<div class="alert-card alert-high">🚨 Negative feedback is increasing. Over 15% of recent reviews are negative.</div>', unsafe_allow_html=True)
                
            if urgent_issues > 0:
                st.markdown(f'<div class="alert-card alert-med">⚠️ You have {urgent_issues} urgent reviews (rating ≤ 2) requiring immediate response.</div>', unsafe_allow_html=True)
                
            if (negative_pct <= 15) and (urgent_issues == 0) and (priority_data.empty or 'High' not in priority_data['Priority'].values):
                st.markdown('<div class="alert-card alert-low">✅ All systems normal. No critical alerts.</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Review Explorer Table
        st.subheader("🔍 Review Explorer")
        
        display_df = df[['Stay_Date', 'Guest_Type', 'Room_Type', 'Review', 'Overall_Rating', 'Sentiment', 'Complaint_Category']]
        
        st.dataframe(display_df, use_container_width=True, height=300)
        
        csv_data = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Filtered Reviews as CSV",
            data=csv_data,
            file_name="hotel_filtered_reviews.csv",
            mime="text/csv",
        )

# --- Main Page: Satisfaction Predictor (ML) ---
elif page == "Satisfaction Predictor":
    st.title("🤖 Satisfaction Predictor")
    st.markdown("Use this machine learning model to estimate guest satisfaction based on service quality and booking details.")
    st.info("ℹ️ **Note:** This model provides an **estimate** based on historical patterns, not a guaranteed outcome.")
    
    # Prepare Data for ML
    ml_df = df_raw.copy()
    
    # Target variable: 4-5 High, 3 Medium, 1-2 Low
    def classify_satisfaction(rating):
        if rating >= 4: return 'High'
        elif rating == 3: return 'Medium'
        else: return 'Low'
        
    ml_df['Satisfaction_Class'] = ml_df['Overall_Rating'].apply(classify_satisfaction)
    
    features = ['Cleanliness_Rating', 'Staff_Rating', 'Food_Rating', 'WiFi_Rating', 'CheckIn_Rating', 'Price', 'Guest_Type', 'Room_Type']
    X = ml_df[features].copy()
    y = ml_df['Satisfaction_Class']
    
    le_guest = LabelEncoder()
    le_room = LabelEncoder()
    X['Guest_Type'] = le_guest.fit_transform(X['Guest_Type'])
    X['Room_Type'] = le_room.fit_transform(X['Room_Type'])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    with st.spinner("Training Random Forest model..."):
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
    st.success(f"Model trained successfully! **Estimated Accuracy: {acc*100:.1f}%**")
    
    st.markdown("---")
    st.markdown("### Estimate Satisfaction for a New Guest")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Expected Service Ratings (1-5)**")
        p_clean = st.slider("Cleanliness", 1, 5, 4)
        p_staff = st.slider("Staff", 1, 5, 4)
        p_food = st.slider("Food", 1, 5, 3)
        p_wifi = st.slider("Wi-Fi", 1, 5, 5)
        p_checkin = st.slider("Check-in", 1, 5, 4)
        
    with col2:
        st.markdown("**Booking Details**")
        p_guest = st.selectbox("Guest Type", le_guest.classes_)
        p_room = st.selectbox("Room Type", le_room.classes_)
        p_price = st.number_input("Price Paid ($)", 50, 1000, 150)
        
    with col3:
        st.markdown("**Prediction Result**")
        if st.button("Predict Satisfaction", type="primary"):
            input_data = pd.DataFrame({
                'Cleanliness_Rating': [p_clean],
                'Staff_Rating': [p_staff],
                'Food_Rating': [p_food],
                'WiFi_Rating': [p_wifi],
                'CheckIn_Rating': [p_checkin],
                'Price': [p_price],
                'Guest_Type': [le_guest.transform([p_guest])[0]],
                'Room_Type': [le_room.transform([p_room])[0]]
            })
            
            prediction = rf_model.predict(input_data)[0]
            probs = rf_model.predict_proba(input_data)[0]
            classes = rf_model.classes_
            prob_dict = dict(zip(classes, probs))
            
            if prediction == 'High':
                st.markdown('<div class="alert-card alert-low">Predicted Satisfaction: <b>High</b> 🌟</div>', unsafe_allow_html=True)
            elif prediction == 'Medium':
                st.markdown('<div class="alert-card alert-med">Predicted Satisfaction: <b>Medium</b> 😐</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="alert-card alert-high">Predicted Satisfaction: <b>Low</b> 😞</div>', unsafe_allow_html=True)
                
            st.markdown("**Confidence Estimates:**")
            for cls, prob in prob_dict.items():
                st.markdown(f"- {cls}: {prob*100:.1f}%")
