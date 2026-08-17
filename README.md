# HotelPulse AI: Hotel Customer Satisfaction Analysis System

HotelPulse AI is a complete, beginner-friendly Streamlit dashboard designed for hotel managers. It analyzes guest reviews, identifies satisfaction trends, and provides actionable insights and smart recommendations based on real customer feedback.

## Features

- **Interactive Dashboard:** Filter by date range, guest type, room type, sentiment (Positive/Neutral/Negative), and overall rating.
- **KPI Tracking:** Real-time visibility into overall satisfaction, sentiment breakdown, and urgent issues, complete with helpful captions.
- **Sentiment Analysis:** Automatically categorizes review text into Positive, Neutral, or Negative using VADER Sentiment Analysis.
- **Trend & Service Analysis:** Visualizes satisfaction over time with dynamic text explaining whether trends are improving or declining. 
- **Service Comparison:** Compares performance across cleanliness, staff, food, Wi-Fi, and check-in services, explicitly highlighting the lowest-rated service.
- **Complaint Detection & Prioritization:** Extracts specific complaints (e.g., Wi-Fi, Cleanliness) and calculates a priority score (High/Medium/Low).
- **Smart Recommendations & Alerts:** Generates 3-5 automated, actionable recommendations and system alerts (e.g., "Wi-Fi is the highest-priority issue") based on current data.
- **Review Explorer & Export:** Detailed view of individual reviews with sorting capabilities and a one-click CSV download for filtered results.
- **Satisfaction Predictor (ML):** Uses a Random Forest classifier to estimate guest satisfaction (High, Medium, Low) based on service ratings and booking details.

## Project Folder Structure

- `app.py`: The main Streamlit dashboard containing the updated UI and logic.
- `data_generator.py`: A script to generate a realistic synthetic dataset (`sample_hotel_reviews.csv`) if one doesn't exist.
- `sample_hotel_reviews.csv`: The dataset containing sample hotel reviews.
- `requirements.txt`: The list of Python dependencies required to run the project.
- `README.md`: This documentation file.

## Setup Instructions

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your computer.

### 2. Install Dependencies
Open your terminal or command prompt, navigate to the `HotelPulse_AI` folder, and install the required packages:

```bash
pip install -r requirements.txt
```

### 3. Run the Application
Start the Streamlit dashboard by running:

```bash
streamlit run app.py
```

*Note: If `sample_hotel_reviews.csv` is not present, `app.py` will automatically call `data_generator.py` to create it on the first run.*
