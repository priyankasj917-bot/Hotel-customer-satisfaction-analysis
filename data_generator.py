import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

def generate_mock_data(num_records=600):
    """
    Generates a realistic sample dataset of hotel reviews.
    """
    print(f"Generating {num_records} synthetic hotel reviews...")
    
    np.random.seed(42)
    random.seed(42)
    
    guest_types = ['Business', 'Family', 'Couple', 'Solo']
    room_types = ['Standard', 'Deluxe', 'Suite']
    
    # Base reviews mapped roughly to rating profiles
    positive_reviews = [
        "Amazing stay! The staff was incredibly friendly and the room was spotless.",
        "Perfect location and great amenities. Will definitely come back.",
        "Loved the food and the Wi-Fi was super fast. Great for my business trip.",
        "Check-in was a breeze. Room was comfortable and clean.",
        "Excellent experience overall. Highly recommend this hotel to anyone."
    ]
    
    neutral_reviews = [
        "It was okay. Nothing special but no major issues either.",
        "Average stay. The room was fine, but the food could be better.",
        "Decent hotel for the price. Location is good.",
        "Wi-Fi was a bit spotty, but the staff was nice.",
        "Met my expectations. Cleanliness was acceptable."
    ]
    
    negative_reviews = [
        "Terrible experience. The room was dirty and smelled bad.",
        "Very rude staff at the reception. Check-in took forever.",
        "The food gave me food poisoning. Never eating here again.",
        "Wi-Fi didn't work at all. Completely unacceptable for a business trip.",
        "Way overpriced for what you get. The bed was uncomfortable and noisy."
    ]
    
    # Complaint specific reviews
    wifi_complaints = ["Wi-Fi kept disconnecting.", "Internet speed is terrible.", "Couldn't connect to Wi-Fi in my room."]
    clean_complaints = ["Bathroom was not clean.", "Found hair on the bed sheets.", "Dusty room."]
    food_complaints = ["Breakfast was cold and tasteless.", "Limited food options.", "Restaurant is too expensive."]
    checkin_complaints = ["Waited an hour for check-in.", "Reception lost my booking.", "Unfriendly check-in staff."]
    staff_complaints = ["Staff was unhelpful.", "Housekeeping ignored my requests.", "Rude management."]
    room_complaints = ["AC was broken.", "Bed was too hard.", "Noisy neighbors and thin walls."]
    price_complaints = ["Hidden fees on the bill.", "Overpriced minibar.", "Not worth the money."]
    
    data = []
    
    # Generate dates over the last year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    for _ in range(num_records):
        guest_type = random.choice(guest_types)
        room_type = random.choice(room_types)
        
        # Determine base satisfaction level to guide ratings and text
        sentiment_profile = np.random.choice(['positive', 'neutral', 'negative'], p=[0.6, 0.25, 0.15])
        
        if sentiment_profile == 'positive':
            base_rating = random.randint(4, 5)
            review_text = random.choice(positive_reviews)
        elif sentiment_profile == 'neutral':
            base_rating = 3
            review_text = random.choice(neutral_reviews)
        else:
            base_rating = random.randint(1, 2)
            review_text = random.choice(negative_reviews)
            # Inject a specific complaint sometimes
            if random.random() < 0.7:
                complaint_type = random.choice([wifi_complaints, clean_complaints, food_complaints, checkin_complaints, staff_complaints, room_complaints, price_complaints])
                review_text += " " + random.choice(complaint_type)
        
        # Add some noise to ratings
        cleanliness = max(1, min(5, base_rating + random.randint(-1, 1)))
        staff = max(1, min(5, base_rating + random.randint(-1, 1)))
        food = max(1, min(5, base_rating + random.randint(-1, 1)))
        wifi = max(1, min(5, base_rating + random.randint(-2, 1))) # Wi-Fi is notoriously variable
        checkin = max(1, min(5, base_rating + random.randint(-1, 1)))
        
        overall = round(np.mean([cleanliness, staff, food, wifi, checkin]))
        
        # Price based on room type
        base_price = {'Standard': 100, 'Deluxe': 180, 'Suite': 300}[room_type]
        price = base_price + random.randint(-20, 50)
        
        # Random date
        random_days = random.randint(0, 365)
        stay_date = (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")
        
        data.append({
            'Review': review_text,
            'Overall_Rating': overall,
            'Cleanliness_Rating': cleanliness,
            'Staff_Rating': staff,
            'Food_Rating': food,
            'WiFi_Rating': wifi,
            'CheckIn_Rating': checkin,
            'Guest_Type': guest_type,
            'Room_Type': room_type,
            'Price': price,
            'Stay_Date': stay_date
        })
        
    df = pd.DataFrame(data)
    
    # Save to current directory
    output_path = 'sample_hotel_reviews.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at {os.path.abspath(output_path)}")

if __name__ == "__main__":
    generate_mock_data()
