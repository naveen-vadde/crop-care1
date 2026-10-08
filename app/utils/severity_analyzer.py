"""
Severity analysis utilities for disease progression tracking
"""
import cv2
import numpy as np
from PIL import Image
import io


def calculate_severity_score(disease_class, confidence):
    """
    Calculate severity score based on disease type and confidence
    Returns a score from 0-100
    """
    # Disease severity weights (some diseases are more severe than others)
    severity_weights = {
        'healthy': 0,
        'Early_blight': 60,
        'Late_blight': 85,
        'Bacterial_spot': 70,
        'Septoria_leaf_spot': 65,
        'Spider_mites': 55,
        'Target_Spot': 60,
        'Tomato_Yellow_Leaf_Curl_Virus': 80,
        'Tomato_mosaic_virus': 75,
        'Leaf_Mold': 50,
        'Powdery_mildew': 65,
        'Black_rot': 75,
        'Cedar_apple_rust': 60,
        'Apple_scab': 55,
        'Common_rust': 50,
        'Northern_Leaf_Blight': 70,
        'Cercospora_leaf_spot': 65,
        'Leaf_scorch': 60,
        'Haunglongbing': 90,
        'Esca': 80,
        'Black_Measles': 80,
        'Leaf_blight': 65,
    }
    
    # Extract disease name from full class string
    base_severity = 50  # default
    for disease_key, weight in severity_weights.items():
        if disease_key.lower() in disease_class.lower():
            base_severity = weight
            break
    
    # Adjust by confidence
    severity_score = base_severity * (confidence / 100.0)
    
    return round(severity_score, 2)


def estimate_affected_area(image_bytes):
    """
    Estimate percentage of leaf area affected by disease using image processing
    Returns percentage (0-100)
    """
    try:
        # Convert bytes to numpy array
        image = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Define range for healthy green color
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        
        # Define range for diseased areas (brown, yellow, spots)
        lower_disease1 = np.array([10, 40, 40])
        upper_disease1 = np.array([30, 255, 255])
        
        lower_disease2 = np.array([0, 40, 40])
        upper_disease2 = np.array([10, 255, 255])
        
        # Create masks
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        disease_mask1 = cv2.inRange(hsv, lower_disease1, upper_disease1)
        disease_mask2 = cv2.inRange(hsv, lower_disease2, upper_disease2)
        disease_mask = cv2.bitwise_or(disease_mask1, disease_mask2)
        
        # Calculate percentages
        total_pixels = img_array.shape[0] * img_array.shape[1]
        green_pixels = cv2.countNonZero(green_mask)
        disease_pixels = cv2.countNonZero(disease_mask)
        
        # Estimate affected area
        if green_pixels + disease_pixels > 0:
            affected_pct = (disease_pixels / (green_pixels + disease_pixels)) * 100
        else:
            affected_pct = 0
        
        # Cap at reasonable values
        affected_pct = min(affected_pct, 95)
        
        return round(affected_pct, 2)
    
    except Exception as e:
        # If image processing fails, return 0
        return 0.0


def analyze_progression(records):
    """
    Analyze disease progression from multiple records
    Returns trend analysis and alerts
    """
    if len(records) < 2:
        return {
            'trend': 'insufficient_data',
            'rate_of_change': 0,
            'alert_level': 'none',
            'message': 'Need at least 2 records to analyze progression'
        }
    
    # Sort by timestamp
    sorted_records = sorted(records, key=lambda x: x['timestamp'])
    
    # Calculate severity change
    first_severity = sorted_records[0]['severity_score']
    last_severity = sorted_records[-1]['severity_score']
    severity_change = last_severity - first_severity
    
    # Calculate time difference in days
    from datetime import datetime
    first_time = datetime.fromisoformat(sorted_records[0]['timestamp'])
    last_time = datetime.fromisoformat(sorted_records[-1]['timestamp'])
    days_elapsed = (last_time - first_time).days
    
    if days_elapsed == 0:
        days_elapsed = 1  # Avoid division by zero
    
    # Rate of change per day
    rate_of_change = severity_change / days_elapsed
    
    # Determine trend
    if severity_change < -5:
        trend = 'improving'
        alert_level = 'good'
        message = f'Disease severity decreased by {abs(severity_change):.1f} points. Treatment appears effective!'
    elif severity_change > 10:
        trend = 'worsening'
        if rate_of_change > 5:
            alert_level = 'critical'
            message = f'⚠️ ALERT: Disease progressing rapidly ({rate_of_change:.1f} points/day). Immediate action recommended!'
        else:
            alert_level = 'warning'
            message = f'Disease severity increased by {severity_change:.1f} points. Monitor closely and consider treatment.'
    else:
        trend = 'stable'
        alert_level = 'normal'
        message = 'Disease progression is stable. Continue current management practices.'
    
    return {
        'trend': trend,
        'rate_of_change': round(rate_of_change, 2),
        'alert_level': alert_level,
        'message': message,
        'severity_change': round(severity_change, 2),
        'days_elapsed': days_elapsed,
        'first_severity': round(first_severity, 2),
        'last_severity': round(last_severity, 2)
    }


def get_severity_color(severity_score):
    """Return color code based on severity score"""
    if severity_score < 20:
        return '#52b788'  # Green - healthy/mild
    elif severity_score < 40:
        return '#95d5b2'  # Light green
    elif severity_score < 60:
        return '#f4a261'  # Orange - moderate
    elif severity_score < 80:
        return '#e76f51'  # Dark orange
    else:
        return '#e63946'  # Red - severe
