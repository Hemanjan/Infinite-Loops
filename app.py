# app.py
from flask import Flask, render_template, request, jsonify
from itinerary_generator import GooglePlacesAPI, TripAdvisorAPI, ReviewProcessor, ItineraryGenerator, UserPreferences
from evaluation import PerformanceEvaluator
import os
from dotenv import load_dotenv
import time

app = Flask(__name__)
load_dotenv()

# Initialize APIs, components and evaluator
try:
    google_api = GooglePlacesAPI()
    tripadvisor_api = TripAdvisorAPI()
    review_processor = ReviewProcessor()
    generator = ItineraryGenerator(google_api, tripadvisor_api, review_processor)
    evaluator = PerformanceEvaluator()
except Exception as e:
    print(f"Error initializing components: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_itinerary', methods=['POST'])
def generate_itinerary():
    try:
        start_time = time.time()
        data = request.json
        
        # Create UserPreferences object
        user_prefs = UserPreferences()
        
        # Add preferences from request
        preferences = data.get('preferences', {})
        for category, value in preferences.items():
            user_prefs.add_preference(category, int(value))
        
        # Set budget level
        user_prefs.set_budget_level(int(data.get('budget', 2)))
        
        # Generate itinerary
        itinerary = generator.generate_itinerary(
            location=data.get('location', ''),
            user_preferences=user_prefs,
            num_days=int(data.get('num_days', 3))
        )
        
        # Log API metrics
        api_time = time.time() - start_time
        evaluator.log_api_request('google', True, api_time)
        
        # Format itinerary for response
        formatted_itinerary = {}
        for day, activities in itinerary.items():
            formatted_itinerary[day] = {}
            for time_slot, place in activities.items():
                if place:
                    formatted_itinerary[day][time_slot] = {
                        'name': place.name,
                        'address': place.address,
                        'rating': place.rating,
                        'type': place.type,
                        'website': place.website,
                        'price_level': place.price_level,
                        'total_reviews': place.total_reviews
                    }
                else:
                    formatted_itinerary[day][time_slot] = None
        
        # Log itinerary generation
        evaluator.log_itinerary_generation(formatted_itinerary)
        
        return jsonify({
            'success': True, 
            'itinerary': formatted_itinerary
        })
        
    except Exception as e:
        # Log failed request
        evaluator.log_api_request('google', False, time.time() - start_time, str(e))
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.json
        rating = int(data.get('rating', 0))
        feedback = data.get('feedback', '')
        
        evaluator.add_user_feedback(rating, feedback)
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    try:
        report = evaluator.generate_evaluation_report()
        evaluator.generate_performance_graphs()
        return jsonify(report)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
