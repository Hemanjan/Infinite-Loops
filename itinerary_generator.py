import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import json
import requests
from typing import List, Dict, Union, Optional, Tuple, Any
import os
from dotenv import load_dotenv
import logging
from datetime import datetime, timedelta
import time
from dataclasses import dataclass
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Download NLTK data
try:
    for package in ['punkt', 'wordnet', 'stopwords', 'vader_lexicon']:
        nltk.download(package, quiet=True)
except Exception as e:
    logger.error(f"Failed to download NLTK data: {e}")
    raise

@dataclass
class Place:
    """Data class for place information"""
    name: str
    address: str
    rating: float
    type: str
    source: str
    place_id: str
    location: Dict[str, float]
    reviews: List[Dict]
    price_level: int
    website: str
    opening_hours: Dict
    photo_url: Optional[str] = None
    total_reviews: int = 0
    score: float = 0.0

class ReviewProcessor:
    """Handles text processing of reviews using NLP techniques"""
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()

    @lru_cache(maxsize=1000)
    def preprocess_text(self, text: str) -> str:
        """Preprocess text with caching for better performance"""
        if not text or not isinstance(text, str):
            return ""
        
        tokens = word_tokenize(text.lower())
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and token.isalnum()]
        return ' '.join(tokens)

    def analyze_reviews_batch(self, reviews: List[str], 
                            batch_size: int = 100) -> Dict[str, List[float]]:
        """Process reviews in batches for better performance"""
        results = {
            'sentiment_scores': [],
            'text_lengths': [],
            'word_counts': []
        }
        
        for i in range(0, len(reviews), batch_size):
            batch = reviews[i:i + batch_size]
            with ThreadPoolExecutor() as executor:
                sentiment_futures = [
                    executor.submit(self.sia.polarity_scores, review) 
                    for review in batch if review
                ]
                
                for future in as_completed(sentiment_futures):
                    try:
                        sentiment = future.result()
                        results['sentiment_scores'].append(sentiment['compound'])
                    except Exception as e:
                        logger.error(f"Error processing review: {e}")
                        results['sentiment_scores'].append(0.0)
            
            # Process text statistics
            for review in batch:
                if review:
                    results['text_lengths'].append(len(review))
                    results['word_counts'].append(len(review.split()))
        
        return results

class APIBase:
    """Base class for API clients"""
    def __init__(self):
        load_dotenv()
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TravelApp/1.0',
            'Accept': 'application/json'
        })

    def _make_request(self, url: str, params: Dict = None, 
                     headers: Dict = None, retries: int = 3) -> Optional[Dict]:
        """Make HTTP request with retry logic"""
        current_headers = self.session.headers.copy()
        if headers:
            current_headers.update(headers)

        for attempt in range(retries):
            try:
                response = self.session.get(
                    url, 
                    params=params, 
                    headers=current_headers,
                    timeout=10
                )
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                if attempt == retries - 1:
                    logger.error(f"Failed to make request after {retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

class GooglePlacesAPI(APIBase):
    """Google Places API client with improved connection management"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("GOOGLE_PLACES_API_KEY")
        if not self.api_key:
            raise ValueError("Google Places API key not found")
        
        # Configure connection pooling
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=20,
            pool_maxsize=20,
            max_retries=3,
            pool_block=False
        )
        self.session.mount('https://', adapter)
        self.session.mount('http://', adapter)
        logger.info("Google Places API initialized successfully")

    def search_places(self, location: str, keywords: List[str], 
                     radius: int = 5000) -> List[Place]:
        """Search places with improved concurrency management"""
        places = []
        base_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
        
        # Process keywords in smaller batches to manage connection pool
        batch_size = 5
        for i in range(0, len(keywords), batch_size):
            batch_keywords = keywords[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_keyword = {
                    executor.submit(
                        self._search_single_keyword,
                        base_url,
                        location,
                        keyword,
                        radius
                    ): keyword for keyword in batch_keywords
                }
                
                for future in as_completed(future_to_keyword):
                    keyword = future_to_keyword[future]
                    try:
                        results = future.result()
                        if results:
                            places.extend(results)
                            logger.info(f"Found {len(results)} places for keyword: {keyword}")
                    except Exception as e:
                        logger.error(f"Error searching for {keyword}: {e}")
            
            # Add a small delay between batches to prevent rate limiting
            time.sleep(1)
        
        return places

    def _search_single_keyword(self, base_url: str, location: str, 
                             keyword: str, radius: int) -> List[Place]:
        """Search for a single keyword"""
        params = {
            'query': f'{keyword} in {location}',
            'radius': radius,
            'key': self.api_key
        }
        
        data = self._make_request(base_url, params)
        if not data or data.get('status') != 'OK':
            return []
            
        places = []
        for place in data.get('results', []):
            try:
                place_details = self.get_place_details(place['place_id'])
                places.append(Place(
                    name=place['name'],
                    address=place['formatted_address'],
                    rating=place.get('rating', 0),
                    type=keyword,
                    source='google',
                    place_id=place['place_id'],
                    location=place['geometry']['location'],
                    reviews=place_details.get('reviews', []),
                    price_level=place_details.get('price_level', 2),
                    website=place_details.get('website', ''),
                    opening_hours=place_details.get('opening_hours', {})
                ))
            except Exception as e:
                logger.error(f"Error processing place {place.get('name')}: {e}")
                
        return places

    def get_place_details(self, place_id: str) -> Dict:
        """Get detailed information about a specific place"""
        url = "https://maps.googleapis.com/maps/api/place/details/json"
        params = {
            'place_id': place_id,
            'fields': 'reviews,price_level,website,opening_hours',
            'key': self.api_key
        }
        
        data = self._make_request(url, params)
        return data.get('result', {}) if data else {}

class TripAdvisorAPI(APIBase):
    """TripAdvisor API client using RapidAPI"""
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("RAPIDAPI_KEY")
        if not self.api_key:
            raise ValueError("RapidAPI key not found in environment variables")
            
        self.headers = {
            "X-RapidAPI-Key": self.api_key,
            "X-RapidAPI-Host": "travel-advisor.p.rapidapi.com"
        }
        logger.info("TripAdvisor API initialized successfully")

    @lru_cache(maxsize=100)
    def _get_location_id(self, location: str) -> Optional[str]:
        """Get location ID with caching and proper error handling"""
        try:
            url = "https://travel-advisor.p.rapidapi.com/locations/search"
            params = {
                "query": location,
                "limit": "1",
                "offset": "0",
                "units": "km",
                "currency": "USD",
                "sort": "relevance",
                "lang": "en_US"
            }
            
            response = self._make_request(url, params, self.headers)
            if response and isinstance(response, dict):
                data = response.get("data", [])
                if data and len(data) > 0:
                    return data[0].get("result_object", {}).get("location_id")
            
            logger.warning(f"No location ID found for {location}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting location ID for {location}: {e}")
            return None

    def search_places(self, location: str, keywords: List[str], 
                     radius: int = 5000) -> List[Place]:
        """Search places using TripAdvisor API with improved error handling"""
        places = []
        location_id = self._get_location_id(location)
        
        if not location_id:
            logger.error(f"Could not find location ID for {location}")
            return places

        try:
            # Search for restaurants
            restaurant_url = "https://travel-advisor.p.rapidapi.com/restaurants/list"
            # Search for attractions
            attraction_url = "https://travel-advisor.p.rapidapi.com/attractions/list"
            
            base_params = {
                "location_id": location_id,
                "currency": "USD",
                "lunit": "km",
                "limit": "30",
                "lang": "en_US"
            }
            
            # Fetch both restaurants and attractions in parallel
            with ThreadPoolExecutor() as executor:
                restaurant_future = executor.submit(
                    self._make_request, restaurant_url, base_params, self.headers)
                attraction_future = executor.submit(
                    self._make_request, attraction_url, base_params, self.headers)
                
                responses = []
                for future in as_completed([restaurant_future, attraction_future]):
                    try:
                        result = future.result()
                        if result and isinstance(result, dict):
                            responses.append(result)
                    except Exception as e:
                        logger.error(f"Error fetching places: {e}")

            # Process responses
            for response in responses:
                data = response.get("data", [])
                if not isinstance(data, list):
                    continue
                    
                for place_data in data:
                    if not isinstance(place_data, dict):
                        continue
                        
                    # Extract place details
                    try:
                        name = place_data.get("name", "")
                        if not name:
                            continue
                            
                        # Check if place matches any keywords
                        categories = place_data.get("cuisine", []) + place_data.get("subcategory", [])
                        categories = [cat.get("name", "").lower() for cat in categories if isinstance(cat, dict)]
                        
                        if keywords and not any(keyword.lower() in " ".join(categories) for keyword in keywords):
                            continue
                            
                        place = Place(
                            name=name,
                            address=place_data.get("address", {}).get("string", ""),
                            rating=float(place_data.get("rating", 0)),
                            type=categories[0] if categories else "attraction",
                            source="tripadvisor",
                            place_id=str(place_data.get("location_id", "")),
                            location={
                                "lat": float(place_data.get("latitude", 0)),
                                "lng": float(place_data.get("longitude", 0))
                            },
                            reviews=self._format_reviews(place_data),
                            price_level=self._convert_price_level(place_data.get("price_level", "")),
                            website=place_data.get("website", ""),
                            opening_hours=self._format_opening_hours(place_data),
                            photo_url=place_data.get("photo", {}).get("images", {}).get("original", {}).get("url", ""),
                            total_reviews=place_data.get("num_reviews", 0)
                        )
                        places.append(place)
                        
                    except Exception as e:
                        logger.error(f"Error processing place data: {e}")
                        continue

            logger.info(f"Found {len(places)} places on TripAdvisor")
            
        except Exception as e:
            logger.error(f"Error fetching TripAdvisor data: {e}")
            
        return places

    def _format_reviews(self, place_data: Dict) -> List[Dict]:
        """Format reviews with proper error handling"""
        reviews = []
        try:
            raw_reviews = place_data.get("reviews", [])
            for review in raw_reviews:
                if not isinstance(review, dict):
                    continue
                    
                reviews.append({
                    "text": review.get("text", ""),
                    "rating": float(review.get("rating", 0)),
                    "time": review.get("published_date", datetime.now().isoformat()),
                    "author_name": review.get("author", {}).get("name", "Anonymous")
                })
        except Exception as e:
            logger.error(f"Error formatting reviews: {e}")
            
        return reviews

    def _format_opening_hours(self, place_data: Dict) -> Dict:
        """Format opening hours with proper error handling"""
        try:
            hours = place_data.get("hours", {})
            periods = []
            
            if hours:
                week_ranges = hours.get("weekday_text", [])
                for day_range in week_ranges:
                    if isinstance(day_range, str) and ":" in day_range:
                        day, times = day_range.split(": ", 1)
                        if times.lower() != "closed":
                            periods.append({
                                "open": {"time": "0800", "day": day},
                                "close": {"time": "2200", "day": day}
                            })
            
            return {
                "open_now": hours.get("open_now", True),
                "status": hours.get("status_text", ""),
                "periods": periods
            }
            
        except Exception as e:
            logger.error(f"Error formatting opening hours: {e}")
            return {"open_now": True, "status": "", "periods": []}

class UserPreferences:
    """Manages user preferences for travel recommendations"""
    def __init__(self):
        self.preferences = {}
        self.budget_level = 2
        self.preferred_times = {
            'morning': ['08:00', '12:00'],
            'afternoon': ['12:00', '17:00'],
            'evening': ['17:00', '21:00'],
            'night': ['21:00', '23:00']
        }

    def add_preference(self, category: str, value: int):
        """Add user preference with importance score (1-5)"""
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError("Preference value must be between 1 and 5")
        self.preferences[category] = value
        logger.info(f"Added preference: {category} = {value}")
        
    def set_budget_level(self, level: int):
        """Set budget level (1-4)"""
        if not isinstance(level, int) or level < 1 or level > 4:
            raise ValueError("Budget level must be between 1 and 4")
        self.budget_level = level
        logger.info(f"Set budget level to {level}")

    def get_keywords_from_preferences(self) -> List[str]:
        """Convert user preferences to relevant keywords"""
        keyword_mapping = {
            'culture': ['museum', 'art gallery', 'historical site', 'theater'],
            'nature': ['park', 'garden', 'hiking trail', 'botanical garden'],
            'food': ['restaurant', 'cafe', 'food market', 'bistro'],
            'shopping': ['mall', 'shopping center', 'market', 'boutique'],
            'entertainment': ['theater', 'cinema', 'concert hall', 'comedy club']
        }
        
        keywords = []
        for category, score in self.preferences.items():
            if score >= 3 and category in keyword_mapping:
                keywords.extend(keyword_mapping[category])
        return list(set(keywords))

    def get_preference_score(self, place_type: str) -> float:
        """Calculate preference score for a place type"""
        type_category_mapping = {
            'museum': 'culture',
            'art_gallery': 'culture',
            'park': 'nature',
            'restaurant': 'food',
            'shopping_mall': 'shopping',
            'movie_theater': 'entertainment'
        }
        
        category = type_category_mapping.get(place_type)
        if category:
            return self.preferences.get(category, 0) / 5.0
        return 0.2

class ItineraryGenerator:
    """Generates personalized travel itineraries"""
    def __init__(self, google_api: GooglePlacesAPI, 
                 tripadvisor_api: TripAdvisorAPI, 
                 review_processor: ReviewProcessor):
        self.google_api = google_api
        self.tripadvisor_api = tripadvisor_api
        self.review_processor = review_processor

    def generate_itinerary(self, location: str, user_preferences: UserPreferences, 
                          num_days: int) -> Dict:
        """Generate a complete itinerary based on user preferences"""
        try:
            # Get search keywords based on user preferences
            keywords = user_preferences.get_keywords_from_preferences()
            logger.info(f"Generated keywords: {keywords}")

            # Search for places using both APIs in parallel
            with ThreadPoolExecutor() as executor:
                google_future = executor.submit(
                    self.google_api.search_places, location, keywords)
                tripadvisor_future = executor.submit(
                    self.tripadvisor_api.search_places, location, keywords)

                places = []
                for future in as_completed([google_future, tripadvisor_future]):
                    try:
                        places.extend(future.result())
                    except Exception as e:
                        logger.error(f"Error fetching places: {e}")

            if not places:
                logger.warning(f"No places found for location: {location}")
                return {}

            # Filter and process places
            places = self._process_places(places, user_preferences)
            places = self._filter_by_budget(places, user_preferences.budget_level)

            # Generate daily itineraries
            return self._create_daily_itineraries(places, num_days, user_preferences)

        except Exception as e:
            logger.error(f"Error generating itinerary: {e}")
            raise

    def _process_places(self, places: List[Place], 
                       user_preferences: UserPreferences) -> List[Place]:
        """Process and score places"""
        processed_places = []
        
        for place in places:
            try:
                # Analyze reviews
                review_texts = [review['text'] for review in place.reviews]
                review_scores = self.review_processor.analyze_reviews_batch(review_texts)
                
                # Calculate review score
                avg_sentiment = np.mean(review_scores['sentiment_scores']) if review_scores['sentiment_scores'] else 0
                avg_rating = place.rating if place.rating > 0 else 3.0  # Default rating if none available
                
                # Calculate preference score
                preference_score = user_preferences.get_preference_score(place.type)
                
                # Combined score calculation
                place.score = (
                    avg_rating * 0.4 +  # Base rating
                    (avg_sentiment + 1) * 2.5 * 0.3 +  # Normalized sentiment (-1 to 1 → 0 to 5)
                    preference_score * 5 * 0.3  # Normalized preference score
                )
                
                processed_places.append(place)
                
            except Exception as e:
                logger.error(f"Error processing place {place.name}: {e}")
                continue
        
        # Sort by score
        return sorted(processed_places, key=lambda x: x.score, reverse=True)

    def _filter_by_budget(self, places: List[Place], budget_level: int) -> List[Place]:
        """Filter places based on budget level"""
        return [place for place in places 
                if place.price_level <= budget_level]

    def _create_daily_itineraries(self, places: List[Place], num_days: int, 
                                user_preferences: UserPreferences) -> Dict:
        """Create optimized daily itineraries with improved variety"""
        itinerary = {}
        used_places = set()
        
        # Categorize places by type
        places_by_type = {}
        for place in places:
            if place.type not in places_by_type:
                places_by_type[place.type] = []
            places_by_type[place.type].append(place)
        
        # Sort each category by score
        for type_places in places_by_type.values():
            type_places.sort(key=lambda x: x.score, reverse=True)
        
        for day in range(1, num_days + 1):
            day_places = {}
            
            # Define ideal types for each time slot
            time_slot_preferences = {
                'morning': ['museum', 'art gallery', 'park', 'garden'],
                'afternoon': ['shopping center', 'mall', 'market', 'botanical garden'],
                'evening': ['restaurant', 'bistro', 'food market'],
                'night': ['theater', 'concert hall', 'comedy club']
            }
            
            for time_slot, preferred_types in time_slot_preferences.items():
                selected_place = None
                
                # Try to find a place of preferred type first
                for p_type in preferred_types:
                    if p_type in places_by_type:
                        available_places = [
                            p for p in places_by_type[p_type]
                            if p.name not in used_places and
                            self._check_opening_hours(p, time_slot, user_preferences.preferred_times)
                        ]
                        if available_places:
                            selected_place = available_places[0]
                            break
                
                # If no preferred type available, use any available place
                if not selected_place:
                    for type_places in places_by_type.values():
                        available_places = [
                            p for p in type_places
                            if p.name not in used_places and
                            self._check_opening_hours(p, time_slot, user_preferences.preferred_times)
                        ]
                        if available_places:
                            selected_place = available_places[0]
                            break
                
                if selected_place:
                    day_places[time_slot] = selected_place
                    used_places.add(selected_place.name)
            
            # Optimize route for the day
            if day_places:
                ordered_places = self._optimize_route(list(day_places.values()))
                itinerary[f'Day {day}'] = {
                    'morning': ordered_places[0] if len(ordered_places) > 0 else None,
                    'afternoon': ordered_places[1] if len(ordered_places) > 1 else None,
                    'evening': ordered_places[2] if len(ordered_places) > 2 else None,
                    'night': ordered_places[3] if len(ordered_places) > 3 else None
                }
        
        return itinerary

    def _check_opening_hours(self, place: Place, time_slot: str, 
                           preferred_times: Dict[str, List[str]]) -> bool:
        """Check if a place is open during the given time slot"""
        if not place.opening_hours:
            return True

        try:
            time_range = preferred_times[time_slot]
            current_date = datetime.now().date()
            start_time = datetime.strptime(f"{current_date} {time_range[0]}", 
                                         "%Y-%m-%d %H:%M").time()
            end_time = datetime.strptime(f"{current_date} {time_range[1]}", 
                                       "%Y-%m-%d %H:%M").time()

            # Check if place is open
            is_open = place.opening_hours.get('open_now', True)
            if not is_open:
                return False

            # Check specific opening hours if available
            periods = place.opening_hours.get('periods', [])
            if not periods:
                return True

            for period in periods:
                if period.get('open') and period.get('close'):
                    period_start = datetime.strptime(period['open']['time'], 
                                                   "%H%M").time()
                    period_end = datetime.strptime(period['close']['time'], 
                                                 "%H%M").time()
                    
                    if (start_time >= period_start and end_time <= period_end):
                        return True

            return False

        except Exception as e:
            logger.error(f"Error checking opening hours: {e}")
            return True

    def _optimize_route(self, places: List[Place]) -> List[Place]:
        """Optimize the route between places using nearest neighbor approach"""
        if not places:
            return places

        optimized = [places[0]]
        remaining = places[1:]

        while remaining:
            last_place = optimized[-1]
            # Find nearest place
            nearest = min(remaining, 
                        key=lambda x: self._calculate_distance(
                            last_place.location, x.location))
            optimized.append(nearest)
            remaining.remove(nearest)

        return optimized

    def _calculate_distance(self, loc1: Dict, loc2: Dict) -> float:
        """Calculate distance between two locations using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        try:
            lat1, lon1 = float(loc1['lat']), float(loc1['lng'])
            lat2, lon2 = float(loc2['lat']), float(loc2['lng'])
            
            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
            
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            
            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error calculating distance: {e}")
            return float('inf')  # Return infinity for invalid coordinates

def main():
    """Main function to demonstrate the application"""
    try:
        # Initialize APIs and components
        google_api = GooglePlacesAPI()
        tripadvisor_api = TripAdvisorAPI()
        review_processor = ReviewProcessor()
        
        # Initialize ItineraryGenerator
        generator = ItineraryGenerator(google_api, tripadvisor_api, review_processor)
        
        # Example preferences
        preferences = {
            'culture': 5,
            'nature': 4,
            'food': 5,
            'shopping': 3,
            'entertainment': 4
        }
        
        user_prefs = UserPreferences()
        for category, value in preferences.items():
            user_prefs.add_preference(category, value)
        user_prefs.set_budget_level(3)
        
        # Generate itinerary
        logger.info("Generating itinerary...")
        itinerary = generator.generate_itinerary(
            location="New York, NY",
            user_preferences=user_prefs,
            num_days=3
        )
        
        # Print results
        if itinerary:
            print("\nGenerated Itinerary:")
            for day, activities in itinerary.items():
                print(f"\n{day}:")
                for time_slot, place in activities.items():
                    if place:
                        print(f"  {time_slot.title()}: {place.name} ({place.rating}/5)")
        else:
            print("No itinerary could be generated.")
            
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise

if __name__ == "__main__":
    main()