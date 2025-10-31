# evaluation.py

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

@dataclass
class APIMetrics:
    """Tracks API performance metrics"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    response_times: List[float] = field(default_factory=list)
    errors: Dict[str, int] = field(default_factory=dict)

    def log_api_call(self, success: bool, response_time: float, error_type: Optional[str] = None):
        """Log an API call with its results"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            if error_type:
                self.errors[error_type] = self.errors.get(error_type, 0) + 1
        self.response_times.append(response_time)

    def get_success_rate(self) -> float:
        """Calculate API success rate"""
        return (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0

    def get_average_response_time(self) -> float:
        """Calculate average response time"""
        return np.mean(self.response_times) if self.response_times else 0

    def generate_report(self) -> Dict:
        """Generate a report of API metrics"""
        return {
            'total_calls': self.total_calls,
            'success_rate': self.get_success_rate(),
            'avg_response_time': self.get_average_response_time(),
            'error_distribution': self.errors
        }

@dataclass
class ItineraryMetrics:
    """Tracks itinerary generation metrics"""
    total_itineraries: int = 0
    total_slots: int = 0
    filled_slots: int = 0
    user_ratings: List[int] = field(default_factory=list)
    user_feedback: List[Dict] = field(default_factory=list)

    def log_itinerary(self, itinerary: Dict):
        """Log metrics for a generated itinerary"""
        self.total_itineraries += 1
        
        for day, activities in itinerary.items():
            for time_slot, place in activities.items():
                self.total_slots += 1
                if place is not None:
                    self.filled_slots += 1

    def add_user_rating(self, rating: int, feedback: str = ""):
        """Add user rating and feedback"""
        if 1 <= rating <= 5:
            self.user_ratings.append(rating)
            self.user_feedback.append({
                'rating': rating,
                'feedback': feedback,
                'timestamp': datetime.now().isoformat()
            })

    def get_completion_rate(self) -> float:
        """Calculate itinerary completion rate"""
        return (self.filled_slots / self.total_slots * 100) if self.total_slots > 0 else 0

    def get_average_rating(self) -> float:
        """Calculate average user rating"""
        return np.mean(self.user_ratings) if self.user_ratings else 0

    def generate_report(self) -> Dict:
        """Generate a report of itinerary metrics"""
        return {
            'total_itineraries': self.total_itineraries,
            'completion_rate': self.get_completion_rate(),
            'average_rating': self.get_average_rating(),
            'total_feedback_received': len(self.user_feedback)
        }

class PerformanceEvaluator:
    """Main class for evaluating system performance"""
    def __init__(self, log_dir: str = "evaluation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        self.api_metrics = {
            'google': APIMetrics(),
            'tripadvisor': APIMetrics()
        }
        self.itinerary_metrics = ItineraryMetrics()
        
        # Set up logging
        logging.basicConfig(
            filename=self.log_dir / 'evaluation.log',
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def log_api_request(self, api_name: str, success: bool, response_time: float, 
                       error_type: Optional[str] = None):
        """Log an API request"""
        if api_name in self.api_metrics:
            self.api_metrics[api_name].log_api_call(success, response_time, error_type)
            logging.info(f"API Call - {api_name}: Success={success}, Time={response_time:.2f}s")

    def log_itinerary_generation(self, itinerary: Dict):
        """Log itinerary generation metrics"""
        self.itinerary_metrics.log_itinerary(itinerary)
        logging.info(f"Itinerary Generated - Slots: {self.itinerary_metrics.total_slots}")

    def add_user_feedback(self, rating: int, feedback: str = ""):
        """Add user feedback"""
        self.itinerary_metrics.add_user_rating(rating, feedback)
        logging.info(f"User Feedback Received - Rating: {rating}")

    def generate_evaluation_report(self) -> Dict:
        """Generate comprehensive evaluation report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'api_performance': {
                name: metrics.generate_report()
                for name, metrics in self.api_metrics.items()
            },
            'itinerary_metrics': self.itinerary_metrics.generate_report()
        }
        
        # Save report to file
        report_path = self.log_dir / f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        return report

    def generate_performance_graphs(self):
        """Generate performance visualization graphs"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create API performance graph
            plt.figure(figsize=(10, 6))
            api_success_rates = {name: metrics.get_success_rate() 
                               for name, metrics in self.api_metrics.items()}
            plt.bar(api_success_rates.keys(), api_success_rates.values())
            plt.title('API Success Rates')
            plt.ylabel('Success Rate (%)')
            plt.savefig(self.log_dir / 'api_success_rates.png')
            plt.close()
            
            # Create user ratings distribution
            if self.itinerary_metrics.user_ratings:
                plt.figure(figsize=(8, 6))
                sns.histplot(self.itinerary_metrics.user_ratings, bins=5)
                plt.title('User Ratings Distribution')
                plt.xlabel('Rating')
                plt.ylabel('Count')
                plt.savefig(self.log_dir / 'user_ratings_distribution.png')
                plt.close()
                
        except ImportError:
            logging.warning("Could not generate graphs: matplotlib or seaborn not installed")

def main():
    """Example usage of the evaluation system"""
    evaluator = PerformanceEvaluator()
    
    # Simulate some API calls
    evaluator.log_api_request('google', True, 0.5)
    evaluator.log_api_request('google', False, 1.2, 'TIMEOUT')
    evaluator.log_api_request('tripadvisor', True, 0.8)
    
    # Simulate itinerary generation
    sample_itinerary = {
        'Day 1': {
            'morning': {'name': 'Museum'},
            'afternoon': {'name': 'Park'},
            'evening': None
        }
    }
    evaluator.log_itinerary_generation(sample_itinerary)
    
    # Simulate user feedback
    evaluator.add_user_feedback(4, "Great itinerary!")
    evaluator.add_user_feedback(5, "Perfect suggestions!")
    
    # Generate and print report
    report = evaluator.generate_evaluation_report()
    print(json.dumps(report, indent=2))
    
    # Generate graphs
    evaluator.generate_performance_graphs()

if __name__ == "__main__":
    main()