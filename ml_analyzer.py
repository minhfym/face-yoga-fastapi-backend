import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple, Optional
import logging
from skimage import filters, measure, feature
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaceYogaMLAnalyzer:
    """
    Advanced ML analyzer using MediaPipe for facial landmark detection
    and OpenCV for comprehensive facial analysis.
    """
    
    def __init__(self):
        """Initialize MediaPipe and OpenCV components."""
        self.mp_face_mesh = None
        self.face_mesh = None
        self.initialized = False
        
        try:
            # Initialize MediaPipe Face Mesh
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.initialized = True
            logger.info("✅ MediaPipe Face Mesh initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize MediaPipe: {e}")
            self.initialized = False
    
    def is_ready(self) -> bool:
        """Check if analyzer is ready."""
        return self.initialized
    
    def analyze_image(self, image_data: bytes) -> Dict:
        """
        Perform comprehensive facial analysis on an image.
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary containing analysis results
        """
        if not self.initialized:
            return {
                "success": False,
                "error": "ML analyzer not initialized",
                "landmarks_detected": 0
            }
        
        try:
            # Convert bytes to OpenCV image
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return {
                    "success": False,
                    "error": "Could not decode image",
                    "landmarks_detected": 0
                }
            
            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            height, width = image.shape[:2]
            
            # Perform MediaPipe face mesh detection
            results = self.face_mesh.process(rgb_image)
            
            if not results.multi_face_landmarks:
                return {
                    "success": False,
                    "error": "No face detected in image",
                    "landmarks_detected": 0
                }
            
            # Extract landmarks (468 points)
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = []
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                landmarks.append((x, y))
            
            # Perform comprehensive analysis
            analysis_results = {
                "success": True,
                "landmarks_detected": len(landmarks),
                "image_dimensions": {"width": width, "height": height},
                "face_analysis": self._analyze_facial_features(image, landmarks),
                "quality_metrics": self._assess_image_quality(image),
                "analysis_metadata": {
                    "analyzer_version": "railway_optimized_v1.0",
                    "mediapipe_landmarks": len(landmarks)
                }
            }
            
            logger.info(f"✅ Analysis completed: {len(landmarks)} landmarks detected")
            return analysis_results
            
        except Exception as e:
            logger.error(f"❌ Analysis failed: {e}")
            return {
                "success": False,
                "error": f"Analysis error: {str(e)}",
                "landmarks_detected": 0
            }
    
    def compare_images(self, before_data: bytes, after_data: bytes) -> Dict:
        """Compare before and after images to show improvement."""
        try:
            # Analyze both images
            before_analysis = self.analyze_image(before_data)
            after_analysis = self.analyze_image(after_data)
            
            if not (before_analysis["success"] and after_analysis["success"]):
                return {
                    "success": False,
                    "error": "Failed to analyze one or both images",
                    "before_success": before_analysis["success"],
                    "after_success": after_analysis["success"]
                }
            
            # Calculate improvements
            before_face = before_analysis["face_analysis"]
            after_face = after_analysis["face_analysis"]
            
            improvements = {
                "wrinkle_improvement": after_face["wrinkle_score"] - before_face["wrinkle_score"],
                "symmetry_improvement": after_face["symmetry_score"] - before_face["symmetry_score"],
                "contour_improvement": after_face["contour_score"] - before_face["contour_score"]
            }
            
            overall_improvement = np.mean(list(improvements.values()))
            
            return {
                "success": True,
                "before_analysis": before_analysis,
                "after_analysis": after_analysis,
                "improvements": improvements,
                "overall_improvement": float(overall_improvement),
                "improvement_percentage": float(max(0, overall_improvement)),
                "recommendations": self._generate_recommendations(improvements),
                "comparison_metadata": {
                    "before_landmarks": before_analysis["landmarks_detected"],
                    "after_landmarks": after_analysis["landmarks_detected"],
                    "analyzer_version": "railway_optimized_v1.0"
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Image comparison failed: {e}")
            return {
                "success": False,
                "error": f"Comparison error: {str(e)}"
            }
    
    def _analyze_facial_features(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> Dict:
        """Analyze facial features using OpenCV."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Basic wrinkle analysis using edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            wrinkle_score = max(0, 100 - (edge_density * 1000))
            
            # Basic symmetry analysis
            landmarks_array = np.array(landmarks)
            center_x = np.mean(landmarks_array[:, 0])
            
            left_landmarks = landmarks_array[landmarks_array[:, 0] < center_x]
            right_landmarks = landmarks_array[landmarks_array[:, 0] > center_x]
            
            if len(left_landmarks) > 0 and len(right_landmarks) > 0:
                left_spread = np.std(left_landmarks[:, 1])
                right_spread = np.std(right_landmarks[:, 1])
                symmetry_score = max(0, 100 - abs(left_spread - right_spread) * 2)
            else:
                symmetry_score = 50.0
            
            # Basic contour analysis
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            contour_strength = laplacian.var()
            contour_score = min(100, contour_strength / 10)
            
            return {
                "wrinkle_score": float(wrinkle_score),
                "symmetry_score": float(symmetry_score),
                "contour_score": float(contour_score),
                "edge_density": float(edge_density),
                "overall_score": float((wrinkle_score + symmetry_score + contour_score) / 3)
            }
            
        except Exception as e:
            logger.error(f"❌ Facial feature analysis failed: {e}")
            return {
                "wrinkle_score": 50.0,
                "symmetry_score": 50.0,
                "contour_score": 50.0,
                "overall_score": 50.0
            }
    
    def _assess_image_quality(self, image: np.ndarray) -> Dict:
        """Assess the quality of the input image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate brightness
            brightness = np.mean(gray)
            
            # Calculate contrast using standard deviation
            contrast = np.std(gray)
            
            # Calculate sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Determine quality score (0-100)
            brightness_score = min(100, max(0, 100 - abs(brightness - 127) * 0.8))
            contrast_score = min(100, contrast * 2)
            sharpness_score = min(100, laplacian_var / 10)
            
            overall_quality = (brightness_score + contrast_score + sharpness_score) / 3
            
            return {
                "brightness": float(brightness),
                "contrast": float(contrast),
                "sharpness": float(laplacian_var),
                "overall_score": float(overall_quality),
                "quality_rating": "excellent" if overall_quality > 80 else 
                               "good" if overall_quality > 60 else 
                               "fair" if overall_quality > 40 else "poor"
            }
            
        except Exception as e:
            logger.error(f"❌ Image quality assessment failed: {e}")
            return {"overall_score": 50.0, "quality_rating": "unknown"}
    
    def _generate_recommendations(self, improvements: Dict) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        if improvements["wrinkle_improvement"] < -5:
            recommendations.append("Focus on forehead and eye area exercises to reduce fine lines")
        elif improvements["wrinkle_improvement"] > 5:
            recommendations.append("Great improvement in skin smoothness! Continue current routine")
        
        if improvements["symmetry_improvement"] < -3:
            recommendations.append("Practice facial symmetry exercises with balanced movements")
        elif improvements["symmetry_improvement"] > 3:
            recommendations.append("Excellent symmetry improvement! Your practice is working")
        
        if improvements["contour_improvement"] < -4:
            recommendations.append("Add jawline and cheekbone definition exercises")
        elif improvements["contour_improvement"] > 4:
            recommendations.append("Outstanding facial definition improvement!")
        
        if not recommendations:
            recommendations.append("Continue your current face yoga practice for maintained results")
        
        # Add general recommendations
        recommendations.extend([
            "Stay hydrated for optimal skin health",
            "Practice consistently for best results",
            "Consider progress photos to track improvements"
        ])
        
        return recommendations[:5]  # Limit to top 5

# Global analyzer instance
analyzer = FaceYogaMLAnalyzer()

def get_analyzer() -> FaceYogaMLAnalyzer:
    """Get the global analyzer instance."""
    return analyzer
