import cv2
import mediapipe as mp
import numpy as np
from skimage import feature, filters, measure
from typing import Dict, List, Tuple, Any
import asyncio
import time
from PIL import Image
import io

class FaceYogaAnalyzer:
    """Advanced face analysis using MediaPipe and OpenCV"""
    
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_mesh = None
        self.initialized = False
        
        # Facial landmark indices for different regions
        self.FACE_REGIONS = {
            'forehead': [10, 151, 9, 10, 151, 9, 10, 151],
            'eye_left': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'eye_right': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'nose': [1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51, 48, 115, 131, 134, 102, 49, 220, 305, 281, 360, 279],
            'mouth': [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
            'cheek_left': [116, 117, 118, 119, 120, 121, 128, 126, 142, 36, 205, 206, 207, 213, 192, 147],
            'cheek_right': [345, 346, 347, 348, 349, 350, 451, 452, 453, 464, 435, 410, 454, 323, 361, 340],
            'jawline': [172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323]
        }
    
    async def initialize(self):
        """Initialize MediaPipe models"""
        try:
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.initialized = True
            print("✅ MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize MediaPipe: {e}")
            raise
    
    def is_initialized(self) -> bool:
        """Check if analyzer is initialized"""
        return self.initialized
    
    def _load_image_from_bytes(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes"""
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('RGB')
        return np.array(image)
    
    def _detect_landmarks(self, image: np.ndarray) -> Tuple[List[Tuple[int, int]], int]:
        """Detect 468 facial landmarks using MediaPipe"""
        if not self.initialized:
            raise RuntimeError("Analyzer not initialized")
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.face_mesh.process(rgb_image)
        
        landmarks = []
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmarks.append((x, y))
        
        return landmarks, len(landmarks)
    
    def _analyze_wrinkles(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze wrinkles using OpenCV edge detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Sobel edge detection for texture analysis
        sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Analyze different facial regions
        region_scores = {}
        for region_name, region_indices in self.FACE_REGIONS.items():
            if len(landmarks) >= max(region_indices, default=0):
                # Create mask for region
                region_points = [landmarks[i] for i in region_indices if i < len(landmarks)]
                if len(region_points) > 2:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.fillPoly(mask, [np.array(region_points)], 255)
                    
                    # Calculate edge density in region
                    region_edges = cv2.bitwise_and(edges, mask)
                    edge_density = np.sum(region_edges > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0
                    
                    # Calculate texture variance
                    region_gray = cv2.bitwise_and(blurred, mask)
                    texture_variance = np.var(region_gray[mask > 0]) if np.sum(mask > 0) > 0 else 0
                    
                    region_scores[region_name] = {
                        'edge_density': float(edge_density),
                        'texture_variance': float(texture_variance),
                        'smoothness_score': float(1.0 - min(edge_density * 2, 1.0))
                    }
        
        # Overall wrinkle score (lower is better)
        overall_score = np.mean([scores['smoothness_score'] for scores in region_scores.values()])
        
        return {
            'overall_score': float(overall_score),
            'region_analysis': region_scores,
            'edge_density': float(np.sum(edges > 0) / edges.size),
            'texture_quality': float(np.std(sobel_magnitude))
        }
    
    def _analyze_symmetry(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze facial symmetry using landmark positions"""
        if len(landmarks) < 468:
            return {'overall_score': 0.5, 'analysis': 'Insufficient landmarks'}
        
        h, w = image.shape[:2]
        center_x = w // 2
        
        # Define symmetric landmark pairs (MediaPipe 468 landmarks)
        symmetric_pairs = [
            (33, 362),    # Eye corners
            (133, 362),   # Eye centers
            (61, 291),    # Mouth corners
            (13, 14),     # Face outline
            (116, 345),   # Cheek points
            (172, 397),   # Jaw points
        ]
        
        symmetry_scores = []
        for left_idx, right_idx in symmetric_pairs:
            if left_idx < len(landmarks) and right_idx < len(landmarks):
                left_point = landmarks[left_idx]
                right_point = landmarks[right_idx]
                
                # Calculate distance from center line
                left_dist = abs(left_point[0] - center_x)
                right_dist = abs(right_point[0] - center_x)
                
                # Calculate symmetry score (1.0 = perfect symmetry)
                max_dist = max(left_dist, right_dist)
                if max_dist > 0:
                    symmetry = 1.0 - abs(left_dist - right_dist) / max_dist
                    symmetry_scores.append(symmetry)
        
        overall_symmetry = np.mean(symmetry_scores) if symmetry_scores else 0.5
        
        # Analyze vertical alignment
        nose_tip = landmarks[1] if len(landmarks) > 1 else (center_x, h//2)
        vertical_alignment = 1.0 - abs(nose_tip[0] - center_x) / (w * 0.1)
        vertical_alignment = max(0.0, min(1.0, vertical_alignment))
        
        return {
            'overall_score': float(overall_symmetry),
            'vertical_alignment': float(vertical_alignment),
            'symmetry_pairs': len(symmetric_pairs),
            'analysis': f'Analyzed {len(symmetry_scores)} symmetric features'
        }
    
    def _analyze_contours(self, image: np.ndarray, landmarks: List[Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze facial contours and definition"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian for edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = laplacian.var()
        
        # Gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Analyze jawline definition if landmarks available
        jawline_score = 0.5
        if len(landmarks) > 200:
            jawline_indices = self.FACE_REGIONS.get('jawline', [])
            if jawline_indices:
                jawline_points = [landmarks[i] for i in jawline_indices if i < len(landmarks)]
                if len(jawline_points) > 3:
                    # Create jawline mask
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.polylines(mask, [np.array(jawline_points)], False, 255, 3)
                    
                    # Calculate edge strength along jawline
                    jawline_edges = cv2.bitwise_and((gradient_magnitude > np.percentile(gradient_magnitude, 75)).astype(np.uint8) * 255, mask)
                    jawline_score = np.sum(jawline_edges > 0) / np.sum(mask > 0) if np.sum(mask > 0) > 0 else 0.5
        
        # Overall contour score
        contour_strength = min(laplacian_var / 1000, 1.0)  # Normalize
        
        return {
            'overall_score': float(contour_strength),
            'jawline_definition': float(jawline_score),
            'edge_strength': float(np.mean(gradient_magnitude)),
            'laplacian_variance': float(laplacian_var)
        }
    
    def _generate_recommendations(self, wrinkle_data: Dict, symmetry_data: Dict, contour_data: Dict) -> List[str]:
        """Generate personalized recommendations based on analysis"""
        recommendations = []
        
        # Wrinkle-based recommendations
        if wrinkle_data['overall_score'] < 0.7:
            recommendations.append("Focus on forehead smoothing exercises")
            recommendations.append("Practice eye area relaxation techniques")
        
        # Symmetry-based recommendations
        if symmetry_data['overall_score'] < 0.8:
            recommendations.append("Work on facial muscle balance exercises")
            recommendations.append("Practice mirror symmetry awareness")
        
        # Contour-based recommendations
        if contour_data['overall_score'] < 0.6:
            recommendations.append("Strengthen jawline with resistance exercises")
            recommendations.append("Practice facial contouring techniques")
        
        # General recommendations
        recommendations.extend([
            "Maintain consistent daily practice",
            "Stay hydrated for optimal skin health",
            "Practice stress-reduction techniques"
        ])
        
        return recommendations[:5]  # Limit to top 5
    
    async def analyze_images(self, before_image_bytes: bytes, after_image_bytes: bytes) -> Dict[str, Any]:
        """Perform complete facial analysis comparison"""
        start_time = time.time()
        
        try:
            # Load images
            before_image = self._load_image_from_bytes(before_image_bytes)
            after_image = self._load_image_from_bytes(after_image_bytes)
            
            # Convert to BGR for OpenCV
            before_bgr = cv2.cvtColor(before_image, cv2.COLOR_RGB2BGR)
            after_bgr = cv2.cvtColor(after_image, cv2.COLOR_RGB2BGR)
            
            # Detect landmarks
            before_landmarks, before_count = self._detect_landmarks(before_bgr)
            after_landmarks, after_count = self._detect_landmarks(after_bgr)
            
            if before_count == 0 or after_count == 0:
                raise ValueError("No face detected in one or both images")
            
            # Analyze before image
            before_wrinkles = self._analyze_wrinkles(before_bgr, before_landmarks)
            before_symmetry = self._analyze_symmetry(before_bgr, before_landmarks)
            before_contours = self._analyze_contours(before_bgr, before_landmarks)
            
            # Analyze after image
            after_wrinkles = self._analyze_wrinkles(after_bgr, after_landmarks)
            after_symmetry = self._analyze_symmetry(after_bgr, after_landmarks)
            after_contours = self._analyze_contours(after_bgr, after_landmarks)
            
            # Calculate improvements
            wrinkle_improvement = after_wrinkles['overall_score'] - before_wrinkles['overall_score']
            symmetry_improvement = after_symmetry['overall_score'] - before_symmetry['overall_score']
            contour_improvement = after_contours['overall_score'] - before_contours['overall_score']
            
            # Overall scores
            overall_before = (before_wrinkles['overall_score'] + before_symmetry['overall_score'] + before_contours['overall_score']) / 3
            overall_after = (after_wrinkles['overall_score'] + after_symmetry['overall_score'] + after_contours['overall_score']) / 3
            
            processing_time = time.time() - start_time
            
            # Generate recommendations
            recommendations = self._generate_recommendations(after_wrinkles, after_symmetry, after_contours)
            
            return {
                'landmarks_count': after_count,
                'facial_regions': {
                    'before': {
                        'wrinkles': before_wrinkles,
                        'symmetry': before_symmetry,
                        'contours': before_contours
                    },
                    'after': {
                        'wrinkles': after_wrinkles,
                        'symmetry': after_symmetry,
                        'contours': after_contours
                    }
                },
                'improvements': {
                    'wrinkle_reduction': float(wrinkle_improvement),
                    'symmetry_improvement': float(symmetry_improvement),
                    'contour_enhancement': float(contour_improvement),
                    'overall_improvement': float(overall_after - overall_before)
                },
                'scores': {
                    'wrinkle_score': float(after_wrinkles['overall_score']),
                    'symmetry_score': float(after_symmetry['overall_score']),
                    'contour_score': float(after_contours['overall_score']),
                    'overall_score': float(overall_after)
                },
                'recommendations': recommendations,
                'analysis_metadata': {
                    'processing_time': float(processing_time),
                    'landmarks_detected': after_count,
                    'analysis_version': '2.0_mediapipe_opencv'
                }
            }
            
        except Exception as e:
            raise RuntimeError(f"Analysis failed: {str(e)}")
