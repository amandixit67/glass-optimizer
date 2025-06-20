import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class EfficiencyTarget:
    """Target efficiency parameters"""
    desired_efficiency: float  # Percentage (0-100)
    total_area_needed: float   # Total area of all pieces needed
    max_sheets: Optional[int] = None  # Maximum number of sheets allowed
    sheet_width: float = 600.0
    sheet_height: float = 400.0

@dataclass
class OptimizationRecommendation:
    """Recommendation for achieving target efficiency"""
    recommended_sheet_size: Tuple[float, float]
    recommended_piece_sizes: List[Tuple[float, float, int]]
    estimated_efficiency: float
    waste_percentage: float
    sheets_required: int
    cost_analysis: Dict
    optimization_strategy: str

class ReverseOptimizer:
    """Reverse optimization tool to find measurements for desired efficiency"""
    
    def __init__(self):
        self.standard_sheet_sizes = [
            (600, 400), (800, 600), (1000, 800), (1200, 1000),
            (400, 300), (500, 400), (700, 500), (900, 700)
        ]
        
    def calculate_required_measurements(self, 
                                      target: EfficiencyTarget,
                                      piece_constraints: Optional[Dict] = None) -> OptimizationRecommendation:
        """
        Calculate required measurements to achieve desired efficiency
        
        Args:
            target: Efficiency target parameters
            piece_constraints: Constraints on piece sizes (min/max dimensions)
            
        Returns:
            OptimizationRecommendation with recommended measurements
        """
        
        # Calculate total sheet area needed
        total_sheet_area = target.total_area_needed / (target.desired_efficiency / 100)
        
        # Find optimal sheet size
        optimal_sheet_size = self._find_optimal_sheet_size(total_sheet_area, target)
        
        # Calculate recommended piece sizes
        recommended_pieces = self._calculate_optimal_piece_sizes(
            target.total_area_needed, 
            optimal_sheet_size, 
            target.desired_efficiency,
            piece_constraints
        )
        
        # Calculate estimated efficiency
        estimated_efficiency = self._calculate_estimated_efficiency(
            recommended_pieces, optimal_sheet_size
        )
        
        # Calculate cost analysis
        cost_analysis = self._calculate_cost_analysis(
            recommended_pieces, optimal_sheet_size, target.desired_efficiency
        )
        
        return OptimizationRecommendation(
            recommended_sheet_size=optimal_sheet_size,
            recommended_piece_sizes=recommended_pieces,
            estimated_efficiency=estimated_efficiency,
            waste_percentage=100 - estimated_efficiency,
            sheets_required=math.ceil(target.total_area_needed / (optimal_sheet_size[0] * optimal_sheet_size[1] * (target.desired_efficiency / 100))),
            cost_analysis=cost_analysis,
            optimization_strategy=self._determine_strategy(target.desired_efficiency)
        )
    
    def _find_optimal_sheet_size(self, total_sheet_area: float, target: EfficiencyTarget) -> Tuple[float, float]:
        """Find the optimal sheet size for the given area"""
        
        # If custom sheet size is specified, use it
        if target.sheet_width and target.sheet_height:
            return (target.sheet_width, target.sheet_height)
        
        # Find the best standard sheet size
        best_sheet = None
        best_waste = float('inf')
        
        for width, height in self.standard_sheet_sizes:
            sheet_area = width * height
            waste = abs(sheet_area - total_sheet_area)
            
            if waste < best_waste:
                best_waste = waste
                best_sheet = (width, height)
        
        return best_sheet or (600, 400)
    
    def _calculate_optimal_piece_sizes(self, 
                                     total_area: float, 
                                     sheet_size: Tuple[float, float],
                                     target_efficiency: float,
                                     constraints: Optional[Dict] = None) -> List[Tuple[float, float, int]]:
        """Calculate optimal piece sizes to achieve target efficiency"""
        
        sheet_width, sheet_height = sheet_size
        sheet_area = sheet_width * sheet_height
        
        # Calculate target used area
        target_used_area = sheet_area * (target_efficiency / 100)
        
        # Generate piece size combinations
        piece_combinations = self._generate_piece_combinations(
            sheet_width, sheet_height, target_used_area, constraints
        )
        
        # Find the best combination
        best_combination = self._find_best_piece_combination(
            piece_combinations, target_used_area, target_efficiency
        )
        
        return best_combination
    
    def _generate_piece_combinations(self, 
                                   sheet_width: float, 
                                   sheet_height: float,
                                   target_area: float,
                                   constraints: Optional[Dict] = None) -> List[List[Tuple[float, float, int]]]:
        """Generate possible piece size combinations"""
        
        min_width = constraints.get('min_width', 50) if constraints else 50
        max_width = constraints.get('max_width', sheet_width) if constraints else sheet_width
        min_height = constraints.get('min_height', 50) if constraints else 50
        max_height = constraints.get('max_height', sheet_height) if constraints else sheet_height
        
        combinations = []
        
        # Generate different piece sizes
        for width in np.arange(min_width, max_width, 50):
            for height in np.arange(min_height, max_height, 50):
                piece_area = width * height
                
                # Calculate how many pieces of this size can fit
                max_pieces = int(target_area / piece_area)
                
                if max_pieces > 0:
                    # Try different quantities
                    for quantity in range(1, min(max_pieces + 1, 20)):
                        total_piece_area = piece_area * quantity
                        
                        if total_piece_area <= target_area * 1.1:  # Allow 10% tolerance
                            combinations.append([(width, height, quantity)])
        
        # Generate combinations of different piece sizes
        mixed_combinations = self._generate_mixed_combinations(
            sheet_width, sheet_height, target_area, constraints
        )
        combinations.extend(mixed_combinations)
        
        return combinations
    
    def _generate_mixed_combinations(self, 
                                   sheet_width: float, 
                                   sheet_height: float,
                                   target_area: float,
                                   constraints: Optional[Dict] = None) -> List[List[Tuple[float, float, int]]]:
        """Generate combinations of different piece sizes"""
        
        combinations = []
        
        # Common facade piece sizes
        common_sizes = [
            (100, 80), (120, 100), (150, 120), (200, 150),
            (80, 60), (90, 70), (110, 90), (130, 110),
            (180, 140), (160, 130), (140, 100), (170, 130)
        ]
        
        # Filter based on constraints
        if constraints:
            min_width = constraints.get('min_width', 50)
            max_width = constraints.get('max_width', sheet_width)
            min_height = constraints.get('min_height', 50)
            max_height = constraints.get('max_height', sheet_height)
            
            common_sizes = [(w, h) for w, h in common_sizes 
                           if min_width <= w <= max_width and min_height <= h <= max_height]
        
        # Generate combinations of 2-4 different piece sizes
        for num_pieces in range(2, 5):
            for i in range(len(common_sizes)):
                for j in range(i + 1, len(common_sizes)):
                    if num_pieces == 2:
                        combination = [
                            (common_sizes[i][0], common_sizes[i][1], 1),
                            (common_sizes[j][0], common_sizes[j][1], 1)
                        ]
                        
                        total_area = sum(w * h * q for w, h, q in combination)
                        if total_area <= target_area * 1.1:
                            combinations.append(combination)
        
        return combinations
    
    def _find_best_piece_combination(self, 
                                   combinations: List[List[Tuple[float, float, int]]],
                                   target_area: float,
                                   target_efficiency: float) -> List[Tuple[float, float, int]]:
        """Find the best piece combination that achieves target efficiency"""
        
        best_combination = []
        best_score = float('inf')
        
        for combination in combinations:
            total_area = sum(w * h * q for w, h, q in combination)
            efficiency = (total_area / target_area) * 100
            
            # Calculate score based on efficiency match and piece variety
            efficiency_diff = abs(efficiency - target_efficiency)
            variety_score = len(combination) * 10  # Prefer more variety
            
            score = efficiency_diff + variety_score
            
            if score < best_score:
                best_score = score
                best_combination = combination
        
        return best_combination if best_combination else [(100, 80, 1)]
    
    def _calculate_estimated_efficiency(self, 
                                      pieces: List[Tuple[float, float, int]], 
                                      sheet_size: Tuple[float, float]) -> float:
        """Calculate estimated efficiency for given pieces and sheet size"""
        
        total_piece_area = sum(w * h * q for w, h, q in pieces)
        sheet_area = sheet_size[0] * sheet_size[1]
        
        return (total_piece_area / sheet_area) * 100
    
    def _calculate_cost_analysis(self, 
                               pieces: List[Tuple[float, float, int]], 
                               sheet_size: Tuple[float, float],
                               target_efficiency: float) -> Dict:
        """Calculate cost analysis for the optimization"""
        
        sheet_area = sheet_size[0] * sheet_size[1]
        total_piece_area = sum(w * h * q for w, h, q in pieces)
        
        # Assume glass cost per square meter (adjust as needed)
        glass_cost_per_sqm = 50  # USD per square meter
        
        # Calculate costs
        sheet_cost = (sheet_area / 10000) * glass_cost_per_sqm  # Convert cm² to m²
        piece_value = (total_piece_area / 10000) * glass_cost_per_sqm
        waste_cost = ((sheet_area - total_piece_area) / 10000) * glass_cost_per_sqm
        
        # Calculate savings compared to lower efficiency
        baseline_efficiency = 70  # Assume baseline efficiency
        baseline_sheets_needed = math.ceil(total_piece_area / (sheet_area * (baseline_efficiency / 100)))
        optimized_sheets_needed = math.ceil(total_piece_area / (sheet_area * (target_efficiency / 100)))
        
        sheets_saved = max(0, baseline_sheets_needed - optimized_sheets_needed)
        cost_savings = sheets_saved * sheet_cost
        
        return {
            'sheet_cost_usd': round(sheet_cost, 2),
            'piece_value_usd': round(piece_value, 2),
            'waste_cost_usd': round(waste_cost, 2),
            'sheets_saved': sheets_saved,
            'cost_savings_usd': round(cost_savings, 2),
            'roi_percentage': round((cost_savings / sheet_cost) * 100, 1) if sheet_cost > 0 else 0
        }
    
    def _determine_strategy(self, target_efficiency: float) -> str:
        """Determine optimization strategy based on target efficiency"""
        
        if target_efficiency >= 90:
            return "High-Efficiency Cutting (Advanced Algorithms)"
        elif target_efficiency >= 80:
            return "Optimized Layout (Genetic Algorithm)"
        elif target_efficiency >= 70:
            return "Standard Optimization (Best-Fit Algorithm)"
        else:
            return "Basic Layout (Greedy Algorithm)"
    
    def get_efficiency_guidelines(self) -> Dict:
        """Get guidelines for different efficiency targets"""
        
        return {
            'high_efficiency': {
                'range': '85-95%',
                'description': 'Maximum material utilization, complex layouts',
                'best_for': 'High-value projects, limited materials',
                'algorithm': 'Genetic Algorithm',
                'setup_time': '5-15 minutes',
                'cost_premium': '10-20%'
            },
            'standard_efficiency': {
                'range': '75-85%',
                'description': 'Good balance of efficiency and simplicity',
                'best_for': 'Most commercial projects',
                'algorithm': 'Best-Fit Algorithm',
                'setup_time': '2-5 minutes',
                'cost_premium': '5-10%'
            },
            'basic_efficiency': {
                'range': '60-75%',
                'description': 'Simple layouts, fast processing',
                'best_for': 'Quick estimates, simple projects',
                'algorithm': 'Greedy Algorithm',
                'setup_time': '30 seconds - 2 minutes',
                'cost_premium': '0-5%'
            }
        }
    
    def suggest_improvements(self, current_efficiency: float, target_efficiency: float) -> List[str]:
        """Suggest improvements to reach target efficiency"""
        
        suggestions = []
        
        if target_efficiency > current_efficiency:
            if target_efficiency - current_efficiency > 20:
                suggestions.append("Consider using larger sheet sizes to reduce waste")
                suggestions.append("Group similar piece sizes together")
                suggestions.append("Use advanced optimization algorithms")
            
            if target_efficiency - current_efficiency > 10:
                suggestions.append("Optimize piece dimensions for better fit")
                suggestions.append("Consider piece rotation for better placement")
                suggestions.append("Use mixed-size cutting strategies")
            
            if target_efficiency - current_efficiency > 5:
                suggestions.append("Fine-tune piece quantities")
                suggestions.append("Adjust piece dimensions slightly")
                suggestions.append("Use best-fit placement algorithms")
        
        return suggestions 