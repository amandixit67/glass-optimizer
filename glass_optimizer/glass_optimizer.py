import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from dataclasses import dataclass
import random
import copy

@dataclass
class GlassPiece:
    """Represents a glass piece with dimensions and quantity"""
    width: float
    height: float
    quantity: int
    id: str = ""
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.width}x{self.height}"

@dataclass
class Sheet:
    """Represents a glass sheet with dimensions"""
    width: float
    height: float
    pieces: List[Tuple[float, float, float, float]] = None  # x, y, width, height
    waste_area: float = 0.0
    
    def __post_init__(self):
        if self.pieces is None:
            self.pieces = []
        self.total_area = self.width * self.height

class GlassOptimizer:
    """AI-powered glass optimization tool using advanced bin packing algorithms"""
    
    def __init__(self, sheet_width: float, sheet_height: float):
        self.sheet_width = sheet_width
        self.sheet_height = sheet_height
        self.sheet_area = sheet_width * sheet_height
        
    def optimize_cutting(self, glass_pieces: List[GlassPiece], 
                        algorithm: str = "genetic") -> Dict:
        """
        Optimize glass cutting to minimize waste
        
        Args:
            glass_pieces: List of glass pieces with dimensions and quantities
            algorithm: Optimization algorithm ('genetic', 'greedy', 'best_fit')
            
        Returns:
            Dictionary with optimization results
        """
        # Expand pieces based on quantity
        expanded_pieces = []
        for piece in glass_pieces:
            for _ in range(piece.quantity):
                expanded_pieces.append(GlassPiece(piece.width, piece.height, 1, piece.id))
        
        if algorithm == "genetic":
            return self._genetic_algorithm(expanded_pieces)
        elif algorithm == "greedy":
            return self._greedy_algorithm(expanded_pieces)
        elif algorithm == "best_fit":
            return self._best_fit_algorithm(expanded_pieces)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    def _genetic_algorithm(self, pieces: List[GlassPiece], 
                          population_size: int = 50, 
                          generations: int = 100) -> Dict:
        """Genetic algorithm for optimal glass cutting"""
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = self._create_random_layout(pieces)
            population.append(individual)
        
        best_solution = None
        best_fitness = float('inf')
        
        for generation in range(generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                fitness = self._calculate_fitness(individual)
                fitness_scores.append(fitness)
                
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_solution = copy.deepcopy(individual)
            
            # Selection
            new_population = []
            for _ in range(population_size // 2):
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2)
                
                # Mutation
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                
                new_population.extend([child1, child2])
            
            population = new_population
        
        return self._create_result_dict(best_solution)
    
    def _create_random_layout(self, pieces: List[GlassPiece]) -> List[Sheet]:
        """Create a random layout of pieces on sheets"""
        shuffled_pieces = pieces.copy()
        random.shuffle(shuffled_pieces)
        
        sheets = []
        current_sheet = Sheet(self.sheet_width, self.sheet_height)
        
        for piece in shuffled_pieces:
            if not self._can_place_piece(current_sheet, piece):
                sheets.append(current_sheet)
                current_sheet = Sheet(self.sheet_width, self.sheet_height)
            
            self._place_piece(current_sheet, piece)
        
        if current_sheet.pieces:
            sheets.append(current_sheet)
        
        return sheets
    
    def _can_place_piece(self, sheet: Sheet, piece: GlassPiece) -> bool:
        """Check if a piece can be placed on the sheet"""
        # Simple placement check - can be improved with more sophisticated algorithms
        used_area = sum(p[2] * p[3] for p in sheet.pieces)
        piece_area = piece.width * piece.height
        return (used_area + piece_area) <= self.sheet_area
    
    def _place_piece(self, sheet: Sheet, piece: GlassPiece):
        """Place a piece on the sheet (simplified placement)"""
        # Find available position (simplified)
        x, y = 0, 0
        for placed_piece in sheet.pieces:
            x = max(x, placed_piece[0] + placed_piece[2])
        
        if x + piece.width <= sheet.width:
            sheet.pieces.append((x, y, piece.width, piece.height))
        else:
            # Try to place below
            y = max(p[1] + p[3] for p in sheet.pieces) if sheet.pieces else 0
            if y + piece.height <= sheet.height:
                sheet.pieces.append((0, y, piece.width, piece.height))
    
    def _calculate_fitness(self, sheets: List[Sheet]) -> float:
        """Calculate fitness (lower is better)"""
        total_waste = 0
        for sheet in sheets:
            used_area = sum(p[2] * p[3] for p in sheet.pieces)
            waste = self.sheet_area - used_area
            total_waste += waste
        
        return total_waste + len(sheets) * 1000  # Penalize number of sheets
    
    def _tournament_selection(self, population: List, fitness_scores: List[float]) -> List[Sheet]:
        """Tournament selection for genetic algorithm"""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[tournament_fitness.index(min(tournament_fitness))]
        return copy.deepcopy(population[winner_idx])
    
    def _crossover(self, parent1: List[Sheet], parent2: List[Sheet]) -> Tuple[List[Sheet], List[Sheet]]:
        """Crossover operation for genetic algorithm"""
        # Simple crossover - can be improved
        return copy.deepcopy(parent1), copy.deepcopy(parent2)
    
    def _mutate(self, individual: List[Sheet]) -> List[Sheet]:
        """Mutation operation for genetic algorithm"""
        # Simple mutation - can be improved
        if random.random() < 0.1:  # 10% mutation rate
            # Swap two pieces
            pass
        return individual
    
    def _greedy_algorithm(self, pieces: List[GlassPiece]) -> Dict:
        """Greedy algorithm for glass cutting"""
        # Sort pieces by area (largest first)
        sorted_pieces = sorted(pieces, key=lambda p: p.width * p.height, reverse=True)
        
        sheets = []
        current_sheet = Sheet(self.sheet_width, self.sheet_height)
        
        for piece in sorted_pieces:
            if not self._can_place_piece(current_sheet, piece):
                sheets.append(current_sheet)
                current_sheet = Sheet(self.sheet_width, self.sheet_height)
            
            self._place_piece(current_sheet, piece)
        
        if current_sheet.pieces:
            sheets.append(current_sheet)
        
        return self._create_result_dict(sheets)
    
    def _best_fit_algorithm(self, pieces: List[GlassPiece]) -> Dict:
        """Best fit algorithm for glass cutting"""
        sheets = []
        
        for piece in pieces:
            best_sheet = None
            best_waste = float('inf')
            
            # Try to find the best sheet to place the piece
            for sheet in sheets:
                if self._can_place_piece(sheet, piece):
                    # Calculate waste if piece is placed
                    used_area = sum(p[2] * p[3] for p in sheet.pieces)
                    new_used_area = used_area + piece.width * piece.height
                    waste = self.sheet_area - new_used_area
                    
                    if waste < best_waste:
                        best_waste = waste
                        best_sheet = sheet
            
            if best_sheet:
                self._place_piece(best_sheet, piece)
            else:
                # Create new sheet
                new_sheet = Sheet(self.sheet_width, self.sheet_height)
                self._place_piece(new_sheet, piece)
                sheets.append(new_sheet)
        
        return self._create_result_dict(sheets)
    
    def _create_result_dict(self, sheets: List[Sheet]) -> Dict:
        """Create result dictionary with optimization metrics"""
        total_pieces = sum(len(sheet.pieces) for sheet in sheets)
        total_used_area = sum(sum(p[2] * p[3] for p in sheet.pieces) for sheet in sheets)
        total_waste_area = sum(self.sheet_area - sum(p[2] * p[3] for p in sheet.pieces) for sheet in sheets)
        
        waste_percentage = (total_waste_area / (len(sheets) * self.sheet_area)) * 100
        
        return {
            'sheets_used': len(sheets),
            'total_pieces': total_pieces,
            'total_used_area': total_used_area,
            'total_waste_area': total_waste_area,
            'waste_percentage': waste_percentage,
            'sheets': sheets,
            'efficiency': ((total_used_area / (len(sheets) * self.sheet_area)) * 100)
        }
    
    def visualize_optimization(self, result: Dict, save_path: Optional[str] = None):
        """Visualize the optimization results"""
        sheets = result['sheets']
        
        fig, axes = plt.subplots(1, min(len(sheets), 4), figsize=(20, 5))
        if len(sheets) == 1:
            axes = [axes]
        
        colors = plt.cm.Set3(np.linspace(0, 1, 20))
        
        for i, sheet in enumerate(sheets[:4]):  # Show max 4 sheets
            ax = axes[i]
            ax.set_xlim(0, sheet.width)
            ax.set_ylim(0, sheet.height)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.set_title(f'Sheet {i+1}')
            
            for j, piece in enumerate(sheet.pieces):
                x, y, w, h = piece
                rect = patches.Rectangle((x, y), w, h, 
                                       linewidth=1, edgecolor='black', 
                                       facecolor=colors[j % len(colors)], alpha=0.7)
                ax.add_patch(rect)
                ax.text(x + w/2, y + h/2, f'{w:.1f}x{h:.1f}', 
                       ha='center', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 