#!/usr/bin/env python3
"""
Test script for the Reverse Optimization Tool
Demonstrates efficiency-based design capabilities
"""

from reverse_optimizer import ReverseOptimizer, EfficiencyTarget
import pandas as pd

def test_reverse_optimization():
    """Test the reverse optimization tool"""
    
    print("🎯 Reverse Optimization Tool - Test Run")
    print("=" * 50)
    
    # Initialize reverse optimizer
    reverse_optimizer = ReverseOptimizer()
    
    # Test case 1: High efficiency target
    print("\n🔴 Test Case 1: High Efficiency (90%)")
    print("-" * 40)
    
    target1 = EfficiencyTarget(
        desired_efficiency=90.0,
        total_area_needed=50000.0,  # 50,000 cm²
        sheet_width=600.0,
        sheet_height=400.0
    )
    
    recommendation1 = reverse_optimizer.calculate_required_measurements(target1)
    
    print(f"Target Efficiency: {target1.desired_efficiency}%")
    print(f"Total Area Needed: {target1.total_area_needed:,.0f} cm²")
    print(f"Recommended Sheet Size: {recommendation1.recommended_sheet_size[0]:.0f}cm × {recommendation1.recommended_sheet_size[1]:.0f}cm")
    print(f"Achieved Efficiency: {recommendation1.estimated_efficiency:.1f}%")
    print(f"Sheets Required: {recommendation1.sheets_required}")
    print(f"Optimization Strategy: {recommendation1.optimization_strategy}")
    
    # Test case 2: Standard efficiency target
    print("\n🟡 Test Case 2: Standard Efficiency (80%)")
    print("-" * 40)
    
    target2 = EfficiencyTarget(
        desired_efficiency=80.0,
        total_area_needed=30000.0,  # 30,000 cm²
        sheet_width=600.0,
        sheet_height=400.0
    )
    
    recommendation2 = reverse_optimizer.calculate_required_measurements(target2)
    
    print(f"Target Efficiency: {target2.desired_efficiency}%")
    print(f"Total Area Needed: {target2.total_area_needed:,.0f} cm²")
    print(f"Recommended Sheet Size: {recommendation2.recommended_sheet_size[0]:.0f}cm × {recommendation2.recommended_sheet_size[1]:.0f}cm")
    print(f"Achieved Efficiency: {recommendation2.estimated_efficiency:.1f}%")
    print(f"Sheets Required: {recommendation2.sheets_required}")
    print(f"Optimization Strategy: {recommendation2.optimization_strategy}")
    
    # Test case 3: Basic efficiency target
    print("\n🟢 Test Case 3: Basic Efficiency (70%)")
    print("-" * 40)
    
    target3 = EfficiencyTarget(
        desired_efficiency=70.0,
        total_area_needed=20000.0,  # 20,000 cm²
        sheet_width=600.0,
        sheet_height=400.0
    )
    
    recommendation3 = reverse_optimizer.calculate_required_measurements(target3)
    
    print(f"Target Efficiency: {target3.desired_efficiency}%")
    print(f"Total Area Needed: {target3.total_area_needed:,.0f} cm²")
    print(f"Recommended Sheet Size: {recommendation3.recommended_sheet_size[0]:.0f}cm × {recommendation3.recommended_sheet_size[1]:.0f}cm")
    print(f"Achieved Efficiency: {recommendation3.estimated_efficiency:.1f}%")
    print(f"Sheets Required: {recommendation3.sheets_required}")
    print(f"Optimization Strategy: {recommendation3.optimization_strategy}")
    
    # Display efficiency guidelines
    print("\n📋 Efficiency Guidelines")
    print("=" * 30)
    
    guidelines = reverse_optimizer.get_efficiency_guidelines()
    
    for level, info in guidelines.items():
        print(f"\n{level.replace('_', ' ').title()}:")
        print(f"  Range: {info['range']}")
        print(f"  Description: {info['description']}")
        print(f"  Best for: {info['best_for']}")
        print(f"  Algorithm: {info['algorithm']}")
        print(f"  Setup Time: {info['setup_time']}")
        print(f"  Cost Premium: {info['cost_premium']}")
    
    # Test piece constraints
    print("\n🔧 Test Case 4: With Piece Constraints")
    print("-" * 40)
    
    target4 = EfficiencyTarget(
        desired_efficiency=85.0,
        total_area_needed=40000.0,
        sheet_width=600.0,
        sheet_height=400.0
    )
    
    piece_constraints = {
        'min_width': 80,
        'max_width': 300,
        'min_height': 60,
        'max_height': 200
    }
    
    recommendation4 = reverse_optimizer.calculate_required_measurements(target4, piece_constraints)
    
    print(f"Target Efficiency: {target4.desired_efficiency}%")
    print(f"Piece Constraints: {piece_constraints}")
    print(f"Recommended Piece Sizes:")
    
    for i, (width, height, quantity) in enumerate(recommendation4.recommended_piece_sizes):
        print(f"  Piece {i+1}: {width:.0f}cm × {height:.0f}cm (Qty: {quantity})")
    
    print(f"Achieved Efficiency: {recommendation4.estimated_efficiency:.1f}%")
    
    # Cost analysis
    print("\n💰 Cost Analysis Example")
    print("-" * 30)
    
    cost_data = recommendation1.cost_analysis
    print(f"Sheet Cost: ${cost_data['sheet_cost_usd']}")
    print(f"Piece Value: ${cost_data['piece_value_usd']}")
    print(f"Waste Cost: ${cost_data['waste_cost_usd']}")
    print(f"Sheets Saved: {cost_data['sheets_saved']}")
    print(f"Cost Savings: ${cost_data['cost_savings_usd']}")
    print(f"ROI: {cost_data['roi_percentage']}%")
    
    return recommendation1, recommendation2, recommendation3, recommendation4

def test_improvement_suggestions():
    """Test improvement suggestions"""
    print("\n💡 Improvement Suggestions")
    print("=" * 30)
    
    reverse_optimizer = ReverseOptimizer()
    
    # Test different scenarios
    scenarios = [
        (60, 90, "Low to High Efficiency"),
        (70, 85, "Standard to High Efficiency"),
        (80, 90, "Good to Excellent Efficiency")
    ]
    
    for current, target, description in scenarios:
        suggestions = reverse_optimizer.suggest_improvements(current, target)
        print(f"\n{description} ({current}% → {target}%):")
        for suggestion in suggestions:
            print(f"  • {suggestion}")

if __name__ == "__main__":
    # Run tests
    recommendations = test_reverse_optimization()
    test_improvement_suggestions()
    
    print("\n✅ Reverse optimization tests completed successfully!")
    print("🚀 To use the full application: streamlit run app.py")
    print("🎯 Try the 'Efficiency-Based Design' tab for interactive optimization!") 