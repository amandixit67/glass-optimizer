#!/usr/bin/env python3
"""
Test script for the Glass Optimization Tool
Demonstrates the tool's capabilities with sample facade data
"""

from glass_optimizer import GlassOptimizer, GlassPiece
import pandas as pd
import time

def test_optimization():
    """Test the optimization tool with sample data"""
    
    print("🔷 AI Glass Optimization Tool - Test Run")
    print("=" * 50)
    
    # Sample facade data
    sample_pieces = [
        GlassPiece(120, 80, 8, "Window_1"),
        GlassPiece(100, 60, 12, "Window_2"),
        GlassPiece(150, 100, 6, "Window_3"),
        GlassPiece(80, 80, 15, "Window_4"),
        GlassPiece(200, 120, 4, "Window_5"),
        GlassPiece(90, 70, 10, "Window_6"),
        GlassPiece(110, 90, 7, "Window_7"),
        GlassPiece(130, 85, 9, "Window_8"),
    ]
    
    # Sheet dimensions (600cm x 400cm)
    sheet_width = 600
    sheet_height = 400
    
    print(f"📋 Input Summary:")
    print(f"   Sheet dimensions: {sheet_width}cm x {sheet_height}cm")
    print(f"   Total pieces: {sum(p.quantity for p in sample_pieces)}")
    print(f"   Total area: {sum(p.width * p.height * p.quantity for p in sample_pieces):,.0f} cm²")
    print()
    
    # Initialize optimizer
    optimizer = GlassOptimizer(sheet_width, sheet_height)
    
    # Test different algorithms
    algorithms = ["greedy", "best_fit", "genetic"]
    
    for algorithm in algorithms:
        print(f"🧠 Testing {algorithm.upper()} Algorithm:")
        print("-" * 30)
        
        start_time = time.time()
        result = optimizer.optimize_cutting(sample_pieces, algorithm)
        end_time = time.time()
        
        print(f"   ⏱️  Execution time: {end_time - start_time:.2f} seconds")
        print(f"   📊 Sheets used: {result['sheets_used']}")
        print(f"   📈 Efficiency: {result['efficiency']:.1f}%")
        print(f"   🗑️  Waste: {result['waste_percentage']:.1f}%")
        print(f"   📏 Waste area: {result['total_waste_area']:,.0f} cm²")
        print()
    
    # Show best result
    print("🏆 Best Result Summary:")
    print("=" * 30)
    
    # Run genetic algorithm for best result
    result = optimizer.optimize_cutting(sample_pieces, "genetic")
    
    print(f"   Sheets Required: {result['sheets_used']}")
    print(f"   Material Efficiency: {result['efficiency']:.1f}%")
    print(f"   Waste Percentage: {result['waste_percentage']:.1f}%")
    print(f"   Total Waste Area: {result['total_waste_area']:,.0f} cm²")
    print(f"   Total Used Area: {result['total_used_area']:,.0f} cm²")
    
    # Calculate cost savings
    sheet_area = sheet_width * sheet_height
    total_sheet_area = result['sheets_used'] * sheet_area
    savings_percentage = (result['total_waste_area'] / total_sheet_area) * 100
    
    print(f"\n💰 Cost Analysis:")
    print(f"   Material Savings: {savings_percentage:.1f}%")
    print(f"   Sheets Saved: {max(0, int(sum(p.width * p.height * p.quantity for p in sample_pieces) / sheet_area) - result['sheets_used'])}")
    
    return result

def test_with_csv_data():
    """Test with CSV data"""
    print("\n📁 Testing with CSV Data:")
    print("=" * 30)
    
    try:
        # Read sample CSV
        df = pd.read_csv('examples/sample_facade.csv')
        
        # Convert to glass pieces
        pieces = []
        for _, row in df.iterrows():
            pieces.append(GlassPiece(
                row['Width'], 
                row['Height'], 
                int(row['Quantity']),
                f"{row['Width']}x{row['Height']}"
            ))
        
        # Run optimization
        optimizer = GlassOptimizer(600, 400)
        result = optimizer.optimize_cutting(pieces, "genetic")
        
        print(f"   CSV pieces loaded: {len(pieces)} types")
        print(f"   Total pieces: {sum(p.quantity for p in pieces)}")
        print(f"   Optimization result: {result['sheets_used']} sheets, {result['efficiency']:.1f}% efficiency")
        
    except Exception as e:
        print(f"   Error reading CSV: {e}")

if __name__ == "__main__":
    # Run tests
    result = test_optimization()
    test_with_csv_data()
    
    print("\n✅ Test completed successfully!")
    print("🚀 To run the full application: streamlit run app.py") 