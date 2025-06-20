import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64
from glass_optimizer import GlassOptimizer, GlassPiece
from reverse_optimizer import ReverseOptimizer, EfficiencyTarget, OptimizationRecommendation
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Page configuration
st.set_page_config(
    page_title="AI Glass Optimization Tool",
    page_icon="🔷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .optimization-result {
        background-color: #e8f4fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
    .tab-content {
        padding: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">🔷 AI Glass Optimization Tool</h1>', unsafe_allow_html=True)
    st.markdown("### Optimize glass cutting to minimize waste and maximize efficiency")
    
    # Create tabs for different functionalities
    tab1, tab2, tab3 = st.tabs(["📐 Standard Optimization", "🎯 Efficiency-Based Design", "📊 Analysis & Reports"])
    
    with tab1:
        standard_optimization()
    
    with tab2:
        efficiency_based_design()
    
    with tab3:
        analysis_and_reports()

def standard_optimization():
    """Standard glass optimization functionality"""
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Sheet dimensions
        st.subheader("Glass Sheet Dimensions")
        sheet_width = st.number_input("Sheet Width (cm)", min_value=100.0, max_value=1000.0, value=600.0, step=10.0)
        sheet_height = st.number_input("Sheet Height (cm)", min_value=100.0, max_value=1000.0, value=400.0, step=10.0)
        
        # Algorithm selection
        st.subheader("Optimization Algorithm")
        algorithm = st.selectbox(
            "Choose Algorithm",
            ["genetic", "greedy", "best_fit"],
            format_func=lambda x: {
                "genetic": "Genetic Algorithm (Best)",
                "greedy": "Greedy Algorithm (Fast)",
                "best_fit": "Best Fit Algorithm (Balanced)"
            }[x]
        )
        
        # Genetic algorithm parameters
        if algorithm == "genetic":
            st.subheader("Genetic Algorithm Parameters")
            population_size = st.slider("Population Size", 20, 100, 50)
            generations = st.slider("Generations", 50, 200, 100)
        
        st.markdown("---")
        st.markdown("### 📊 Features")
        st.markdown("• **Multiple Algorithms**: Choose from genetic, greedy, or best-fit algorithms")
        st.markdown("• **Visual Results**: See optimized layouts with waste analysis")
        st.markdown("• **Cost Analysis**: Calculate material savings and efficiency")
        st.markdown("• **Export Results**: Download optimization reports")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Input Glass Pieces")
        
        # Input method selection
        input_method = st.radio(
            "Choose input method:",
            ["Manual Entry", "Upload Excel/CSV", "Sample Data"]
        )
        
        glass_pieces = []
        
        if input_method == "Manual Entry":
            glass_pieces = manual_input()
        elif input_method == "Upload Excel/CSV":
            glass_pieces = file_upload_input()
        else:  # Sample Data
            glass_pieces = sample_data()
        
        if glass_pieces:
            # Display input summary
            st.subheader("📊 Input Summary")
            display_input_summary(glass_pieces, sheet_width, sheet_height)
            
            # Optimization button
            if st.button("🚀 Run Optimization", type="primary", use_container_width=True):
                with st.spinner("Optimizing glass cutting layout..."):
                    # Initialize optimizer
                    optimizer = GlassOptimizer(sheet_width, sheet_height)
                    
                    # Run optimization
                    if algorithm == "genetic":
                        result = optimizer.optimize_cutting(glass_pieces, algorithm)
                    else:
                        result = optimizer.optimize_cutting(glass_pieces, algorithm)
                    
                    # Display results
                    display_optimization_results(result, optimizer, glass_pieces)
    
    with col2:
        st.header("💡 Tips")
        st.markdown("""
        **For Best Results:**
        • Use Genetic Algorithm for complex layouts
        • Group similar sizes together
        • Consider rotation of pieces
        • Larger pieces first often works better
        
        **Cost Savings:**
        • Reduce waste by 15-30%
        • Save on material costs
        • Optimize production time
        • Improve resource utilization
        """)

def efficiency_based_design():
    """Efficiency-based design functionality"""
    st.header("🎯 Efficiency-Based Design")
    st.markdown("### Calculate required measurements for your desired efficiency target")
    
    # Initialize reverse optimizer
    reverse_optimizer = ReverseOptimizer()
    
    # Get efficiency guidelines
    guidelines = reverse_optimizer.get_efficiency_guidelines()
    
    # Display efficiency guidelines
    with st.expander("📋 Efficiency Guidelines", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🔴 High Efficiency (85-95%)**")
            st.markdown(f"• **Best for**: {guidelines['high_efficiency']['best_for']}")
            st.markdown(f"• **Algorithm**: {guidelines['high_efficiency']['algorithm']}")
            st.markdown(f"• **Setup Time**: {guidelines['high_efficiency']['setup_time']}")
            st.markdown(f"• **Cost Premium**: {guidelines['high_efficiency']['cost_premium']}")
        
        with col2:
            st.markdown("**🟡 Standard Efficiency (75-85%)**")
            st.markdown(f"• **Best for**: {guidelines['standard_efficiency']['best_for']}")
            st.markdown(f"• **Algorithm**: {guidelines['standard_efficiency']['algorithm']}")
            st.markdown(f"• **Setup Time**: {guidelines['standard_efficiency']['setup_time']}")
            st.markdown(f"• **Cost Premium**: {guidelines['standard_efficiency']['cost_premium']}")
        
        with col3:
            st.markdown("**🟢 Basic Efficiency (60-75%)**")
            st.markdown(f"• **Best for**: {guidelines['basic_efficiency']['best_for']}")
            st.markdown(f"• **Algorithm**: {guidelines['basic_efficiency']['algorithm']}")
            st.markdown(f"• **Setup Time**: {guidelines['basic_efficiency']['setup_time']}")
            st.markdown(f"• **Cost Premium**: {guidelines['basic_efficiency']['cost_premium']}")
    
    # Input form for efficiency-based design
    st.subheader("🎯 Set Your Efficiency Target")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Efficiency target inputs
        desired_efficiency = st.slider(
            "Desired Efficiency (%)", 
            min_value=60.0, 
            max_value=95.0, 
            value=80.0, 
            step=5.0,
            help="Higher efficiency means less waste but may require more complex layouts"
        )
        
        total_area_needed = st.number_input(
            "Total Area Needed (cm²)", 
            min_value=1000.0, 
            max_value=1000000.0, 
            value=50000.0, 
            step=1000.0,
            help="Total area of all glass pieces you need"
        )
        
        max_sheets = st.number_input(
            "Maximum Sheets Allowed (Optional)", 
            min_value=1, 
            max_value=100, 
            value=None,
            help="Leave empty for unlimited sheets"
        )
    
    with col2:
        # Sheet size preferences
        st.subheader("📏 Sheet Size Preferences")
        
        use_custom_sheet = st.checkbox("Use Custom Sheet Size")
        
        if use_custom_sheet:
            sheet_width = st.number_input("Custom Sheet Width (cm)", min_value=100.0, max_value=2000.0, value=600.0, step=10.0)
            sheet_height = st.number_input("Custom Sheet Height (cm)", min_value=100.0, max_value=2000.0, value=400.0, step=10.0)
        else:
            sheet_width = 600.0
            sheet_height = 400.0
            st.info("Using standard sheet size: 600cm × 400cm")
        
        # Piece constraints
        st.subheader("🔧 Piece Constraints")
        min_width = st.number_input("Minimum Piece Width (cm)", min_value=10.0, max_value=500.0, value=50.0, step=10.0)
        min_height = st.number_input("Minimum Piece Height (cm)", min_value=10.0, max_value=500.0, value=50.0, step=10.0)
        max_width = st.number_input("Maximum Piece Width (cm)", min_value=100.0, max_value=1000.0, value=sheet_width, step=10.0)
        max_height = st.number_input("Maximum Piece Height (cm)", min_value=100.0, max_value=1000.0, value=sheet_height, step=10.0)
    
    # Calculate button
    if st.button("🧮 Calculate Required Measurements", type="primary", use_container_width=True):
        with st.spinner("Calculating optimal measurements..."):
            # Create efficiency target
            target = EfficiencyTarget(
                desired_efficiency=desired_efficiency,
                total_area_needed=total_area_needed,
                max_sheets=max_sheets,
                sheet_width=sheet_width,
                sheet_height=sheet_height
            )
            
            # Create piece constraints
            piece_constraints = {
                'min_width': min_width,
                'max_width': max_width,
                'min_height': min_height,
                'max_height': max_height
            }
            
            # Calculate recommendations
            recommendation = reverse_optimizer.calculate_required_measurements(target, piece_constraints)
            
            # Display results
            display_efficiency_recommendations(recommendation, target)

def display_efficiency_recommendations(recommendation: OptimizationRecommendation, target: EfficiencyTarget):
    """Display efficiency-based recommendations"""
    st.header("🎯 Optimization Recommendations")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Target Efficiency", f"{target.desired_efficiency:.1f}%")
    with col2:
        st.metric("Achieved Efficiency", f"{recommendation.estimated_efficiency:.1f}%")
    with col3:
        st.metric("Sheets Required", recommendation.sheets_required)
    with col4:
        st.metric("Waste Percentage", f"{recommendation.waste_percentage:.1f}%")
    
    # Detailed recommendations
    with st.expander("📐 Recommended Sheet Size", expanded=True):
        st.markdown(f"**Optimal Sheet Dimensions**: {recommendation.recommended_sheet_size[0]:.0f}cm × {recommendation.recommended_sheet_size[1]:.0f}cm**")
        st.markdown(f"**Sheet Area**: {recommendation.recommended_sheet_size[0] * recommendation.recommended_sheet_size[1]:,.0f} cm²")
        
        # Visual representation
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_xlim(0, recommendation.recommended_sheet_size[0])
        ax.set_ylim(0, recommendation.recommended_sheet_size[1])
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Recommended Sheet Size: {recommendation.recommended_sheet_size[0]:.0f}cm × {recommendation.recommended_sheet_size[1]:.0f}cm')
        
        # Draw sheet outline
        rect = patches.Rectangle((0, 0), recommendation.recommended_sheet_size[0], recommendation.recommended_sheet_size[1],
                               linewidth=2, edgecolor='blue', facecolor='lightblue', alpha=0.3)
        ax.add_patch(rect)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Recommended piece sizes
    with st.expander("📋 Recommended Piece Sizes", expanded=True):
        st.markdown("**Optimal piece sizes to achieve your efficiency target:**")
        
        piece_data = []
        for i, (width, height, quantity) in enumerate(recommendation.recommended_piece_sizes):
            piece_data.append({
                'Piece': f"Piece {i+1}",
                'Width (cm)': width,
                'Height (cm)': height,
                'Quantity': quantity,
                'Area (cm²)': width * height,
                'Total Area (cm²)': width * height * quantity
            })
        
        df_pieces = pd.DataFrame(piece_data)
        st.dataframe(df_pieces, use_container_width=True)
        
        # Pie chart of piece distribution
        fig = px.pie(
            df_pieces, 
            values='Total Area (cm²)', 
            names='Piece',
            title="Piece Area Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Cost analysis
    with st.expander("💰 Cost Analysis", expanded=True):
        cost_data = recommendation.cost_analysis
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sheet Cost", f"${cost_data['sheet_cost_usd']}")
            st.metric("Piece Value", f"${cost_data['piece_value_usd']}")
        
        with col2:
            st.metric("Waste Cost", f"${cost_data['waste_cost_usd']}")
            st.metric("Sheets Saved", cost_data['sheets_saved'])
        
        with col3:
            st.metric("Cost Savings", f"${cost_data['cost_savings_usd']}")
            st.metric("ROI", f"{cost_data['roi_percentage']}%")
        
        # Cost breakdown chart
        fig = go.Figure(data=[
            go.Bar(name='Costs', x=['Sheet Cost', 'Waste Cost'], y=[cost_data['sheet_cost_usd'], cost_data['waste_cost_usd']], marker_color=['blue', 'red']),
            go.Bar(name='Value', x=['Piece Value'], y=[cost_data['piece_value_usd']], marker_color='green')
        ])
        fig.update_layout(title="Cost Breakdown", barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    # Optimization strategy
    with st.expander("🧠 Optimization Strategy", expanded=True):
        st.markdown(f"**Recommended Strategy**: {recommendation.optimization_strategy}")
        
        # Efficiency comparison
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=recommendation.estimated_efficiency,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Achieved Efficiency"},
            delta={'reference': target.desired_efficiency},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 60], 'color': "lightgray"},
                    {'range': [60, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': target.desired_efficiency
                }
            }
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Export recommendations
    st.subheader("📥 Export Recommendations")
    export_efficiency_recommendations(recommendation, target)

def export_efficiency_recommendations(recommendation: OptimizationRecommendation, target: EfficiencyTarget):
    """Export efficiency recommendations"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Export to Excel
        if st.button("📊 Export to Excel"):
            # Create detailed report
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                # Sheet recommendations
                sheet_data = {
                    'Metric': ['Recommended Width', 'Recommended Height', 'Sheet Area', 'Target Efficiency', 'Achieved Efficiency'],
                    'Value': [
                        f"{recommendation.recommended_sheet_size[0]:.0f} cm",
                        f"{recommendation.recommended_sheet_size[1]:.0f} cm",
                        f"{recommendation.recommended_sheet_size[0] * recommendation.recommended_sheet_size[1]:,.0f} cm²",
                        f"{target.desired_efficiency:.1f}%",
                        f"{recommendation.estimated_efficiency:.1f}%"
                    ]
                }
                pd.DataFrame(sheet_data).to_excel(writer, sheet_name='Sheet_Recommendations', index=False)
                
                # Piece recommendations
                piece_data = []
                for i, (width, height, quantity) in enumerate(recommendation.recommended_piece_sizes):
                    piece_data.append({
                        'Piece': f"Piece {i+1}",
                        'Width (cm)': width,
                        'Height (cm)': height,
                        'Quantity': quantity,
                        'Area (cm²)': width * height,
                        'Total Area (cm²)': width * height * quantity
                    })
                pd.DataFrame(piece_data).to_excel(writer, sheet_name='Piece_Recommendations', index=False)
                
                # Cost analysis
                cost_data = recommendation.cost_analysis
                cost_df = pd.DataFrame([
                    {'Metric': k.replace('_', ' ').title(), 'Value': v}
                    for k, v in cost_data.items()
                ])
                cost_df.to_excel(writer, sheet_name='Cost_Analysis', index=False)
            
            output.seek(0)
            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name="efficiency_recommendations.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        # Export to CSV
        if st.button("📄 Export to CSV"):
            # Create CSV report
            piece_data = []
            for i, (width, height, quantity) in enumerate(recommendation.recommended_piece_sizes):
                piece_data.append({
                    'Piece': f"Piece {i+1}",
                    'Width': width,
                    'Height': height,
                    'Quantity': quantity,
                    'Area': width * height,
                    'Total_Area': width * height * quantity
                })
            
            df_csv = pd.DataFrame(piece_data)
            csv = df_csv.to_csv(index=False)
            
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name="efficiency_recommendations.csv",
                mime="text/csv"
            )

def analysis_and_reports():
    """Analysis and reports functionality"""
    st.header("📊 Analysis & Reports")
    st.markdown("### Advanced analytics and reporting tools")
    
    # Placeholder for future analytics features
    st.info("📈 Advanced analytics and reporting features coming soon!")
    st.markdown("""
    **Planned Features:**
    - Historical efficiency trends
    - Cost savings analysis over time
    - Performance benchmarking
    - Custom report generation
    - Data visualization dashboards
    """)

def manual_input():
    """Manual input for glass pieces"""
    st.subheader("Enter Glass Pieces")
    
    pieces = []
    num_pieces = st.number_input("Number of different piece types", min_value=1, max_value=20, value=3)
    
    for i in range(num_pieces):
        st.markdown(f"**Piece {i+1}**")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            width = st.number_input(f"Width (cm)", min_value=10.0, max_value=600.0, value=100.0, key=f"w{i}")
        with col2:
            height = st.number_input(f"Height (cm)", min_value=10.0, max_value=400.0, value=80.0, key=f"h{i}")
        with col3:
            quantity = st.number_input(f"Quantity", min_value=1, max_value=100, value=5, key=f"q{i}")
        
        pieces.append(GlassPiece(width, height, quantity, f"Piece_{i+1}"))
    
    return pieces

def file_upload_input():
    """File upload input for glass pieces"""
    st.subheader("Upload File")
    
    uploaded_file = st.file_uploader(
        "Choose Excel or CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="File should have columns: Width, Height, Quantity"
    )
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Display preview
            st.write("**File Preview:**")
            st.dataframe(df.head())
            
            # Validate columns
            required_cols = ['Width', 'Height', 'Quantity']
            if all(col in df.columns for col in required_cols):
                pieces = []
                for _, row in df.iterrows():
                    pieces.append(GlassPiece(
                        row['Width'], 
                        row['Height'], 
                        int(row['Quantity']),
                        f"{row['Width']}x{row['Height']}"
                    ))
                return pieces
            else:
                st.error("File must contain columns: Width, Height, Quantity")
                return []
                
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return []
    
    return []

def sample_data():
    """Sample data for demonstration"""
    st.subheader("Sample Facade Design")
    st.info("Using sample facade data for demonstration")
    
    sample_pieces = [
        GlassPiece(120, 80, 8, "Window_1"),
        GlassPiece(100, 60, 12, "Window_2"),
        GlassPiece(150, 100, 6, "Window_3"),
        GlassPiece(80, 80, 15, "Window_4"),
        GlassPiece(200, 120, 4, "Window_5"),
    ]
    
    return sample_pieces

def display_input_summary(pieces, sheet_width, sheet_height):
    """Display summary of input data"""
    total_pieces = sum(p.quantity for p in pieces)
    total_area = sum(p.width * p.height * p.quantity for p in pieces)
    sheet_area = sheet_width * sheet_height
    estimated_sheets = max(1, int(total_area / sheet_area) + 1)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Pieces", total_pieces)
    with col2:
        st.metric("Total Area (cm²)", f"{total_area:,.0f}")
    with col3:
        st.metric("Sheet Area (cm²)", f"{sheet_area:,.0f}")
    with col4:
        st.metric("Est. Sheets", estimated_sheets)
    
    # Create input data table
    df = pd.DataFrame([
        {
            'Piece ID': p.id,
            'Width (cm)': p.width,
            'Height (cm)': p.height,
            'Quantity': p.quantity,
            'Area (cm²)': p.width * p.height,
            'Total Area (cm²)': p.width * p.height * p.quantity
        }
        for p in pieces
    ])
    
    st.dataframe(df, use_container_width=True)

def display_optimization_results(result, optimizer, original_pieces):
    """Display optimization results"""
    st.header("🎯 Optimization Results")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Sheets Used", result['sheets_used'])
    with col2:
        st.metric("Efficiency", f"{result['efficiency']:.1f}%")
    with col3:
        st.metric("Waste", f"{result['waste_percentage']:.1f}%")
    with col4:
        st.metric("Waste Area (cm²)", f"{result['total_waste_area']:,.0f}")
    
    # Detailed results
    with st.expander("📊 Detailed Analysis", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            # Efficiency chart
            fig = go.Figure()
            fig.add_trace(go.Indicator(
                mode="gauge+number+delta",
                value=result['efficiency'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Material Efficiency"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Waste analysis
            fig = px.pie(
                values=[result['total_used_area'], result['total_waste_area']],
                names=['Used Area', 'Waste Area'],
                title="Area Utilization",
                color_discrete_sequence=['green', 'red']
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    # Visual layout
    st.subheader("📐 Optimized Layout")
    visualize_layout(result, optimizer)
    
    # Export results
    st.subheader("📥 Export Results")
    export_results(result, original_pieces)

def visualize_layout(result, optimizer):
    """Visualize the optimized layout"""
    sheets = result['sheets']
    
    if len(sheets) <= 4:
        cols = st.columns(len(sheets))
        for i, sheet in enumerate(sheets):
            with cols[i]:
                st.markdown(f"**Sheet {i+1}**")
                
                # Create visualization
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_xlim(0, sheet.width)
                ax.set_ylim(0, sheet.height)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(sheet.pieces)))
                
                for j, piece in enumerate(sheet.pieces):
                    x, y, w, h = piece
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1, edgecolor='black',
                        facecolor=colors[j], alpha=0.7
                    )
                    ax.add_patch(rect)
                    ax.text(x + w/2, y + h/2, f'{w:.0f}×{h:.0f}',
                           ha='center', va='center', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
    else:
        st.info(f"Showing first 4 sheets out of {len(sheets)} total sheets")
        cols = st.columns(4)
        for i in range(4):
            with cols[i]:
                sheet = sheets[i]
                st.markdown(f"**Sheet {i+1}**")
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.set_xlim(0, sheet.width)
                ax.set_ylim(0, sheet.height)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(sheet.pieces)))
                
                for j, piece in enumerate(sheet.pieces):
                    x, y, w, h = piece
                    rect = patches.Rectangle(
                        (x, y), w, h,
                        linewidth=1, edgecolor='black',
                        facecolor=colors[j], alpha=0.7
                    )
                    ax.add_patch(rect)
                    ax.text(x + w/2, y + h/2, f'{w:.0f}×{h:.0f}',
                           ha='center', va='center', fontsize=8)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

def export_results(result, original_pieces):
    """Export optimization results"""
    col1, col2 = st.columns(2)
    
    with col1:
        # Export to Excel
        if st.button("📊 Export to Excel"):
            # Create detailed report
            report_data = []
            for i, sheet in enumerate(result['sheets']):
                for j, piece in enumerate(sheet.pieces):
                    x, y, w, h = piece
                    report_data.append({
                        'Sheet': i + 1,
                        'Piece': j + 1,
                        'X Position': x,
                        'Y Position': y,
                        'Width': w,
                        'Height': h,
                        'Area': w * h
                    })
            
            df_report = pd.DataFrame(report_data)
            
            # Create Excel file with multiple sheets
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df_report.to_excel(writer, sheet_name='Cutting_Layout', index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Sheets Used', 'Total Pieces', 'Efficiency (%)', 'Waste (%)', 'Total Waste Area (cm²)'],
                    'Value': [
                        result['sheets_used'],
                        result['total_pieces'],
                        f"{result['efficiency']:.1f}",
                        f"{result['waste_percentage']:.1f}",
                        f"{result['total_waste_area']:,.0f}"
                    ]
                }
                pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
            
            output.seek(0)
            st.download_button(
                label="Download Excel Report",
                data=output.getvalue(),
                file_name="glass_optimization_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        # Export to CSV
        if st.button("📄 Export to CSV"):
            # Create CSV report
            report_data = []
            for i, sheet in enumerate(result['sheets']):
                for j, piece in enumerate(sheet.pieces):
                    x, y, w, h = piece
                    report_data.append({
                        'Sheet': i + 1,
                        'Piece': j + 1,
                        'X': x,
                        'Y': y,
                        'Width': w,
                        'Height': h,
                        'Area': w * h
                    })
            
            df_csv = pd.DataFrame(report_data)
            csv = df_csv.to_csv(index=False)
            
            st.download_button(
                label="Download CSV Report",
                data=csv,
                file_name="glass_optimization_report.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 