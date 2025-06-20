# 🔷 AI Glass Optimization Tool

An intelligent web application that optimizes glass cutting layouts to minimize waste and maximize efficiency for glass manufacturing companies.

## 🎯 Features

### Core Optimization
- **Multiple Algorithms**: Choose from Genetic Algorithm, Greedy Algorithm, or Best Fit Algorithm
- **2D Bin Packing**: Advanced optimization for rectangular glass pieces
- **Waste Minimization**: Reduce material waste by 15-30%
- **Visual Layouts**: See optimized cutting patterns with detailed visualizations

### 🆕 Efficiency-Based Design (NEW!)
- **Reverse Optimization**: Calculate required measurements for desired efficiency
- **Target Efficiency**: Set your desired efficiency percentage (60-95%)
- **Smart Recommendations**: Get optimal sheet sizes and piece dimensions
- **Cost Analysis**: Calculate potential savings and ROI
- **Piece Constraints**: Set minimum/maximum piece dimensions

### User Interface
- **Web-based Application**: Easy-to-use Streamlit interface with tabs
- **Multiple Input Methods**: Manual entry, Excel/CSV upload, or sample data
- **Real-time Analysis**: Instant optimization results and metrics
- **Export Capabilities**: Download reports in Excel or CSV format

### Cost Analysis
- **Efficiency Metrics**: Material utilization percentage
- **Waste Calculation**: Precise waste area and percentage
- **Cost Savings**: Estimate material cost reductions
- **Production Optimization**: Reduce production time and resources

## 🚀 Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd glass_optimizer
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run app.py
```

4. **Open your browser** and navigate to `http://localhost:8501`

### Usage

#### 📐 Standard Optimization Tab
1. **Configure Sheet Dimensions**
   - Set the width and height of your glass sheets
   - Choose your preferred optimization algorithm

2. **Input Glass Pieces**
   - **Manual Entry**: Enter dimensions and quantities directly
   - **File Upload**: Upload Excel/CSV files with your facade data
   - **Sample Data**: Use provided sample data for testing

3. **Run Optimization**
   - Click "Run Optimization" to start the process
   - View real-time results and visualizations

#### 🎯 Efficiency-Based Design Tab (NEW!)
1. **Set Efficiency Target**
   - Choose desired efficiency (60-95%)
   - Enter total area needed
   - Set piece constraints if needed

2. **Get Recommendations**
   - View recommended sheet sizes
   - See optimal piece dimensions
   - Analyze cost implications

3. **Export Results**
   - Download detailed recommendations
   - Get cutting instructions

## 📊 Input Format

### Manual Entry
Enter the following for each glass piece:
- **Width** (cm): Horizontal dimension
- **Height** (cm): Vertical dimension  
- **Quantity**: Number of pieces needed

### Excel/CSV Upload
Your file should contain these columns:
```csv
Width,Height,Quantity
120,80,8
100,60,12
150,100,6
```

## 🧠 Algorithms

### 1. Genetic Algorithm (Recommended)
- **Best for**: Complex layouts with many different piece sizes
- **Advantages**: Finds near-optimal solutions, handles complex constraints
- **Parameters**: Population size, number of generations
- **Performance**: Slower but more accurate

### 2. Greedy Algorithm
- **Best for**: Simple layouts, quick results
- **Advantages**: Fast execution, good for initial estimates
- **Strategy**: Places largest pieces first
- **Performance**: Very fast, moderate accuracy

### 3. Best Fit Algorithm
- **Best for**: Balanced approach between speed and accuracy
- **Advantages**: Good efficiency, reasonable speed
- **Strategy**: Finds best sheet for each piece
- **Performance**: Medium speed and accuracy

## 🎯 Efficiency-Based Design

### Efficiency Levels

#### 🔴 High Efficiency (85-95%)
- **Best for**: High-value projects, limited materials
- **Algorithm**: Genetic Algorithm
- **Setup Time**: 5-15 minutes
- **Cost Premium**: 10-20%
- **Description**: Maximum material utilization, complex layouts

#### 🟡 Standard Efficiency (75-85%)
- **Best for**: Most commercial projects
- **Algorithm**: Best-Fit Algorithm
- **Setup Time**: 2-5 minutes
- **Cost Premium**: 5-10%
- **Description**: Good balance of efficiency and simplicity

#### 🟢 Basic Efficiency (60-75%)
- **Best for**: Quick estimates, simple projects
- **Algorithm**: Greedy Algorithm
- **Setup Time**: 30 seconds - 2 minutes
- **Cost Premium**: 0-5%
- **Description**: Simple layouts, fast processing

### How It Works
1. **Input Target Efficiency**: Set your desired efficiency percentage
2. **Specify Area Needed**: Enter total area of glass pieces required
3. **Set Constraints**: Define piece size limits if needed
4. **Get Recommendations**: Receive optimal sheet sizes and piece dimensions
5. **Analyze Costs**: See potential savings and ROI

## 📈 Results Analysis

### Key Metrics
- **Sheets Used**: Number of glass sheets required
- **Efficiency**: Material utilization percentage
- **Waste Percentage**: Percentage of material wasted
- **Waste Area**: Total wasted area in square centimeters

### Visualizations
- **Layout Diagrams**: See how pieces are arranged on each sheet
- **Efficiency Gauge**: Visual representation of material efficiency
- **Waste Analysis**: Pie chart showing used vs. wasted area
- **Cost Breakdown**: Bar charts showing cost distribution

## 💰 Cost Benefits

### Typical Savings
- **Material Waste Reduction**: 15-30% less waste
- **Cost Savings**: Significant reduction in material costs
- **Production Efficiency**: Faster cutting and reduced setup time
- **Resource Optimization**: Better utilization of available materials

### ROI Calculation
```
Annual Savings = (Waste Reduction % × Annual Material Cost) + (Time Savings × Labor Cost)
```

## 🔧 Technical Details

### Architecture
- **Backend**: Python with advanced optimization algorithms
- **Frontend**: Streamlit web application with tabbed interface
- **Visualization**: Matplotlib and Plotly for charts and layouts
- **Data Processing**: Pandas for data manipulation

### Optimization Techniques
- **2D Bin Packing**: Advanced algorithms for rectangular piece placement
- **Genetic Algorithm**: Evolutionary optimization with selection, crossover, and mutation
- **Heuristic Methods**: Greedy and best-fit approaches for different scenarios
- **Reverse Optimization**: Calculate measurements for target efficiency

### Performance
- **Small Projects** (< 50 pieces): Instant results
- **Medium Projects** (50-200 pieces): 5-30 seconds
- **Large Projects** (> 200 pieces): 1-5 minutes

## 📁 Project Structure

```
glass_optimizer/
├── app.py                 # Main Streamlit application
├── glass_optimizer.py     # Core optimization algorithms
├── reverse_optimizer.py   # Efficiency-based design module
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── test_optimizer.py     # Standard optimization tests
├── test_reverse_optimizer.py # Efficiency-based design tests
├── run_app.bat           # Windows batch launcher
├── launch_app.py         # Python launcher script
├── run_app.ps1           # PowerShell launcher
└── examples/             # Sample data and examples
    └── sample_facade.csv # Sample facade data
```

## 🛠️ Customization

### Adding New Algorithms
1. Implement your algorithm in `glass_optimizer.py`
2. Add the algorithm option in `app.py`
3. Update the algorithm selection dropdown

### Modifying Sheet Constraints
- Edit sheet dimensions in the sidebar
- Add custom constraints in the optimization logic
- Implement rotation capabilities for pieces

### Custom Visualizations
- Modify the visualization functions in `app.py`
- Add new chart types using Plotly or Matplotlib
- Customize color schemes and layouts

## 🧪 Testing

### Test Standard Optimization
```bash
python test_optimizer.py
```

### Test Efficiency-Based Design
```bash
python test_reverse_optimizer.py
```

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For support or questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for common solutions

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🎉 Acknowledgments

- Built with Streamlit for the web interface
- Uses advanced optimization algorithms for glass cutting
- Inspired by real-world manufacturing challenges
- Designed for practical industrial applications

---

**Ready to optimize your glass cutting process? Start the application and see the savings!** 🚀 