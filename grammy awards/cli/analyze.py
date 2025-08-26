import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

def generate_report(output_file):
    """Generate HTML analysis report"""
    try:
        df = pd.read_csv('data/Grammy Award Nominees and Winners 1958-2024.csv')
        
        # Generate plots
        plt.figure(figsize=(10,5))
        df['Year'].value_counts().sort_index().plot()
        plt.title("Nominations Over Time")
        plt.savefig('images/nominations_trend.png')
        
        # Create HTML
        html = f"""
        <html>
        <head><title>Grammy Analysis Report</title></head>
        <body>
            <h1>Grammy Awards Analysis</h1>
            <p>Generated: {datetime.now()}</p>
            
            <h2>Nominations Trend</h2>
            <img src="images/nominations_trend.png" width="800">
            
            <h2>Statistics</h2>
            <pre>{df.describe().to_html()}</pre>
        </body>
        </html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html)
            
        print(f"✅ Report generated: {output_file}")
    
    except Exception as e:
        print(f"❌ Error generating report: {str(e)}")