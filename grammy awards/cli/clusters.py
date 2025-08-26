import pickle
import pandas as pd

def show_clusters(show_all=False):
    """Display award category clusters"""
    try:
        with open('models/award_clusterer.pkl', 'rb') as f:
            kmeans = pickle.load(f)
        
        df = pd.read_csv('data/Grammy Award Nominees and Winners 1958-2024.csv')
        awards = df['Award Name'].unique()
        
        print("\n🎧 Grammy Award Clusters")
        print("-----------------------")
        
        for i in range(5):
            cluster_awards = awards[kmeans.labels_ == i]
            print(f"\nCluster {i+1} ({len(cluster_awards)} awards):")
            print(", ".join(cluster_awards[:3]) + ("..." if not show_all else ""))
            
            if show_all:
                for award in cluster_awards:
                    print(f"  - {award}")
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")