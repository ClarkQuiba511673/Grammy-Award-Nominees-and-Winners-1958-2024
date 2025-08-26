import argparse
from cli.predict import predict_winner
from cli.clusters import show_clusters
from cli.analyze import generate_report

def main():
    parser = argparse.ArgumentParser(description="Grammy Awards CLI Tool")
    subparsers = parser.add_subparsers(dest='command')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict')
    predict_parser.add_argument('--year', type=int, required=True)
    predict_parser.add_argument('--category', type=str, required=True)
    
    # Clusters command
    cluster_parser = subparsers.add_parser('clusters')
    cluster_parser.add_argument('--show-all', action='store_true')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze')
    analyze_parser.add_argument('--output', type=str, default="report.html")
    
    args = parser.parse_args()
    
    if args.command == 'predict':
        predict_winner(args.year, args.category)
    elif args.command == 'clusters':
        show_clusters(args.show_all)
    elif args.command == 'analyze':
        generate_report(args.output)

if __name__ == "__main__":
    main()