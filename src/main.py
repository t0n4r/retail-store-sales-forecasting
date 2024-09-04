from data_preprocessing import load_data, preprocess_data
from feature_engineering import engineer_features

def main():
    train_path = 'data/train.csv'
    stores_path = 'data/stores.csv'
    
    train, stores = load_data(train_path, stores_path)
    df = preprocess_data(train, stores)
    df = engineer_features(df)
    
    print(df.head())
    print(df.info())

    df.to_csv('data/preprocessed_data.csv', index=False)

if __name__ == "__main__":
    main()