import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Function to load data (instead of @st.cache_data decorator)
def get_data():
    data = pd.read_csv(r"D:\python\jupyter\bengaluru_house_prices.csv")
    return data

# Load the data
df = get_data()
o = df.copy()

# Data cleaning and preparation
df.drop(['area_type', 'society', 'balcony', 'availability'], axis=1, inplace=True)
df.dropna(inplace=True)
df['bhk'] = df['size'].apply(lambda x: x.split(' ')[0])
df['bhk'] = df['bhk'].astype('int64')

def convert_sqft_to_num(x):
    token = x.split('-')
    if len(token) == 2:
        return (float(token[0]) + float(token[1])) / 2
    try:
        return float(x)
    except:
        return None

df1 = df.copy()
df1['total_sqft'] = df1['total_sqft'].apply(convert_sqft_to_num)
df2 = df1.copy()
df2['price_per_sqft'] = df2['price'] * 100000 / df2['total_sqft']
df2.location = df2.location.apply(lambda x: x.strip())
l = df2.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats_less_than_10 = l[l <= 10]
df2.location = df2.location.apply(lambda x: 'other' if x in location_stats_less_than_10 else x)
df3 = df2[~(df2.total_sqft / df2.bhk < 300)]

def remove_pps_outliers(df):
    df_out = pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft > (m - st)) & (subdf.price_per_sqft <= (m + st))]
        df_out = pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df3 = remove_pps_outliers(df3)

def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk - 1)
            if stats and stats['count'] > 5:
                exclude_indices = np.append(exclude_indices,
                                            bhk_df[bhk_df.price_per_sqft < (stats['mean'])].index.values)
    return df.drop(exclude_indices, axis='index')

df4 = remove_bhk_outliers(df3)
df4 = df4[df4.bath < df4.bhk + 2]
df5 = df4.drop(['size', 'price_per_sqft'], axis=1)
dummies = pd.get_dummies(df5.location)
df5 = pd.concat([df5, dummies.drop('other', axis=1)], axis=1)
df5.drop('location', axis=1, inplace=True)
X = df5.drop('price', axis=1)
y = df5['price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
model = LinearRegression()
model.fit(X_train, y_train)

def predict_price(location, sqft, bath, bhk):
    loc_index = np.where(X.columns == location)[0][0]
    
    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1
    
    input_df = pd.DataFrame([x], columns=X.columns)
    
    return model.predict(input_df)[0]
# Terminal interface
def main():
    print("--Welcome to the Home Price Predictor!--\n")
    print("Instructions: ")
    print("1. Use valid inputs")
    print("2. Give only meaning full inputs else you get wrong results")

    # Display the available locations
    available_locations = X.columns[4:]  # Location columns start at index 4
    print("\nAvailable locations:")
    for loc in available_locations:
        print(loc)

    # Get inputs from the user
    location = input("\nEnter the location from the list above: ")
    while location not in available_locations:
        print("Invalid location. Please select from the available options.")
        location = input("Enter the location: ")

    sqft = float(input("Enter the total square feet of the house: "))
    bath = int(input("Enter the number of bathrooms: "))
    bhk = int(input("Enter the number of bedrooms (BHK): "))

    # Predict the price
    price = predict_price(location, sqft, bath, bhk)
    print(f"\nThe predicted price of the house is: {price} Lakhs")

if __name__ == "__main__":
    main()
