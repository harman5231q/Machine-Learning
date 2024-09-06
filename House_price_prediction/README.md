

# Home Price Prediction Model

This project is a **Home Price Prediction** tool built using Python. It utilizes a dataset of Bengaluru house prices to predict the price of a house based on its location, square footage, number of bathrooms, and number of bedrooms (BHK). The model is trained using a **Linear Regression** algorithm.

## Features
- **Data Cleaning**: Removes unnecessary columns, handles missing values, and converts features like `size` and `total_sqft`.
- **Outlier Removal**: Detects and removes outliers in `price_per_sqft` and `BHK` price ranges to improve model accuracy.
- **Feature Engineering**: Creates additional features like `price_per_sqft`, and applies **One-Hot Encoding** to the `location` column.
- **Model Training**: Uses **Linear Regression** to train the model on the cleaned dataset.
- **Prediction Interface**: A command-line interface to predict house prices by taking inputs such as location, square feet, number of bathrooms, and BHK.

## Dataset
The dataset used in this project is a CSV file containing the following columns:
- `area_type`
- `availability`
- `location`
- `size` (BHK and area size)
- `total_sqft`
- `society`
- `bath` (number of bathrooms)
- `balcony`
- `price` (in lakhs)

> Note: Certain columns like `area_type`, `society`, and `availability` were dropped during the data cleaning process as they were not necessary for the model.

## Installation

### Prerequisites:
- Python 3.x
- `numpy`
- `pandas`
- `scikit-learn`

## Usage

1. When you run the script, a list of available locations will be displayed.
2. Enter the location of your choice from the list.
3. Provide the total square feet of the house.
4. Enter the number of bathrooms and bedrooms (BHK).
5. The model will output the predicted price in lakhs.

## Example
```bash
--Welcome to the Home Price Predictor!--

Instructions:
1. Use valid inputs
2. Provide meaningful inputs, or you may get incorrect results

Available locations:
location_1
location_2
location_3
...

Enter the location from the list above: location_1
Enter the total square feet of the house: 1200
Enter the number of bathrooms: 2
Enter the number of bedrooms (BHK): 3

The predicted price of the house is: 85 Lakhs
```

## License
This project is licensed under the MIT License.

---

You can customize this further as needed!
