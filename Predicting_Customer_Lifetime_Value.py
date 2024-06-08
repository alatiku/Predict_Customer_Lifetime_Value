# Load all required packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import missingno as msno
from IPython.display import display, HTML
from tabulate import tabulate
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

retail_data = pd.read_excel("Online Retail.xlsx",
                            parse_dates = ["InvoiceDate"])

# Check missing values for each column of the data
# Plot the sum of each columns row that are non-null
msno.bar(
    retail_data,
    sort = "ascending",
    color = "#1f77b4",
    labels = True,
    figsize = (8, 4),
    fontsize = 10)
plt.show()

# Filter out rows with Negative or Zero Quantities
retail_clean = retail_data[retail_data["Quantity"] > 0]
# Filter out rows with Negative or Zero UnitPrice
retail_clean = retail_clean[retail_clean["UnitPrice"] > 0]
# Remove Description Column
retail_clean.drop(columns = ["Description"], inplace = True)
# Drop Rows with No CustomerIDs
retail_clean.dropna(subset = ["CustomerID"], inplace = True)
# Drop duplicate rows
retail_clean.drop_duplicates(inplace=True)
# Create Total_Price Column for items bought
retail_clean["TotalPrice"] = retail_clean["Quantity"] * retail_clean["UnitPrice"] 
# Check the structure of the cleaned data
print(tabulate(retail_clean.info(), 
              headers = "keys", tablefmt = "html"))


# Number of Customers, stocked items and purchases in the retail Data
count_cust = len(retail_clean["CustomerID"].value_counts())
count_stock = len(retail_clean["StockCode"].value_counts())
count_purchases = len(retail_clean["InvoiceNo"].value_counts())

counts_data = pd.DataFrame({" ": ["Customers", "Stock", "Purchases"],
                            "Counts": [count_cust, count_stock,
                            count_purchases]
                        })
# Set "Names" column as index
counts_data.set_index(" ", inplace = True)
# Set figure size
fig, ax = plt.subplots(figsize=(8, 4))
# Plot the counts data in a horizontal bar chart
counts_data.plot(kind = "barh", legend = None, ax = ax)
# Set axis labels
ax.set_xlabel("Counts")
ax.set_ylabel(" ")
ax.xaxis.grid(True)
# Show the plot
plt.show()


country_cus = pd.DataFrame(retail_clean.groupby("Country")["CustomerID"].nunique().rename("No_of_Customers"))
country_pur = pd.DataFrame(retail_clean.groupby("Country")["InvoiceNo"].nunique().rename("NumPurchases"))
country_rev = pd.DataFrame(retail_clean.groupby("Country")["TotalPrice"].agg("sum").rename("TotalRevenue").round(2))
# Join all the dataframes above into 1
country_data = country_cus.merge(country_pur, 
                                  on = "Country").merge(country_rev, 
                                  on = "Country")
# Calculate Average revenue per customer for each country
country_data["Avg_Revenue_per_Customer"] = (country_data["TotalRevenue"] / country_data["No_of_Customers"]).round(2)
# Calculate Average purchase value each country
country_data["Avg_Purchase_Value"] = (country_data["TotalRevenue"] / country_data["NumPurchases"]).round(2)
# Calculate Purchase Frequency for each country
country_data["Purchase_Frequency"] = (country_data["NumPurchases"] / country_data["No_of_Customers"]).round(2)


stock_data = retail_clean.groupby("StockCode").agg({"TotalPrice": "sum", "Quantity": "sum"}).sort_values(["TotalPrice", "Quantity"], ascending = False)
# Print the stock_stats
# Convert the first 50 rows of stock_data to HTML table using tabulate
table_html3 = tabulate(stock_data.head(50),
                        headers = "keys", tablefmt = "html")


# Group by Customer to calculate total spending and number of purchases
customer_data = retail_clean.groupby("CustomerID").agg({"TotalPrice": "sum", "InvoiceNo": "nunique"})
# Rename the columns created above
customer_data.rename(columns={"Total_Purchase_Value": "TotalPrice", "InvoiceNo": "NumPurchases"}, inplace=True)
# Calculate the average revenue per customer
customer_data["AvgPurchaseValue"] = (customer_data["TotalPrice"] / customer_data["NumPurchases"]).round(2)
# Add Country data for each customer
customer_country = retail_clean[["CustomerID", "Country"]].drop_duplicates()
customer_data = customer_country.merge(customer_data, on = "CustomerID", 
                                        how = "left")
# Print the customer_data
# Convert the first 100 rows of customer_data to HTML table using tabulate
table_html4 = tabulate(customer_data.head(100).sort_values(["TotalPrice", "NumPurchases", "AvgPurchaseValue"], ascending = False),
                        headers = "keys", tablefmt = "html"


# Calculate first and last purchase date for each customer
purchase_period = retail_clean.groupby("CustomerID").agg({"InvoiceDate": ["min", "max"]}).reset_index()
# Rename the columns
purchase_period.columns =["CustomerID", "FirstPurchase", "LastPurchase"]
# Combine customer_date with purchase_period
customer_data = customer_data.merge(purchase_period, on = "CustomerID")
# Calculate the purchasing period in days for each customer and convert to yearly fraction
customer_data["LifeSpan"] = (((customer_data["LastPurchase"] - customer_data["FirstPurchase"]).dt.days)/365).round(2)
# Calculate number of purchases per year for each customer
customer_data["YearlyPurchases"] = (customer_data["NumPurchases"] / customer_data["LifeSpan"]).round(2)
# Replace infinite values in YearlyPurchases column with Zero
customer_data["YearlyPurchases"] = customer_data["YearlyPurchases"].replace(np.inf, 0)
# Calculate CLV for each customer as AvgPurchaseValue * YearlyPurchases
customer_data["CLV"] = (customer_data["AvgPurchaseValue"] * customer_data["YearlyPurchases"]).round(2)
# Print customer_data
# Convert the first 100 rows of customer_data to HTML table using tabulate
table_html4 = tabulate(customer_data.head(100).sort_values(["TotalPrice", "NumPurchases", "AvgPurchaseValue", "LifeSpan", "YearlyPurchases", "CLV"], ascending = False),
                        headers = "keys", tablefmt = "html")


# Drop the CustomerID, Country, FirstPurchase and LastPurchase columns
customer_data.drop(columns = ["FirstPurchase", "LastPurchase", 
                            "CustomerID", "Country"], inplace = True)
plt.figure(figsize=(8, 8))
sns.heatmap(customer_data.corr(), annot = True, 
            linewidths = 0.5, fmt = ".2f");
plt.xticks(rotation = 15, fontsize = 6)
plt.yticks(fontsize = 6)
plt.show()


# Split train-test
# Set X as the data without the target column, while y is only the target
X = customer_data.drop(columns = ["CLV"]).values
y = customer_data["CLV"].values
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                              test_size = 0.3,
                                              random_state = 42)


# Initialize Linear Regression model
reg = LinearRegression()
# Train model
reg.fit(X_train, y_train)
# Predict using model
reg_y_pred = reg.predict(X_test)
# Evaluate model
reg_accuracy = r2_score(y_test, reg_y_pred).round(2)
req_mse = root_mean_squared_error(y_test, reg_y_pred).round(2)
reg_mae = mean_absolute_error(y_test, reg_y_pred).round(2)
# Print model metrics
print(f'Linear Regression - Accuracy: {reg_accuracy * 100}%, RMSE: {req_mse}, MAE: {reg_mae}')


# Initialize RF model
rf_model = RandomForestRegressor(n_estimators = 500, 
                                  max_depth = 4,
                                  min_samples_split = 4,
                                  max_features = None,
                                  bootstrap = True,
                                  criterion = 'friedman_mse',
                                  random_state = 42)
# Train model
rf_model.fit(X_train, y_train)
# Predict using model
rf_y_pred = rf_model.predict(X_test)
# Evaluate model
rf_accuracy = r2_score(y_test, rf_y_pred).round(2)
rf_mse = mean_squared_error(y_test, rf_y_pred).round(2)
# Print model metrics
print(f'Random Forest - Accuracy: {rf_accuracy * 100}%, MSE: {rf_mse}')


# Initialize XGB model
xgb_model = xgb.XGBRegressor(objective = "reg:squarederror",
                        n_estimators = 500,
                        learning_rate = 0.2,
                        max_depth = 4,
                        subsample = 0.7,
                        seed = 1234)
# Train model
xgb_model.fit(X_train, y_train)
# Predict using model
xgb_y_pred = xgb_model.predict(X_test)
# Evaluate model
xgb_accuracy = r2_score(y_test, xgb_y_pred).round(2)
xgb_mse = mean_squared_error(y_test, xgb_y_pred).round(2)
# Print model metrics
print(f'XGB Regressor - Accuracy: {xgb_accuracy * 100}%, MSE: {xgb_mse}')


# Define parameter grid
param_grid_rf = {
    "n_estimators": [100, 200, 300, 400],
    "max_features": [None, "sqrt", "log2"],
    "max_depth": [1, 2, 3, None],
    "min_samples_split": [2, 3, 5, 6]}
# Initialize Random Forest model
rf_model_tune = RandomForestRegressor(random_state = 42)
# Perform Grid Search on model
grid_search_rf = GridSearchCV(estimator = rf_model_tune, 
                              param_grid = param_grid_rf,
                              # Perform Cross-Validation
                              cv = 5,
                              # Utilize all the CPU cores available
                              n_jobs = -1,
                              # Show no info about grid search
                              verbose = 0)
grid_search_rf.fit(X_train, y_train)
# Best parameters and model evaluation
best_rf = grid_search_rf.best_estimator_
y_pred_rf_tune = best_rf.predict(X_test)
# Evaluate metrics of best model selected
mse_rf_tune = mean_squared_error(y_test, y_pred_rf_tune).round(2)
r2_rf_tune = r2_score(y_test, y_pred_rf_tune).round(2)
# Print model selected metrics
print(f"Best Random Forest Parameters:")
print(f"{grid_search_rf.best_params_}")
print(f"Random Forest - Accuracy: {r2_rf_tune * 100}%, MSE: {mse_rf_tune}")


# Define parameter grid
param_grid_xgb = {
    "n_estimators": [100, 200, 300, 400],
	"learning_rate": [0.1, 0.3, 0.4, 0.5],
    "max_depth": [2, 3, 4, 5],
    "subsample": [0.5, 0.6, 0.8, 0.9]}
# Initialize XGBoost model
xgb_model_tune = xgb.XGBRegressor(objective = "reg:squarederror", 
                              random_state = 42)
# Perform Randomized Search model
random_search_xgb = RandomizedSearchCV(estimator = xgb_model_tune,
                                      param_distributions = param_grid_xgb,
                                      # Number to sample and evaluate
                                      n_iter = 50,
                                      # Perform Cross-Validation
                                      cv = 5,
                                      # Utilize all CPU cores available
                                      n_jobs = -1,
                                      # Show no info about grid search
                                      verbose = 0, 
                                      random_state = 42)
random_search_xgb.fit(X_train, y_train)
# Best parameters and model evaluation
best_xgb = random_search_xgb.best_estimator_
y_pred_xgb_tune = best_xgb.predict(X_test)
# Evaluate metrics of best model selected
mse_xgb_tune = mean_squared_error(y_test, y_pred_xgb_tune).round(2)
r2_xgb_tune = r2_score(y_test, y_pred_xgb_tune).round(2)
# Print model selected metrics
print(f"Best XGBoost Parameters:")
print(f"{random_search_xgb.best_params_}")
print(f"XGBoost - Accuracy: {r2_xgb_tune * 100}%, MSE: {mse_xgb_tune}")



# Extract feature importance from the best RF model
rf_importances = best_rf.feature_importances_
# Create a dataframe for feature importance
rf_features = pd.DataFrame({
    'Feature': customer_data.drop(columns=['CLV']).columns,
    'Importance': rf_importances
})
# Sort the dataframe by importance
rf_features.sort_values(by='Importance', ascending=False, inplace=True)
# Plot the feature importance
plt.figure(figsize=(8, 4))
plt.bar(rf_features['Feature'], rf_features['Importance'])
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.xticks(fontsize = 6)
plt.title('Random Forest Feature Importance')
plt.show()


# Extract feature importance from the best XGBoost model
xgb_importances = best_xgb.feature_importances_
# Create a dataframe for feature importance
xgb_features = pd.DataFrame({
    'Feature': customer_data.drop(columns=['CLV']).columns,
    'Importance': xgb_importances
})
# Sort the dataframe by importance
xgb_features.sort_values(by='Importance', ascending=False, inplace=True)
# Plot the feature importance
plt.figure(figsize=(8, 4))
plt.bar(xgb_features['Feature'], xgb_features['Importance'])
plt.ylabel('Importance')
plt.xlabel('Feature')
plt.xticks(fontsize = 6)
plt.title('XGBoost Feature Importance')
plt.show()