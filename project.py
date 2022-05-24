#!/usr/bin/env python
# coding: utf-8

# # FIT5196 Assessment 2
# 
# #### Student Name: Ibrahim Al-Hindi
# #### Date Created: 20/09/2021
# #### Date Last Edited: 03/10/2021
# 
# Environment Details:
# - Python: 3.8.10
# - Anaconda: 4.10.3
# 
# Libraries Used:
# - pandas
# - numpy
# - matplotlib
# - nltk
# - re
# - datetime
# - networkx
# - sklearn
# - ast
# 
# ## 1. Introduction
# 
# This assignment is concerned with exhibiting the critical importance of data wrangling and data cleansing for a given dataset. Three files created from the same origin will be used in this report. The files contain data for a food delivery service and contain information such as restaurant, customer, date, price, and others. The first section of the report will examine the first file which contains a multitude of errors of various types, the task is to identify and fix all the errors present in the file. The second section of the report concerns a file that contains missing data and the task is to impute those missing values. The third section is analysing a file with outliers in one of its columns, where the task is to identify and remove the rows with the outliers. It is important to note that all three files stem from the same source file, as such some of the files will be used to tackle issues in other files in some of the sections contained in this report.

# ## 2. Import Libraries and Load Data Files

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import datetime as dt
import networkx as nx
from sklearn.linear_model import LinearRegression
from ast import literal_eval
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dirty = pd.read_csv("./data/dirty.csv")
missing = pd.read_csv("./data/missing.csv")
outliers = pd.read_csv("./data/outlier.csv")


# As our analysis will require us to use the three files interchangeably, we will verify that all three files contain the same unique restaurants according to the `restaurant_id` column

# In[ ]:


print("dirty:   ", sorted(list(dirty["restaurant_id"].unique())))
print("missing: ", sorted(list(missing["restaurant_id"].unique())))
print("outliers:", sorted(list(outliers["restaurant_id"].unique())))


# The three files contain the same restaurants

# ## 3. Dirty Data
# 
# The first part of the report will examine the `dirty` file which contains errors and proceed to fix those errors.
# 
# ### 3.1 Data Exploration
# 
# A preliminary exploration will be performed

# Let's take a look at the first and last five rows of the data

# In[ ]:


dirty


# In[ ]:


dirty.shape


# The data consists of 500 observations of 18 columns

# Let's examine the types of variables in our data

# In[ ]:


dirty.info()


# We have 10 numerical and 8 categorical variables

# Let's examine the summary statistics of the numerical variables

# In[ ]:


dirty.describe()


# Already we can see some errors such as the maximum of **145.004298** under `customer_lat` and, conversely, the minimum of **-37.814714** under `customer_lon`. Let's examine the summary statistics of the categorical variables

# In[ ]:


dirty.describe(include = "O")


# Further exploration will be performed to discover and solve errors further along the report.
# 
# The errors will now be located and corrected. The columns to be examined and corrected if necessary (in no specific order) are:
# 
# - customer_lat
# - customer_lon
# - date
# - is_peak_time
# - is_weekend
# - carrier_vehicle
# - coupon_discount
# - order_price
# - shortest_distance_to_customer
# - travel_time_minutes
# - restaurant_rating
# 
# ### 3.2 restaurant_rating
# 
# Referencing the table above examining the categorical variables, we can see that we have five unique `restaurant_id`. We should therefore have only five unique restaurant ratings. We actually have the following number of unique restaurant ratings:

# In[ ]:


len(dirty["restaurant_rating"].unique())


# We can see the ratings for each restaurant by generating the unique `restaurant_rating` after grouping by `restaurant_id`

# In[ ]:


original_ratings = dirty.groupby(dirty["restaurant_id"])["restaurant_rating"].unique()
original_ratings


# As we have 2 restaurant ratings per `restaurant_id` instead of one, the true restaurant ratings will be generated and inserted into the data. To calculate the true restaurant rating for each restaurant, `SentimentIntensityAnalyzer` from `nltk.sentiment.vader` will be implemented on the reviews of each restaurant to calculate the average polarity (avg_polarity) per restaurant, then the rating will be calculated using the following formula:
# 
# **10 * (avg_polarity + 0.9623) / (1.9533)**
# 
# First, the file containing information regarding each restaurant including their reviews will be read in

# In[ ]:


rest_data = pd.read_csv("./supplementary_data/restaurant_data_student.csv")
rest_data.head()


# We will filter `rest_data` to only contain the `restaurant_id`s present in the dirty data

# In[ ]:


rest_data_match = rest_data[rest_data["restaurant_id"].isin(dirty["restaurant_id"])]
rest_data_match


# All the columns except for `restaurant_id` and `reviews_list` will be dropped

# In[ ]:


rest_data_rating = rest_data_match.drop(columns = ["restaurant_name", "menu_items", "lat", "lon", "rest_nodes"])
rest_data_rating


# We now have the five restaurants we need with their reviews.

# We will now utilise the sentiment analyzer. First we initialize a sentiment analyzer object

# In[ ]:


# initialize
sid = SentimentIntensityAnalyzer()


# The function `rat_gen` will be created whereby the list of reviews for a restaurant will be passed in as an argument and will return a rating for that restaurant

# In[ ]:


def rat_gen(reviews):
    # split the one string of reviews into separate reviews using regex
    reviews_sep = re.split(r"[\'\"], [\'\"]", reviews)
    # extract the compound score from the polarity score of each review
    total = [sid.polarity_scores(sentence)["compound"] for sentence in reviews_sep]
    # calculate average
    avg = sum(total)/len(total)
    # calculate rating using formula given above
    rating = round((10 * (avg + 0.9623) / (1.9533)), ndigits=2)
    return rating


# `rest_data_rating` will be merged to `dirty` so that the `reviews_list` column gets added

# In[ ]:


dirty = pd.merge(dirty, rest_data_rating, on = "restaurant_id", how = "left")
dirty.head()


# The `restaurant_rating` column values will now be updated by applying the `rat_gen` function to the `reviews_list` column

# In[ ]:


dirty["restaurant_rating"] = dirty["reviews_list"].apply(rat_gen)
dirty.head()


# Let's verify that we now have five unique ratings

# In[ ]:


len(dirty["restaurant_rating"].unique())


# In[ ]:


new_ratings = dirty.groupby(dirty["restaurant_id"])["restaurant_rating"].unique()
new_ratings


# We can verify that the new ratings generated are correct as they all correspond to the first rating for each restaurant prior to the adjustment

# In[ ]:


original_ratings


# We now have five unique ratings corresponding to the five unique restaurants
# 
# We can now drop the `reviews_list` column

# In[ ]:


dirty.drop(columns = "reviews_list", inplace = True)
dirty.head()


# ### 3.3 customer_lat and customer_lon
# 
# To find any errors in `customer_lat` and `customer_lon`, histograms of the values for each will be plotted

# In[ ]:


dirty["customer_lat"].hist()


# In[ ]:


dirty["customer_lon"].hist()


# We can see that there are some observations with a latitude greater than 0 and a longitude smaller than 100. Let's retrieve the rows that have either a large `customer_lat` or a small `customer_lon`.
# 
# First we will create a mask filter that will return the rows where either `customer_lat` is greater than 0 or `customer_lon` is less than 100

# In[ ]:


mask = (dirty["customer_lat"] > 0) | (dirty["customer_lon"] < 100)
dirty[mask]


# It seems that for these six rows, the `customer_lat` and `customer_lon` values were swapped due to data entry error. The values will have to be swapped to the correct columns.

# In[ ]:


dirty.loc[mask, "customer_lat"], dirty.loc[mask, "customer_lon"] = dirty.loc[mask, "customer_lon"], dirty.loc[mask, "customer_lat"]


# Check that the incorrect rows are now correct:

# In[ ]:


dirty[mask]


# In[ ]:


dirty[(dirty["customer_lat"] > 0) | (dirty["customer_lon"] < 100)]


# Check histograms of both measurements are correct:

# In[ ]:


dirty["customer_lat"].hist()


# In[ ]:


dirty["customer_lon"].hist()


# The values are now in their respective correct columns

# Check that we have 491 unique latitudes and longitudes corresponding to 491 unique customers as indicated by the `describe` function performed in the beginning of the report

# In[ ]:


print("unique customers:   ", len(dirty["customer_id"].unique()))
print("unique customer_lat:", len(dirty["customer_lat"].unique()))
print("unique customer_lon:", len(dirty["customer_lon"].unique()))


# ### 3.4 date
# 
# The first check for the date column is using regex to make sure all the dates are:
# - in the correct format
# - a valid date
# 
# the pattern is
# 
# **(2020-(?:(?:02-(?:0[1-9]|[12][0-9]))|(?:(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30))|(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|31))))**
# 
# This pattern ensures that the months have the correct number of days (for example, February has 29, January has 31, April has 30, etc), and it ensures the format is yyyy-mm-dd.
# 
# We can find the rows where the `date` value does not match the regex pattern, and is therefore incorrect. The `str.match` function is applied to the `date` column which returns either **True** or **False** for each column

# In[ ]:


pattern = r"(2020-(?:(?:02-(?:0[1-9]|[12][0-9]))|(?:(?:0[469]|11)-(?:0[1-9]|[12][0-9]|30))|(?:(?:0[13578]|1[02])-(?:0[1-9]|[12][0-9]|3[01]))))"


# In[ ]:


dirty[~dirty["date"].str.match(pattern)]


# We can see two types of errors in the date column:
# - rows **28, 182 376, and 388** have a completely different format of **day mon d time year**
# - the remaining six rows have swapped day and month locations **yyyy-dd-mm** instead of the correct **yyyy-mm-dd**
# 
# Let's first tackle the first type of error. We notice that these date values have a `:` in them

# In[ ]:


dirty[dirty["date"].str.contains(":")]


# We will now replace these values with the correct format. First, we will pass the above filter into a mask variable, then the `date` of the above rows will be converted to datetime using `pd.to_datetime`. To convert them back to string to match the remaining `date` values in the column, we will apply `strftime` from the `datetime` library

# In[ ]:


mask = dirty["date"].str.contains(":")
dirty.loc[mask, "date"] = pd.to_datetime(dirty.loc[mask, "date"]).dt.strftime("%Y-%m-%d")


# In[ ]:


dirty[mask]


# The `date` values in the above rows are now correct.
# 
# The remaining rows with the second type of error are given below:

# In[ ]:


dirty[~dirty["date"].str.match(pattern)]


# We will apply a lambda function to the incorrect values in the `date` column where the date will be split on `-` and reordered to where the format is year-month-day like so:
# 
# - Original date: 2020-27-07
# - Split on "-": [2020, 27, 07]
# - Reorder through indexing: [2020, 27, 07][0]-[2020, 27, 07][2]-[2020, 27, 07][1]
# - Output: 2020-07-27

# In[ ]:


# set mask
mask = ~dirty["date"].str.match(pattern)


# In[ ]:


dirty.loc[mask, "date"] = dirty.loc[mask, "date"].apply(lambda x: f"{x.split('-')[0]}-{x.split('-')[2]}-{x.split('-')[1]}")


# Let's double check our work

# In[ ]:


dirty[~dirty["date"].str.match(pattern)]


# We have no more incorrect dates

# ### 3.5 is_peak_time
# 
# First, we will check if any value exists in this column other than 0 or 1

# In[ ]:


print(dirty["is_peak_time"].unique())


# We can see that indeed only 0 or 1 exist in this column.
# 
# Next, we will check the values of `is_peak_time` against the `time` column to check whether the time in each row was indeed inside or outside the peak time. The peak periods are the intervals **12:00:00pm - 1:59:59pm** and **6:00:00pm - 7:59:59pm**
# 
# We will pass the filter of peak times into a `mask` variable

# In[ ]:


mask = (((dirty["time"] >= "12:00:00") & (dirty["time"] < "14:00:00")) | ((dirty["time"] >= "18:00:00") & (dirty["time"] < "20:00:00")))


# Now we will check if any `is_peak_time` is equal to 0 during the peak times

# In[ ]:


dirty[mask & (dirty["is_peak_time"] == 0)]


# ORD126860's `is_peak_time` is incorrectly coded as 0. We will recode it as 1

# In[ ]:


dirty.loc[mask & (dirty["is_peak_time"] == 0), "is_peak_time"] = 1


# In[ ]:


dirty.loc[300, "is_peak_time"]


# Similarly, we will check if any `is_peak_time` is equal to 1 outside of peak times

# In[ ]:


dirty[~mask & (dirty["is_peak_time"] == 1)]


# The above orders have been incorrectly coded as 1. They will be corrected to 0

# In[ ]:


dirty.loc[~mask & (dirty["is_peak_time"] == 1), "is_peak_time"] = 0


# In[ ]:


dirty.loc[[51, 194, 315, 330, 459], "is_peak_time"]


# We will perform one final check

# In[ ]:


dirty[(mask & (dirty["is_peak_time"] == 0)) | (~mask & (dirty["is_peak_time"] == 1))]


# All `is_peak_time` values are correctly coded now

# ### 3.6 is_weekend
# 
# First, we will check if any values exist in this column other than 0 or 1

# In[ ]:


print(dirty["is_weekend"].unique())


# We can see that indeed only 0 or 1 exist in this column.
# 
# Next, we will check the `is_weekend` values against `date` to verify that the date is in fact a weekend. To accomplish this, we will first create a new temporary `day` column that contains the `date` column values converted to datetime using the `datetime` library

# In[ ]:


dirty["day"] = pd.to_datetime(dirty["date"])
dirty["day"].head()


# Next, we will retrieve the name of the day using the `day_name` function from the `datetime` library

# In[ ]:


dirty["day"]= dirty["day"].dt.day_name()
dirty["day"].head()


# Now we can check whether any `is_weekend` values were incorrectly coded. A cross-tabulation of `day` vs `is_weekend` will be generated to check

# In[ ]:


pd.crosstab(dirty["day"],dirty["is_weekend"])


# We can see that Friday, Monday, Thursday, and Wednesday were incorrectly coded as weekends on some occasions.
# 
# We will pass the weekend filter into a mask variable

# In[ ]:


mask = ((dirty["day"] == "Saturday") | (dirty["day"] == "Sunday"))
dirty[mask].head()


# The weekdays that have been incorrectly recorded as weekends will be corrected

# In[ ]:


dirty.loc[~mask & (dirty["is_weekend"] == 1), "is_weekend"] = 0


# We will perform one final check

# In[ ]:


pd.crosstab(dirty["day"],dirty["is_weekend"])


# All `is_weekend` values are correctly coded. We can now drop the `day` column

# In[ ]:


dirty.drop(columns = "day", inplace = True)
dirty.head(5)


# ### 3.7  coupon_discount and order_price
# 
# `order_price` is calculated by applying the percentage of `coupon_discount` to the price derived from the order's `shopping_cart`. Therefore, for each order the original price prior to the discount will be calculated, this original price will then be used to derive the price of each item for each restaurant using linear algebra. Since the `order_price` in `dirty` is the column we are verifiying, we will run the process on the `missing` data which contains no errors to derive the item prices, we will then apply those item prices to the `dirty` dataframe to verify our `order_price` column.
# 
# #### 3.7.1 Calculate Item Prices
# 
# We will create a copy of `missing` to work on with only the required columns

# In[ ]:


price_calc_df = missing[["restaurant_id", "shopping_cart", "coupon_discount", "order_price"]].copy()
price_calc_df.head()


# Now we will create a new column `original_price` that contains the price of each order prior to applying `coupon_discount`. The original price is calculated using the following formula:
# 
# **original_price = order_price / (1 - (coupon_discount / 100))**

# In[ ]:


price_calc_df["original_price"] = round(price_calc_df["order_price"] / (1 - (price_calc_df["coupon_discount"] / 100)), 2)
price_calc_df.head()


# Using the function `literal_eval` from the `ast` package ([source](https://www.codegrepper.com/code-examples/python/python+extract+list+from+string)), we can convert the string which contains the list of items under `shopping_cart` into a pure list as so:

# In[ ]:


price_calc_df.loc[0, "shopping_cart"]


# In[ ]:


# type before applying literal_eval
type(price_calc_df.loc[0, "shopping_cart"])


# In[ ]:


# type after applying literal_eval
type(literal_eval(price_calc_df.loc[0, "shopping_cart"]))


# In[ ]:


price_calc_df["shopping_cart"] = price_calc_df["shopping_cart"].apply(literal_eval)
price_calc_df.head()


# We will build a dictionary containing the menu for each restaurant and the price for each item. Initially, the price for all the items will be set to 0

# In[ ]:


rest_menu = {}

# iterate over unique restaurants in restaurant_id
for rest in price_calc_df["restaurant_id"].unique():
    # create temporary dataframe containing only current restaurant_id
    temp_df = price_calc_df[price_calc_df["restaurant_id"] == rest].copy()   
    # for each row under shopping_cart, extract the first item from each tuple in that shopping cart.
    # place inside set to only have unique items
    menu = {item[0] for cart in temp_df["shopping_cart"] for item in cart}
    # create dictionary of items and price
    menu_prices = {item: 0 for item in menu}
    # insert restaurant_id as key and the menu as the value
    rest_menu[rest] = menu_prices
    
rest_menu


# To calculate the price of each item for each restaurant, will need to use a linear system of equations. To calculate the first few items for each restaurant, we will need to find two orders with the same menu items and plug them into our linear system. To be able to find orders with the same menu items, we will create a new column `basket_set` that contains the menu items from each order. It will be placed inside a set so that the order of the items doesn't matter when making comparisons

# In[ ]:


price_calc_df["basket_set"] = price_calc_df["shopping_cart"].apply(lambda x: {item[0] for item in x})
price_calc_df.head()


# For each restaurant now, we will generate the value_counts of `basket_set` to show the number of times the same `basket_set` is present. We will then pick the set with the largest number of repitions, that contains the same number of items as the number of repitions to simultaneously calculate as many items as possible using the linear system of equations. We will start with `restaurant_id` **1781**

# In[ ]:


# create new dataframe filtered for REST1781
price_calc_df_1781 = price_calc_df[price_calc_df["restaurant_id"] == "REST1781"].copy()
price_calc_df_1781.head()


# In[ ]:


# generate REST1781 menu for reference
rest_menu["REST1781"]


# In[ ]:


# generate value_counts for basket_set
price_calc_df_1781["basket_set"].value_counts()


# We will first choose the set {beef burger, mango salad, chicken burger} as it is the set with the highest repitions with the same number of items (3).
# 
# Retrieve the rows with that set to choose from

# In[ ]:


price_calc_df_1781[price_calc_df_1781["basket_set"] == {"beef burger", "mango salad", "chicken burger"}]


# We will extract the `shopping_cart` column as a list so we can better view the carts

# In[ ]:


list(price_calc_df_1781[price_calc_df_1781["basket_set"] == {"beef burger", "mango salad", "chicken burger"}]["shopping_cart"])


# We will now plug in the values for each item and the original price into the linear system. We will use the function `linalg.solve` from the `numpy` package

# In[ ]:


# enter number of items from each cart where each cart is a separate list into an array
a = np.array([[3, 2, 1], [3, 3, 2], [1, 3, 1]])
# enter original_price for each cart into an array
b = np.array([90.63, 122.5, 89.79])
# solve linear equation
x = np.linalg.solve(a, b)
x


# In[ ]:


# set prices of items using unpacking
chicken_burger, mango_salad, beef_burger = x


# In[ ]:


# update menu prices
rest_menu["REST1781"]["chicken burger"] = chicken_burger
rest_menu["REST1781"]["mango salad"] = mango_salad
rest_menu["REST1781"]["beef burger"] = beef_burger
rest_menu["REST1781"]


# We will calculate the prices of the remaining items the same way

# In[ ]:


price_calc_df_1781[price_calc_df_1781["basket_set"] == {"pizza", "veggie pizza"}]


# In[ ]:


list(price_calc_df_1781[price_calc_df_1781["basket_set"] == {"pizza", "veggie pizza"}]["shopping_cart"])


# In[ ]:


a = np.array([[3, 2], [1, 3]])
b = np.array([74.67, 79.84])
x = np.linalg.solve(a, b)
veggie_pizza, pizza = x
rest_menu["REST1781"]["veggie pizza"] = veggie_pizza
rest_menu["REST1781"]["pizza"] = pizza


# For the remaining items, we can solve the equation if we know the price of the other item in the cart

# In[ ]:


price_calc_df_1781[price_calc_df_1781["basket_set"] == {"peri peri chicken wings", "chicken burger"}]


# In[ ]:


price_calc_df_1781.loc[115, "shopping_cart"]


# We already know the price of a chicken burger, so we will plug it in into row 115 and subtract the price from the `original_price` to come up with the price of peri peri chicken wings

# In[ ]:


peri_peri = (48.26 - rest_menu["REST1781"]["chicken burger"]) / 2
peri_peri


# In[ ]:


rest_menu["REST1781"]["peri peri chicken wings"] = peri_peri


# The remaining items will be calculated the same way

# In[ ]:


price_calc_df_1781[price_calc_df_1781["basket_set"] == {"coffee", "pizza"}]


# In[ ]:


price_calc_df_1781.loc[227, "shopping_cart"]


# In[ ]:


coffee = (92.22 - rest_menu["REST1781"]["pizza"]) / 3
rest_menu["REST1781"]["coffee"] = coffee


# In[ ]:


price_calc_df_1781[price_calc_df_1781["basket_set"] == {"mango salad", "burgers"}]


# In[ ]:


price_calc_df_1781.loc[350, "shopping_cart"]


# In[ ]:


burgers = (51.68 - (2 * rest_menu["REST1781"]["mango salad"]))
rest_menu["REST1781"]["burgers"] = burgers


# We now have the prices for all the items for REST1781

# In[ ]:


rest_menu["REST1781"]


# Now we will calculate the item prices for the remaining restaurants using the same process

# In[ ]:


price_calc_df_1801 = price_calc_df[price_calc_df["restaurant_id"] == "REST1801"].copy()


# In[ ]:


rest_menu["REST1801"]


# In[ ]:


price_calc_df_1801["basket_set"].value_counts()


# In[ ]:


price_calc_df_1801[price_calc_df_1801["basket_set"] == {"naan", "chicken boneless biryani", "crispy chicken"}]


# In[ ]:


list(price_calc_df_1801[price_calc_df_1801["basket_set"] == {"naan", "chicken boneless biryani", "crispy chicken"}]["shopping_cart"])


# In[ ]:


a = np.array([[3, 2, 1], [1, 1, 2], [2, 2, 1]])
b = np.array([106.5, 61.13, 83.8])
x = np.linalg.solve(a, b)
chicken_boneless_biryani, crispy_chicken, naan = x
rest_menu["REST1801"]["chicken boneless biryani"] = chicken_boneless_biryani
rest_menu["REST1801"]["crispy chicken"] = crispy_chicken
rest_menu["REST1801"]["naan"] = naan


# In[ ]:


price_calc_df_1801[price_calc_df_1801["basket_set"] == {"panner tikka", "tandoori chicken"}]


# In[ ]:


a = np.array([[1, 1], [3, 2]])
b = np.array([25.06, 69.02])
x = np.linalg.solve(a, b)
tandoori_chicken, panner_tikka = x
rest_menu["REST1801"]["tandoori chicken"] = tandoori_chicken
rest_menu["REST1801"]["panner tikka"] = panner_tikka


# In[ ]:


price_calc_df_1801[price_calc_df_1801["basket_set"] == {"chicken guntur", "chicken boneless biryani"}]


# In[ ]:


price_calc_df_1801.loc[341, "shopping_cart"]


# In[ ]:


chicken_guntur = 32.23 - rest_menu["REST1801"]["chicken boneless biryani"]
rest_menu["REST1801"]["chicken guntur"] = chicken_guntur


# In[ ]:


price_calc_df_1801[price_calc_df_1801["basket_set"] == {"naan", "hyderabadi biryani"}]


# In[ ]:


hyderabadi_biryani = 33.28 - (2 * rest_menu["REST1801"]["naan"])
rest_menu["REST1801"]["hyderabadi biryani"] = hyderabadi_biryani


# In[ ]:


rest_menu["REST1801"]


# In[ ]:


price_calc_df_1450 = price_calc_df[price_calc_df["restaurant_id"] == "REST1450"].copy()


# In[ ]:


rest_menu["REST1450"]


# In[ ]:


price_calc_df_1450["basket_set"].value_counts()


# In[ ]:


price_calc_df_1450[price_calc_df_1450["basket_set"] == {"rasgulla", "puri sabzi", "jalebi", "samosa"}]


# In[ ]:


list(price_calc_df_1450[price_calc_df_1450["basket_set"] == {"rasgulla", "puri sabzi", "jalebi", "samosa"}]["shopping_cart"])


# In[ ]:


a = np.array([[1, 2, 3, 1], [2, 2, 2, 1], [1, 1, 3, 1], [1, 1, 1, 3]])
b = np.array([121.23, 109.35, 108.44, 104])
x = np.linalg.solve(a, b)
jalebi, samosa, rasgulla, puri_sabzi = x
rest_menu["REST1450"]["rasgulla"] = rasgulla
rest_menu["REST1450"]["puri sabzi"] = puri_sabzi
rest_menu["REST1450"]["jalebi"] = jalebi
rest_menu["REST1450"]["samosa"] = samosa


# In[ ]:


price_calc_df_1450[price_calc_df_1450["basket_set"] == {"kachori", "poha"}]


# In[ ]:


a = np.array([[3, 1], [1, 2]])
b = np.array([32.82, 24.84])
x = np.linalg.solve(a, b)
kachori, poha = x
rest_menu["REST1450"]["kachori"] = kachori
rest_menu["REST1450"]["poha"] = poha


# In[ ]:


rest_menu["REST1450"]


# In[ ]:


price_calc_df_1858 = price_calc_df[price_calc_df["restaurant_id"] == "REST1858"].copy()


# In[ ]:


rest_menu["REST1858"]


# In[ ]:


price_calc_df_1858["basket_set"].value_counts().head(20)


# In[ ]:


price_calc_df_1858[price_calc_df_1858["basket_set"] == {"mackerel fry", "crab toast", "roast beef"}]


# In[ ]:


list(price_calc_df_1858[price_calc_df_1858["basket_set"] == {"mackerel fry", "crab toast", "roast beef"}]["shopping_cart"])


# In[ ]:


a = np.array([[2, 3, 3], [1, 3, 2], [3, 3, 1]])
b = np.array([112.01, 85.12, 75.18])
x = np.linalg.solve(a, b)
crab_toast, mackerel_fry, roast_beef = x
rest_menu["REST1858"]["crab toast"] = crab_toast
rest_menu["REST1858"]["mackerel fry"] = mackerel_fry
rest_menu["REST1858"]["roast beef"] = roast_beef


# In[ ]:


price_calc_df_1858[price_calc_df_1858["basket_set"] == {"fish", "appam", "kerala parotta"}]


# In[ ]:


a = np.array([[1, 1, 2], [2, 3, 1], [3, 1, 1]])
b = np.array([67.58, 98.21, 90.41])
x = np.linalg.solve(a, b)
appam, kerala_parotta, fish = x
rest_menu["REST1858"]["appam"] = appam
rest_menu["REST1858"]["kerala parotta"] = kerala_parotta
rest_menu["REST1858"]["fish"] = fish


# In[ ]:


price_calc_df_1858[price_calc_df_1858["basket_set"] == {"mackerel fry", "beef fry"}]


# In[ ]:


beef_fry = 47.71 - (3 * rest_menu["REST1858"]["mackerel fry"])
rest_menu["REST1858"]["beef fry"] = beef_fry


# In[ ]:


rest_menu["REST1858"]


# In[ ]:


price_calc_df_0936 = price_calc_df[price_calc_df["restaurant_id"] == "REST0936"].copy()


# In[ ]:


rest_menu["REST0936"]


# In[ ]:


price_calc_df_0936["basket_set"].value_counts().head(40)


# In[ ]:


price_calc_df_0936[price_calc_df_0936["basket_set"] == {"tender coconut ice cream", "mocha chingri", "cheesecake", "begun pora"}]


# In[ ]:


list(price_calc_df_0936[price_calc_df_0936["basket_set"] == {"tender coconut ice cream", "mocha chingri", "cheesecake", "begun pora"}]["shopping_cart"])


# In[ ]:


a = np.array([[1, 3, 1, 2], [1, 1, 3, 3], [1, 2, 1, 2], [2, 3, 2, 1]])
b = np.array([121.14, 154.34, 108.57, 145.17])
x = np.linalg.solve(a, b)
begun_pora, cheesecake, mocha_chingri, tender_coconut_ice_cream = x
rest_menu["REST0936"]["begun pora"] = begun_pora
rest_menu["REST0936"]["cheesecake"] = cheesecake
rest_menu["REST0936"]["mocha chingri"] = mocha_chingri
rest_menu["REST0936"]["tender coconut ice cream"] = tender_coconut_ice_cream


# In[ ]:


price_calc_df_0936[price_calc_df_0936["basket_set"] == {"banana chips", "paratha"}]


# In[ ]:


a = np.array([[2,3], [1, 1]])
b = np.array([67.28, 28.86])
x = np.linalg.solve(a, b)
banana_chips, paratha = x
rest_menu["REST0936"]["banana chips"] = banana_chips
rest_menu["REST0936"]["paratha"] = paratha


# In[ ]:


price_calc_df_0936[price_calc_df_0936["basket_set"] == {"saffron pilaf", "begun pora"}]


# In[ ]:


saffron_pilaf = (96.24 - (2 * rest_menu["REST0936"]["begun pora"])) / 2
rest_menu["REST0936"]["saffron pilaf"] = saffron_pilaf


# In[ ]:


rest_menu["REST0936"]


# We now have all the item prices for all the restaurants

# In[ ]:


rest_menu


# We will round the prices to two decimal places

# In[ ]:


# access restaurant
for key, value in rest_menu.items():
    # access menu of restaurant
    for key2, value2 in value.items():
        # round price to two decimal places
        rest_menu[key][key2] = round(value2, 2)


# In[ ]:


rest_menu


# We will create a function `calc_price` where the new price using the dictionary `rest_menu` will be created for each row

# In[ ]:


def calc_price(row):
    # grab the restaurant_id, shopping_cart, and coupon_discount from each row
    rest = row["restaurant_id"]
    basket = row["shopping_cart"]
    discount = row["coupon_discount"]
    total = 0
    
    for item in basket:
        food = item[0]
        quantity = item[1]
        # grab the price for the item from the relevant restaurant dictionary and multiply by the quantity
        item_total = rest_menu[rest][food] * quantity
        # update total
        total += item_total
        
    return round(total, 2)


# In[ ]:


# apply calc_price to price_calc_df
price_calc_df["new_price"] = price_calc_df.apply(calc_price, axis = 1)
price_calc_df.head()


# We will verify if the process was correct by ensuring that all `new_price` are equal to `order_price`

# In[ ]:


price_calc_df[price_calc_df["original_price"] != price_calc_df["new_price"]]


# We can see that the two columns are equal. We will run our prices on a copy of `outliers` to double-check. The necessary columns will be created

# In[ ]:


out = outliers.copy()
out["shopping_cart"] = out["shopping_cart"].apply(literal_eval)
out["original_price"] = round(out["order_price"] / (1 - (out["coupon_discount"] / 100)), 2)
out["new_price"] = out.apply(calc_price, axis = 1)


# In[ ]:


out[out["original_price"] != out["new_price"]]


# `original_price` is equal to `new_price` for all the rows in `outliers` as well, therefore we are confident in our item prices. We can use our `rest_menu` dictionary to check the prices in our `dirty` dataframe.
# 
# #### 3.7.2 Check Prices in `dirty`
# 
# First we will need to create a new `shopping_list` column that converts the string `shopping_cart` into a list as was done above

# In[ ]:


dirty["shopping_list"] = dirty["shopping_cart"].apply(literal_eval)
dirty.head()


# We will also create a new `original_price` column as was done in the `missing` data to obtain the prices prior to discount. This is done so that when we encounter errors in the `order_price` column, we can use `original_price` to determine if the error is actually located in the `coupon_discount` column vs the `order_price` column. This will be illustrated later on

# In[ ]:


dirty["original_price"] = round(dirty["order_price"] / (1 - (dirty["coupon_discount"] / 100)), 2)
dirty.head()


# Now we will apply `calc_price` to `dirty` to create the updated prices. However, we will first need to update `calc_price` to account for the newly created `shopping_list` instead of `shopping_cart`. The updated function is called `calc_price2`

# In[ ]:


def calc_price2(row):
    # grab the restaurant_id, shopping_cart, and coupon_discount from each row
    rest = row["restaurant_id"]
    basket = row["shopping_list"]
    discount = row["coupon_discount"]
    total = 0
    
    for item in basket:
        food = item[0]
        quantity = item[1]
        # grab the price for the item from the relevant restaurant dictionary and multiply by the quantity
        item_total = rest_menu[rest][food] * quantity
        # update total
        total += item_total
        
    return round(total, 2)


# In[ ]:


dirty["new_price"] = dirty.apply(calc_price2, axis = 1)
dirty.head()


# Now we can locate the rows with the errors

# In[ ]:


dirty[dirty["original_price"] != dirty["new_price"]]


# We can see the above rows have incorrect `original_price`, and consequently incorrect `order_price`. However, before we assume the error is due to `original_price` alone, we need to check that the correct `coupon_discount` was applied to the price. This is done by calculating the actual discount applied by using the following formula:
# 
# **(1 - (order_price / new_price)) * 100**
# 
# The result will be outputted to a new column `discount_applied`.
# 
# If, for a given row, `coupon_discount` is not equal to `discount_applied` **and** `discount_applied` is equal to one of the allowed values of disounts [0, 10, 20, 40, 45], then `order_price` is correct and the error is in `coupon_discount`.
# 
# If, however, the `discount_applied` is not equal to one of the accepted values, then the error is in fact solely due to an incorrect `order_price`.
# 
# We will verify that all the value in the `coupon_discount` columns are acceptable by being one of [0, 10, 20, 40, 45]

# In[ ]:


sorted(list(dirty["coupon_discount"].unique()))


# All the values in `coupon_discount` are acceptable.
# 
# Two new columns will now be created:
# 
# - `order_price_round`: `order_price` rounded to two decimals places instead of three. This is done to avoid discrepancies due to rounding errors
# - `discount_applied`: as explained above

# In[ ]:


dirty["order_price_round"] = round(dirty["order_price"], 2)
dirty["discount_applied"] = ((1 - (dirty["order_price_round"] / dirty["new_price"])) * 100).apply(round)
dirty.head()


# We will locate the rows with incorrect `coupon_discount` as described above

# In[ ]:


dirty[(dirty["coupon_discount"] != dirty["discount_applied"]) & (dirty["discount_applied"].isin(dirty["coupon_discount"].unique()))]


# The above rows have the incorrect `coupon_discount` applied to the `original_price`, the actual discounts are presented in the `discount_applied` column, which are also of acceptable value. As such, the `coupon_discount` for these rows will be set to equal `discount_applied`.
# 
# First we will pass the above filter to a mask variable

# In[ ]:


mask = (dirty["coupon_discount"] != dirty["discount_applied"]) & (dirty["discount_applied"].isin(dirty["coupon_discount"].unique()))


# In[ ]:


# set coupon_discount equal to discount_applied
dirty.loc[mask, "coupon_discount"] = dirty.loc[mask, "discount_applied"]


# Check again

# In[ ]:


dirty[(dirty["coupon_discount"] != dirty["discount_applied"]) & (dirty["discount_applied"].isin(dirty["coupon_discount"].unique()))]


# The `coupon_discount` column is now free of errors.
# 
# To find the errors in `order_price`, a new column called `new_price_disc` will be created which is equal to `new_price` after applying `coupon_discount`. This new column will be compared to `order_price_round` to discover the errors.

# In[ ]:


dirty["new_price_disc"] = round(dirty["new_price"] * (1 - (dirty["coupon_discount"] / 100)), 2)
dirty.head()


# In[ ]:


# compare absolute value of difference to two decimal places to account for rounding discrepancies
dirty[abs(dirty["order_price"] - dirty["new_price_disc"]) > 0.01]


# The above rows have incorrect `order_price` and will be updated to equal the correct `new_price_disc`.
# 
# Set a mask variable equal to the above filter

# In[ ]:


mask = abs(dirty["order_price"] - dirty["new_price_disc"]) > 0.01


# In[ ]:


# set order_price equal to new_price_disc
dirty.loc[mask, "order_price"] = dirty.loc[mask, "new_price_disc"]


# Check again

# In[ ]:


dirty[abs(dirty["order_price"] - dirty["new_price_disc"]) > 0.01]


# The `order_price` column is now correct

# Remove unneeded columns from `dirty`

# In[ ]:


dirty.drop(columns = ["shopping_list", "original_price", "new_price", "order_price_round", "discount_applied", "new_price_disc"], inplace = True)
dirty.head()


# ### 3.8 shortest_distance_to_customer
# 
# We will now validate the `shortest_distance_to_customer`. To accomplish this, the shortest distance from the restaurant to the customer will be calculated using the **Dijkstra algorithm**, this algorithm will be implemented using the `networkx` library.
# 
# But first, let's prepare our data to contain all the required information. The information required are the nodes for both restaurants and customers.
# 
# #### 3.8.1 Incorporate Restaurant and Customer Nodes
# 
# For the restaurants, we can obtain the nodes using the `rest_data_match` dataframe created in **section 3.2** reproduced here

# In[ ]:


rest_data_match


# We will drop all the unneeded columns

# In[ ]:


rest_data_nodes = rest_data_match.drop(columns = ["restaurant_name", "menu_items", "reviews_list", "lat", "lon"])
rest_data_nodes


# Now we will merge `rest_nodes` into our `dirty` data

# In[ ]:


dirty = pd.merge(dirty, rest_data_nodes, on = "restaurant_id", how = "left")
dirty.head()


# For the customers, first we need to load the `nodes` file which contains all the nodes corresponding to latitudes and longitudes

# In[ ]:


nodes_file = pd.read_csv("./supplementary_data/nodes.csv")
nodes_file.head()


# Now we will only retrieve the nodes with latitudes and longitude that match the customers' latitudes and longitudes present in the `dirty` data. However, we must ensure that the decimal places for the latitudes and longitudes have the same decimal places (7) so that they can be matched

# In[ ]:


dirty["customer_lat"] = dirty["customer_lat"].round(7)
dirty["customer_lon"] = dirty["customer_lon"].round(7)
nodes_file["lat"] = nodes_file["lat"].round(7)
nodes_file["lon"] = nodes_file["lon"].round(7)


# Retrieve the nodes with latitudes and longitudes that simultaneously exist in `dirty` `customer_lat` and `customer_lon`

# In[ ]:


nodes_match = nodes_file[(nodes_file["lat"].isin(dirty["customer_lat"])) & (nodes_file["lon"].isin(dirty["customer_lon"]))]
nodes_match.head()


# We can see that the number of unique nodes equals the number of unique customers

# In[ ]:


print("unique nodes:    ", len(nodes_match["node"].unique()))
print("unique customers:", len(dirty["customer_id"].unique()))


# Now we will merge these nodes to the `dirty` dataframe

# In[ ]:


dirty = pd.merge(dirty, nodes_match, left_on = ["customer_lat", "customer_lon"], right_on = ["lat", "lon"], how = "left")
dirty.head()


# We will drop the `lat` and `lon` columns because we no longer need them

# In[ ]:


dirty.drop(columns = ["lat", "lon"], inplace = True)
dirty.head()


# We will rename the `rest_nodes` and `node` column to make them clearer regarding who they refer to

# In[ ]:


dirty.rename(columns = {"rest_nodes":"restaurant_node", "node":"customer_node"}, inplace = True)
dirty.head()


# Now we can compute the shortest distance for each order using the Dijkstra Algorithm

# #### 3.8.2 Compute Shortest Distance
# 
# To compute the shortest distance using the Dijkstra algorithm, we will use the `networkx` package. First we need to build the graph of nodes and edges

# In[ ]:


# create empty graph
G = nx.Graph()

# add nodes by passing in the node column from the nodes_file
G.add_nodes_from(nodes_file["node"])


# The next step in building the graph is adding the edges. First, we need to import the edges data

# In[ ]:


edges = pd.read_csv("./supplementary_data/edges.csv", usecols = ["u", "v", "distance(m)"])
edges.head()


# Each row in the edges dataframe contains two nodes and the distance in metres between them. To add the edges with their respective weight of distance to the graph, the edges and the weight need to be in the correct format of 
# 
# `(u, v, {'weight': distance}`
# 
# Therefore, we will apply a lambda function to each row in the edges dataframe that will output the required format to a new `edge` column so that it can be passed to the edges of the graph. This is done by indexing each row represented by the `x` with the name of the column required

# In[ ]:


edges["edge"] = edges.apply(lambda x: (int(x["u"]), int(x["v"]), {"weight": x["distance(m)"]}), axis = 1)
edges.head()


# We will now add the edges to the graph

# In[ ]:


G.add_edges_from(edges["edge"])


# For each row in the `dirty` dataframe, we will calculate the shortest distance using the function `single_source_dijkstra` from `networkx`. For each row, the source and target nodes inputted to the function will be the `restaurant_node` and `customer_node` respectively. The calculated distance will be outputted to a new column `dijkstra_distance`.
# 
# The `single_source_dijkstra` function returns a tuple where the first element is the shortest distance calculated, and the second element is a list of the edges in that shortest distance, therefore for each row, we will retrieve only the shortest distance by indexing the result of the function by [0]. The result will also be rounded to the nearest metre

# In[ ]:


dirty["dijkstra_distance"] = dirty.apply(
    lambda x: round(nx.single_source_dijkstra(G, x["restaurant_node"], x["customer_node"])[0]), axis = 1)


# In[ ]:


dirty[dirty["dijkstra_distance"] != dirty["shortest_distance_to_customer"]]


# The above rows have incorrect `shortest_distance_to_customer` values.

# Since only 12 rows from 500 are incorrect, we can be confident that our method is correct as the algorithm correctly calculated the great majority of rows at 488 rows.
# 
# We will now set the incorrect values of `shortest_distance_to_customer` equal to the correct values in `dijkstra_distance`.
# 
# We will first set a mask variable equal to the filter above

# In[ ]:


mask = dirty["dijkstra_distance"] != dirty["shortest_distance_to_customer"]


# In[ ]:


dirty.loc[mask, "shortest_distance_to_customer"] = dirty.loc[mask, "dijkstra_distance"]


# In[ ]:


dirty[dirty["dijkstra_distance"] != dirty["shortest_distance_to_customer"]]


# The values under `shortest_distance_to_customer` are now correct.
# 
# We will now remove the columns no longer needed

# In[ ]:


dirty.drop(columns = ["restaurant_node", "customer_node", "dijkstra_distance"], inplace = True)
dirty.head()


# ### 3.9 travel_time_minutes and carrier_vehicle
# 
# `travel_time_minutes` depends on both `shortest_distance_to_customer` and the speed according to `carrier_vehicle`. The following speeds apply according to `carrier_vehicle`:
# 
# - Motorbike: 30KM/hour
# - Bike: 12KM/hour
# - Car: 25KM/hour
# 
# The formula to calculate `travel_time_minutes` is `shortest_distance_to_customer` divided by speed according to `carrier_vehicle`
# 
# However, it is important to note and adjust for 2 things:
# 
# 1. The speed is in KM but `shortest_distance_to_customer` is in metres, therefore `shortest_distance_to_customer` needs to be converted to KM by dividing by 1,000
# 2. The speed is per hour but `travel_time_minutes` is in minutes, therefore a new speed figure per minute will be calculated by dividing by 60
# 
# First, we will check the values in `carrier_vehicle` are one of **Motorbike, Bike, Car**

# In[ ]:


list(dirty["carrier_vehicle"].unique())


# Some rows contain the vehicle type in lowercase, these will now be capitalised using `str.capitalize`

# In[ ]:


dirty["carrier_vehicle"] = dirty["carrier_vehicle"].str.capitalize()


# Let's check that all the values in `carrier_vehicle` are now capitalized

# In[ ]:


list(dirty["carrier_vehicle"].unique())


# A dictionary called `vehicle_speed` will be created that holds each vehicle and it's corresponding speed per minute

# In[ ]:


# divide speed by 60 to convert to per minute figure
vehicle_speeds = {"Motorbike": (30/60), "Bike": (12/60), "Car": (25/60)}


# A new column called `new_travel_time` will be created which will hold the output of a lambda function where for each row, the time to the nearest minute according to the vehicle used will be calculated by dividing the `shortest_distance_to_customer` by the speed according to the `carrier_vehicle` retrieved from the `vehicle_speeds` dictionary. We divide `shortest_distance_to_customer` by 1,000 to convert to metres

# In[ ]:


dirty["new_travel_time"] = dirty.apply(lambda x: round((x["shortest_distance_to_customer"] / 1000) / 
                                                       vehicle_speeds[x["carrier_vehicle"]]), axis = 1)


# We will extract the incorrect rows by comparing `travel_time_minutes` to `new_travel_time`

# In[ ]:


dirty[dirty["travel_time_minutes"] != dirty["new_travel_time"]]


# The above rows have differing `travel_time_minutes` and `new_travel_time`. However, before assuming the error is in `travel_time_minutes` itself, we must first check if the error is in fact in `carrier_vehicle`, this will be done by generating the time figure if the other vehicles were used other than the one specified in the row. If we find that the given `travel_time_minutes` equals the travel time if another vehicle is used, then this indicates that the error is in the `carrier_vehicle` column rather than the `travel_time_minutes` column.

# We will create three new columns which calculates the time according to the different vehicle speeds:
# 
# - `motor_time`: time calculated by using motorbike speed
# - `bike_time`: time calculated by using bike speed
# - `car_time`: time calculated by using car speed

# In[ ]:


dirty["motor_time"] = ((dirty["shortest_distance_to_customer"]/1000) / vehicle_speeds["Motorbike"]).apply(round)
dirty["bike_time"] = ((dirty["shortest_distance_to_customer"]/1000) / vehicle_speeds["Bike"]).apply(round)
dirty["car_time"] = ((dirty["shortest_distance_to_customer"]/1000) / vehicle_speeds["Car"]).apply(round)


# Now we will locate the rows where `travel_time_minutes` does not equal `new_travel_time` **AND** `travel_time_minutes` equals one of the other times if another vehicle was used

# In[ ]:


dirty[(dirty["travel_time_minutes"] != dirty["new_travel_time"]) &
      ((dirty["travel_time_minutes"] == dirty["motor_time"]) | (dirty["travel_time_minutes"] == dirty["bike_time"]) | (dirty["travel_time_minutes"] == dirty["car_time"]))]


# We can see for the above rows, `travel_time_minutes` does not equal `new_travel_time` but it does equal one of the other times if another vehicle was used. We will update the `carrier_vehicle` for those rows by creating an `update_vehicle` function which will compare `travel_time_minutes` to the different vehicles times, if the two times match, then `carrier_vehicle` will be updated to the vehicle which produced that time

# In[ ]:


def update_vehicle(row):
    # obtain necessary values
    original_time = row["travel_time_minutes"]
    motor_time = row["motor_time"]
    bike_time = row["bike_time"]
    car_time = row["car_time"]
    
    # update carrier_vehicle according to time matched
    if original_time == motor_time:
        return "Motorbike"
        
    elif original_time == bike_time:
        return "Bike"
        
    else:
        return "Car"


# Pass the above filter into a mask variable

# In[ ]:


mask = (dirty["travel_time_minutes"] != dirty["new_travel_time"]) & ((dirty["travel_time_minutes"] == dirty["motor_time"]) | (dirty["travel_time_minutes"] == dirty["bike_time"]) | (dirty["travel_time_minutes"] == dirty["car_time"]))


# Alter the values of `carrier_vehicle` in the rows above by applying `update_vehicle` function

# In[ ]:


dirty.loc[mask, "carrier_vehicle"] = dirty.loc[mask].apply(update_vehicle, axis = 1)


# We will recalculate `new_travel_time` with the updated `carrier_vehicle` values

# In[ ]:


dirty["new_travel_time"] = dirty.apply(lambda x: round((x["shortest_distance_to_customer"] / 1000) / 
                                                       vehicle_speeds[x["carrier_vehicle"]]), axis = 1)
dirty.head()


# Check that error is not due to incorrect `carrier_vehicle`

# In[ ]:


dirty[(dirty["travel_time_minutes"] != dirty["new_travel_time"]) &
      ((dirty["travel_time_minutes"] == dirty["motor_time"]) | (dirty["travel_time_minutes"] == dirty["bike_time"]) | (dirty["travel_time_minutes"] == dirty["car_time"]))]


# The remaining errors are solely due to incorrect `travel_time_minutes`. We will extract the incorrect rows by comparing `travel_time_minutes` to `new_travel_time` again

# In[ ]:


dirty[dirty["travel_time_minutes"] != dirty["new_travel_time"]]


# We will now set the incorrect values of `travel_time_minutes` equal to the correct values in `new_travel_time`.
# 
# We will first set a mask variable equal to the filter above

# In[ ]:


mask = dirty["travel_time_minutes"] != dirty["new_travel_time"]


# In[ ]:


dirty.loc[mask, "travel_time_minutes"] = dirty.loc[mask, "new_travel_time"]


# In[ ]:


dirty[dirty["travel_time_minutes"] != dirty["new_travel_time"]]


# The values under `travel_time_minutes` are now correct.
# 
# Since only 17 rows from 500 were changed, we can be confident that our method is correct as the function correctly calculated the greaty majority of rows at 483 rows.
# 
# We will now remove the additional columns created as they are no longer needed

# In[ ]:


dirty.drop(columns = ["new_travel_time", "motor_time", "bike_time", "car_time"], inplace = True)
dirty.head()


# All the errors in the `dirty` file are now fixed

# In[ ]:


# save updated dirty csv
#dirty.to_csv("dirty_solution.csv", index = False)


# ## 4. Missing Data
# 
# The second part of this report examines the `missing` file where some columns contain missing values, and then imputing those missing values
# 
# ### 4.1 Data Exploration
# 
# A preliminary exploration will be performed.

# In[ ]:


missing.shape


# The data consists of 500 observations of 18 columns. Let's take a look at the five first and last rows of the data

# In[ ]:


missing


# Let's examine the types of variables in our data

# In[ ]:


missing.info()


# Let's discover where the missing values are

# In[ ]:


missing.isnull().sum()


# We can see that `shortest_distance_to_customer` and `delivery_charges` are both missing in 150 observations, while `restaurant_rating` is missing totally. The missing values will be imputed for each of these variables

# ### 4.2 shortest_distance_to_customer
# 
# `shortest_distance_to_customer` will be calculated in a similar fashion to how it was calculated in section **3.8**.
# 
# First we will merge the restaurant and customer nodes to the `missing` dataframe. For the restaurants, will use `rest_data_nodes`, which was derived using the `dirty` dataframe, from **3.8.1** reproduced here

# In[ ]:


rest_data_nodes


# We will merge `rest_nodes` into our `missing` data

# In[ ]:


missing = pd.merge(missing, rest_data_nodes, on = "restaurant_id", how = "left")
missing.head()


# For the customers, we will use the `nodes_file` created in **3.8.1** reproduced here

# In[ ]:


nodes_file.head()


# Now we will only retrieve the nodes with latitudes and longitude that match the customers' latitudes and longitudes present in the `missing` data. However, we must ensure that the decimal places for the latitudes and longitudes have the same decimal places (7) so that they can be matched

# In[ ]:


missing["customer_lat"] = missing["customer_lat"].round(7)
missing["customer_lon"] = missing["customer_lon"].round(7)


# Retrive the nodes with latitudes and longitudes that simultaneously exist in `missing` `customer_lat` and `customer_lon`

# In[ ]:


nodes_match_missing = nodes_file[(nodes_file["lat"].isin(missing["customer_lat"])) & (nodes_file["lon"].isin(missing["customer_lon"]))]
nodes_match_missing.head()


# We can see that the number of unique nodes equal the number of unique customers

# In[ ]:


print("unique nodes    :", len(nodes_match_missing["node"].unique()))
print("unique customers:", len(missing["customer_id"].unique()))


# Now we will merge these nodes to the `missing` dataframe

# In[ ]:


missing = pd.merge(missing, nodes_match_missing, left_on = ["customer_lat", "customer_lon"], right_on = ["lat", "lon"], how = "left")
missing.head()


# We will drop the `lat` and `lon` columns because we no longer need them

# In[ ]:


missing.drop(columns = ["lat", "lon"], inplace = True)
missing.head()


# We will rename the `rest_nodes` and `node` column to make them clearer regarding who they refer to

# In[ ]:


missing.rename(columns = {"rest_nodes":"restaurant_node", "node":"customer_node"}, inplace = True)
missing.head()


# Now we will calculate the shortest distance for the missing values using `nx.single_source_dijkstra` from the `networkx` package as in section **3.8.2**

# In[ ]:


missing.loc[missing["shortest_distance_to_customer"].isnull(), "shortest_distance_to_customer"] = missing[missing["shortest_distance_to_customer"].isnull()].apply(
    lambda x: round(nx.single_source_dijkstra(G, x["restaurant_node"], x["customer_node"])[0]), axis = 1)


# In[ ]:


missing.head()


# In[ ]:


missing["shortest_distance_to_customer"].isnull().sum()


# We can now see that there are no more missing values in the `shortest_distance_to_customer` column.
# 
# We will drop the `restaurant_node` and `customer_node` columns since we don't need them anymore

# In[ ]:


missing.drop(columns = ["restaurant_node", "customer_node"], inplace = True)
missing.head()


# ### 4.3 restaurant_rating
# 
# Since we established the values of `restaurant_id` in `missing` are the same as those in `dirty`, the `restaurant_rating` for each restaurant will be the same. We will create a series `dirty_ratings` that groups each `restaurant_id` with its `restaurant_rating` from `dirty`

# In[ ]:


dirty_ratings = dirty.groupby("restaurant_id")["restaurant_rating"].unique()
dirty_ratings


# In[ ]:


# convert restaurant_rating to float
dirty_ratings = dirty_ratings.apply(lambda x: x[0])
dirty_ratings


# `restaurant_rating` in `missing` will now be imputed by applying a lambda function that retrieves the `restaurant_rating` from `dirty_ratings` through indexing it using the value of `restaurant_id` from each row in `missing`

# In[ ]:


missing["restaurant_rating"] = missing.apply(lambda x: dirty_ratings[x["restaurant_id"]], axis = 1)
missing.head()


# In[ ]:


missing["restaurant_rating"].isnull().sum()


# In[ ]:


missing.groupby("restaurant_id")["restaurant_rating"].unique()


# We can now see that there are no more missing values in the `restaurant_rating` column

# ### 4.4 delivery_charges
# 
# The `delivery_charges` for each row will be calculated using a linear model fitted using the following variables:
# 
# - travel_time_minutes
# - is_peak_time
# - is_weekend
# - restaurant_rating
# 
# The linear model will be trained using the `dirty` dataframe

# In[ ]:


# initialize
lm_impute = LinearRegression()

# fit the linear model by using the explanatory columns to predict the delivery_charges column
lm_impute.fit(dirty[["travel_time_minutes", "is_peak_time", "is_weekend", "restaurant_rating"]], dirty["delivery_charges"])


# We will test the model by running it on the `missing` data that contains the `delivery_charges`, we will call this new dataframe `test_df` and will only contain the required columns

# In[ ]:


test_df = missing[["travel_time_minutes", "is_peak_time", "is_weekend", "restaurant_rating", "delivery_charges"]].copy()

# drop rows with empty delivery_charges
test_df.dropna(subset = ["delivery_charges"], inplace = True)
test_df.head()


# In[ ]:


test_df.shape


# We can test how well our model performs by computing the r-squared when the fitted `delivery_charges` are compared to the true `delivery_charges` in `test_df`

# In[ ]:


# set the test explanatory data
X_test = test_df.drop(columns = "delivery_charges")


# The r-squared computed is

# In[ ]:


print('r-squared for this model = ', lm_impute.score(X_test,test_df["delivery_charges"]))


# This is an extremely high r-squared, which indicates that our model is excellent for predicting `delivery_charges` (maybe too excellent and indicates over-fitting, but we will not investigate this further in this report).
# 
# We will impute the missing `delivery_charges` in the `missing` dataframe using the model we fit

# In[ ]:


# impute missing values by running the model on the rows that contain the missing values and the explanatory columns only
missing.loc[missing["delivery_charges"].isnull(), "delivery_charges"] = lm_impute.predict(missing.loc[missing["delivery_charges"].isnull(),
                                                                                                      [column for column in test_df.columns if column != "delivery_charges"]])


# In[ ]:


missing.head()


# In[ ]:


missing["delivery_charges"].isnull().sum()


# We can now see that there are no more missing values in the `delivery_charges` column

# In[ ]:


missing.isnull().sum()


# All the missing values have been imputed

# In[ ]:


# save updated missing csv
#missing.to_csv("missing_solution.csv", index = False)


# ## 5. Outliers
# 
# The `outliers` file will now be investigated. Outliers are restricted to the `delivery_charges` column. This section will investigate these outliers and will remove them from the data.
# 
# To discover the outliers, a new column of delivery charges will be created using the linear model developed in the previous section. Then, the difference between the original `delivery_charges` and the new ones will be calculated to generate the residuals. A boxplot of the residuals will then be generated to identify the outliers and remove them.
# 
# ### 5.1 Explore delivery_charges
# 
# `delivery_charges` column in `outliers` will be investigated

# In[ ]:


outliers.head()


# In[ ]:


outliers.shape


# The `outliers` data is missing the `restaurant_rating` column present in the previous files. This column will have to be added to this data as it is needed when calculating the new delivery charges using the linear model.
# 
# Let's examine the summary statistics of `delivery_charges`

# In[ ]:


outliers["delivery_charges"].describe()


# Graphical summaries of `delivery_charges` will be created

# In[ ]:


outliers["delivery_charges"].hist()


# In[ ]:


outliers.boxplot(column='delivery_charges')


# Judging by the summary statistics and graphs above, outliers seem to be present at values above 8. However, further investigation is required

# ### 5.2 Add restaurant_rating
# 
# `delivery_charges` is calculated based on four variables:
# 
# - travel_time_minutes
# - is_peak_time
# - is_weekend
# - restaurant_rating
# 
# Since `restaurant_rating` is missing, it will be added to the data. We will use the `dirty_ratings` from **section 4.3** to create the `restaurant_rating` column for each `restaurnat_id` in `outliers`. `dirty_ratings` will be reproduced here for reference

# In[ ]:


dirty_ratings


# In[ ]:


# compute missing restaurant_rating as was done in section 4.3
outliers["restaurant_rating"] = outliers.apply(lambda x: dirty_ratings[x["restaurant_id"]], axis = 1)
outliers.head()


# The `restaurant_rating` column has now been included in `outliers`

# ### 5.3 Identify Outliers
# 
# The column `new_charges` will be created using the linear model from **section 4.4**

# In[ ]:


# predict using explanatory columns
outliers["new_charges"] = lm_impute.predict(outliers[[column for column in test_df.columns if column != "delivery_charges"]])
outliers.head()


# We will now calculate the `residual` by calculating the difference between `delivery_charges` and `new_charges`

# In[ ]:


outliers["residual"] = outliers["delivery_charges"] - outliers["new_charges"]
outliers.head()


# Now we will create a boxplot of the residuals to locate the outliers

# In[ ]:


outliers.boxplot(column='residual')


# Judging by the boxplot, any observations where the value of the `residual` is greater than 1 or less than -1 is considered an outlier.
# 
# Let's extract those observations

# In[ ]:


out = outliers[abs(outliers["residual"]) > 1]
out


# In[ ]:


out.shape


# We have 30 outlier observations.
# 
# We will now drop these observations

# In[ ]:


outliers.drop(index = out.index, inplace = True)
outliers.shape


# We have successfully removed the outliers.
# 
# We'll remove the columns we no longer need

# In[ ]:


outliers.drop(columns = ["restaurant_rating", "new_charges", "residual"], inplace = True)
outliers.head()


# In[ ]:


# save updated outliers csv
#outliers.to_csv("outlier_solution.csv", index = False)


# ## 6. Conclusion
# 
# This assignment displayed various methods and techniques relating to data wrangling and data cleansing. For all three tasks, the `pandas` library was the main workhorse and backbone of all the work performed.
# 
# Regarding the `dirty` file, errors were discovered by either examining the columns in relation to themselves only by identifying the values were acceptable, as was the case with the `date` column, or by cross referencing the columns with other columns to verify their validity, such as with the `is_weekend` and `order_price` columns. Some errors were fixed in a straightforward fashion such as ensuring that `peak_time` were in fact inside the peak times according to the `time` column. While others required a more complicated procedure such as using the `SentimentIntensityAnalyzer` from the `nltk` library to generate the restaurant ratings or using the `Dijkstra algorithm` from the `networkx` library to compute the `shortest_distance_to_customer`.
# 
# Concerning the `missing` file, methods of imputing the missing values included using prior-learned knowledge as was done when imputing `restaurant_rating`, as well as the use of linear regression to create a model that predicted the missing `delivery_charges`.
# 
# Finally for the `outliers` file, the linear regression model was again used to calculated predicted values, which were then used to calculate the residual for each observation, and using a boxplot of the residuals, the outliers were successfully identified and removed.

# ## 7. References
# 
# - Aric A. Hagberg, Daniel A. Schult and Pieter J. Swart, “Exploring network structure, dynamics, and function using NetworkX”, in Proceedings of the 7th Python in Science Conference (SciPy2008), Gäel Varoquaux, Travis Vaught, and Jarrod Millman (Eds), (Pasadena, CA USA), pp. 11–15, Aug 2008
# - Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.
# - Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).
# - J. D. Hunter, "Matplotlib: A 2D Graphics Environment", Computing in Science & Engineering, vol. 9, no. 3, pp. 90-95, 2007.
# - McKinney, W., & others. (2010). Data structures for statistical computing in python. In Proceedings of the 9th Python in Science Conference (Vol. 445, pp. 51–56)
# - Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
# - Van Rossum, G. (2020). The Python Library Reference, release 3.8.2. Python Software Foundation.
