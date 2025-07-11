#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis (EDA) for Real Estate Housing Dataset
# 
# ### =======================
# 
# ### 1. Import Required Libraries
# ### 2. Load the Dataset
# ### 3. Initial Data Overview
# ### 4. Data Cleaning
# ####     * Handle missing values
# ####     * Remove duplicates
# ####     * Fix anomalies
# ### 5. Univariate Analysis & Bivariate Analysis
# ####      * Histograms, KDE plots
# ### 6. Multivariate Analysis
# ####      * Correlation Matrix
# ####      * Pairplot / Scatter plots
# ####      *  Discrete & Continuous features
# ####      *  Scatter plots, Scatter plots with line 
# ####      *  Categorical vs SalePrice Boxplots
# ####      *  Pairplot & Correlation Heatmap
# ### 7.  Feature Engineering
# ####      * Price per SqFt
# ####      * House Age
# ####      * Total Bathrooms
# ####      *  Linear Regression
# ####      *  Kmeans Custering
# ####      *  Cluster Visualization
# ### 8.  Size Impact
# ####      *  Bedrooms, Bathrooms, Square Footage vs Price
# ### 9.  Market Trends
# ####      * Time series analysis with YrSold and MoSold
# ### 10. Customer Preferences & Amenities
# ####      * Amenities vs Sale Price
# #### Clustering (if required)
# ### 11.  Recommendations & Insights Summary
# 
# 

# ### Step 1: Import Libraries
# ### =======================
# 

# In[144]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# In[7]:


warnings.filterwarnings("ignore")
sns.set(style='whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


print("Step 1 Complete: Imported all essential libraries for analysis and visualization.")


# ### =======================
# ### Step 2: Load the Dataset
# ### =======================
# 

# In[9]:


df = pd.read_csv("housing_data.csv")
print("✅ Step 2 Complete: Loaded dataset with shape:", df.shape)


# ### =======================
# ### Step 3: Initial Overview
# ### =======================

# In[10]:


print(df.info())
print(df.describe().T)


# In[11]:


print("Missing Values per Column:\n", df.isnull().sum()[df.isnull().sum() > 0])


#  ### Step 3 Complete:  Insight: Dataset has 81 columns and 1460 records. Some missing values detected in 'Electrical' and 'GarageYrBlt'.

# ### =======================
# ### Step 4: Handle Missing Values
# ### =======================
# 

# In[12]:


df.dropna(subset=['Electrical'], inplace=True)
df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)


# ### Step 4 Complete: Handled missing values via dropping or median imputation.

# ### =======================
# ### Step 4.1: Remove Duplicates
# ### =======================

# In[13]:


initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f" Step 5 Complete: Removed {initial_shape[0] - df.shape[0]} duplicate rows.")


# ### Step 4.1 Complete: Removed all duplicate values.

# ### =======================
# ### Step 4.2: Outlier Detection and Handling (IQR Method With Comparison)
# ### =======================

# In[14]:


print(type(df))


# In[15]:


df_original = df.copy()  # before applying outlier removal


# ### Box Plot Before Outlier Remove

# In[16]:


plt.figure(figsize=(7, 5))
sns.boxplot(df_original['SalePrice'])
plt.title("Boxplot of SalePrice (Before Outlier Removal)")
plt.show()


# ### Insight: Grapgh Represneting before outliers removal.

# ### Remove outliers
# 

# In[17]:


def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]


before = df.shape[0]
df = remove_outliers(df, 'SalePrice')
after = df.shape[0]
print(f"✅ Step 6 Complete: Removed {before - after} outliers from 'SalePrice' using IQR method.")


# ### Insight: Removed outliers from saleprice using IQR method.

# ### : Boxplot AFTER outlier removal

# In[18]:


plt.figure(figsize=(7, 5))
sns.boxplot(df['SalePrice'])  
plt.title("For SalePrice (After Outlier Removal)")
plt.show()


# ### Insight: Grapgh Represneting after outliers removal.

# 
# ### Insight: SalePrice outliers removed to reduce skewness and stabilize the model input.

# ### =======================
# ## Step 5: Univariate Analysis
# ### =======================

# ##  step 5.1 Continuous Feature Analysis  (SalePrice, GrLivArea, TotalBsmtSF, GarageArea)
# ### In this step we will apply univariate analysis for every continuous feature one by one.

# ### 1.  For  Sale Prize  

# In[19]:


plt.figure(figsize=(10, 5))
sns.histplot(df['SalePrice'], kde=True, color='tan', edgecolor='black')
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")


# ### Insight: The distribution of SalePrice is right-skewed, indicating that most houses are moderately priced, with fewer high-priced outliers.

# In[20]:


plt.figure(figsize=(10, 5))
sns.boxplot(df['SalePrice'], color='#9fc8c8')
plt.title("Boxplot of SalePrice")
plt.show()


# ### Insight: The boxplot of SalePrice reveals a wide range with several high-value outliers, highlighting price variability in the dataset.

# ###  Skewness and Kurtosis For SalePrice

# In[21]:


print("Skewness:", skew(df['SalePrice']))
print("Kurtosis:", kurtosis(df['SalePrice']))


# ### 2.  For GrLivArea

# In[22]:


plt.figure(figsize=(10, 4))
sns.histplot(df['GrLivArea'], kde=True, color='#2066a8', bins=30, edgecolor='black')
plt.title("Distribution of GrLivArea")
plt.xlabel("GrLivArea")
plt.ylabel("Frequency")
plt.show()


# ### insight: The distribution of GrLivArea is right-skewed, showing that most homes have moderate living areas, with a few very large properties extending the tail.

# In[23]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df['GrLivArea'], color='yellowgreen')
plt.title("Boxplot of GrLivArea")
plt.show()


# ### The boxplot of GrLivArea highlights a concentrated range of typical home sizes with several large outliers, indicating a few unusually spacious homes.

# ### Skewness and Kurtosis For GrLivArea

# In[24]:


print("Skewness:", skew(df['GrLivArea']))
print("Kurtosis:", kurtosis(df['GrLivArea']))


# ### 3. For TotalBsmtSF
# 

# In[25]:


plt.figure(figsize=(10, 5))
sns.histplot(df['TotalBsmtSF'], kde=True, bins=30, color='#c46666', edgecolor='black')
plt.title("Distribution of TotalBsmtSF")
plt.xlabel("TotalBsmtSF")
plt.ylabel("Frequency")
plt.show()


# ### Insight:The distribution of TotalBsmtSF is right-skewed, showing that most homes have smaller basement areas, with a few having exceptionally large basements.

# In[26]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df['TotalBsmtSF'], color='#8cc5e3')
plt.title("Boxplot of TotalBsmtSF")
plt.show()



# ### Insight: The boxplot of TotalBsmtSF shows a concentrated spread of basement sizes with several high-end outliers, indicating some homes have exceptionally large basements.

# ### Skewness and Kurtosis For TotalBsmtSF

# In[27]:


print("Skewness:", skew(df['TotalBsmtSF']))
print("Kurtosis:", kurtosis(df['TotalBsmtSF']))


#  ### 4. For GarageArea

# In[28]:


plt.figure(figsize=(10, 5))
sns.histplot(df['GarageArea'], kde=True, color='mediumseagreen', edgecolor='black')
plt.title("Distribution of GarageArea")
plt.xlabel("GarageArea")
plt.ylabel("Frequency")
plt.show()


# ### Insight: The distribution of GarageArea is right-skewed, suggesting that most homes have moderately sized garages, with a few properties featuring very large garage spaces.

# In[29]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df['GarageArea'], color='#9fc8c8')
plt.title("Boxplot of GarageArea")
plt.show()


# ### Insight: The boxplot of GarageArea shows a tight central range with several large outliers, indicating that while most garages are standard-sized, some homes have significantly larger garage spaces.

# ### Skewness and Kurtosis For GarageArea

# In[30]:


print("Skewness:", skew(df['GarageArea']))
print("Kurtosis:", kurtosis(df['GarageArea']))


# #### As we have more than 40+ categorical features like Exterior1st, Functional, GarageQual etc. so we will not perform all the features. We will work with high-impact or frequently analyzed categorical variables MSZoning, MSSubClass, Neighborhood, HouseStyle, Street, Alley, LotShape etc.
# #### We will perform Filter only those with <10 unique values for cleaner visuals and also perform numeric columns with fewer than 10 unique values becuase Some numeric columns are actually categorical in nature (like MSSubClass, OverallQual, Fireplaces)
# 
# ##   Step 5.2 Categorical Feature Analysis (MSZoning, MSSubClass, Street, Alley, LotShape, Neighborhood, HouseStyle)
# 
# ### In this step we will apply Univariate Analysis  for every categorical variable one by one.

# ### ===============================
# ### 1.  MSZoning (Zoning classification)
# ### ===============================
# 

# In[31]:


plt.figure(figsize=(6, 4))
sns.countplot(x='MSZoning', data=df, palette='deep')
plt.title("Bar Chart: MSZoning")
plt.ylabel("Count")
plt.show()


# ### Insight: The bar chart of MSZoning shows that the Residential Low Density (RL) zone is the most common, indicating a preference for low-density residential areas in the dataset.

# In[32]:


## colors = plt.cm.Paired.colors
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
df['MSZoning'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6), colors=colors)
plt.title("Pie Chart: MSZoning")
plt.ylabel("")
plt.show()


# ### Insight: The pie chart of MSZoning reveals that RL (Residential Low Density) dominates the dataset, making up the largest proportion of zoning types, followed by RM and others.

# ### ===============================
# ### 2.  MSSubClass 
# ### ===============================
# 

# In[33]:


plt.figure(figsize=(10, 4))
sns.countplot(x='MSSubClass', data=df, order=df['MSSubClass'].value_counts().index)
plt.title("Bar Chart: MSSubClass")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# ### Insight: The bar chart of MSSubClass indicates that 20 (1-STORY 1946+) is the most common building class, reflecting a dominance of modern single-story homes in the dataset.

# In[34]:


df['MSSubClass'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title("Pie Chart: MSSubClass")
plt.ylabel("")
plt.show()


# ### Insight: the pie chart of MSSubClass shows that 1-story homes built in 1946 and later (20) constitute the largest share of the dataset, with other building classes contributing smaller proportions.Most homes are in class SC20 (1-story) and SC60 (2-story newer)

# ### -------------------------------
# ### 3. Street
# ### -------------------------------

# In[35]:


plt.figure(figsize=(6, 4))
sns.countplot(x='Street', data=df)
plt.title("Bar Chart: Street")
plt.ylabel("Count")
plt.show()


# ### Insight: The bar chart of Street shows that the vast majority of homes are located on paved streets (Pave), with very few on gravel roads (Grvl).

# In[36]:


df['Street'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(5, 5))
plt.title("Pie Chart: Street")
plt.ylabel("")
plt.show()


# ### Insight: The pie chart of Street reveals that over 99% of properties are on paved streets, indicating that unpaved roads are extremely rare in the dataset.

# ### -------------------------------
# ### 4. Alley
# ### -------------------------------

# In[37]:


plt.figure(figsize=(6, 4))
sns.countplot(x='Alley', data=df)
plt.title("Bar Chart: Alley Access")
plt.ylabel("Count")
plt.show()


# ### insight: The bar chart of Alley shows that most properties have missing alley data, indicating that alley access is uncommon or often not recorded in the dataset.

# In[38]:


df['Alley'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6))
plt.title("Pie Chart: Alley Access")
plt.ylabel("")
plt.show()


# ### insight: The pie chart of Alley shows that among the recorded values, Gravel (Grvl) access is slightly more common than Paved (Pave), but overall, alley access is rare, with most entries missing. or can say that Most properties have 'No alley access' or 'None'.

# ### -------------------------------
# ### 5.LotShape
# ### -------------------------------

# In[39]:


plt.figure(figsize=(8, 4))
sns.countplot(x='LotShape', data=df)
plt.title("Bar Chart: LotShape")
plt.ylabel("Count")
plt.show()


# ### insight: The bar chart of LotShape indicates that regular-shaped lots (Reg) are the most common, followed by slightly irregular ones, suggesting a general preference for uniform lot shapes in the dataset.

# In[40]:


df['LotShape'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6))
plt.title("Pie Chart: LotShape")
plt.ylabel("")


# ### Insight: The pie chart of LotShape reveals that regular lots (Reg) make up the majority, while other shapes like slightly irregular (IR1) and more irregular shapes are less common, highlighting a preference for standard lot configurations.

# ### -------------------------------
# ### 6.Neighborhood
# ### -------------------------------
# 

# In[41]:


plt.figure(figsize=(16, 6))
sns.countplot(x='Neighborhood', data=df, order=df['Neighborhood'].value_counts().index)
plt.title("Bar Chart: Neighborhood")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.show()


# ### insight: The bar chart of Neighborhood shows that 'NAmes' is the most represented neighborhood, followed by others like 'CollgCr', 'OldTown', and 'Edwards', indicating these are the most common residential areas in the dataset.

# In[42]:


top_neighborhoods = df['Neighborhood'].value_counts().nlargest(8)
other_count = df['Neighborhood'].value_counts().sum() - top_neighborhoods.sum()
pie_data = top_neighborhoods.append(pd.Series({'Other': other_count}))


# ### insight: here we are  grouping all other less frequent neighborhoods into an "Other" category.
# 
# 

# In[43]:


plt.figure(figsize=(7, 7))
pie_data.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Pie Chart: Top 8 Neighborhoods")
plt.ylabel("")
plt.show()


# ### Insight: The pie chart of the Top 8 Neighborhoods shows that a few key areas—such as 'NAmes', 'CollgCr', and 'OldTown'—contain the majority of homes and most common neighbourhood, while the remaining neighborhoods collectively form a smaller portion grouped as "Other".

# ### -------------------------------
# ### 7.HouseStyle
# ### -------------------------------
# 

# In[44]:


plt.figure(figsize=(10, 5))
sns.countplot(x='HouseStyle', data=df)
plt.title("Bar Chart: HouseStyle")
plt.ylabel("Count")
plt.show()



# ### Insight: The bar chart of HouseStyle shows that 1-Story and 2-Story homes are the most common types in the dataset, indicating a strong buyer or builder preference for traditional single- and two-story residential layouts.
# 

# In[45]:


df['HouseStyle'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(7, 7))
plt.title("Pie Chart: HouseStyle")
plt.ylabel("")
plt.show()



# ### Insight: The pie chart of HouseStyle reveals that 1-Story homes dominate the dataset, followed by 2-Story and 1.5Fin styles, highlighting a strong preference for simpler, vertically compact residential designs.

# ##  Step 5.3 Categorical columns with < 10 unique values

# ### ===============================
# ### 1. Low Cardinality Categorical Features (<10 unique)
# ### ===============================

# In[46]:


cat_cols = df.select_dtypes(include='object').columns
low_card_cat_cols = [col for col in cat_cols if df[col].nunique() < 10]

print(" Low Cardinality Categorical Columns:", low_card_cat_cols)


# In[47]:


for col in low_card_cat_cols:
    print(f"\n Categorical Feature: {col}")
    
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Bar Chart: {col}")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
    
   


# ### Insight: This loop automatically creates bar charts for low cardinality categorical features (features with few unique values) in your dataset like-
# ### MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition
# 

# In[48]:


plt.figure(figsize=(8, 8))
df[col].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=sns.color_palette('deep'),
    startangle=90
)


# ### Insight: This code generates a pie chart for a categorical feature (col) to show the proportional distribution of its categories:

# #### Most Common Category 

# In[49]:


plt.title(f"Pie Chart: {col}")
plt.ylabel("")
plt.show()
print(f" Insight: Most common category is '{df[col].mode()[0]}'")


# ### Insight: printed insight showing the most common category in the feature which is 'Normal'.
# 
# 

# ### ===============================
# ### Step 5.4 Numeric Columns with < 10 Unique Values (Likely Categorical)
# ### ===============================

# In[50]:


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
low_card_num_cols = [col for col in num_cols if df[col].nunique() < 10]

print("\n Numeric Columns with < 10 Unique Values (Likely Categorical):", low_card_num_cols)


# ### FInd  Numerical Cateogrical Features via Barchart

# In[51]:


for col in low_card_num_cols:
    print(f"\n📊 Numeric-Categorical Feature: {col}")
    
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, data=df, order=sorted(df[col].dropna().unique()))
    plt.title(f"Bar Chart: {col}")
    plt.ylabel("Count")
    plt.show()


# ### Insights: Here we Get all the  Numerical Cateogrical Features Like - 'OverallCond', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'Fireplaces', 'GarageCars', 'PoolArea', 'YrSold'

# ###  Most frequent value in YrSold via Pie Chart

# In[52]:


plt.figure(figsize=(8, 8))
df[col].value_counts().sort_index().plot.pie(
    autopct='%1.1f%%',
    colors=sns.color_palette('coolwarm'),
    startangle=90
)
plt.title(f"Pie Chart: {col}")
plt.ylabel("")
plt.show()

print(f" Insight: Most frequent value in {col} is {df[col].mode()[0]}")


# ### Insight: Most frequent value in SaleCondition is Normal
# 

# ## Step 5.5  Bivariate Analysis

# ### Numerical Feature vs Target (Scatterplot)

# In[53]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df, color='#d8a6a6')
plt.title("GrLivArea vs SalePrice")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price ($)")
plt.show()


# In[54]:


print(" Insight: Homes with more above-ground living area (GrLivArea) generally sell for higher prices.")


# ## Insight : Homes with more above-ground living area (GrLivArea) generally sell for higher prices.

# ### Categorical Feature vs Target (Boxplot)

# In[55]:


plt.figure(figsize=(10, 5))
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.title("OverallQual vs SalePrice")
plt.xlabel("Overall Quality Rating")
plt.ylabel("Sale Price ($)")
plt.show()


# In[56]:


print(" Insight: Higher overall quality ratings (OverallQual) are clearly associated with higher SalePrice.")


# ###  Insight for categorical feature: Insight: Higher overall quality ratings (OverallQual) are clearly associated with higher SalePrice.

# ### Grouped Bar Plot: Neighborhood average price

# In[57]:


plt.figure(figsize=(10, 8))
df.groupby('Neighborhood')['SalePrice'].mean().sort_values().plot(kind='barh', color='cadetblue')
plt.title("Average SalePrice by Neighborhood")
plt.xlabel("Average SalePrice")
plt.ylabel("Neighborhood")
plt.show()


# In[58]:


print(" Insight: Neighborhood plays a significant role in housing price. Some areas consistently command higher prices.")


#  ### Insight for grouped categorical : Neighborhood plays a significant role in housing price. Some areas consistently command higher prices.

# ### Insights for over all analysis:
# #### GrLivArea, GarageCars, KitchenQual, and OverallQual show strong individual relationships with SalePrice.
# 
# #### Neighborhood, CentralAir, and Seasonality (via MoSold) reflect buyer preferences and market behaviors.
# 
# #### Combining numeric and categorical features in bivariate analysis paints a richer story than univariate views alone.

# ### =======================
# ### Step 6: Multivariate Analysis
# ### =======================

# #### ------------------------------------------------
# #### 6.1: Scatter Plot Analysis – Discrete vs Continuous
# #### ------------------------------------------------

# In[121]:


import seaborn as sns
import matplotlib.pyplot as plt
num_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('SalePrice')
discrete_features = [col for col in num_cols if df[col].nunique() < 15]
continuous_features = [col for col in num_cols if df[col].nunique() >= 15]

print("🔹 Discrete Numeric Features:", discrete_features)


# ### Insight:  Here are the List of Discreate Features Value. Now we will see values via scatter plot for all the features.
# #### As we have more a huge numbers of columns in our database so i decided to limit the features. for Discreate Numeric columns we will go with less than 15 unique values.

# ### Strip plot for discrete numeric features

# In[137]:


for col in discrete_features:
    plt.figure(figsize=(7, 4))
    sns.scatterplot(x=df[col], y=df['SalePrice'], color='#3594cc', alpha=0.7, s=60)
    plt.title(f"Scatter Plot: SalePrice vs {col} (Discrete)")
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.grid(True)
    plt.show()
    print(f"✅ {col}: {df[col].nunique()} unique values")


# ### Insight :
# #### OverallQual: 10 unique values
# #### OverallCond: 9 unique values
# ####  BsmtFullBath: 4 unique values
#  #### BsmtHalfBath: 3 unique values
# #### FullBath: 4 unique values
# #### HalfBath: 3 unique values
# #### BedroomAbvGr: 8 unique values
# #### KitchenAbvGr: 4 unique values
# #### TotRmsAbvGrd: 12 unique values
# #### Fireplaces: 4 unique values
# #### GarageCars: 5 unique values
#  #### PoolArea: 7 unique values
# #### YrSold: 5 unique values
# #### TotalBathrooms: 10 unique values

# In[123]:


print(" Continuous Numeric Features:", continuous_features)


# ### Insight:  Here are the List of Continuous Features Value. Now we will see values via  plot for all the features.

# ### Scatter plot for continuous numeric features

# In[132]:


for col in continuous_features:
    plt.figure(figsize=(9, 4))
    sns.scatterplot(x=col, y='SalePrice', data=df, alpha=0.6 , color='#c46666')
    plt.title(f"Scatter Plot: SalePrice vs {col} (Continuous)")
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.show()
    print(f" {col}: {df[col].nunique()} unique values")


# ### Insights:
# #### LotFrontage: 108 unique values
# #### LotArea: 1016 unique values
# ####  YearBuilt: 111 unique values
#  #### YearRemodAdd: 61 unique values
# #### MasVnrArea: 297 unique values
# ####  BsmtFinSF1: 598 unique values
# ####  BsmtFinSF2: 143 unique values
# #### BsmtUnfSF: 751 unique values
# #### TotalBsmtSF: 675 unique values
# #### 1stFlrSF: 709 unique values
# #### 2ndFlrSF: 395 unique values
#  #### LowQualFinSF: 23 unique values
# #### GrLivArea: 810 unique values
# #### GarageYrBlt: 97 unique values
# #### GarageArea: 411 unique values
# #### WoodDeckSF: 255 unique values
# #### OpenPorchSF: 193 unique values
# #### EnclosedPorch: 118 unique values
# #### 3SsnPorch: 18 unique values
# ####  ScreenPorch: 74 unique values
# ####   MiscVal: 21 unique values
# ####   PricePerSqFt: 1372 unique values
# ####   HouseAge: 121 unique values
# ####  SalePriceLog: 605 unique values
# ####  GarageScore: 454 unique values
# ####  PredictedPrice: 1304 unique values
# ####   Residual: 1396 unique values
# ####  TotalSF: 909 unique values
# ####  price_per_sqft: 1372 unique values
# 
# 

# ### ------------------------------------------------
# ### Step 6.2: Scatter Plots with Line Trends (selected features)
# ### ------------------------------------------------

# In[78]:


selected_trend_features = ['OverallQual', 'Fireplaces', 'GrLivArea', 'TotalBsmtSF']


# In[ ]:





# In[138]:


for col in selected_trend_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=col, y='SalePrice', data=df, alpha=0.4)

    trend_data = df.groupby(col)['SalePrice'].mean().reset_index()
    sns.lineplot(x=trend_data[col], y=trend_data['SalePrice'], color='red', marker='o')

    plt.title(f"SalePrice vs {col} with Trend Line")
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.grid(True)
    plt.show()
    print(f" {col}: Scatter + trend line plotted")


# ### Insight: This loop creates scatter plots with red trend lines for every feature in selected_trend_features, helping  visually assess how each variable affects SalePrice.

# ####  ------------------------------------------------
# ### Step 6.3: Categorical vs SalePrice (Barplots)
# #### ------------------------------------------------

# In[139]:


cat_cols = df.select_dtypes(include='object').columns
print(" Categorical Features:", list(cat_cols))


# In[141]:


cat_cols = df.select_dtypes(include='object').columns
print(" Categorical Features:", list(cat_cols))

for col in cat_cols:
    if df[col].nunique() < 15:
        plt.figure(figsize=(8, 4))
        sns.barplot(x=col, y='SalePrice', data=df, estimator=np.mean, palette='coolwarm')
        plt.title(f"Average SalePrice by {col}")
        plt.xticks(rotation=45)
        plt.ylabel("Mean SalePrice")
        plt.grid(True)
        plt.show()
        print(f"✅ {col}: {df[col].nunique()} unique categories")
 


# ### Insights: Unique Categories
# 
# #### MSZoning: 5 unique categories
# #### Street: 2 unique categories
# ####  Alley: 3 unique categories
#  #### LotShape: 4 unique categories
# #### LandContour: 4 unique categories
# ####  Utilities: 2 unique categories
# ####  LotConfig: 5 unique categories
# #### LandSlope: 3 unique categories
# #### Condition1: 9 unique categories
# #### Condition2: 8 unique categories
# #### BldgType: 5 unique categories
#  #### HouseStyle: 8 unique categories
# #### RoofStyle: 6 unique categories
# #### RoofMatl: 8 unique categories
# #### ExterQual: 4 unique categories
# #### ExterCond: 5 unique categories
# #### Foundation: 6 unique categories
# #### BsmtQual: 5 unique categories
# #### BsmtCond: 5 unique categories
# ####  BsmtExposure: 4 unique categories
# ####  BsmtFinType1: 7 unique categories
# ####   BsmtFinType2: 7 unique categories
# ####   Heating: 6 unique categories
# ####  HeatingQC: 5 unique categories
# ####  CentralAir: 2 unique categories
# ####  Electrical: 5 unique categories
# ####    KitchenQual: 4 unique categories
# ####  Functional: 7 unique categories
# ####  FireplaceQu: 6 unique categories
# ####  GarageType: 7 unique categories
# ####  GarageFinish: 4 unique categories
# ####  GarageQual: 6 unique categories
# ####  GarageCond: 6 unique categories
# ####  PoolQC: 4 unique categories
# ####  Fence: 5 unique categories
# ####  MiscFeature: 5 unique categories
# ####  MoSold: 12 unique categories
# ####  SaleType: 9 unique categories
# ####  SaleCondition: 6 unique categories
# 
# 
# 
# 
# 
# 

# ### ------------------------------------------------
# ### 9.3: Pairplot & Correlation Heatmap
# ### ------------------------------------------------

# In[180]:


top_corr = df.corr()['SalePrice'].abs().sort_values(ascending=False).head(5).index.tolist()

sns.pairplot(df[top_corr])
plt.suptitle("Pairplot of Top Correlated Features", y=1.02)
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(df[top_corr].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Top 5 Features)")
plt.show()

print(": All multivariate views executed.")


# ### Insight: Multivariate Analysis by generating a pairplot and heatmap using the top 5 most correlated features with SalePrice.
# 
# 

# ### Correlation Heatmap (Correlation Matrix)

# #### As we have Relation between numerical features and sale price.

# In[99]:


import seaborn as sns
import matplotlib.pyplot as plt

numeric_df = df.select_dtypes(include=['int64', 'float64'])


corr_matrix = numeric_df.corr()
saleprice_corr = corr_matrix['SalePrice'].sort_values(ascending=False)


top_corr_features = saleprice_corr[1:16]  # exclude SalePrice itself

plt.figure(figsize=(10, 6))
sns.heatmap(top_corr_features.to_frame(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Top Numerical Features Correlated with SalePrice")
plt.show()


# ### Insight: This code visualizes the top 15 numerical features most correlated with SalePrice using a vertical heatmap.

# In[59]:


plt.figure(figsize=(10,5))
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', color='#c46666')
plt.title("GrLivArea vs SalePrice")
plt.show()


# ### Insight: The scatter plot of GrLivArea vs SalePrice shows a strong positive linear relationship, indicating that homes with more above-ground living area tend to sell for higher prices, though a few high-area outliers with lower prices are also visible.

# ### Top correlations

# In[60]:


correlation_matrix = df.corr()
top_corr = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)[1:11]
print("Top 10 features most correlated with SalePrice:\n", top_corr)


# ### Insight: Top 10 features most correlated with SalePrice:
#  #### OverallQual     0.784435
#  #### GrLivArea       0.661326
# #### GarageCars      0.628061
# #### GarageArea      0.607239
#  #### FullBath        0.577549
#  #### YearBuilt       0.564888
# #### TotalBsmtSF     0.543939
# #### YearRemodAdd    0.541437
# #### 1stFlrSF        0.522960
# #### GarageYrBlt     0.476080

# In[61]:


top_corr.plot(kind='bar', figsize=(10,5), color='teal')
plt.title("Top 10 Correlated Features with SalePrice")
plt.grid(True)
plt.show()


# 
# ### The bar plot of the top 10 features correlated with SalePrice highlights that variables like OverallQual, GrLivArea, and GarageCars have the strongest positive correlations, reinforcing their importance in predicting house prices.

# ### Drop highly correlated features

# In[62]:


threshold = 0.9
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
df.drop(columns=to_drop, inplace=True)
print(f" Insight: Dropped features with high multicollinearity: {to_drop}")
print(" Insight: GrLivArea and GarageCars are highly correlated with SalePrice.")


# ### Insight: Dropped features with high multicollinearity: [ ]
#  ### Insight: GrLivArea and GarageCars are highly correlated with SalePrice.

# ### =======================
# ### Step 7: Feature Engineering
# ### =======================

# ### Step 7.1: Common features

# In[84]:


## PricePerSqFt: Price per square foot of living area
df['PricePerSqFt'] = df['SalePrice'] / df['GrLivArea']
print("✅ Feature: PricePerSqFt — helps compare pricing across different house sizes.")


# ### Average Price Per SqFt by Neighborhood

# In[182]:


import matplotlib.pyplot as plt
import seaborn as sns
price_sqft_avg = df.groupby('Neighborhood')['PricePerSqFt'].mean().sort_values(ascending=False)
plt.figure(figsize=(14, 6))
sns.barplot(x=price_sqft_avg.index, y=price_sqft_avg.values, palette='bright')
plt.title("Average Price Per SqFt by Neighborhood")
plt.ylabel("Price Per SqFt")
plt.xlabel("Neighborhood")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Inisght: The bar chart of PricePerSqFt by neighborhood reveals that certain areas like 'NridgHt' and 'StoneBr' command higher price per square foot, indicating their premium market status.

# In[86]:


## HouseAge: Age of house at time of sale
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
print(" Feature: HouseAge  — helps assess depreciation or vintage premium.")


# ### Average HouseAge by Neighborhood

# In[183]:


import matplotlib.pyplot as plt
import seaborn as sns
avg_age = df.groupby('Neighborhood')['HouseAge'].mean().sort_values()
plt.figure(figsize=(14, 6))
sns.barplot(x=avg_age.index, y=avg_age.values, palette='coolwarm')
plt.title("Average House Age by Neighborhood")
plt.ylabel("Average House Age (Years)")
plt.xlabel("Neighborhood")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Insight: The bar chart of HouseAge by neighborhood reveals that some areas feature newer constructions, while others have older, more established homes, reflecting variation in development timelines across neighborhoods.

# In[88]:


## TotalBathrooms: Combined full and half bathrooms (including basement)
df['TotalBathrooms'] = (df['FullBath'] + df['HalfBath'] * 0.5 +
                        df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5)
print(" Feature: TotalBathrooms — reflects total usable bathroom capacity.")


# ### Average SalePrice by TotalBathrooms

# In[187]:


import matplotlib.pyplot as plt
import seaborn as sns
bath_price_avg = df.groupby('TotalBathrooms')['SalePrice'].mean().reset_index()
plt.figure(figsize=(11, 4))
sns.barplot(x='TotalBathrooms', y='SalePrice', data=bath_price_avg, palette='crest')
plt.title("Average SalePrice by Total Bathrooms")
plt.xlabel("Total Bathrooms")
plt.ylabel("Average SalePrice")
plt.tight_layout()
plt.show()


# ### Insight: The bar chart of TotalBathrooms vs SalePrice shows a clear upward trend, indicating that homes with more bathrooms generally command higher sale prices, making it a key value-adding feature.

# ### Step 7.2: Garage Feature Function

# In[91]:


##  Garage Feature — derived using a function
def add_garage_features(data):
    data['HasGarage'] = (data['GarageArea'] > 0).astype(int)
    data['GarageScore'] = data['GarageCars'] * data['GarageArea']
    data['GarageAge'] = data['YrSold'] - data['GarageYrBlt']
    return data
df = add_garage_features(df)


print("- HasGarage: Binary indicator of garage presence")
print("- GarageScore: Composite score of garage size and capacity")
print("- GarageAge: Indicates how modern the garage is")


# ### Insights
# #### PricePerSqFt	Compares pricing across homes regardless of size
# #### HouseAge	Shows if older homes sell for more/less
# #### TotalBathrooms	Total usable bathroom units (including halves)
# #### HasGarage	Helps distinguish homes with no garage
# #### GarageScore	Combines car capacity & size — quality proxy
# #### GarageAge	Age of garage — older ones may lower value or require upgrades

# In[104]:


# Calculate total square footage
df['TotalSF'] = df['1stFlrSF'] + df['2ndFlrSF'] + df['TotalBsmtSF']

# Calculate price per square foot
df['price_per_sqft'] = df['SalePrice'] / df['GrLivArea']

# Display summary statistics for the new features
print(df[['TotalSF', 'price_per_sqft']].describe())


# ### Step 7.3: Linear Regression as Feature

# In[114]:


# 8. Linear Regression Model
# We predict house prices using multiple linear regression.

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
# Train the model
model = LinearRegression()
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Output results
print("✅ Linear Regression Model Results")
print("R² Score:", round(r2, 3))
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))

# Coefficients
coeff_df = pd.DataFrame(model.coef_, index=X.columns, columns=['Coefficient'])
print("\n📊 Model Coefficients:")
print(coeff_df)


# ### Insights: Linear Regression Model Results
# ### R² Score: 0.778
# ### MAE: 20906.36
# ### RMSE: 27692.08
# 
# ### Model Coefficients:
#  ###                Coefficient
# ### GrLivArea          27.004284
# ### TotalBathrooms  14513.593032
# ### GarageCars      18073.532774
# ### OverallQual     20268.375775

# ### Step 7.4: KMeans Clustering

# In[117]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# Select features
cluster_data = df[['GrLivArea', 'TotalSF', 'OverallQual', 'SalePrice']].dropna()

# Scale the data
scaler = StandardScaler()
scaled = scaler.fit_transform(cluster_data)

# Apply KMeans (e.g., 3 clusters)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled)

print(" KMeans clustering completed and labels added to 'Cluster' column.")


# ### Insight: Features Used for Clustering:
# ### GrLivArea – Above-ground living area
# ### TotalSF – Total square footage (possibly including basement + floors)
# ### OverallQual – Overall material and finish quality
# ### SalePrice – Target variable for pricing
# 
# ### Data Preparation:
# #### Selected features relevant to house value.
# #### Removed rows with missing values.
# 
# ### Standardization:
# #### Used StandardScaler to bring all features to the same scale.
# 
# ### Clustering:
# #### Applied KMeans with 3 clusters (n_clusters=3).
# #### Cluster labels were stored in a new column df['Cluster'].
# 
# 
# ### KMeans clustering grouped houses into 3 distinct segments based on size, quality, and price, enabling deeper insights into market segments and property types.
# 

# ###  Visualize clusters with GrLivArea vs SalePrice

# In[119]:


plt.figure(figsize=(10, 5))
sns.scatterplot(
    x='GrLivArea', y='SalePrice',
    hue='Cluster',
    palette='Set2',
    data=df
)
plt.title("KMeans Clustering: House cluster based on Size, Quality & Price")
plt.xlabel("GrLivArea")
plt.ylabel("SalePrice")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()


# ### Insight: The scatter plot of GrLivArea vs SalePrice colored by KMeans Cluster reveals distinct housing groups, showing that properties naturally segment based on living area, quality, and price, which reflects varying market tiers or buyer profiles.

# ### Step 7.5: Cluster Visualization

# In[96]:


df['PriceCluster'] = df['PriceCluster'].astype('category')
avg_price_by_cluster = df.groupby('PriceCluster')['SalePrice'].mean()
plt.figure(figsize=(7, 4))
sns.barplot(x=avg_price_by_cluster.index, y=avg_price_by_cluster.values, palette='coolwarm')
plt.title("Average SalePrice by PriceCluster")
plt.ylabel("SalePrice")
plt.xlabel("Cluster")
plt.grid(True)
plt.show()


# ### Insight: The bar chart of Average SalePrice by PriceCluster clearly shows that each cluster represents a distinct pricing tier, confirming that the KMeans algorithm effectively segmented the housing market based on value-related attributes.

# ### =======================
# ### Step 8: Skewnwss and Kurtosis
# ### =======================

# In[64]:


import numpy as np
from scipy.stats import skew, kurtosis
print("Skewness of SalePrice:", skew(df['SalePrice']))
print("Kurtosis of SalePrice:", kurtosis(df['SalePrice']))
df['SalePriceLog'] = np.log1p(df['SalePrice'])
print("✅ Step 11 Complete: Applied log transformation to correct right-skewed distribution.")


# ### Insight: We found skewness and kurtosis behalf of Sale Price.
# #### Skewness of SalePrice: 0.6786693738399334
# #### Kurtosis of SalePrice: 0.08403037099969923

# ### =======================
# ### Step 9: Size and Feature Impact
# ### =======================

# ### Step 9.1 Size-related Features — Scatter Plots

# In[148]:


import seaborn as sns
import matplotlib.pyplot as plt

size_features = ['GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'LotArea']
for feature in size_features:
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=df[feature], y=df['SalePrice'], alpha=0.6, color='firebrick')
    plt.title(f"{feature} vs SalePrice")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ### Features -> GrLivArea', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GarageArea', 'LotArea
# ### Insights: 
# #### GrLivArea often has the cleanest linear relationship with SalePrice.
# 
# #### LotArea may show high variance at larger sizes (some points far above or below trend).
# 
# #### Outliers ( very large homes with low price) become easy to detect.
# 
# #### This is ideal for understanding how size variables individually influence price.
# 
# ### GrLivArea vs SalePrice
# #### One of the strongest predictors of SalePrice.
# #### Points align diagonally with a clear upward slope.
# #### Possible outliers: very large homes (e.g., > 4000 sq ft) with lower-than-expected prices.
# 
# ### TotalBsmtSF vs SalePrice
# #### mpact depends on basement finish/utility.
# #### Some large basements may not lead to higher prices.
# #### A few homes with 0 basement area will appear as a flat band near x=0
# 
# ### 1stFlrSF vs SalePrice
# #### Correlated with TotalBsmtSF since larger homes often have bigger basements and 1st floors.
# #### Outliers may exist for large homes with unusually low prices.
# 
# ### 2ndFlrSF vs SalePrice
# #### Trend exists, but data is bimodal: homes with and without a 2nd floor.
# #### Homes with large 2nd floors usually have higher sale prices.
# 
# ### GarageArea vs SalePrice
# #### Some large garage areas with low prices (detached or unfinished garages).
# #### Capped effect: beyond ~1000 sq ft garage, SalePrice doesn’t increase much.
# 

# ### =======================
# ### Step 10: Market Trends & Customer Preferences
# ### =======================

# ### OverallQual vs SalePrice
# 

# In[163]:


plt.figure(figsize=(9,5))
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.title('SalePrice by Overall Quality')
plt.show()


# ### Insight: The boxplot of SalePrice by OverallQual shows a strong upward trend, indicating that homes with higher overall quality ratings consistently achieve higher sale prices.

# ### TotalBathrooms vs SalePrice
# 

# In[165]:


plt.figure(figsize=(9,5))
sns.scatterplot(x=df['TotalBathrooms'], y=df['SalePrice'])
plt.title('Total Bathrooms vs SalePrice')
plt.show()


# ### Insight: The scatter plot of TotalBathrooms vs SalePrice shows a positive correlation, indicating that homes with more bathrooms generally tend to sell for higher prices, though with some variability at higher counts

# ### Living Space Preference

# #### GrLivArea vs SalePrice

# In[177]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Bin GrLivArea into 6 intervals
df['GrLivArea_bin'] = pd.cut(df['GrLivArea'], bins=6)


plt.figure(figsize=(10, 5))
sns.boxplot(x='GrLivArea_bin', y='SalePrice', data=df, palette='Set3', showfliers=False)
plt.title('SalePrice by Binned GrLivArea')
plt.xlabel('GrLivArea Range')
plt.ylabel('SalePrice')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Insight: The boxplot of SalePrice by binned GrLivArea shows a clear upward price trend across larger living area ranges, confirming that bigger homes consistently command higher prices, even when outliers are removed.

# #### GarageCars vs SalePrice

# In[169]:


plt.figure(figsize=(9,5))
sns.boxplot(x='GarageCars', y='SalePrice', data=df)
plt.title('Garage Capacity vs SalePrice')
plt.show()


# ### Insight: The boxplot of GarageCars vs SalePrice shows that homes with more garage spaces generally sell for higher prices, with the most significant price jump observed between 1-car and 2-car garages.

# ### Exterior & Lot Preferences

# #### LotArea vs SalePrice

# In[171]:


import numpy as np
plt.figure(figsize=(9,5))
sns.scatterplot(x=np.log1p(df['LotArea']), y=df['SalePrice'])
plt.title("Log Lot Area vs SalePrice")
plt.show()


# ### Insight: The scatter plot of log-transformed LotArea vs SalePrice reveals a clearer positive relationship, reducing the effect of extreme lot size outliers and showing that larger lots tend to correlate with higher prices.

# #### fence (Categorical) vs SalePrice

# In[172]:


plt.figure(figsize=(9,5))
sns.boxplot(x='Fence', y='SalePrice', data=df)
plt.title('Fence Type vs SalePrice')
plt.xticks(rotation=45)
plt.show()


# ### Insight: The boxplot of Fence type vs SalePrice indicates that homes with better-quality fences (like GdPrv) tend to have higher sale prices, though many entries lack fence data, suggesting it's not a common or consistently valued feature.

# ### Location-Based Preferences

# #### Neighborhood vs SalePrice

# In[176]:


plt.figure(figsize=(12,5))
sns.boxplot(x='Neighborhood', y='SalePrice', data=df)
plt.xticks(rotation=45)
plt.title("Neighborhood vs SalePrice")
plt.show()


# ### Insight: The boxplot of Neighborhood vs SalePrice shows significant variation in housing prices across neighborhoods, with areas like 'NridgHt' and 'StoneBr' commanding substantially higher prices, highlighting the strong influence of location on property value.

# #### Sale Condition / Type

# In[175]:


plt.figure(figsize=(10,5))
sns.boxplot(x='SaleCondition', y='SalePrice', data=df)
plt.title('Sale Condition vs SalePrice')
plt.xticks(rotation=45)
plt.show()


# ### Insight: This boxplot visualizes how different sale conditions affect the distribution of house sale prices.

# In[174]:


## Structure Prefernce
plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='YrSold', y='SalePrice')
plt.title("Year Sold vs SalePrice")
plt.show()


# ### Insight: This boxplot shows the variation in sale prices of houses for each year they were sold.

# In[150]:


import matplotlib.pyplot as plt
import seaborn as sns

# Group by MoSold and compute average SalePrice
monthly_avg = df.groupby('MoSold')['SalePrice'].mean().reset_index()

# Plot the bar plot
plt.figure(figsize=(8, 4))
sns.barplot(data=monthly_avg, x='MoSold', y='SalePrice', palette='viridis')
plt.title("Average SalePrice by Month Sold")
plt.xlabel("Month Sold")
plt.ylabel("Average SalePrice")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ### Insight:This bar plot displays the average house sale price for each month of the year, highlighting seasonal trends in sales.

# In[157]:


import matplotlib.pyplot as plt
import seaborn as sns

# Compute average SalePrice for each CentralAir category
air_avg = df.groupby('CentralAir')['SalePrice'].mean().reset_index()


plt.figure(figsize=(6, 4))
sns.barplot(data=air_avg, x='CentralAir', y='SalePrice', palette='deep')
plt.title("Average SalePrice by Central Air Conditioning")
plt.xlabel("Central Air (Y = Yes, N = No)")
plt.ylabel("Average SalePrice")
plt.tight_layout()
plt.show()



# ### Insight: This bar plot compares the average house sale prices between homes with and without central air conditioning.

# In[153]:


plt.figure(figsize=(8,4))
sns.scatterplot(data=df[df['PoolArea'] > 0], x='PoolArea', y='SalePrice')
plt.title("Pool Area vs SalePrice")
plt.show()



# ### Insight:This scatter plot shows the relationship between pool area size and house sale price for homes that have a pool.

# In[154]:


plt.figure(figsize=(8,4))
sns.scatterplot(data=df, x='GarageArea', y='SalePrice' , color='#a00000')
plt.title("Garage Area vs SalePrice")
plt.show()



# In[ ]:


print(" Insight: Amenities like central air, garages, and pools positively influence sale price.")
print(" Insight: Sale year/month slightly affect price; market timing matters.")


# ### Insight: Amenities like central air, garages, and pools positively influence sale price.
#  ### Insight: Sale year/month slightly affect price; market timing matters.

# ### Market Trends
# #### Yearly Sales Pattern:
# #### SalePrice peaked around 2007, with slight drops afterward — indicating impact of market cycles.
# #### Homes sold in summer months (May–July) consistently show higher average prices, highlighting seasonal buying trends.
# #### Monthly Trends:
# #### June and July are peak months in both price and volume.
# #### Homes sold in winter months (Dec–Feb) typically go for lower prices, possibly due to lower demand.
# #### Trend Insight:
# #### Incorporating YrSold and MoSold into modeling helps capture temporal effects on pricing.
# 
# 
# 
# ### Customer Preferences
# #### Cooling Systems:
# #### Homes with Central Air Conditioning (CentralAir = Y) sell for notably higher prices, reflecting strong buyer preference.
# #### Garage & Parking:
# #### Both Garage Area and Garage Capacity (GarageCars) show clear positive influence on SalePrice.
# #### 2-car garages appear to be the sweet spot for maximizing value.
# 
# ### Amenities & Comfort:
# #### Homes with pools, higher-quality kitchens (KitchenQual), and more bathrooms attract premium prices.
# #### However, some features like pools or fences show high variability — indicating taste-specific appeal.
# 
# ### Structural Preferences:
# #### GrLivArea and OverallQual are among the strongest predictors of SalePrice.
# #### Buyers are willing to pay more for well-maintained and larger homes.
# 
# ### Neighborhood Matters:
# #### Significant price variation across neighborhoods.
# #### Certain localities consistently yield above-average sale prices, revealing location as a key factor in valuation.
# 
# ### Key Takeaways
# #### Customer preferences align with practical features: space, quality, comfort, and cooling.
# #### Seasonality and year of sale significantly affect pricing, suggesting external market forces play a role.
# #### Including variables like CentralAir, GarageCars, OverallQual, and Neighborhood will likely boost model performance.
# #### Feature importance is both logical and data-supported, giving confidence in using them for downstream modeling.
# 
# 

# ### =======================
# ###  Step 11: Final Summary and Conclusion
# ### =======================

# #### Cleaned dataset with missing values and outliers handled.
# ####  SalePrice shows strong correlation with GrLivArea, OverallQual, GarageCars, and TotalBathrooms.
# ####  Engineered useful features like PricePerSqFt and HouseAge.
# ####  Visualizations confirm logical pricing trends and customer preference patterns.

# ###  Final Summary:
# ####  Cleaned dataset with missing values and outliers handled.
# ####  SalePrice shows strong correlation with GrLivArea, OverallQual, GarageCars, and TotalBathrooms.
# ####  Engineered useful features like PricePerSqFt and HouseAge.
# #### Visualizations confirm logical pricing trends and customer preference patterns.
# 
# ### Conclusion:
# #### This EDA provided actionable insights into what affects house prices. Size, quality, and features like air conditioning and garage space are crucial.
# 
# #### These insights will assist in building predictive models and informing pricing strategies in the real estate market.
# 

# In[ ]:




