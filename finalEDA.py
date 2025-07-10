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
# ### 5. Univariate Analysis
# ####      * Histograms, KDE plots
# ### 6. Multivariate Analysis
# ####      * Correlation Matrix
# ####      * Pairplot / Scatter plots
# ### 7.  Feature Engineering
# ####      * Price per SqFt
# ####      * House Age
# ####      * Total Bathrooms
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

# In[96]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
import warnings


# In[97]:


warnings.filterwarnings("ignore")
sns.set(style='whitegrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[98]:


print("Step 1 Complete: Imported all essential libraries for analysis and visualization.")


# ### =======================
# ### Step 2: Load the Dataset
# ### =======================
# 

# In[99]:


df = pd.read_csv("housing_data.csv")
print("✅ Step 2 Complete: Loaded dataset with shape:", df.shape)


# ### =======================
# ### Step 3: Initial Overview
# ### =======================

# In[100]:


print(df.info())
print(df.describe().T)


# In[101]:


print("Missing Values per Column:\n", df.isnull().sum()[df.isnull().sum() > 0])


#  ### Step 3 Complete:  Insight: Dataset has 81 columns and 1460 records. Some missing values detected in 'Electrical' and 'GarageYrBlt'.

# ### =======================
# ### Step 4: Handle Missing Values
# ### =======================
# 

# In[102]:


df.dropna(subset=['Electrical'], inplace=True)
df['GarageYrBlt'].fillna(df['GarageYrBlt'].median(), inplace=True)


# ### Step 4 Complete: Handled missing values via dropping or median imputation.

# ### =======================
# ### Step 5: Remove Duplicates
# ### =======================

# In[103]:


initial_shape = df.shape
df.drop_duplicates(inplace=True)
print(f" Step 5 Complete: Removed {initial_shape[0] - df.shape[0]} duplicate rows.")


# ### Step 5 Complete: Removed all duplicate values.

# ### =======================
# ### Step 6: Outlier Detection and Handling (IQR Method With Comparison)
# ### =======================

# In[104]:


print(type(df))


# In[105]:


df_original = df.copy()  # before applying outlier removal


# ### Box Plot Before Outlier Remove

# In[106]:


plt.figure(figsize=(7, 5))
sns.boxplot(df_original['SalePrice'])
plt.title("Boxplot of SalePrice (Before Outlier Removal)")
plt.show()


# ### Remove outliers
# 

# In[107]:


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

# In[108]:


plt.figure(figsize=(7, 5))
sns.boxplot(df['SalePrice'])  
plt.title("For SalePrice (After Outlier Removal)")
plt.show()


# ### Step 6 Completed:
# ### Insight: SalePrice outliers removed to reduce skewness and stabilize the model input.

# ### =======================
# ## Univariate Analysis
# ### =======================

# ## Continuous Feature Analysis  (SalePrice, GrLivArea, TotalBsmtSF, GarageArea)
# ### In this step we will apply univariate analysis for every continuous feature one by one.

# ### 1.  For  Sale Prize  

# In[109]:


plt.figure(figsize=(10, 5))
sns.histplot(df['SalePrice'], kde=True, color='tan', edgecolor='black')
plt.title("Distribution of SalePrice")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")


# In[110]:


plt.figure(figsize=(10, 5))
sns.boxplot(df['SalePrice'], color='#9fc8c8')
plt.title("Boxplot of SalePrice")
plt.show()


# ###  Skewness and Kurtosis For SalePrice

# In[111]:


print("Skewness:", skew(df['SalePrice']))
print("Kurtosis:", kurtosis(df['SalePrice']))


# ### 2.  For GrLivArea

# In[112]:


plt.figure(figsize=(10, 4))
sns.histplot(df['GrLivArea'], kde=True, color='#2066a8', bins=30, edgecolor='black')
plt.title("Distribution of GrLivArea")
plt.xlabel("GrLivArea")
plt.ylabel("Frequency")
plt.show()


# In[113]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df['GrLivArea'], color='yellowgreen')
plt.title("Boxplot of GrLivArea")
plt.show()


# ### Skewness and Kurtosis For GrLivArea

# In[114]:


print("Skewness:", skew(df['GrLivArea']))
print("Kurtosis:", kurtosis(df['GrLivArea']))


# ### 3. For TotalBsmtSF
# 

# In[115]:


plt.figure(figsize=(10, 5))
sns.histplot(df['TotalBsmtSF'], kde=True, bins=30, color='#c46666', edgecolor='black')
plt.title("Distribution of TotalBsmtSF")
plt.xlabel("TotalBsmtSF")
plt.ylabel("Frequency")
plt.show()


# In[116]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df['TotalBsmtSF'], color='#8cc5e3')
plt.title("Boxplot of TotalBsmtSF")
plt.show()



# ### Skewness and Kurtosis For TotalBsmtSF

# In[117]:


print("Skewness:", skew(df['TotalBsmtSF']))
print("Kurtosis:", kurtosis(df['TotalBsmtSF']))


#  ### 4. For GarageArea

# In[118]:


plt.figure(figsize=(10, 5))
sns.histplot(df['GarageArea'], kde=True, color='mediumseagreen', edgecolor='black')
plt.title("Distribution of GarageArea")
plt.xlabel("GarageArea")
plt.ylabel("Frequency")
plt.show()


# In[169]:


plt.figure(figsize=(10, 5))
sns.boxplot(x=df['GarageArea'], color='#9fc8c8')
plt.title("Boxplot of GarageArea")
plt.show()


# ### Skewness and Kurtosis For GarageArea

# In[120]:


print("Skewness:", skew(df['GarageArea']))
print("Kurtosis:", kurtosis(df['GarageArea']))


# #### As we have more than 40+ categorical features like Exterior1st, Functional, GarageQual etc. so we will not perform all the features. We will work with high-impact or frequently analyzed categorical variables MSZoning, MSSubClass, Neighborhood, HouseStyle, Street, Alley, LotShape etc.
# #### We will perform Filter only those with <10 unique values for cleaner visuals and also perform numeric columns with fewer than 10 unique values becuase Some numeric columns are actually categorical in nature (like MSSubClass, OverallQual, Fireplaces)
# 
# ##  Categorical Feature Analysis (MSZoning, MSSubClass, Street, Alley, LotShape, Neighborhood, HouseStyle)
# 
# ### In this step we will apply Univariate Analysis  for every categorical variable one by one.

# ### ===============================
# ### 1.  MSZoning (Zoning classification)
# ### ===============================
# 

# In[121]:


plt.figure(figsize=(6, 4))
sns.countplot(x='MSZoning', data=df, palette='deep')
plt.title("Bar Chart: MSZoning")
plt.ylabel("Count")
plt.show()


# In[122]:


## colors = plt.cm.Paired.colors
colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3', '#a6d854']
df['MSZoning'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6), colors=colors)
plt.title("Pie Chart: MSZoning")
plt.ylabel("")
plt.show()


# ###  Insight: 'RL' is the dominant zoning class.

# ### ===============================
# ### 2.  MSSubClass 
# ### ===============================
# 

# In[123]:


plt.figure(figsize=(10, 4))
sns.countplot(x='MSSubClass', data=df, order=df['MSSubClass'].value_counts().index)
plt.title("Bar Chart: MSSubClass")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# In[124]:


df['MSSubClass'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(8, 8))
plt.title("Pie Chart: MSSubClass")
plt.ylabel("")
plt.show()


# ### Insight: Most homes are in class SC20 (1-story) and SC60 (2-story newer)

# ### -------------------------------
# ### 3. Street
# ### -------------------------------

# In[125]:


plt.figure(figsize=(6, 4))
sns.countplot(x='Street', data=df)
plt.title("Bar Chart: Street")
plt.ylabel("Count")
plt.show()


# In[126]:


df['Street'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(5, 5))
plt.title("Pie Chart: Street")
plt.ylabel("")
plt.show()


# ### Insight: Vast majority of properties are on paved streets.

# ### -------------------------------
# ### 4. Alley
# ### -------------------------------

# In[127]:


plt.figure(figsize=(6, 4))
sns.countplot(x='Alley', data=df)
plt.title("Bar Chart: Alley Access")
plt.ylabel("Count")
plt.show()


# In[128]:


df['Alley'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6))
plt.title("Pie Chart: Alley Access")
plt.ylabel("")
plt.show()


# ### Insight: Most properties have 'No alley access' or 'None'.

# ### -------------------------------
# ### 5.LotShape
# ### -------------------------------

# In[129]:


plt.figure(figsize=(8, 4))
sns.countplot(x='LotShape', data=df)
plt.title("Bar Chart: LotShape")
plt.ylabel("Count")
plt.show()


# In[130]:


df['LotShape'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(6, 6))
plt.title("Pie Chart: LotShape")
plt.ylabel("")


# ###  Insight: 'Regular' (Reg) lot shapes dominate the dataset.

# ### -------------------------------
# ### 6.Neighborhood
# ### -------------------------------
# 

# In[131]:


plt.figure(figsize=(16, 6))
sns.countplot(x='Neighborhood', data=df, order=df['Neighborhood'].value_counts().index)
plt.title("Bar Chart: Neighborhood")
plt.xticks(rotation=45)
plt.ylabel("Count")
plt.show()


# In[132]:


top_neighborhoods = df['Neighborhood'].value_counts().nlargest(8)
other_count = df['Neighborhood'].value_counts().sum() - top_neighborhoods.sum()
pie_data = top_neighborhoods.append(pd.Series({'Other': other_count}))


# In[133]:


plt.figure(figsize=(7, 7))
pie_data.plot.pie(autopct='%1.1f%%', startangle=90)
plt.title("Pie Chart: Top 8 Neighborhoods")
plt.ylabel("")
plt.show()


# ### Insight: NAmes and CollgCr are the most common neighborhoods.

# ### -------------------------------
# ### 7.HouseStyle
# ### -------------------------------
# 

# In[134]:


plt.figure(figsize=(10, 5))
sns.countplot(x='HouseStyle', data=df)
plt.title("Bar Chart: HouseStyle")
plt.ylabel("Count")
plt.show()



# In[135]:


df['HouseStyle'].value_counts().plot.pie(autopct='%1.1f%%', figsize=(7, 7))
plt.title("Pie Chart: HouseStyle")
plt.ylabel("")
plt.show()



# ### Insight: 1Story and 2Story homes dominate the dataset.

# ## Categorical columns with < 10 unique values

# ### ===============================
# ### 1. Low Cardinality Categorical Features (<10 unique)
# ### ===============================

# In[136]:


cat_cols = df.select_dtypes(include='object').columns
low_card_cat_cols = [col for col in cat_cols if df[col].nunique() < 10]

print(" Low Cardinality Categorical Columns:", low_card_cat_cols)


# In[137]:


for col in low_card_cat_cols:
    print(f"\n Categorical Feature: {col}")
    
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, data=df, order=df[col].value_counts().index)
    plt.title(f"Bar Chart: {col}")
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.show()
    
   


# In[138]:


plt.figure(figsize=(8, 8))
df[col].value_counts().plot.pie(
    autopct='%1.1f%%',
    colors=sns.color_palette('deep'),
    startangle=90
)


# In[139]:


plt.title(f"Pie Chart: {col}")
plt.ylabel("")
plt.show()
print(f" Insight: Most common category is '{df[col].mode()[0]}'")


# ###  Insight: Most common category is 'Normal'.

# ### ===============================
# ### 2. Numeric Columns with < 10 Unique Values (Likely Categorical)
# ### ===============================

# In[140]:


num_cols = df.select_dtypes(include=['int64', 'float64']).columns
low_card_num_cols = [col for col in num_cols if df[col].nunique() < 10]

print("\n Numeric Columns with < 10 Unique Values (Likely Categorical):", low_card_num_cols)


# In[141]:


for col in low_card_num_cols:
    print(f"\n📊 Numeric-Categorical Feature: {col}")
    
    plt.figure(figsize=(10, 4))
    sns.countplot(x=col, data=df, order=sorted(df[col].dropna().unique()))
    plt.title(f"Bar Chart: {col}")
    plt.ylabel("Count")
    plt.show()


# In[142]:


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

# ## Bivariate Analysis

# ### Numerical Feature vs Target (Scatterplot)

# In[170]:


plt.figure(figsize=(8, 5))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df, color='#d8a6a6')
plt.title("GrLivArea vs SalePrice")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price ($)")
plt.show()


# In[144]:


print(" Insight: Homes with more above-ground living area (GrLivArea) generally sell for higher prices.")


# ## Insight : Homes with more above-ground living area (GrLivArea) generally sell for higher prices.

# ### Categorical Feature vs Target (Boxplot)

# In[145]:


plt.figure(figsize=(10, 5))
sns.boxplot(x='OverallQual', y='SalePrice', data=df)
plt.title("OverallQual vs SalePrice")
plt.xlabel("Overall Quality Rating")
plt.ylabel("Sale Price ($)")
plt.show()


# In[146]:


print(" Insight: Higher overall quality ratings (OverallQual) are clearly associated with higher SalePrice.")


# ###  Insight for categorical feature: Insight: Higher overall quality ratings (OverallQual) are clearly associated with higher SalePrice.

# ### Grouped Bar Plot: Neighborhood average price

# In[164]:


plt.figure(figsize=(10, 8))
df.groupby('Neighborhood')['SalePrice'].mean().sort_values().plot(kind='barh', color='cadetblue')
plt.title("Average SalePrice by Neighborhood")
plt.xlabel("Average SalePrice")
plt.ylabel("Neighborhood")
plt.show()


# In[148]:


print(" Insight: Neighborhood plays a significant role in housing price. Some areas consistently command higher prices.")


#  ### Insight for grouped categorical : Neighborhood plays a significant role in housing price. Some areas consistently command higher prices.

# ### =======================
# ### Step 9: Multivariate Analysis
# ### =======================

# In[149]:


plt.figure(figsize=(18,8))
sns.heatmap(df.corr(), cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()


# In[168]:


plt.figure(figsize=(10,5))
sns.scatterplot(data=df, x='GrLivArea', y='SalePrice', color='#c46666')
plt.title("GrLivArea vs SalePrice")
plt.show()


# ### Top correlations

# In[151]:


correlation_matrix = df.corr()
top_corr = correlation_matrix['SalePrice'].abs().sort_values(ascending=False)[1:11]
print("Top 10 features most correlated with SalePrice:\n", top_corr)


# In[152]:


top_corr.plot(kind='bar', figsize=(10,5), color='teal')
plt.title("Top 10 Correlated Features with SalePrice")
plt.grid(True)
plt.show()


# ### Drop highly correlated features

# In[153]:


threshold = 0.9
upper = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
df.drop(columns=to_drop, inplace=True)
print(f" Insight: Dropped features with high multicollinearity: {to_drop}")
print(" Insight: GrLivArea and GarageCars are highly correlated with SalePrice.")


# ### Insight: Dropped features with high multicollinearity: [ ]
#  ### Insight: GrLivArea and GarageCars are highly correlated with SalePrice.

# ### =======================
# ### Step 10: Feature Engineering
# ### =======================

# In[154]:


df['PricePerSqFt'] = df['SalePrice'] / df['GrLivArea']
df['HouseAge'] = df['YrSold'] - df['YearBuilt']
df['TotalBathrooms'] = (df['FullBath'] + df['HalfBath'] * 0.5 +
                        df['BsmtFullBath'] + df['BsmtHalfBath'] * 0.5)

print("SalePrice Mean:", df['SalePrice'].mean())
print("SalePrice Median:", df['SalePrice'].median())
print("SalePrice Mode:", df['SalePrice'].mode()[0])
print(" Insight: Created new features: PricePerSqFt, HouseAge, TotalBathrooms.")
print(" Insight: Price per square foot and number of bathrooms help refine value prediction.")


# ### Insight: Created new features: PricePerSqFt, HouseAge, TotalBathrooms.
#  ### Insight: Price per square foot and number of bathrooms help refine value prediction.
# 

# ### =======================
# ### Step 11: Size and Feature Impact
# ### =======================

# In[155]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='BedroomAbvGr', y='SalePrice')
plt.title("Bedrooms vs SalePrice")
plt.show()


# In[156]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='TotalBathrooms', y='SalePrice')
plt.title("TotalBathrooms vs SalePrice")
plt.show()
print(" Insight: More bathrooms generally lead to higher prices. Bedrooms have less clear impact.")


# ### =======================
# ### Step 12: Market Trends & Customer Preferences
# ### =======================

# In[157]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='YrSold', y='SalePrice')
plt.title("Year Sold vs SalePrice")
plt.show()


# In[158]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='MoSold', y='SalePrice')
plt.title("Month Sold vs SalePrice")
plt.xticks(rotation=45)
plt.show()


# In[159]:


plt.figure(figsize=(10,5))
sns.boxplot(data=df, x='CentralAir', y='SalePrice')
plt.title("Central Air vs SalePrice")
plt.show()



# In[160]:


plt.figure(figsize=(10,5))
sns.scatterplot(data=df[df['PoolArea'] > 0], x='PoolArea', y='SalePrice')
plt.title("Pool Area vs SalePrice")
plt.show()



# In[172]:


plt.figure(figsize=(10,5))
sns.scatterplot(data=df, x='GarageArea', y='SalePrice' , color='#a00000')
plt.title("Garage Area vs SalePrice")
plt.show()



# In[162]:


print(" Insight: Amenities like central air, garages, and pools positively influence sale price.")
print(" Insight: Sale year/month slightly affect price; market timing matters.")


# ### Insight: Amenities like central air, garages, and pools positively influence sale price.
#  ### Insight: Sale year/month slightly affect price; market timing matters.

# ### =======================
# ### Final Summary and Conclusion
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




