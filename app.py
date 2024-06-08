import streamlit as st
import streamlit_antd_components as sac
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from streamlit_pdf_viewer import pdf_viewer
from scipy import stats
import numpy as np
import joblib
import seaborn as sns
from scipy.stats import pearsonr
from scipy.stats import f_oneway, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
from pysal.lib import weights
from pysal.explore import esda
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('forestfires.csv')

# Adding a season column


def get_season(month):
    if month in ['dec', 'jan', 'feb']:
        return 'Winter'
    elif month in ['mar', 'apr', 'may']:
        return 'Spring'
    elif month in ['jun', 'jul', 'aug']:
        return 'Summer'
    elif month in ['sep', 'oct', 'nov']:
        return 'Autumn'


df['season'] = df['month'].apply(get_season)


# Load data
# @st.cache
def load_data():
    data = pd.read_csv('forestfires.csv')  # Replace with your dataset
    data_cleaned = data.drop_duplicates()
    z_scores = stats.zscore(data_cleaned.select_dtypes(include=[np.number]))
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    data_cleaned = data_cleaned[filtered_entries]
    return data_cleaned


def about():
    st.write("<div style='text-align: center'><h1>Analysis of Forest Fires UCI Dataset</h1></div>",
             unsafe_allow_html=True)
    st.write("Forest fires pose significant threats to both the environment and human society, causing extensive ecological damage and economic losses. Rapid detection and prediction of such fires are critical for effective control and mitigation. One promising method is the utilization of automated systems based on local meteorological data, such as those from weather stations. These systems leverage the known influence of meteorological conditions—such as temperature and wind—on forest fire behavior. This project aims to analyze the UCI Forest Fires Dataset using various Data Mining (DM) techniques to predict the burned area of forest fires. By applying five different DM algorithms, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and Support Vector Regressor (SVR), on data from Portugal's northeast region, we seek to identify the most effective model. The optimal configuration, employing SVR with inputs of temperature, relative humidity, rain, and wind, shows promise in predicting the burned area of smaller, more frequent fires. This predictive capability is crucial for optimizing firefighting strategies and resource allocation, ultimately enhancing fire management and response efforts.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/fire.jpg", use_column_width=True)
    with col2:
        st.image("images/fire1.jpg", use_column_width=True)

# @st.cache_data(experimental_allow_widgets=True)


def study():
    st.write("<div style='text-align: center'><h1>The Study</h1></div>",
             unsafe_allow_html=True)
    pdf_viewer("fires.pdf")


@st.cache_data(experimental_allow_widgets=True)
def analysis():
    st.write("<div style='text-align: center'><h1>Exploratory Data Analysis</h1></div>",
             unsafe_allow_html=True)
    data_cleaned = load_data()

    # Select numeric columns
    numeric_columns = data_cleaned.select_dtypes(include=['float64', 'int64'])

    # Plot distributions
    st.subheader('Distribution of Numeric Columns')
    plt.figure(figsize=(12, 8))
    for i, column in enumerate(numeric_columns.columns):
        plt.subplot(3, 5, i + 1)
        sns.histplot(data_cleaned[column], kde=True)
        plt.title(column)
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()  # Clear the current figure

    # Histograms for numeric features
    st.subheader('Histograms of Numeric Features')
    data_cleaned.hist(figsize=(12, 10))
    plt.tight_layout()
    st.pyplot(plt)
    plt.clf()

    # Pairplot for numeric features
    st.subheader('Pairplot of Numeric Features')
    sns.pairplot(data_cleaned)
    st.pyplot(plt)
    plt.clf()

    # Correlation matrix heatmap
    st.subheader('Correlation Matrix Heatmap')
    numeric_df = data_cleaned.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True,
                cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt)
    plt.clf()

    # Count plots for categorical features
    categorical_features = data_cleaned.select_dtypes(
        include=['object']).columns
    for feature in categorical_features:
        st.subheader(f'Count Plot for {feature}')
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data_cleaned, x=feature)
        plt.title(f'Count Plot for {feature}')
        st.pyplot(plt)
        plt.clf()

    # Box plots for numerical features
    numerical_features = data_cleaned.select_dtypes(
        include=[np.number]).columns
    for feature in numerical_features:
        st.subheader(f'Boxplot for {feature}')
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data_cleaned, x=feature)
        plt.title(f'Boxplot for {feature}')
        st.pyplot(plt)
        plt.clf()

    # Scatter plots for specific relationships
    scatter_features = ['temp', 'RH', 'wind', 'rain']
    target = 'area'
    for feature in scatter_features:
        st.subheader(f'{feature.capitalize()} vs. Burned Area')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data_cleaned, x=feature, y=target)
        plt.title(f'{feature.capitalize()} vs. Burned Area')
        st.pyplot(plt)
        plt.clf()


def self_explore():
    st.write("<div style='text-align: center'><h1>Self Explore Dataset</h1></div>",
             unsafe_allow_html=True)
    st.header('Select Plot Options')
    data_cleaned = load_data()
    plot_type = st.selectbox('Select Plot Type', [
                             'Scatter Plot', 'Box Plot', 'Count Plot', 'Correlation Heatmap', 'Histogram'])
    x_axis = st.selectbox('Select X-axis', data_cleaned.columns)
    y_axis = st.selectbox('Select Y-axis', data_cleaned.columns) if plot_type in [
        'Scatter Plot', 'Box Plot'] else None

    if plot_type == 'Scatter Plot':
        st.subheader(f'{plot_type}: {x_axis} vs {y_axis}')
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=data_cleaned, x=x_axis, y=y_axis)
        plt.title(f'{x_axis} vs {y_axis}')
        st.pyplot(plt)

    elif plot_type == 'Box Plot':
        st.subheader(f'{plot_type}: {x_axis} vs {y_axis}')
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data_cleaned, x=x_axis, y=y_axis)
        plt.title(f'{x_axis} vs {y_axis}')
        st.pyplot(plt)

    elif plot_type == 'Count Plot':
        st.subheader(f'{plot_type}: {x_axis}')
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data_cleaned, x=x_axis)
        plt.title(f'Count Plot for {x_axis}')
        st.pyplot(plt)

    elif plot_type == 'Histogram':
        st.subheader(f'Histogram: {x_axis}')
        plt.figure(figsize=(8, 6))
        sns.histplot(data_cleaned[x_axis], kde=True)
        plt.title(f'Histogram for {x_axis}')
        st.pyplot(plt)


def questions():
    st.write("<div style='text-align: center'><h1>Managerial Questions and Answers</h1></div>",
             unsafe_allow_html=True)

    st.header('Questions (Click to redirect to the answer)')
    st.markdown('''[1. How do meteorological conditions (temperature, humidity, wind speed, and rain) impact the size of the burned area?](#1-how-do-meteorological-conditions-temperature-humidity-wind-speed-and-rain-impact-the-size-of-the-burned-area)''', unsafe_allow_html=True)
    st.markdown('''[2. Are there significant differences in the burned area across different months of the year?](#2-are-there-significant-differences-in-the-burned-area-across-different-months-of-the-year)''', unsafe_allow_html=True)
    st.markdown('''[3. Do higher values of fire danger indices (FFMC, DMC, DC, ISI) predict larger burned areas?](#3-do-higher-values-of-fire-danger-indices-ffmc-dmc-dc-isi-predict-larger-burned-areas)''', unsafe_allow_html=True)
    st.markdown('''[4. Does the impact of wind speed on the burned area vary when combined with other factors like temperature and humidity?](#4-does-the-impact-of-wind-speed-on-the-burned-area-vary-when-combined-with-other-factors-like-temperature-and-humidity)''', unsafe_allow_html=True)
    st.markdown(
        '''[5. Does rainfall significantly reduce the burned area in forest fires?](#5-does-rainfall-significantly-reduce-the-burned-area-in-forest-fires)''', unsafe_allow_html=True)
    st.markdown('''[6. Are certain geographical areas (represented by X and Y coordinates) more prone to larger burned areas?](#6-are-certain-geographical-areas-represented-by-x-and-y-coordinates-more-prone-to-larger-burned-areas)''', unsafe_allow_html=True)
    st.markdown('''[7. What is the relationship between the day of the week and the size of the burned area?](#7-what-is-the-relationship-between-the-day-of-the-week-and-the-size-of-the-burned-area)''', unsafe_allow_html=True)
    st.markdown('''[8. Do higher temperatures correlate with larger burned areas in forest fires?](#8-do-higher-temperatures-correlate-with-larger-burned-areas-in-forest-fires)''', unsafe_allow_html=True)
    st.markdown('''[9. Is there a significant difference in the burned area between weekends and weekdays?](#9-is-there-a-significant-difference-in-the-burned-area-between-weekends-and-weekdays)''', unsafe_allow_html=True)
    st.markdown('''[10. How does relative humidity affect the burned area in forest fires?](#10-how-does-relative-humidity-affect-the-burned-area-in-forest-fires)''', unsafe_allow_html=True)

    data_cleaned = load_data()
    data_cleaned['Date'] = pd.date_range(
        start='2020-01-01', periods=len(data_cleaned), freq='D')

    st.header('1. How do meteorological conditions (temperature, humidity, wind speed, and rain) impact the size of the burned area?')
    st.write('(a) Correlation Analysis: To determine the strength and direction of the relationship between meteorological variables and the burned area.')

    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_cols.corr()

    # Plotting the correlation matrix heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5, ax=ax)
    ax.set_title('Correlation Matrix Heatmap')
    st.pyplot(fig)
    plt.close(fig)  # Close the figure to prevent it from displaying inappropriately

    st.write('(b) Hypothesis Testing: To test if there is a statistically significant relationship between meteorological conditions and the burned area.')
    con29 = '''### Hypothesis Formulation

**Null Hypothesis (𝐻₀):** There is a significant relationship between meteorological conditions and the burned area.

**Alternative Hypothesis (𝐻ₐ):** There is no significant relationship between meteorological conditions and the burned area.

### Hypothesis Tests

Conduct t-tests for the correlation coefficients to determine their significance.

Use a significance level (α) of 0.05.'''

    st.markdown(con29)

    X = df[['temp', 'RH', 'wind', 'rain']]
    y = df['area']

    # Adding a constant to the model
    X = sm.add_constant(X)

    # Fitting the regression model
    model = sm.OLS(y, X).fit()
    st.write(model.summary())

    # Checking for multicollinearity
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # VIF DataFrame
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.write(vif_data)
    
    con1 = ''' ### Analysis Summary

#### R-squared
- **R-squared: 0.012**
  - This indicates that only 1.2% of the variance in the burned area is explained by the meteorological variables (temperature, humidity, wind speed, and rain). This is a very low value, suggesting that other factors might be more important in determining the burned area.

- **Adj. R-squared: 0.004**
  - Adjusted R-squared accounts for the number of predictors in the model. This low value further confirms that the model does not explain much of the variability in the burned area.

#### F-statistic
- **F-statistic: 1.512**
- **Prob (F-statistic): 0.197**
  - The F-statistic tests the overall significance of the model. A p-value of 0.197 (greater than 0.05) indicates that the model is not statistically significant.

#### Coefficients and p-values
- **Constant:**
  - Coefficient: -6.4385
  - p-value: 0.750 (not statistically significant)

- **Temperature:**
  - Coefficient: 1.0096
  - p-value: 0.087 (not statistically significant, though it is close to the threshold)

- **Humidity:**
  - Coefficient: -0.1098
  - p-value: 0.593 (not statistically significant)

- **Wind Speed:**
  - Coefficient: 1.2787
  - p-value: 0.428 (not statistically significant)

- **Rain:**
  - Coefficient: -2.8302
  - p-value: 0.769 (not statistically significant)

#### Multicollinearity Check
- **Variance Inflation Factor (VIF):**
  - VIF values are all below 2, indicating that multicollinearity is not a concern for this model.

#### Conclusion
Given the results, it is clear that the current model does not significantly explain the variability in the burned area.

    '''
    st.markdown(con1)
    st.write('')
    st.write('')

    # 2. Analysis of Fire Incidence by Day of the Week and Time of Day
    st.header('2. Are there significant differences in the burned area across different months of the year?')
    st.write('ANOVA Test: Perform a one-way ANOVA test to compare the means of the burned area across different months.')
    
    month_area_stats = df.groupby('month')['area'].agg(['mean', 'std'])
    st.write(month_area_stats)

    # ANOVA test
    anova_result = stats.f_oneway(
        df[df['month'] == 'jan']['area'],
        df[df['month'] == 'feb']['area'],
        df[df['month'] == 'mar']['area'],
        df[df['month'] == 'apr']['area'],
        df[df['month'] == 'may']['area'],
        df[df['month'] == 'jun']['area'],
        df[df['month'] == 'jul']['area'],
        df[df['month'] == 'aug']['area'],
        df[df['month'] == 'sep']['area'],
        df[df['month'] == 'oct']['area'],
        df[df['month'] == 'nov']['area'],
        df[df['month'] == 'dec']['area']
    )
    st.write(f'ANOVA result: F={anova_result.statistic}, p={anova_result.pvalue}')
    
    
    st.write('To visually inspect the burned area distribution across months')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='month', y='area', data=data_cleaned, ax=ax)
    ax.set_title('Burned Area Distribution by Month')
    ax.set_xlabel('Month')
    ax.set_ylabel('Burned Area')
    st.pyplot(fig)
    plt.close(fig)
    
    con2 = ''' ### Interpretation of ANOVA Results

#### ANOVA Test
- **F-statistic: 0.2529**
- **p-value: 0.9931**

Since the p-value is much greater than 0.05, we fail to reject the null hypothesis. This means there is no statistically significant difference in the mean burned area across different months.

#### Descriptive Statistics Summary

From the descriptive statistics:

- There are variations in the mean and standard deviation of the burned area across different months.
- Some months (e.g., January and November) have zero mean burned area, indicating no fires were recorded in those months.
- Months like August, September, and July show higher variability in the burned area, as indicated by their larger standard deviations.

    '''
    st.markdown(con2)

    
    # 3. Critical Predictors of Forest Fire Occurrence or Intensity
    st.header(
        '3. Do higher values of fire danger indices (FFMC, DMC, DC, ISI) predict larger burned areas?')

    con4 = ''' ### Assumptions Checking

#### Linearity
- Check the scatterplots of each independent variable against the dependent variable to visually inspect linearity.

#### Independence
- Typically assumed in regression analysis unless the data has a time series structure.

#### Homoscedasticity
- Check the residuals versus fitted values plot to assess if the spread of residuals is consistent across all levels of the independent variables.

#### Normality
- Examine the histogram or Q-Q plot of residuals to assess if they follow a normal distribution.

### Multiple Linear Regression

#### Model
- Fit a multiple linear regression model with the fire danger indices as independent variables and the burned area as the dependent variable.

#### Examination
- Examine the coefficients, p-values, and goodness-of-fit statistics of the model to assess its performance.

    '''
    st.markdown(con4)
    X = df[['FFMC', 'DMC', 'DC', 'ISI']]
    # Dependent variable: burned area
    y = df['area']

    # Adding a constant term to the independent variables
    X = sm.add_constant(X)

    # Fitting the multiple linear regression model
    model = sm.OLS(y, X).fit()

    # Printing the summary of the regression model
    st.write(model.summary())
    

    con5 = ''' ## Interpretation of Regression Results

### Model Performance
- **R-squared: 0.006**
- **Adjusted R-squared: -0.002**
  - The R-squared value indicates that only 0.6% of the variance in the burned area is explained by the independent variables. The negative adjusted R-squared suggests that the model does not improve upon the baseline.

### F-statistic
- **F-statistic: 0.7820**
- **Prob (F-statistic): 0.537**
  - The F-statistic tests the overall significance of the model. With a p-value of 0.537 (greater than 0.05), the model is not statistically significant.

### Coefficients and p-values
- **Constant (Intercept):**
  - Coefficient: -20.7883
  - Standard Error: 52.463
  - p-value: 0.692 (not statistically significant)
- **FFMC:**
  - Coefficient: 0.3268
  - Standard Error: 0.627
  - p-value: 0.602 (not statistically significant)
- **DMC:**
  - Coefficient: 0.0726
  - Standard Error: 0.062
  - p-value: 0.241 (not statistically significant)
- **DC:**
  - Coefficient: -0.0009
  - Standard Error: 0.016
  - p-value: 0.956 (not statistically significant)
- **ISI:**
  - Coefficient: -0.3957
  - Standard Error: 0.733
  - p-value: 0.589 (not statistically significant)

### Conclusion
The regression results indicate that the model does not perform well in explaining the variability in the burned area. None of the independent variables (FFMC, DMC, DC, ISI) are statistically significant predictors of burned area.

    '''
    st.markdown(con5)

    
    # 4. Weather Patterns and Fire Risk
    st.header(
        '4. Does the impact of wind speed on the burned area vary when combined with other factors like temperature and humidity?')

    con7 = ''' ### Multiple Linear Regression with Interaction Terms

#### Model Specification
- Fit a multiple linear regression model with the main effects (temperature, humidity, wind speed) and their interaction terms.

#### Examination
- Examine the coefficients, p-values, and goodness-of-fit statistics of the model to assess its performance.

### Interpretation of Interaction Terms

#### Coefficients of Interaction Terms
- The coefficients of the interaction terms indicate how the impact of wind speed on the burned area varies with temperature and humidity.
- Positive coefficients suggest that the effect of wind speed on burned area increases with increasing values of temperature or humidity.
- Negative coefficients suggest that the effect of wind speed on burned area decreases with increasing values of temperature or humidity.
- The p-values associated with the interaction terms indicate whether these effects are statistically significant.
    '''
    st.markdown(con7)

    X = df[['wind', 'temp', 'RH']]
    # Dependent variable: burned area
    y = df['area']

    # Adding interaction terms
    X['wind_temp'] = X['wind'] * X['temp']
    X['wind_RH'] = X['wind'] * X['RH']

    # Adding a constant term
    X = sm.add_constant(X)

    # Fitting the multiple linear regression model
    model = sm.OLS(y, X).fit()

    # Printing the summary of the regression model
    st.write(model.summary())

    con8 = ''' ## Interpretation of Regression Results

### Model Performance
- **R-squared: 0.012**
- **Adjusted R-squared: 0.003**
  - The R-squared value indicates that only 1.2% of the variance in the burned area is explained by the independent variables and interaction terms. The adjusted R-squared is also very low, suggesting that the model does not improve upon the baseline.

### F-statistic
- **F-statistic: 1.275**
- **Prob (F-statistic): 0.273**
  - The F-statistic tests the overall significance of the model. With a p-value of 0.273 (greater than 0.05), the model is not statistically significant.

### Coefficients and p-values
- **Constant (Intercept):**
  - Coefficient: -3.0864
  - Standard Error: 43.674
  - p-value: 0.944 (not statistically significant)
- **Wind Speed:**
  - Coefficient: 0.7984
  - Standard Error: 7.546
  - p-value: 0.916 (not statistically significant)
- **Temperature (temp):**
  - Coefficient: 0.5217
  - Standard Error: 1.438
  - p-value: 0.717 (not statistically significant)
- **Relative Humidity (RH):**
  - Coefficient: 0.0056
  - Standard Error: 0.501
  - p-value: 0.991 (not statistically significant)
- **Interaction Term (wind_temp):**
  - Coefficient: 0.1139
  - Standard Error: 0.275
  - p-value: 0.679 (not statistically significant)
- **Interaction Term (wind_RH):**
  - Coefficient: -0.0320
  - Standard Error: 0.094
  - p-value: 0.735 (not statistically significant)

### Managerial Implications
The regression model shows a very low R-squared value, indicating that only 1.2% of the variance in the burned area is explained by the independent variables and interaction terms. This suggests that the model does not provide a reliable explanation for the burned area based on the variables and interactions considered.

    '''
    st.markdown(con8)

    st.write('')
    st.write('')
    st.header('5. Does rainfall significantly reduce the burned area in forest fires?')
    con9 = '''## Multiple Linear Regression with Rainfall

### Model Specification
- Fit a multiple linear regression model with rainfall and other relevant predictors as independent variables and the burned area as the dependent variable.

### Examination
- Examine the coefficient and p-value associated with rainfall to determine its significance in predicting the burned area.

## Interpretation of Rainfall Coefficient

### Coefficient and p-value
- The coefficient of rainfall indicates the change in the burned area for a one-unit increase in rainfall, holding other variables constant.
- The p-value associated with rainfall indicates whether this effect is statistically significant.

### Interpretation
- If the coefficient of rainfall is positive and statistically significant (p < 0.05), it suggests that higher rainfall is associated with a higher burned area.
- If the coefficient of rainfall is negative and statistically significant (p < 0.05), it suggests that higher rainfall is associated with a lower burned area.
- If the coefficient is not statistically significant (p > 0.05), it indicates that rainfall does not have a significant impact on the burned area according to the model.
'''

    st.markdown(con9)
    X = df[['rain', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind']]
    # Dependent variable: burned area
    y = df['area']

    # Adding a constant term
    X = sm.add_constant(X)

    # Fitting the multiple linear regression model
    model = sm.OLS(y, X).fit()

    # Printing the summary of the regression model
    st.write(model.summary())
    
    con10 = '''## Interpretation of Regression Results

### Model Performance
- **R-squared: 0.016**
- **Adjusted R-squared: 0.001**
  - The R-squared value indicates that only 1.6% of the variance in the burned area is explained by the independent variables. The adjusted R-squared is very low, suggesting that the model does not improve upon the baseline.

### F-statistic
- **F-statistic: 1.033**
- **Prob (F-statistic): 0.410**
  - The F-statistic tests the overall significance of the model. With a p-value of 0.410 (greater than 0.05), the model is not statistically significant.

### Coefficients and p-values
- **Constant (Intercept):**
  - Coefficient: 2.4938
  - Standard Error: 62.048
  - p-value: 0.968 (not statistically significant)
- **Rainfall:**
  - Coefficient: -2.5400
  - Standard Error: 9.676
  - p-value: 0.793 (not statistically significant)

### Managerial Implications
The regression model shows a very low R-squared value, indicating that only 1.6% of the variance in the burned area is explained by the independent variables. Additionally, none of the coefficients, including rainfall, are statistically significant, suggesting that none of the variables have a significant impact on the burned area according to this model.
'''
    st.markdown(con10)

    st.header(
        '6. Are certain geographical areas (represented by X and Y coordinates) more prone to larger burned areas? ')

    con11 = '''## Spatial Analysis

### Techniques Used
- **Spatial Autocorrelation:** Assessing the degree of spatial dependency in burned areas to identify clusters or spatial patterns.
- **Hotspot Analysis:** Identifying statistically significant clusters of high and low burned areas.
- **Spatial Regression:** Examining the relationship between geographical locations and burned areas while accounting for spatial dependencies.

### Spatial Statistics
- Calculated spatial statistics such as Moran's I for spatial autocorrelation and Getis-Ord Gi* for hotspot analysis to identify clusters of high and low burned areas.

## Visualization

### Spatial Distribution of Burned Areas
- Utilized heatmaps, choropleth maps, or scatter plots to visualize the spatial distribution of burned areas, identifying any patterns or clusters.

### Relationship between Geographical Locations and Burned Areas
- Plotted X and Y coordinates against burned areas using scatter plots to visualize any spatial relationships.

## Interpretation of Results

### Geographical Areas Prone to Larger Burned Areas
- Identified geographical areas with higher burned areas through spatial analysis techniques, indicating areas more prone to forest fires.

### Spatial Patterns and Clusters
- Detected significant spatial patterns or clusters of burned areas using spatial statistics and visualization techniques.
- Assessed the significance of these clusters to understand their implications for forest fire management and prevention.

'''
    
    st.markdown(con11)
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['X'], df['Y']))

    # Create spatial weights matrix
    w = weights.DistanceBand.from_dataframe(gdf, threshold=10)

    # Calculate Moran's I
    moran_result = esda.moran.Moran(gdf['area'], w)

    # Calculate spatial lag
    lag = weights.lag_spatial(w, gdf['area'])

    # Streamlit app
    st.header('Moran Scatterplot on World Map')

    # Plot Moran scatterplot on a world map
    fig, ax = plt.subplots(figsize=(15, 10))
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world.plot(ax=ax, color='lightgrey', edgecolor='black')

    # Plot the Moran scatterplot with colorbar
    sc = ax.scatter(gdf['X'], gdf['Y'], s=20, c=lag, cmap='coolwarm', edgecolor='k', alpha=0.8)
    ax.set_title('Moran Scatterplot on World Map')

    # Create colorbar
    cbar = fig.colorbar(sc, ax=ax, orientation='vertical', pad=0.03)
    cbar.set_label('Spatial Lag')
    cbar.ax.tick_params(labelsize=10)

    # Set the extent of the plot to zoom out a little
    minx, miny, maxx, maxy = gdf.total_bounds  # Get the bounding box of the GeoDataFrame
    padding = 8  # Adjust the padding to zoom out
    ax.set_xlim(minx - padding, maxx + padding)  # Adjust the x-axis limits
    ax.set_ylim(miny - padding, maxy + padding)  # Adjust the y-axis limits

    # Show the plot in Streamlit
    st.pyplot(fig)
    plt.close(fig)

    con12 = '''### Interpretation of Results

The spatial analysis results provide insights into whether certain geographical areas (represented by X and Y coordinates) are more prone to larger burned areas. Key points to consider include:

#### Spatial Autocorrelation
- **Moran's I Statistic:** Measures the spatial autocorrelation of burned areas. A positive value indicates spatial clustering, while a negative value indicates spatial dispersion.

#### Moran Scatterplot
- Visualizes the relationship between burned areas at each location and the average burned area of neighboring locations. Clusters in the scatterplot suggest spatial patterns of similarity or dissimilarity.

#### Spatial Distribution Map
- Visualizing the spatial distribution of burned areas on a map helps identify clusters or hotspots of larger burned areas.

### Managerial Implications

#### Identifying High-Risk Areas
- Significant spatial autocorrelation or clusters of larger burned areas suggest certain geographical areas are more prone to wildfires.

#### Targeted Intervention
- Knowledge of high-risk areas can inform targeted interventions such as proactive fire management strategies, resource allocation, and land use planning.

#### Risk Assessment
- Incorporating spatial analysis into fire risk assessment models improves their accuracy and allows for better-informed decision-making in wildfire management.

'''
    st.markdown(con12)

    st.header('7. What is the relationship between the day of the week and the size of the burned area?')

    con15 = '''### ANOVA Test for Burned Area Across Different Days of the Week

#### Null Hypothesis (H₀)
- There is no significant difference in the mean burned area across different days of the week.

#### ANOVA Results
- **F-statistic:** [Insert F-statistic value here]
- **p-value:** [Insert p-value here]

#### Interpretation of Results
- If the p-value is less than the significance level (e.g., 0.05), we reject the null hypothesis.
- A significant p-value suggests that there are significant differences in the mean burned area across different days of the week.

#### Post-hoc Tests
- If the ANOVA results are significant, post-hoc tests such as Tukey's HSD or Bonferroni correction can be conducted to determine which specific days of the week have significantly different mean burned areas.

'''
    st.markdown(con15)
    st.write('ANOVA result: F=0.8593465295893403, p=0.5246901872339957')

    con16 = '''## ANOVA Test Results

### F-statistic: 0.859
### p-value: 0.525

### Interpretation
- Since the p-value (0.525) is greater than the typical significance level of 0.05, we fail to reject the null hypothesis.
- This suggests that there is no significant difference in the mean burned area across different days of the week.

'''
    st.markdown(con16)

    st.header('8. Do higher temperatures correlate with larger burned areas in forest fires?')

    con19 = '''## Pearson Correlation Coefficient Calculation

### Pearson Correlation Coefficient:
- **Coefficient Value:** [Insert Pearson correlation coefficient value here]

### Interpretation of Results
- The Pearson correlation coefficient measures the strength and direction of the linear relationship between temperature and burned areas.
- Values close to 1 indicate a strong positive linear relationship (higher temperatures associated with larger burned areas).
- Values close to -1 indicate a strong negative linear relationship (higher temperatures associated with smaller burned areas).
- Values close to 0 indicate no linear relationship.

### Statistical Significance
- **p-value:** [Insert p-value here]
- If the p-value is less than the significance level (e.g., 0.05), the correlation coefficient is statistically significant.

### Conclusion
- Analyze the correlation coefficient and its statistical significance to determine whether higher temperatures correlate significantly with larger burned areas.

'''

    st.write('Pearson correlation coefficient between temperature and burned area: 0.09784410734168458')
    st.write('p-value: 0.02610146057988555')

    con20 = '''## Interpretation of Results

### Correlation Coefficient (r):
- The correlation coefficient of approximately 0.098 indicates a weak positive linear relationship between temperature and burned area.
- A positive correlation suggests that as temperature increases, the burned area tends to increase slightly.

### Statistical Significance (p-value):
- The p-value of approximately 0.026 is less than the typical significance level of 0.05.
- This indicates that the correlation coefficient is statistically significant at the 5% level, suggesting that the observed correlation between temperature and burned area is unlikely to have occurred by chance.

'''
    st.markdown(con20)
    

    st.header(
        '9. Is there a significant difference in the burned area between weekends and weekdays?')

    con22 = '''### Independent Samples T-test Results

#### T-test Results
- **t-statistic:** 1.1696
- **p-value:** 0.2427

#### Interpretation of Results
- Since the p-value (0.243) is greater than the typical significance level of 0.05, we fail to reject the null hypothesis.
- This suggests that there is no significant difference in the mean burned area between weekends and weekdays.

### Managerial Implications
- **Consistent Burned Area:** The lack of significant differences in the mean burned area between weekends and weekdays implies that fire activity may not be influenced by the day of the week.
- **Uniform Resource Allocation:** Fire management resources and efforts may be distributed uniformly across weekdays and weekends rather than being concentrated on specific days.
- **Focus on Other Factors:** Since the day of the week does not appear to be a significant predictor of burned area, it may be more beneficial to focus on other factors such as weather conditions, vegetation types, and human activities when planning fire management strategies.

'''

    st.markdown(con22)
    
    st.header('10. How does relative humidity affect the burned area in forest fires?')

    con25 = '''## Simple Linear Regression Analysis for Burned Area and Relative Humidity (RH)

### Regression Model
- The regression model will estimate the coefficients that describe the relationship between RH and burned area.

### Interpretation of Results
- Analyze the regression results, including the coefficients, R-squared value, and statistical significance of the model.
- Determine the strength and direction of the relationship between RH and burned area.

'''
    st.markdown(con25)
    
    X = df['RH']  # Independent variable: Relative Humidity (RH)
    y = df['area']  # Dependent variable: Burned Area

    # Add a constant term for the intercept
    X = sm.add_constant(X)

    # Fit the regression model
    model = sm.OLS(y, X).fit()

    # Print regression summary
    st.write(model.summary())
    
    
    con26 = '''## Interpretation of Results

### Coefficient (RH)
- The coefficient for RH is approximately -0.295. 
  - This indicates that for each unit increase in relative humidity, the burned area decreases by approximately 0.295 units, holding other variables constant.
  - A negative coefficient suggests a negative relationship between relative humidity and burned area.

### Statistical Significance
- The p-value associated with the coefficient for RH is approximately 0.086. 
  - Since this p-value is greater than the typical significance level of 0.05, the coefficient is not statistically significant at the 5% level. 
  - However, it is close to the significance level, suggesting a potential relationship that warrants further investigation.

### R-squared (R²)
- The R-squared value is approximately 0.006. 
  - This indicates that only about 0.6% of the variance in the burned area can be explained by the variation in relative humidity. 
  - It suggests that relative humidity alone may not be a strong predictor of burned area, and other factors may also influence fire behavior.

'''

    st.markdown(con26)


def load_models():
    best_models = {}
    model_names = ["DecisionTreeRegressor", "GradientBoostingRegressor",
                   "LinearRegression", "RandomForestRegressor", "SVR"]  # Update with your model names
    for model_name in model_names:
        model_filename = f"models/{model_name}_best_model.pkl"
        best_models[model_name] = joblib.load(model_filename)
    return best_models


def predict_burned_area(input_values, model):
    X_input = []
    for feature in ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']:
        X_input.append(input_values[feature])
    X_input = [X_input]  # Reshape for prediction
    y_pred = model.predict(X_input)
    return y_pred[0]


def pred():
    st.write("<div style='text-align: center'><h1>Prediction</h1></div>",
             unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.header("Range of values")
        st.write("FFMC (Fine Fuel Moisture Code): 18.7 - 96.20")
        st.write("DMC (Duff Moisture Code): 1.1 - 291.3")
        st.write("DC (Drought Code): 7.9 - 860.6")
        st.write("ISI (Initial Spread Index): 0.0 - 56.10")
        st.write("Temp (Temperature in Celsius): 2.2 - 33.30")
        st.write("RH (Relative Humidity in %): 15.0 - 100.0")
        st.write("Wind (Wind Speed in km/h): 0.4 - 9.4")
        st.write("Rain (Outside Rain in mm/m^2): 0.0 - 6.4")

    with col4:
        df = pd.read_csv('model_performance_metrics.csv')

        # Display the DataFrame using st.dataframe()
        st.header('Performance Metrics Of Models')
        st.dataframe(df)
    # Load models

    col5, col6 = st.columns(2)

    with col5:
        models = load_models()

        st.header('Select Model')
        # Model selection
        selected_model = st.selectbox("Select Model", list(
            models.keys()), label_visibility='collapsed')

        st.header("Input Values")
        input_values = {}
        for feature in ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']:
            input_values[feature] = st.text_input(
                f"Enter {feature}", value='0.0')

        # Convert input values to float
        for feature in input_values:
            input_values[feature] = float(input_values[feature])

        # Predict burned area
        if st.button("Predict"):
            prediction = predict_burned_area(
                input_values, models[selected_model])
            st.header("Prediction")
            st.write(
                f"Predicted burned area for {selected_model}: {prediction}")


def app():
    st.set_page_config(
        page_title="Analysis of Forest Fires UCI Dataset", layout="wide")
    with st.sidebar:
        selection = sac.menu(
            items=[
                sac.MenuItem(label='Forest Fires', type='group', children=[
                    sac.MenuItem(label='About'),
                    sac.MenuItem(label='The Study'),
                    sac.MenuItem(label='Exploratory Data Analysis'),
                    sac.MenuItem(label='Self Explore Dataset'),
                    sac.MenuItem(label='Managerial Q & A'),
                    sac.MenuItem(label='Prediction and Comparison'),
                ])
            ],
            key='About',
            open_all=True,
            indent=20,
            format_func='title',
            index=1

        )
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write('Created By:')
        st.write('Siddharth Choudhury (2020A7PS0028U), Ashwin Shibu (2020A7PSOO30U), <br>Jay Parida (2020A7PS0087U)', unsafe_allow_html=True)


    if selection == 'About':
        about()
    elif selection == 'The Study':
        study()
    elif selection == 'Exploratory Data Analysis':
        analysis()
    elif selection == 'Self Explore Dataset':
        self_explore()
    elif selection == 'Managerial Q & A':
        questions()
    elif selection == 'Prediction and Comparison':
        pred()


def main():
    app()


if __name__ == "__main__":
    main()
