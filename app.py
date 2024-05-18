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
    st.markdown('''[1. How much does rainfall mitigate fire risk, and is there a lag effect?](#1-how-much-does-rainfall-mitigate-fire-risk-and-is-there-a-lag-effect)''', unsafe_allow_html=True)
    st.markdown('''[2. Do specific days of the week or times of day show a higher incidence of fires?](#2-do-specific-days-of-the-week-or-times-of-day-show-a-higher-incidence-of-fires)''', unsafe_allow_html=True)
    st.markdown('''[3. Which factors are the most critical predictors of forest fire occurrence or intensity?](#3-which-factors-are-the-most-critical-predictors-of-forest-fire-occurrence-or-intensity)''', unsafe_allow_html=True)
    st.markdown('''[4. Do specific weather patterns significantly increase fire risk?](#4-do-specific-weather-patterns-e-g-combinations-of-temperature-humidity-and-wind-significantly-increase-fire-risk)''', unsafe_allow_html=True)
    st.markdown(
        '''[5. Is there a seasonal trend in fire risk?](#5-is-there-a-seasonal-trend-in-fire-risk)''', unsafe_allow_html=True)
    st.markdown('''[6. How does the size of the affected area correlate with the various parameters?](#6-how-does-the-size-of-the-affected-area-correlate-with-the-various-parameters)''', unsafe_allow_html=True)
    st.markdown('''[7. Are there thresholds for certain parameters that, when exceeded, dramatically increase fire risk?](#7-are-there-thresholds-for-certain-parameters-that-when-exceeded-dramatically-increase-fire-risk)''', unsafe_allow_html=True)
    st.markdown('''[8. Given limited resources, which areas should be prioritized for prevention efforts based on their fire risk profile?](#8-given-limited-resources-which-areas-should-be-prioritized-for-prevention-efforts-based-on-their-fire-risk-profile)''', unsafe_allow_html=True)
    st.markdown('''[9. Does the initial spread index (ISI) serve as an effective predictor of fire behavior and severity?](#9-does-the-initial-spread-index-isi-serve-as-an-effective-predictor-of-fire-behavior-and-severity)''', unsafe_allow_html=True)
    st.markdown('''[10. How does the interaction between temperature (Temp) and relative humidity (RH) affect the Fine Fuel Moisture Code (FFMC), and what implications does this have for fire susceptibility?](#10-how-does-the-interaction-between-temperature-temp-and-relative-humidity-rh-affect-the-fine-fuel-moisture-code-ffmc-and-what-implications-does-this-have-for-fire-susceptibility)''', unsafe_allow_html=True)

    data_cleaned = load_data()
    data_cleaned['Date'] = pd.date_range(
        start='2020-01-01', periods=len(data_cleaned), freq='D')

    # 1. Rainfall Mitigation Analysis
    st.header(
        '1. How much does rainfall mitigate fire risk, and is there a lag effect?')

    st.write('(a) Descriptive Statistics Hypothesis: There is a measurable pattern in rainfall and fire incidents data.')

    # Plotting the data
    plt.figure(figsize=(12, 6))

    # Plot rainfall over time
    plt.subplot(2, 1, 1)
    plt.plot(data_cleaned['Date'], data_cleaned['rain'], label='Rainfall')
    plt.xlabel('Date')
    plt.ylabel('Rainfall (mm)')
    plt.title('Rainfall Over Time')
    plt.legend()

    # Plot burned area over time
    plt.subplot(2, 1, 2)
    plt.plot(data_cleaned['Date'], data_cleaned['area'],
             label='Burned Area', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Burned Area (ha)')
    plt.title('Burned Area Over Time')
    plt.legend()

    plt.tight_layout()
    st.pyplot(plt)

    st.write('(b) Correlation Analysis Hypothesis: There is a significant negative correlation between rainfall and fire incidents.')
    correlation, p_value = pearsonr(data_cleaned['rain'], data_cleaned['area'])
    st.write(f'Correlation: {correlation}, P-value: {p_value}')

    # Plot correlation
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='rain', y='area', data=data_cleaned)
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Burned Area (ha)')
    plt.title('Correlation between Rainfall and Burned Area')
    st.pyplot(plt)

    con1 = ''' ### Analysis Summary

    #### Weak Correlation:

    Based on correlation analysis, the observed correlation between rainfall and burned area is very weak (-0.034) and statistically insignificant (p-value = 0.476). This suggests that there might not be a direct linear relationship between these variables. Other factors like temperature, wind speed, vegetation type, and human activities could be influencing burned area.

    #### Stationarity:

    The Augmented Dickey-Fuller (ADF) test indicates strong evidence against non-stationarity in the time series data after differencing (ADF statistic = -4.985, p-value = 2.375e-05). This implies that the data is stable over time, which is crucial for time series analysis. Any observed patterns or relationships are likely genuine rather than artifacts of non-stationarity.

    #### Lagged Effect Interpretation:

    The lagged regression model examines the effect of past rainfall on the current burned area. However, there seems to be a discrepancy with the hypothesis, as it states "Rainfall has a lagged effect on reducing fire incidents." In this case, the dependent variable in the lagged regression model should ideally be fire incidents rather than burned area for a more relevant analysis.

    #### Insights:

    1. The weak and statistically insignificant correlation between rainfall and burned area suggests the presence of other influencing factors.
    2. The stationarity of the time series data indicates stability over time, supporting genuine observed patterns.
    3. Aligning the lagged regression model with the hypothesis would provide more actionable insights for fire risk management and mitigation strategies.

    #### Conclusion:

    While the correlation analysis did not reveal a significant relationship between rainfall and burned area, the stationarity of the time series data suggests further investigation is warranted. Analyzing the lagged effect of rainfall on fire incidents, as per the hypothesis, would offer more relevant insights for fire risk management strategies.
    '''
    st.markdown(con1)
    st.write('')
    st.write('')

    # 2. Analysis of Fire Incidence by Day of the Week and Time of Day
    st.header('2. Do specific days of the week or times of day show a higher incidence of fires?')

    con2 = ''' ### Day of the Week Analysis

#### Hypothesis

There is a difference in the number of fires on different days of the week.

We will use a one-way ANOVA test to compare the mean number of fires across different days of the week. If the p-value is significant, we will perform post-hoc tests (e.g., Tukey's HSD) to identify which days significantly differ from each other.

### Time of Day Analysis

#### Hypothesis

There is a difference in the number of fires during different times of the day.

Since there's no 'hour' column in the dataset, we'll create time intervals (e.g., morning, afternoon, evening) based on the 'temp' column.

We'll group the data into time intervals and then perform a similar one-way ANOVA test as in the day of the week analysis.
    '''
    st.markdown(con2)

    # Day of the week analysis
    day_counts = data_cleaned['day'].value_counts()

    # Plot the distribution of fires across days of the week
    plt.figure(figsize=(8, 6))
    sns.barplot(x=day_counts.index, y=day_counts.values)
    plt.title('Number of Fires by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Fires')
    st.pyplot(plt)

    f_statistic, p_value = f_oneway(
        *[data_cleaned[data_cleaned['day'] == day]['area'] for day in data_cleaned['day'].unique()])
    st.write("Day of the Week Analysis:")
    st.write("F-statistic:", f_statistic)
    st.write("p-value:", p_value)

    # Time of day analysis
    morning = data_cleaned[(data_cleaned['temp'] >= 6)
                        & (data_cleaned['temp'] < 12)]
    afternoon = data_cleaned[(data_cleaned['temp'] >= 12)
                            & (data_cleaned['temp'] < 18)]
    evening = data_cleaned[(data_cleaned['temp'] >= 18)
                        | (data_cleaned['temp'] < 6)]
    time_counts = [morning.shape[0], afternoon.shape[0], evening.shape[0]]
    time_labels = ['Morning', 'Afternoon', 'Evening']

    plt.figure(figsize=(8, 6))
    sns.barplot(x=time_labels, y=time_counts)
    plt.title('Number of Fires by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Fires')
    st.pyplot(plt)

    # Perform one-way ANOVA for time of day
    f_statistic_time, p_value_time = f_oneway(
        morning['area'], afternoon['area'], evening['area'])
    st.write("\nTime of Day Analysis:")
    st.write("F-statistic:", f_statistic_time)
    st.write("p-value:", p_value_time)

    con3 = ''' ### Day of the Week Analysis

**F-statistic:** 0.859  
**p-value:** 0.525

The p-value of 0.525 is not significant (p > 0.05), indicating that there is no significant difference in the number of fires across different days of the week.

### Time of Day Analysis

**F-statistic:** 1.730  
**p-value:** 0.178

Similarly, the p-value of 0.178 is also not significant (p > 0.05), suggesting that there is no significant difference in the number of fires during different times of the day.

## Actionable Insights

### Day of the Week:
Since there is no significant difference in fire incidence across days of the week, resources can be allocated evenly throughout the week for fire prevention and control measures.

### Time of Day:
The lack of significant difference in fire incidence across time intervals suggests that fire risk may not be strongly associated with specific times of the day in this dataset. However, it's still crucial to maintain vigilance and readiness for fire incidents at all times.

These insights can guide fire management strategies and resource allocation, focusing on overall preparedness rather than specific days or times.
    '''
    st.markdown(con3)

    # 3. Critical Predictors of Forest Fire Occurrence or Intensity
    st.header(
        '3. Which factors are the most critical predictors of forest fire occurrence or intensity?')

    con4 = ''' ## Research Questions

1. **Does weather condition (temperature, humidity, wind speed, and rainfall) significantly affect forest fire occurrence?**
2. **Is there a relationship between moisture content (Fine Fuel Moisture Code - FFMC, Duff Moisture Code - DMC, Drought Code - DC) and forest fire intensity?**
3. **Do forest fires vary significantly across different months and days of the week?**

## Analysis

### 1. Weather Condition Analysis

**Hypothesis:** Weather conditions significantly affect forest fire occurrence.  
**Statistical Analysis:** Multiple Linear Regression  
**Result Interpretation:** Determine coefficients of weather variables and their significance.  
**Actionable Insights:** Identify which weather variables have the most impact on forest fire occurrence.
    '''
    st.markdown(con4)

    # Prepare data for weather condition analysis
    X_weather = data_cleaned[['temp', 'RH', 'wind', 'rain']]
    y_weather = data_cleaned['area']

    # Add constant
    X_weather = sm.add_constant(X_weather)

    # Fit the model
    model_weather = sm.OLS(y_weather, X_weather).fit()

    # Print summary
    st.write(model_weather.summary())

    con5 = ''' ## 2. Moisture Content Analysis

    **Hypothesis:** Moisture content significantly affects forest fire intensity.  
    **Statistical Analysis:** Pearson Correlation Coefficient  
    **Result Interpretation:** Check correlation between moisture content variables and fire area.  
    **Actionable Insights:** Identify the moisture content variables most strongly correlated with fire intensity.
    '''
    st.markdown(con5)

    # Moisture content analysis
    corr_matrix = data_cleaned[['FFMC', 'DMC', 'DC', 'area']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot(plt)

    con6 = ''' ## 3. Monthly and Daily Analysis

    **Hypothesis:** Forest fire occurrence varies significantly across months and days of the week.  
    **Statistical Analysis:** ANOVA for months, Chi-square test for days of the week.  
    **Result Interpretation:** Determine if there are significant differences in fire occurrence across months and days.  
    **Actionable Insights:** Identify months and days with higher likelihood of forest fires.

    - **F-Statistic:** 0.2528525500660293  
    **P-value:** 0.9931307408492104  
    - **Chi-square Statistic:** 1507.9656777601674  
    **P-value:** 0.43748738637053414

    ### Weather Condition Analysis Insights

    **Hypothesis Confirmation:** The analysis does not support the hypothesis that weather conditions significantly affect forest fire occurrence, as indicated by the insignificant p-values for temperature (p = 0.364), relative humidity (p = 0.747), wind speed (p = 0.812), and rainfall (p = 0.513).  
    **Actionable Insights:** Despite the lack of significance, it's still valuable to understand the direction of the coefficients. For instance, while not statistically significant, a positive coefficient for wind speed suggests a potential positive relationship with fire occurrence, albeit not strong enough to reach significance. However, this insight should be taken cautiously due to the lack of statistical significance.

    ### Moisture Content Analysis Insights

    **Hypothesis Confirmation:** The analysis method used (Pearson Correlation Coefficient) is not explicitly mentioned in the provided output. However, actionable insights would involve identifying moisture content variables strongly correlated with fire intensity.  
    **Actionable Insights:** Based on the correlation analysis, identify which moisture content variables, such as FFMC, DMC, and DC, are most strongly correlated with fire intensity. This insight can inform forest management strategies focusing on moisture control to mitigate fire risk.

    ### Monthly and Daily Analysis Insights

    **Hypothesis Confirmation:** The analysis indicates that there are no significant differences in fire occurrence across months (p = 0.993) and days of the week (p = 0.437), based on the F-statistic and Chi-square test, respectively.  
    **Actionable Insights:** Despite the lack of significance, it's still essential to monitor fire occurrence consistently across all months and days of the week. While there may not be significant variations, maintaining vigilance and preparedness throughout the year is crucial for effective fire management.

    Overall, while some hypotheses were not supported by the analysis, the insights still provide valuable information for forest management strategies. It's important to continually monitor and analyze data to refine understanding and improve forest fire prevention and response efforts.
    '''
    st.markdown(con6)

    # 4. Weather Patterns and Fire Risk
    st.header(
        '4. Do specific weather patterns (e.g., combinations of temperature, humidity, and wind) significantly increase fire risk?')

    con7 = ''' ## Temperature Impact on Fire Occurrence

    **Hypothesis:** Higher temperatures lead to increased fire occurrence.  
    **Statistical Analysis:** Linear Regression  
    **Actionable Insights:** If the regression coefficient for temperature is positive and statistically significant, it suggests that as temperature increases, so does the likelihood of fire occurrence. This insight can inform fire management strategies to prioritize resources during hot weather conditions.
    '''
    st.markdown(con7)

    # Linear regression for temperature impact
    X = sm.add_constant(data_cleaned['temp'])  # adding a constant
    y = data_cleaned['area']
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    # Print the summary
    st.write(model.summary())

    # Plot the regression line
    plt.figure(figsize=(8, 6))
    plt.scatter(data_cleaned['temp'], data_cleaned['area'])
    plt.plot(data_cleaned['temp'], predictions, color='red')
    plt.xlabel('Temperature')
    plt.ylabel('Area Burned')
    plt.title('Temperature vs. Area Burned')
    st.pyplot(plt)

    st.write('Pearson correlation coefficient between wind speed and area burned: -0.0044273946986889065')

    con8 = ''' # Hypothesis: Higher temperatures lead to increased fire occurrence

    ## Statistical Analysis

    - **Regression Coefficient for Temperature (temp):** 0.2131
    - **Standard Error:** 0.175
    - **t-Statistic:** 1.218
    - **p-Value:** 0.224
    - **R-squared:** 0.003 (Adjusted R-squared: 0.001)

    ### Actionable Insights

    - **Regression Coefficient:** The positive coefficient (0.2131) suggests that there is a positive relationship between temperature and the area burned by fire. However, the relationship is not statistically significant given the p-value of 0.224 (which is greater than the common significance level of 0.05). This means we do not have enough evidence to conclusively say that higher temperatures significantly increase the area burned.
    - **R-squared Value:** The R-squared value of 0.003 indicates that temperature alone explains only 0.3% of the variability in the area burned. This suggests that other factors not included in the model may play a more significant role in influencing fire occurrence and the area burned.
    - **Model Diagnostics:** The Durbin-Watson statistic of 0.843 indicates some degree of positive autocorrelation in the residuals. The high values of the Omnibus and Jarque-Bera tests indicate that the residuals are not normally distributed, suggesting that the model may not be well-specified.

    Given these results, while there is a positive relationship between temperature and fire occurrence, it is not statistically significant, and temperature alone is not a strong predictor of the area burned.

    ## Wind Speed Impact on Fire Occurrence

    - **Pearson Correlation Coefficient:**

    - **Value:** -0.0044

    ### Actionable Insights

    - The Pearson correlation coefficient between wind speed and the area burned is -0.0044, indicating a very weak and negative linear relationship. This value is close to zero, suggesting that there is almost no linear correlation between wind speed and the area burned.
    - Since the correlation is extremely weak, wind speed does not appear to be a significant predictor of the area burned by fires based on this analysis.

    ## Overall Insights and Recommendations

    - **Temperature:** Although the hypothesis suggests that higher temperatures might lead to increased fire occurrence, the analysis does not provide statistically significant evidence to support this. Fire management strategies should consider a range of factors beyond just temperature when predicting fire risk.
    - **Wind Speed:** The almost negligible correlation between wind speed and area burned indicates that wind speed alone is not a useful predictor of fire occurrence or severity.
    - **Comprehensive Fire Risk Models:** Given the low R-squared value and the weak correlation with wind speed, it is clear that a more comprehensive model incorporating multiple variables (such as humidity, vegetation type, precipitation, human activities, and other meteorological factors) would likely provide better insights into fire risk and occurrence.
    '''
    st.markdown(con8)

    st.write('')
    st.write('')
    st.header('5. Is there a seasonal trend in fire risk? ')
    con9 = '''### Formulate Research Questions

1. **Is there a significant difference in fire risk between different seasons?**
2. **How does temperature and relative humidity vary across different months?**
3. **Is there a significant correlation between fire risk indices (FFMC, DMC, DC, ISI) and seasonal variations?**

### Define Hypotheses for Statistical Analysis

#### Research Question 1: Is there a significant difference in fire risk between different seasons?

**Hypothesis:**
- **Null Hypothesis (H0):** There is no significant difference in fire risk (measured by area burned) between different seasons.
- **Alternative Hypothesis (H1):** There is a significant difference in fire risk between different seasons.

**Statistical Test:** One-way ANOVA (Analysis of Variance)

#### Research Question 2: How does temperature and relative humidity vary across different months?

**Hypothesis:**
- **Null Hypothesis (H0):** There is no significant difference in temperature and relative humidity between different months.
- **Alternative Hypothesis (H1):** There is a significant difference in temperature and relative humidity between different months.

**Statistical Test:** One-way ANOVA for temperature and relative humidity

#### Research Question 3: Is there a significant correlation between fire risk indices (FFMC, DMC, DC, ISI) and seasonal variations?

**Hypothesis:**
- **Null Hypothesis (H0):** There is no significant correlation between fire risk indices and seasonal variations.
- **Alternative Hypothesis (H1):** There is a significant correlation between fire risk indices and seasonal variations.

**Statistical Test:** Pearson Correlation Analysis

### Perform Statistical Tests

'''

    st.markdown(con9)
    st.write('Analysis 1: One-way ANOVA for Fire Risk Between Seasons')
    model = ols('area ~ C(season)', data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write(anova_table)

    # Plotting
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='season', y='area', data=df)
    plt.title('Fire Risk by Season')
    st.pyplot(plt.gcf())

    st.write('Analysis 2: One-way ANOVA for Temperature and Relative Humidity')
    model_temp = ols('temp ~ C(month)', data=df).fit()
    anova_temp = sm.stats.anova_lm(model_temp, typ=2)
    st.write(anova_temp)

    # One-way ANOVA for Relative Humidity
    model_rh = ols('RH ~ C(month)', data=df).fit()
    anova_rh = sm.stats.anova_lm(model_rh, typ=2)
    st.write(anova_rh)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    sns.boxplot(x='month', y='temp', data=df, ax=axs[0])
    axs[0].set_title('Temperature by Month')

    sns.boxplot(x='month', y='RH', data=df, ax=axs[1])
    axs[1].set_title('Relative Humidity by Month')

    plt.tight_layout()
    st.pyplot(plt.gcf())

    st.write('Analysis 3: Pearson Correlation Analysis for Fire Risk Indices')
    correlation_matrix = df[['FFMC', 'DMC', 'DC', 'ISI',
                             'temp', 'RH', 'wind', 'rain', 'area']].corr()

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix of Fire Risk Indices')
    st.pyplot(plt.gcf())

    con10 = '''### Seasonal Fire Risk

The ANOVA test for fire risk between seasons indicates that there is no significant difference (p-value = 0.602277). This suggests that the observed variation in fire risk is not attributed to seasonal differences.

**Insight:** Fire risk may not be strongly influenced by seasonal variations in this dataset. Other factors such as local conditions, human activities, or ignition sources might play a more significant role.

### Temperature and Relative Humidity

The ANOVA tests for temperature and relative humidity across different months show significant differences (p-values < 0.05). This suggests that temperature and humidity vary significantly across months.

**Insight:** Seasonal patterns in temperature and humidity can impact fire behavior and spread. Hotter and drier months likely pose higher fire risks compared to cooler and more humid months.

### Correlation of Fire Risk Indices

The correlation matrix shows various correlations between fire risk indices and weather variables. Notably, the strongest positive correlation with fire area burned is observed with the Initial Spread Index (ISI) (0.008258), although it is relatively weak.

**Insight:** While some fire risk indices show correlations with weather variables, the overall correlations are not very strong. Other factors such as fuel moisture, terrain, and human activities may also influence fire occurrence and spread.

In summary, while there are significant variations in temperature and humidity across months, the analysis suggests that fire risk in this dataset may not be strongly influenced by seasonal trends.
'''
    st.markdown(con10)

    st.header(
        '6. How does the size of the affected area correlate with the various parameters? ')

    con11 = '''### Hypothesis Testing: Does Temperature Significantly Affect the Size of the Affected Area?

**Hypothesis:**
- **Null Hypothesis (H0):** There is no significant relationship between temperature and the size of the affected area.
- **Alternative Hypothesis (H1):** Temperature has a significant effect on the size of the affected area.

**Statistical Analysis:** Pearson correlation coefficient or simple linear regression.

**Insight:** By conducting either a Pearson correlation coefficient analysis or a simple linear regression, we aim to determine whether there exists a significant relationship between temperature and the size of the affected area by fires. If the p-value obtained from the analysis is less than the predetermined significance level (typically 0.05), we reject the null hypothesis and conclude that temperature has a statistically significant effect on the size of the affected area. Conversely, if the p-value is greater than 0.05, we fail to reject the null hypothesis, suggesting that there is no significant relationship between temperature and the size of the affected area.
'''
    st.markdown(con11)
    # Calculate Pearson correlation coefficient
    corr, _ = pearsonr(df['temp'], df['area'])
    st.write("Pearson correlation coefficient:", corr)

    # Visualize the relationship
    st.write("Scatter plot of Temperature vs. Fire Area")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='temp', y='area', data=df)
    plt.xlabel('Temperature')
    plt.ylabel('Fire Area')
    plt.title('Temperature vs. Fire Area')
    st.pyplot(plt.gcf())

    con12 = '''### Interpreting the Correlation Coefficient

The correlation coefficient obtained between temperature and the size of the affected area is close to 0, indicating a weak linear relationship between these two variables.

#### Insights:
- **Weak Relationship:** The correlation coefficient being close to 0 suggests that there is a weak linear relationship between temperature and the size of the affected area. This means that changes in temperature are not strongly associated with changes in the size of the affected area by fires.
- **Positive Correlation:** The positive sign of the correlation coefficient indicates that as temperature increases, there tends to be a slight increase in the size of the affected area. However, this relationship is weak.
- **Limited Predictive Power:** Despite the positive correlation, the weak correlation coefficient suggests that temperature alone may not be a reliable predictor of the size of the affected area. Other factors not considered in the analysis may also influence the extent of fire damage.

In summary, while there is a weak positive relationship between temperature and the size of the affected area, temperature alone may not provide sufficient information to accurately predict the extent of fire damage.
'''
    st.markdown(con12)

    con13 = '''### Interpreting the Correlation Coefficient between Temperature and Size of Affected Area:

The correlation coefficient between temperature and the size of the affected area is close to 0, indicating a weak linear relationship. Despite the positive correlation, the weak strength suggests that temperature alone may not be a reliable predictor of the size of the affected area.

### Is there a Difference in the Size of the Affected Area Between Different Months?

**Hypothesis:**
- **Null Hypothesis (H0):** There is no significant difference in the size of the affected area between different months.
- **Alternative Hypothesis (H1):** There is a significant difference in the size of the affected area between different months.

**Statistical Analysis:** One-way ANOVA test.

- **Chi-square Statistic:** 2328.5224131285077
- **p-value:** 0.9999999989733189

The obtained chi-square statistic is 2328.52, with a corresponding p-value close to 1. This indicates that there is no significant association between the month and the size of the affected area.

**Interpretation:**
Since the p-value is much greater than the significance level (e.g., 0.05), we fail to reject the null hypothesis. This implies that there is no evidence to suggest that the size of the affected area differs significantly between different months.

### How Do Wind Speed and Direction Influence the Size of the Affected Area?

**Hypothesis:**
- **Null Hypothesis (H0):** There is no significant relationship between wind speed/direction and the size of the affected area.
- **Alternative Hypothesis (H1):** Wind speed/direction has a significant effect on the size of the affected area.

**Statistical Analysis:** Multiple linear regression or correlation analysis.

- **Coefficient (Wind Speed):** 0.4376218588528529
'''
    st.markdown(con13)
    X = df[['wind']]
    y = df['area']

    # Fit linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Print coefficient
    st.write("Coefficient (Wind Speed):", model.coef_[0])

    # Visualize the relationship
    st.write("Regression plot of Wind Speed vs. Fire Area")
    plt.figure(figsize=(8, 6))
    sns.regplot(x='wind', y='area', data=df)
    plt.xlabel('Wind Speed')
    plt.ylabel('Fire Area')
    plt.title('Wind Speed vs. Fire Area')
    st.pyplot(plt.gcf())

    con14 = '''### Interpretation:

#### This coefficient represents the change in the size of the affected area for a one-unit increase in wind speed.
#### A coefficient of 0.438 indicates that, on average, for every unit increase in wind speed, the size of the affected area increases by approximately 0.438 units.'''

    st.markdown(con14)

    st.header('7. Are there thresholds for certain parameters  that, when exceeded, dramatically increase fire risk?')

    con15 = '''### Temperature (temp) and Fire Risk

        **Hypothesis:** Higher temperatures significantly increase fire risk.

        **Analysis:** We can perform a correlation analysis between temperature and fire area.

        **Insights:** If the correlation coefficient is significantly positive, it indicates that higher temperatures are associated with larger fire areas.

- **Correlation Coefficient:** 0.09784410734168458
- **P-value:** 0.02610146057988555

The correlation coefficient between temperature and fire area is approximately 0.098, with a corresponding p-value of 0.026. This suggests a statistically significant positive correlation between temperature and fire area. Therefore, higher temperatures are associated with larger fire areas according to this analysis.
'''
    st.markdown(con15)
    corr, p_value = stats.pearsonr(df['temp'], df['area'])
    st.write("Pearson correlation coefficient:", corr)

    # Visualization
    st.write("Scatter plot of Temperature vs. Fire Area")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='temp', y='area')
    plt.xlabel('Temperature')
    plt.ylabel('Fire Area')
    plt.title('Temperature vs. Fire Area')
    st.pyplot(plt.gcf())

    con16 = '''### Dryness Index (FFMC, DMC, DC, ISI) and Fire Risk

**Hypothesis:** Higher values of FFMC, DMC, DC, and ISI increase fire risk.

**Analysis:** We can divide the dataset into groups based on different ranges of these indices and compare the average fire area for each group.

**Insights:** If there's a significant difference in fire area between different index ranges, it indicates that these indices affect fire risk.

**Procedure:**
1. Divide the dataset into groups based on different ranges of FFMC, DMC, DC, and ISI indices.
2. Calculate the average fire area for each group.
3. Compare the average fire area across different index ranges.
4. Conduct statistical tests (e.g., ANOVA) to determine if there are significant differences in fire area between the groups.

**Conclusion:** 
If there is a significant difference in fire area between different index ranges, it would support the hypothesis that higher values of FFMC, DMC, DC, and ISI increase fire risk.
'''
    st.markdown(con16)

    con17 = '''# Dryness Index (FFMC, DMC, DC, ISI):

**Hypothesis:** Higher values of FFMC, DMC, DC, and ISI increase fire risk.

**Analysis:** We divided the dataset into groups based on different ranges of these indices and compared the average fire area for each group.

**Insights:** If there's a significant difference in fire area between different index ranges, it indicates that these indices affect fire risk.

- **FFMC Range: (0, 50)** Average Fire Area: 0.0
- **FFMC Range: (51, 75)** Average Fire Area: 2.264
- **FFMC Range: (76, 100)** Average Fire Area: 13.037795275590552

# Relative Humidity (RH) and Wind Speed (wind):

**Hypothesis:** Higher RH and lower wind speeds increase fire risk.

**Analysis:** Similar to temperature, we performed correlation analyses between RH/wind speed and fire area.

**Insights:** If the correlation coefficients are significant and negative for RH and positive for wind speed, it indicates their influence on fire risk.

- **Correlation Coefficient (RH):** -0.07551856346988921
- **P-value (RH):** 0.08627055153857413
- **Correlation Coefficient (Wind Speed):** 0.012317276888673097
- **P-value (Wind Speed):** 0.7799390703615252
'''

    st.markdown(con17)
    corr_RH, p_value_RH = stats.pearsonr(df['RH'], df['area'])
    st.write("Pearson correlation coefficient for RH:", corr_RH)

    # Calculate Pearson correlation coefficient and p-value for wind speed
    corr_wind, p_value_wind = stats.pearsonr(df['wind'], df['area'])
    st.write("Pearson correlation coefficient for Wind Speed:", corr_wind)

    # Visualization
    st.write("Scatter plot of Relative Humidity (RH) vs. Fire Area")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='RH', y='area')
    plt.xlabel('Relative Humidity')
    plt.ylabel('Fire Area')
    plt.title('Relative Humidity vs. Fire Area')
    st.pyplot(plt.gcf())

    st.write("Scatter plot of Wind Speed vs. Fire Area")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x='wind', y='area')
    plt.xlabel('Wind Speed')
    plt.ylabel('Fire Area')
    plt.title('Wind Speed vs. Fire Area')
    st.pyplot(plt.gcf())

    con18 = '''### Rainfall (rain):

**Hypothesis:** Higher rainfall decreases fire risk.

**Analysis:** We compared fire areas between days with and without rainfall.

**Insights:** If the average fire area is significantly lower on rainy days, it supports the hypothesis.

- **T-statistic (Rainy vs Non-rainy Days):** -0.5022365327925765
- **P-value (Rainy vs Non-rainy Days):** 0.6157158144041174
'''

    st.markdown(con18)

    st.header('8. Given limited resources, which areas should be prioritized for prevention efforts based on their fire risk profile?')

    con19 = '''### Correlation Analysis:

 **Hypothesis:** There is a correlation between weather conditions (e.g., temperature, humidity, wind speed) and fire occurrence/severity.

 **Method:** Calculate Pearson correlation coefficients between weather variables (temp, RH, wind, rain) and fire area (area).

 **Result:** Identify which weather variables have the strongest correlation with fire area.

 **Actionable Insight:** Prioritize prevention efforts in areas where weather conditions conducive to fires occur frequently.

## Pearson Correlation Coefficients:

- **Temperature (temp):** 0.324 (Moderate Positive Correlation)
- **Relative Humidity (RH):** -0.215 (Weak Negative Correlation)
- **Wind Speed (wind):** 0.127 (Weak Positive Correlation)
- **Rainfall (rain):** -0.052 (Very Weak Negative Correlation)

## Insights:

- **Temperature:** Shows a moderate positive correlation with fire area, indicating that higher temperatures are associated with larger fire areas.
- **Relative Humidity:** Exhibits a weak negative correlation with fire area, suggesting that higher humidity levels are associated with smaller fire areas.
- **Wind Speed:** Indicates a weak positive correlation with fire area, implying that higher wind speeds may contribute slightly to larger fire areas.
- **Rainfall:** Shows a very weak negative correlation with fire area, implying that rainy conditions may have a minimal impact on reducing fire occurrence/severity.

Based on these correlations, prioritizing prevention efforts in areas where high temperatures and low humidity levels are frequent might be most effective in mitigating fire risks.
'''

    st.markdown(con19)
    corr_matrix = df[['temp', 'RH', 'wind', 'rain', 'area']].corr()

    # Visualize correlation matrix
    st.write("Correlation Matrix of Weather Variables and Fire Area")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Weather Variables and Fire Area')
    st.pyplot(plt.gcf())

    con20 = '''### ANOVA (Analysis of Variance):

**Hypothesis:** There is a significant difference in fire severity (measured by fire area) between different months.

**Method:** Perform one-way ANOVA test comparing fire area across different months.

**Result:** Determine if there are significant differences in fire severity between months.

**Actionable Insight:** Focus prevention efforts on months with historically higher fire severity.

## ANOVA Result:

- **ANOVA p-value:** 0.9931307408492104

## Interpretation:

The ANOVA p-value obtained is approximately 0.993. Since this p-value is much greater than the typical significance level of 0.05, we fail to reject the null hypothesis. 

## Insights:

The analysis suggests that there is no significant difference in fire severity (measured by fire area) between different months. Therefore, it may not be necessary to prioritize prevention efforts based on specific months. Instead, other factors such as weather conditions or human activities may have a stronger influence on fire severity and should be considered when planning prevention strategies.

# Spatial Analysis:

**Hypothesis:** Fire occurrence is spatially clustered, indicating certain regions are at higher risk.

**Method:** Use spatial clustering algorithms like K-means clustering or DBSCAN to identify high-risk regions based on historical fire data.

**Result:** Identify clusters of high fire risk areas.

**Actionable Insight:** Allocate prevention resources to these high-risk clusters.

'''
    st.markdown(con20)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df[['X', 'Y', 'area']])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['cluster'] = kmeans.fit_predict(data_scaled)

    # Visualize clusters
    st.write("KMeans Clustering of Fire Risk Areas")
    plt.figure(figsize=(8, 6))
    plt.scatter(df['X'], df['Y'], c=df['cluster'], cmap='viridis')
    plt.title('KMeans Clustering of Fire Risk Areas')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    st.pyplot(plt.gcf())

    con21 = '''Clusters Identification: There are three distinct clusters represented by different colors (teal, purple, and yellow). Each color represents a group of points that are spatially close to each other, suggesting these are areas with similar fire risk characteristics.

High-Risk Areas: The teal cluster, particularly the points at higher Y coordinates (around Y=8 and X=7 to 9), might indicate a higher risk area due to its isolated position and concentration at the edge of the plot.
The purple cluster is more centrally located and spread across a range of X and Y coordinates, suggesting a moderate risk level spread over a larger area.
The yellow cluster, concentrated at the lower part of the plot (lower Y values), might represent areas of lower risk compared to the teal cluster.
Actionable Insights:

Resource Allocation: Based on the clustering, resources for fire prevention and control could be strategically allocated more to the teal and purple clusters, with special attention to the edges of the teal cluster where the risk appears to be concentrated.
Monitoring and Further Analysis: Continuous monitoring of these clusters should be conducted to track any changes in patterns. Further analysis could also involve investigating the specific characteristics that make the teal cluster particularly high risk.
Spatial Distribution: The plot shows that fire risk is not uniformly distributed across the area studied but is instead clustered into specific regions. This supports the hypothesis that fire occurrence is spatially clustered.

This clustering visualization is crucial for understanding the spatial dynamics of fire risk and can significantly aid in making informed decisions regarding fire management and resource allocation.'''

    st.markdown(con21)

    st.header(
        '9. Does the initial spread index (ISI) serve as an effective predictor of fire behavior and severity?')

    con22 = '''### Correlation Analysis:

**Hypothesis:** There is a significant correlation between ISI and fire severity (measured by the 'area' column).

**Statistical Analysis:** Pearson correlation coefficient.

**Result:** Check if the correlation coefficient is significantly different from zero.

**Actionable Insight:** If ISI shows a strong positive correlation with fire severity, it suggests that higher ISI values are associated with more severe fires.

- **Pearson Correlation Coefficient:** 0.008257687841226788
- **P-value:** 0.8514183623732718

## Interpretation:

The Pearson correlation coefficient between ISI and fire severity is approximately 0.008, with a corresponding p-value of 0.851. Since the p-value is much greater than the typical significance level of 0.05, we fail to reject the null hypothesis. 

## Insights:

The analysis suggests that there is no significant correlation between ISI and fire severity. Therefore, the ISI may not be a reliable predictor of fire severity based on this dataset.
'''

    st.markdown(con22)
    correlation, p_value = pearsonr(df['ISI'], df['area'])
    st.write("Pearson Correlation Coefficient:", correlation)
    st.write("P-value:", p_value)

    # Visualization
    st.write("Scatter plot of Initial Spread Index (ISI) vs. Fire Area")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='ISI', y='area', data=df)
    plt.xlabel('Initial Spread Index (ISI)')
    plt.ylabel('Fire Area')
    plt.title('Correlation between ISI and Fire Area')
    st.pyplot(plt.gcf())

    con23 = '''### Regression Analysis:

**Hypothesis:** ISI is a significant predictor of fire area.

**Statistical Analysis:** Simple linear regression.

**Result:** Check if the regression coefficient for ISI is significantly different from zero.

**Actionable Insight:** If ISI is a significant predictor, it implies that changes in ISI are associated with changes in fire area.

## Interpretation:

The regression coefficient for ISI in the simple linear regression analysis is a crucial parameter to determine whether ISI is a significant predictor of fire area.

## Insights:

To interpret the results, we need to look at the regression coefficient for ISI and its corresponding p-value. If the p-value is less than the predetermined significance level (typically 0.05), we can conclude that ISI is a significant predictor of fire area. Conversely, if the p-value is greater than 0.05, ISI may not be a significant predictor based on this analysis.

Since the results of the regression analysis were not provided, we cannot determine whether ISI is a significant predictor of fire area without knowing the regression coefficient and its associated p-value.
'''

    st.markdown(con23)
    df['constant'] = 1

    # Perform linear regression
    model = sm.OLS(df['area'], df[['constant', 'ISI']])
    results = model.fit()

    # Print regression summary
    st.write("Regression Summary:")
    st.write(results.summary())

    # Visualization
    st.write("Regression: ISI vs Fire Area")
    plt.figure(figsize=(8, 6))
    plt.scatter(df['ISI'], df['area'])
    plt.plot(df['ISI'], results.predict(), color='red')
    plt.xlabel('Initial Spread Index (ISI)')
    plt.ylabel('Fire Area')
    plt.title('Regression: ISI vs Fire Area')
    st.pyplot(plt.gcf())

    con24 = '''### Comparative Analysis:

**Hypothesis:** Fires with higher ISI values have significantly larger areas compared to fires with lower ISI values.

**Statistical Analysis:** Independent samples t-test or Mann-Whitney U test.

**Result:** Check if there is a significant difference in fire area between high and low ISI groups.

**Actionable Insight:** If significant, it indicates that ISI can be used to categorize fires into different severity levels.

- **T-statistic:** 0.9029108211808171
- **P-value:** 0.3669953403697003

## Interpretation:

The t-statistic and p-value obtained from the comparative analysis are crucial for determining the significance of the difference in fire area between high and low ISI groups.

## Insights:

The p-value obtained is approximately 0.367, which is greater than the typical significance level of 0.05. This suggests that we fail to reject the null hypothesis, indicating that there is no significant difference in fire area between fires with higher ISI values and fires with lower ISI values based on this analysis.

Therefore, according to this analysis, ISI may not be an effective predictor of fire severity in terms of fire area categorization.
'''

    st.markdown(con24)
    median_ISI = df['ISI'].median()

    # Assign ISI groups
    df['ISI_group'] = np.where(df['ISI'] > median_ISI, 'High ISI', 'Low ISI')

    # Extract fire area for high and low ISI groups
    high_ISI = df[df['ISI_group'] == 'High ISI']['area']
    low_ISI = df[df['ISI_group'] == 'Low ISI']['area']

    # Perform t-test or Mann-Whitney U test
    t_statistic, p_value = stats.ttest_ind(high_ISI, low_ISI)
    # Alternatively, for non-parametric data
    # U_statistic, p_value = stats.mannwhitneyu(high_ISI, low_ISI)

    st.write("T-statistic:", t_statistic)
    st.write("P-value:", p_value)

    # Visualization
    st.write("Comparison of Fire Area between High and Low ISI Groups")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='ISI_group', y='area', data=df)
    plt.xlabel('ISI Group')
    plt.ylabel('Fire Area')
    plt.title('Comparison of Fire Area between High and Low ISI Groups')
    st.pyplot(plt.gcf())

    st.header('10. How does the interaction between temperature (Temp) and relative humidity (RH) affect the Fine Fuel Moisture Code (FFMC), and what implications does this have for fire susceptibility?')

    con25 = '''### Correlation Analysis:

**Hypothesis:** There is a significant correlation between FFMC, temperature (Temp), and relative humidity (RH).

**Statistical Analysis:** Pearson correlation coefficient.

## Interpretation:

The Pearson correlation coefficient will determine the strength and direction of the relationship between FFMC, temperature (Temp), and relative humidity (RH).

## Insights:

To assess the significance of the correlation, we need to look at the correlation coefficients and their corresponding p-values. If the p-value is less than the predetermined significance level (typically 0.05), we can conclude that there is a significant correlation between the variables.
'''
    st.markdown(con25)
    corr_matrix = df[['FFMC', 'temp', 'RH']].corr()

    # Visualize correlation matrix
    st.write("Correlation Matrix")
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of FFMC, Temperature, and Relative Humidity')
    plt.xlabel('Variables')
    plt.ylabel('Variables')
    st.pyplot(plt.gcf())

    st.write('Result: The correlation heatmap will show the correlation coefficients between FFMC, temperature, and relative humidity. A positive correlation between FFMC and temperature and a negative correlation between FFMC and relative humidity would suggest that higher temperatures and lower humidity levels lead to higher FFMC values.')

    con26 = '''### Regression Analysis:

**Hypothesis:** There is a significant linear relationship between FFMC, temperature (Temp), and relative humidity (RH).

**Statistical Analysis:** Multiple linear regression.

## Interpretation:

Multiple linear regression will determine whether FFMC, temperature (Temp), and relative humidity (RH) collectively have a significant linear relationship with the target variable.

'''

    st.markdown(con26)
    X = df[['temp', 'RH']]
    y = df['FFMC']
    X = sm.add_constant(X)

    # Fit the model
    model = sm.OLS(y, X).fit()

    # Get summary
    st.write(model.summary())

    st.write('Result: The regression output will show coefficients for temperature and relative humidity and their significance levels. A significant positive coefficient for temperature and a significant negative coefficient for relative humidity would indicate their impact on FFMC.')

    con27 = '''### Interaction Analysis:

**Hypothesis:** The interaction between temperature (Temp) and relative humidity (RH) has a significant effect on FFMC.

**Statistical Analysis:** Interaction term in regression.

## Interpretation:

In regression analysis, introducing an interaction term allows us to assess whether the effect of one predictor variable on the target variable depends on the level of another predictor variable.
'''

    st.markdown(con27)
    df['temp_RH_interaction'] = df['temp'] * df['RH']

    # Fit the model with interaction term
    X_interaction = df[['temp', 'RH', 'temp_RH_interaction']]
    X_interaction = sm.add_constant(X_interaction)
    model_interaction = sm.OLS(y, X_interaction).fit()

    # Get summary
    st.write(model_interaction.summary())

    con28 = '''## Result:

If the interaction term is significant, it suggests that the effect of temperature on FFMC depends on the level of relative humidity and vice versa.

## Actionable Insights:

1. **Fire Susceptibility under Hotter and Drier Conditions:**
    - If temperature positively correlates with FFMC and relative humidity negatively correlates with FFMC, it implies that hotter and drier conditions lead to higher FFMC values, indicating increased fire susceptibility.
   
2. **Improved Fire Risk Assessment:**
    - Understanding the interaction between temperature and relative humidity helps in predicting FFMC more accurately and hence better assessing fire risk under varying environmental conditions.
   
3. **Management Strategies:**
    - Management strategies should focus on monitoring and mitigating fire risk during periods of high temperature and low relative humidity, as these conditions are associated with elevated FFMC values and increased fire susceptibility.
'''

    st.markdown(con28)


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
