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
    st.write("<div style='text-align: center'><h1>Analysis of Forest Fires UCI Dataset</h1></div>", unsafe_allow_html=True)
    st.write("Forest fires pose significant threats to both the environment and human society, causing extensive ecological damage and economic losses. Rapid detection and prediction of such fires are critical for effective control and mitigation. One promising method is the utilization of automated systems based on local meteorological data, such as those from weather stations. These systems leverage the known influence of meteorological conditions—such as temperature and wind—on forest fire behavior. This project aims to analyze the UCI Forest Fires Dataset using various Data Mining (DM) techniques to predict the burned area of forest fires. By applying five different DM algorithms, including Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor, and Support Vector Regressor (SVR), on data from Portugal's northeast region, we seek to identify the most effective model. The optimal configuration, employing SVR with inputs of temperature, relative humidity, rain, and wind, shows promise in predicting the burned area of smaller, more frequent fires. This predictive capability is crucial for optimizing firefighting strategies and resource allocation, ultimately enhancing fire management and response efforts.")
    col1, col2 = st.columns(2)
    with col1:
        st.image("images/fire.jpg", use_column_width=True)
    with col2:
        st.image("images/fire1.jpg", use_column_width=True)

# @st.cache_data(experimental_allow_widgets=True)      
def study():
    st.write("<div style='text-align: center'><h1>The Study</h1></div>", unsafe_allow_html=True)
    pdf_viewer("fires.pdf")

@st.cache_data(experimental_allow_widgets=True)
def analysis():
    st.write("<div style='text-align: center'><h1>Exploratory Data Analysis</h1></div>", unsafe_allow_html=True)
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
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix Heatmap')
    st.pyplot(plt)
    plt.clf()

    # Count plots for categorical features
    categorical_features = data_cleaned.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        st.subheader(f'Count Plot for {feature}')
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data_cleaned, x=feature)
        plt.title(f'Count Plot for {feature}')
        st.pyplot(plt)
        plt.clf()

    # Box plots for numerical features
    numerical_features = data_cleaned.select_dtypes(include=[np.number]).columns
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
    st.write("<div style='text-align: center'><h1>Self Explore Dataset</h1></div>", unsafe_allow_html=True)
    st.header('Select Plot Options')
    data_cleaned = load_data()
    plot_type = st.selectbox('Select Plot Type', ['Scatter Plot', 'Box Plot', 'Count Plot', 'Correlation Heatmap', 'Histogram'])
    x_axis = st.selectbox('Select X-axis', data_cleaned.columns)
    y_axis = st.selectbox('Select Y-axis', data_cleaned.columns) if plot_type in ['Scatter Plot', 'Box Plot'] else None

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

    
def qestions():
    st.write("<div style='text-align: center'><h1>Managerial questions and Answers</h1></div>", unsafe_allow_html=True)
    data_cleaned = load_data()
    st.subheader('1. How much does rainfall mitigate fire risk, and is there a lag effect')
    st.write('(a) Descriptive Statistics Hypothesis: There is a measurable pattern in rainfall and fire incidents data.')
    data_cleaned['Date'] = pd.date_range(start='2020-01-01', periods=len(data_cleaned), freq='D')

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
    plt.plot(data_cleaned['Date'], data_cleaned['area'], label='Burned Area', color='orange')
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
    sns.scatterplot(x='rain', y='area', data=data_cleaned)
    plt.xlabel('Rainfall (mm)')
    plt.ylabel('Burned Area (ha)')
    plt.title('Correlation between Rainfall and Burned Area')
    st.pyplot(plt)
    
    st.write('')
    
    con1 = '''### Analysis Summary

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
    st.subheader('2. Do specific days of the week or times of day show a higher incidence of fires?')

    con2 = '''## Day of the Week Analysis

### Hypothesis
There is a difference in the number of fires on different days of the week.

We will use a one-way ANOVA test to compare the mean number of fires across different days of the week. If the p-value is significant, we will perform post-hoc tests (e.g., Tukey's HSD) to identify which days significantly differ from each other.

## Time of Day Analysis

### Hypothesis
There is a difference in the number of fires during different times of the day.

Since there's no 'hour' column in the dataset, we'll create time intervals (e.g., morning, afternoon, evening) based on the 'temp' column.

We'll group the data into time intervals and then perform a similar one-way ANOVA test as in the day of the week analysis.

'''

    st.markdown(con2)
    day_counts = data_cleaned['day'].value_counts()

    # Plot the distribution of fires across days of the week
    sns.barplot(x=day_counts.index, y=day_counts.values)
    plt.title('Number of Fires by Day of the Week')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Fires')
    st.pyplot(plt)
    f_statistic, p_value = f_oneway(*[data_cleaned[data_cleaned['day'] == day]['area'] for day in data_cleaned['day'].unique()])
    st.write("Day of the Week Analysis:")
    st.write("F-statistic:", f_statistic)
    st.write("p-value:", p_value)
    morning = data_cleaned[(data_cleaned['temp'] >= 6) & (data_cleaned['temp'] < 12)]
    afternoon = data_cleaned[(data_cleaned['temp'] >= 12) & (data_cleaned['temp'] < 18)]
    evening = data_cleaned[(data_cleaned['temp'] >= 18) | (data_cleaned['temp'] < 6)] 
    time_counts = [morning.shape[0], afternoon.shape[0], evening.shape[0]]
    time_labels = ['Morning', 'Afternoon', 'Evening']
    sns.barplot(x=time_labels, y=time_counts)
    plt.title('Number of Fires by Time of Day')
    plt.xlabel('Time of Day')
    plt.ylabel('Number of Fires')
    st.pyplot(plt)

    # Perform one-way ANOVA
    f_statistic_time, p_value_time = f_oneway(morning['area'], afternoon['area'], evening['area'])
    st.write("\nTime of Day Analysis:")
    st.write("F-statistic:", f_statistic_time)
    st.write("p-value:", p_value_time)

    con3 = '''## Day of the Week Analysis

**F-statistic:** 0.859  
**p-value:** 0.525

The p-value of 0.525 is not significant (p > 0.05), indicating that there is no significant difference in the number of fires across different days of the week.

## Time of Day Analysis

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
    
    st.subheader('3. Which factors are the most critical predictors of forest fire occurrence or intensity? ')
    
    con4 = '''## Research Questions

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
    # Prepare data
    X_weather = data_cleaned[['temp', 'RH', 'wind', 'rain']]
    y_weather = data_cleaned['area']

    # Add constant
    X_weather = sm.add_constant(X_weather)

    # Fit the model
    model_weather = sm.OLS(y_weather, X_weather).fit()

    # Print summary
    st.write(model_weather.summary())

    con5 = '''## 2. Moisture Content Analysis

**Hypothesis:** Moisture content significantly affects forest fire intensity.  
**Statistical Analysis:** Pearson Correlation Coefficient  
**Result Interpretation:** Check correlation between moisture content variables and fire area.  
**Actionable Insights:** Identify the moisture content variables most strongly correlated with fire intensity.
'''

    st.markdown(con5)
    corr_matrix = data_cleaned[['FFMC', 'DMC', 'DC', 'area']].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    st.pyplot(plt)
    
    con6 = '''## 3. Monthly and Daily Analysis

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
    
    st.subheader('4. Do specific weather patterns (e.g., combinations of temperature, humidity, and wind) significantly increase fire risk?')
    
    con7 = '''## Temperature Impact on Fire Occurrence

**Hypothesis:** Higher temperatures lead to increased fire occurrence.  
**Statistical Analysis:** Linear Regression  
**Actionable Insights:** If the regression coefficient for temperature is positive and statistically significant, it suggests that as temperature increases, so does the likelihood of fire occurrence. This insight can inform fire management strategies to prioritize resources during hot weather conditions.
'''
    st.markdown(con7)
    X = sm.add_constant(data_cleaned['temp']) # adding a constant
    y = data_cleaned['area']
    model = sm.OLS(y, X).fit()
    predictions = model.predict(X)

    # Print the summary
    st.write(model.summary())

    # Plot the regression line
    plt.scatter(data_cleaned['temp'], data_cleaned['area'])
    plt.plot(data_cleaned['temp'], predictions, color='red')
    plt.xlabel('Temperature')
    plt.ylabel('Area Burned')
    plt.title('Temperature vs. Area Burned')
    st.pyplot(plt)
    
    


    








def load_models():
    best_models = {}
    model_names = ["DecisionTreeRegressor", "GradientBoostingRegressor", "LinearRegression", "RandomForestRegressor", "SVR"]  # Update with your model names
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
    st.write("<div style='text-align: center'><h1>Prediction</h1></div>", unsafe_allow_html=True)
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
        selected_model = st.selectbox("Select Model", list(models.keys()), label_visibility='collapsed')
        
        st.header("Input Values")
        input_values = {}
        for feature in ['FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']:
            input_values[feature] = st.text_input(f"Enter {feature}", value='0.0')

        
        # Convert input values to float
        for feature in input_values:
            input_values[feature] = float(input_values[feature])

        # Predict burned area
        if st.button("Predict"):
            prediction = predict_burned_area(input_values, models[selected_model])
            st.header("Prediction")
            st.write(f"Predicted burned area for {selected_model}: {prediction}")
   


def app():
    st.set_page_config(page_title="Analysis of Forest Fires UCI Dataset", layout="wide")
    with st.sidebar:
        selection = sac.menu(
            items=[
                sac.MenuItem(label='Forest Fires', type='group', children=[
                    sac.MenuItem(label='About'),
                    sac.MenuItem(label='The Study'),
                    sac.MenuItem(label='Exploratory Data Analysis'),
                    sac.MenuItem(label='Self Explore Dataset'),
                    sac.MenuItem(label='Managerial questions and Answers'),
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
        st.write('Siddharth Choudhury (2020A7PS0028U), Ashwin Shibu (2020A7PSOO30U), Jay Parida (2020A7PS0087U)')

    if selection == 'About':
        about()
    elif selection == 'The Study':
        study()
    elif selection == 'Exploratory Data Analysis':
        analysis()
    elif selection == 'Self Explore Dataset':
        self_explore()
    elif selection == 'Managerial questions and Answers':
        qestions()
    elif selection == 'Prediction and Comparison':
        pred()

def main():
    app()

if __name__ == "__main__":
    main()
