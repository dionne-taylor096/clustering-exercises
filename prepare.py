#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def summarize_data(df):
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['column_name'] = summary['index']
    summary = summary[['column_name', 'dtypes']]
    summary['missing_values'] = df.isnull().sum().values
    summary['unique_values'] = df.nunique().values
    summary['count'] = df.count().values
    summary['mean'] = df.mean().values
    summary['std'] = df.std().values
    summary['min'] = df.min().values
    summary['25%'] = df.quantile(0.25).values
    summary['50%'] = df.quantile(0.5).values
    summary['75%'] = df.quantile(0.75).values
    summary['max'] = df.max().values
    return summary


# In[ ]:


def scale_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the 'county' column
    scaler.fit(df[['total_property_tax']])

    # Transform the 'county' column using the fitted scaler
    df['total_property_tax'] = scaler.transform(df[['total_property_tax']])
    
    return df


# In[ ]:


def inverse_scale_data(df):
    # Create an instance of MinMaxScaler
    scaler = MinMaxScaler()

    # Fit the scaler to the 'county' column
    scaler.fit(df[['county']])

    # Transform the 'county' column using the fitted scaler
    df['county_scaler'] = scaler.inverse_transform(df[['county_scaled']])
    
    return df


# In[ ]:


def make_sample(df, frac=0.001):
    """
    This function takes a DataFrame as input and returns a random sample of the data.
    
    Parameters
    ----------
    df : DataFrame
        The input DataFrame to take a random sample from.
    frac : float, optional
        The fraction of the data to take as a sample. Default is 0.001 (0.1%).
        
    Returns
    -------
    DataFrame
        The random sample of the input DataFrame.
    """
    sample_df = df.sample(frac=frac)
    return sample_df

# Example usage
# sample_df = make_sample(df, frac=0.001)


# In[ ]:


def quantile_transformer(df, column_to_transform, output_distribution='normal'):
    """
    This function takes a DataFrame and a column name to transform using QuantileTransformer.
    The transformed column is added to the DataFrame with '_transformed' suffix.

    Parameters
    ----------
    df : DataFrame
        The input DataFrame containing the column to be transformed.
    column_to_transform : str
        The name of the column to be transformed.
    output_distribution : str, optional
        The output distribution of the transformed data. Default is 'normal'.
    """
    # Create a copy of the input DataFrame to avoid modifying the original
    df_transformed = df.copy()

    # Initialize the QuantileTransformer
    quantile_transformer = QuantileTransformer(output_distribution=output_distribution)

    # Fit the transformer and transform the specified column
    transformed_data = quantile_transformer.fit_transform(df_transformed[[column_to_transform]])

    # Replace the original column with the transformed data
    df_transformed[f'{column_to_transform}_transformed'] = transformed_data

    # Visualize the original and transformed data
    sns.displot(data=df_transformed, x='total_sqft', y=column_to_transform, label='Original')
    sns.displot(data=df_transformed, x='total_sqft', y=f'{column_to_transform}_transformed', label='Transformed')
    plt.legend()
    plt.show()

    return df_transformed

# Example usage
# transformed_df = quantile_transformer(sample_df, 'assessed_property_value')


# In[ ]:


# Initialize the QuantileTransformer
quantile_transformer = QuantileTransformer()

# Fit the transformer and transform only the 'taxvaluedollarcnt' column
taxval_transformed = quantile_transformer.fit_transform(sample_df[['assessed_property_value']])

# Replace the original 'taxvaluedollarcnt' column with the transformed data
sample_df['assessed_property_value_transformed'] = taxval_transformed

# Visualize the original and transformed data
sns.displot(data=sample_df, x='total_sqft', y='assessed_property_value', label='Original')
sns.displot(data=sample_df, x='total_sqft', y='assessed_property_value_transformed', label='Transformed')
plt.legend()
plt.show()


# In[ ]:


import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(sample_df, test_size=0.2, random_state=None):
    """
    This function takes a DataFrame as input and returns train, validate, and test DataFrames.
    
    Parameters
    ----------
    df : DataFrame
        The input DataFrame to split.
    test_size : float, optional
        The proportion of the dataset to include in the test split. Default is 0.2.
    random_state : int, optional
        Controls the shuffling applied to the data before applying the split.
    """
    # First, split the data into train+validate and test sets
    train_validate, test = train_test_split(sample_df, test_size=test_size, random_state=random_state)
    
    # Next, split the train+validate set into train and validate sets
    train, validate = train_test_split(train_validate, test_size=test_size, random_state=random_state)
    
    return train, validate, test


# In[ ]:


def is_continuous(series, unique_count_threshold=100):
    return (series.dtype in ['int64', 'float64']) and (series.nunique() > unique_count_threshold)

def is_categorical(series, unique_count_threshold=10):
    return (series.dtype == 'object') or (series.nunique() <= unique_count_threshold)


# In[ ]:


def compare_scalers(df, features, target, test_size=0.2, random_state=42):
    
    # Define the target variable and features
    target = 'total_property_tax'
    features = [col for col in df.columns if col != target]
    
    X = df[features]
    y = df[target]

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Define the scalers to test
    scalers = {
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler()
    }

    # Initialize a linear regression model
    lm = LinearRegression()

    results = []

    # Iterate over each scaler, apply it to the data, and evaluate the model
    for scaler_name, scaler in scalers.items():
        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Fit the model on the scaled data
        lm.fit(X_train_scaled, y_train)

        # Calculate the test mean squared error
        y_pred = lm.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Store the results
        results.append({
            'Scaler': scaler_name,
            'MSE': mse,
            'RMSE': rmse
        })

