#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable
from tqdm.notebook import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader,ConcatDataset, TensorDataset,Subset,Dataset
from sklearn.model_selection import train_test_split
import warnings
import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pyts.image import GramianAngularField as GADF
import plotly.graph_objs as go
import plotly.io as pio
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from pyts.image import GramianAngularField
from torchvision.models import resnet18
import torch.optim as optim
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# In[6]:


def find_next_cusip(cusips, current_cusip):
  try:
    current_index = np.where(cusips == current_cusip)[0]
    print(current_index)
    if current_index < len(cusips) - 1:
      return cusips[current_index + 1]
    else:
      return None
  except ValueError:
    return None # Current cusip not found in the list


# In[8]:


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[10]:


import pandas as pd
import plotly.graph_objects as go

def plot_moving_averages_with_dropdown(df: pd.DataFrame, mode='S'):
    # Ensure 'MthCalDt' is a datetime object
    df['MthCalDt'] = pd.to_datetime(df['MthCalDt'])

    # Create traces
    if mode=='S':
      traces = {
          'MthRet': go.Scatter(x=df['MthCalDt'], y=df['MthRet'], mode='lines', name='Mth_Ret'),
          'MA_3': go.Scatter(x=df['MthCalDt'], y=df['SMA_3'], mode='lines', name='SMA_3'),
          'MA_6': go.Scatter(x=df['MthCalDt'], y=df['SMA_6'], mode='lines', name='SMA_6'),
          'MA_12': go.Scatter(x=df['MthCalDt'], y=df['SMA_12'], mode='lines', name='SMA_12')
      }
    else:
      traces = {
          'MthRet': go.Scatter(x=df['MthCalDt'], y=df['MthRet'], mode='lines', name='Mth_Ret'),
          'MA_3': go.Scatter(x=df['MthCalDt'], y=df['EMA_3'], mode='lines', name='EMA_3'),
          'MA_6': go.Scatter(x=df['MthCalDt'], y=df['EMA_6'], mode='lines', name='EMA_6'),
          'MA_12': go.Scatter(x=df['MthCalDt'], y=df['EMA_12'], mode='lines', name='EMA_12')
      }


    # Create figure with all traces
    fig = go.Figure(data=list(traces.values()))

    text_mode = 'Simple' if mode=='S' else 'Exponential'

    # Add dropdown menu to toggle traces
    fig.update_layout(
        title=f"{text_mode} Moving Averages for current CUSIP: ",
        xaxis_title='MthCalDt',
        yaxis_title='Value',
        updatemenus=[
            {
                'buttons': [
                    {'method': 'restyle',
                     'label': name,
                     'args': ['visible', [name == k for k in traces.keys()]]}
                    for name in traces.keys()
                ] + [{
                    'method': 'restyle',
                    'label': 'Show All',
                    'args': ['visible', [True] * len(traces)]
                }],
                'direction': 'down',
                'showactive': True,
            }
        ],
        legend=dict(itemclick='toggle', itemdoubleclick='toggleothers')
    )

    fig.show()


# In[ ]:
def plot_returns(test_df):
  plt.figure(figsize=(18,10))
  plt.plot(test_df['MthCalDt'], test_df['MthRet'])
  plt.xlabel('Return')
  plt.ylabel('Date')
  plt.title('Returns for stock test')
  

  
def test_ml(x_train, y_train, x_test, y_test, meth, model_kwargs=None, fit_kwargs=None):
    if model_kwargs is None:
        model_kwargs = {}
    if fit_kwargs is None:
        fit_kwargs = {}

    ml = meth(**model_kwargs)
    ml.fit(x_train, y_train, **fit_kwargs)

    y_pred = ml.predict(x_test)
    print("R^2 score: ", r2_score(y_test, y_pred))

    plt.figure(figsize=(12, 8))
    plt.scatter(y_test, y_pred, label='Predictions', alpha=0.8)

    # Add ideal dashed line (Predicted == Actual)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal (Predicted = Actual)')

    # Labels and title
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('ACTUAL VS PREDICTED')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return y_pred, y_test, x_test



def tune_hyperparameters(model_class, param_grid, x_train, y_train, scoring='r2', cv=5, verbose=1, n_jobs=-1):
   
    model = model_class()
    grid = GridSearchCV(estimator=model, param_grid=param_grid,
                        scoring=scoring, cv=cv, verbose=verbose, n_jobs=n_jobs)
    grid.fit(x_train, y_train)

    return grid.best_estimator_, grid.best_params_, grid

def report_metrics(y_pred, y_true, name):
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        print(f"--- {name} vs y_true ---")
        print(f"MAE  : {mae:.5f}")
        print(f"RMSE : {rmse:.5f}")
        print(f"R²    : {r2:.5f}")
        print()


def full_model_pipeline(
    model_class,
    x_train,
    y_train,
    x_test,
    y_test,
    param_grid=None,
    fixed_params=None,
    grid_search=True,
    model_name=None
):
    """
    Trains, tunes, evaluates, and compares regression models.

    Parameters
    ----------
    model_class : callable
        The ML model class (e.g., LinearRegression, ElasticNet, XGBRegressor).

    x_train, y_train : array-like
        Training data.

    x_test, y_test : array-like
        Test data.

    param_grid : dict, optional
        Parameter grid for GridSearchCV.

    fixed_params : dict, optional
        Parameters to pass directly to the model constructor.

    grid_search : bool
        If True and param_grid is provided, run GridSearchCV.

    model_name : str, optional
        Label to use in plot and metric printing.
    """
    model_name = model_name or model_class.__name__

    # Grid search if enabled
    if grid_search and param_grid is not None:
        print(f"Running GridSearchCV for {model_name}...")
        best_esti, best_params, grid = tune_hyperparameters(model_class, param_grid, x_train, y_train)       
        print("Best parameters found:")
        print(grid.best_params_)
        fixed_params = grid.best_params_
    else:
        # Use fixed hyperparameters
        fixed_params = fixed_params or {}

    y_pred, y_test, x_test = test_ml(x_train, y_train, x_test, y_test, model_class, fixed_params)

    print(f"\n--- {model_name} ---")
    report_metrics(y_pred, y_test, model_name)
        
    
    return model_class(**fixed_params), y_pred, y_test


def plot_preds(y_test, y_pred):
  plt.scatter(y_test, y_pred)
  plt.xlabel("Actual Returns")
  plt.ylabel("Predicted Returns")
  plt.title("Test Set Prediction vs Actual")
  plt.axhline(0, color='gray')
  plt.axvline(0, color='gray')
  plt.show()

def feat_analysis(x_train, y_train, model):
  model.fit(x_train, y_train)

  if hasattr(model, 'feature_importances_'):
      importances = model.feature_importances_
      label = "Feature Importances"
  elif hasattr(model, 'coef_'):
      importances = model.coef_
      label = "Feature Coefficients"
  else:
      print(f"⚠️ Model {type(model).__name__} does not expose feature importance.")
      return

  feature_importances = pd.DataFrame({
      'Feature': x_train.columns,
      'Importance': importances
  }).sort_values(by='Importance', key=np.abs, ascending=False)

  plt.figure(figsize=(12, 6))
  bars = plt.bar(feature_importances['Feature'], feature_importances['Importance'])
  plt.axhline(0, color='black', linewidth=1)
  plt.title(label)
  plt.xlabel("Feature")
  plt.ylabel("Coefficient")
  plt.xticks(rotation=90)
  plt.grid(axis='y', linestyle='--', alpha=0.5)
  plt.tight_layout()
  plt.show()

  print(feature_importances)


def plot_preds_line(y_test, y_pred):
  # Optional: convert to 1D arrays
  y_test = np.array(y_test).flatten()
  y_pred = np.array(y_pred).flatten()

  plt.figure(figsize=(12, 5))
  plt.plot(y_test, label="Actual Returns", linestyle='-', linewidth=2)
  plt.plot(y_pred, label="Predicted Returns", linestyle='--', linewidth=2)

  plt.xlabel("Test Sample Index")
  plt.ylabel("Returns")
  plt.title("Predicted vs Actual Returns")
  plt.legend()
  plt.grid(True)
  plt.tight_layout()
  plt.show()









