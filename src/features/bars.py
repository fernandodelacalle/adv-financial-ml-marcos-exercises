import pandas as pd
import numpy as np

def tick_bar(df, m):
    return df.iloc[::m]

def volume_bar(df, m):
    aux = df.reset_index()    
    idx = []
    vol_acum = []
    c_v = 0
    for i, v in aux.vol.items():
        c_v = c_v + v 
        if c_v >= m:
            idx.append(i)
            vol_acum.append(c_v)
            c_v = 0
    volume_bar = aux.loc[idx]
    volume_bar.loc[idx, 'cum_vol'] = vol_acum 
    volume_bar = volume_bar.set_index('date')
    return volume_bar

def dollar_bar(df, m):
    aux = df.reset_index()    
    idx = []
    d_acum = []
    c_dv = 0
    for i, dv in aux.dollar_vol.items():
        c_dv = c_dv + dv 
        if c_dv >= m:
            idx.append(i)
            d_acum.append(c_dv)
            c_dv = 0 
    dollar_bar = aux.loc[idx]
    dollar_bar.loc[idx, 'cum_dollar_vol'] = d_acum 
    dollar_bar = dollar_bar.set_index('date')
    return dollar_bar

def volume_bar_cum(df, m):
    aux = df.reset_index()
    cum_v = aux.vol.cumsum()  
    th = m
    idx = []
    for i, c_v in cum_v.items():
        if c_v >= th:
            th = th + m
            idx.append(i)
    return aux.loc[idx].set_index('date')

def dollar_bar_cum(df, m):
    aux = df.reset_index()
    cum_dv = aux.dollar_vol.cumsum()  
    th = m
    idx = []
    for i, c_dv in cum_dv.items():
        if c_dv >= th:
            th = th + m
            idx.append(i)
    return aux.loc[idx].set_index('date')

