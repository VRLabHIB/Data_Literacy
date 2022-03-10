# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:44:55 2022

@author: Stark
"""

import webbrowser
import os
import pandas as pd

def df_html(df):
    """HTML table with pagination and other goodies"""
    df_html = df.to_html()
    return df_html
    
def show_df(df):
    path = os.path.abspath('data.html')
    url = 'file://' + path

    with open(path, 'w') as f:
        f.write(df_html(df))
    webbrowser.open(url)
    