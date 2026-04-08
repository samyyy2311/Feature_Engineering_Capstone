# helpers.py
# reusable feature engineering functions for hotel bookings dataset

import numpy as np
import pandas as pd

def add_ratio_features(df):
    df = df.copy()
    df['price_per_person'] = df['adr'] / (df['adults'] + df['children'] + 1)
    df['special_requests_rate'] = df['total_of_special_requests'] / (
        df['stays_in_week_nights'] + df['stays_in_weekend_nights'] + 1
    )
    return df

def add_interaction_features(df):
    df = df.copy()
    df['adr_x_lead'] = df['adr'] * df['lead_time']
    df['booking_changes_x_lead'] = df['booking_changes'] * df['lead_time']
    return df

def add_simple_features(df):
    df = df.copy()
    df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['is_family'] = ((df['children'] + df['babies']) > 0).astype(int)
    df['high_value_customer'] = (df['adr'] > df['adr'].median()).astype(int)
    return df

def add_datetime_features(df):
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df = df.copy()
    df['arrival_month_num'] = df['arrival_date_month'].map(month_map)
    df['arrival_weekday'] = pd.to_datetime({
        'year': df['arrival_date_year'],
        'month': df['arrival_month_num'],
        'day': df['arrival_date_day_of_month'].fillna(1)
    }, errors='coerce').dt.dayofweek
    df['is_weekend_arrival'] = (df['arrival_weekday'] >= 5).astype(int)
    df['arrival_quarter'] = pd.to_datetime({
        'year': df['arrival_date_year'],
        'month': df['arrival_month_num'],
        'day': df['arrival_date_day_of_month'].fillna(1)
    }, errors='coerce').dt.quarter
    df['lead_time_bucket'] = pd.cut(
        df['lead_time'].fillna(0),
        bins=[-1, 7, 30, 90, 180, 999],
        labels=['same_week', '1_month', '3_month', '6_month', '6plus_month']
    )
    return df

def add_aggregated_features(df_train, df_full):
    country_mean_adr = df_train.groupby('country')['adr'].mean()
    seg_cancel_rate = df_train.groupby('market_segment')['is_canceled'].mean()
    df_full = df_full.copy()
    df_full['country_mean_adr'] = df_full['country'].map(country_mean_adr).fillna(df_train['adr'].mean())
    df_full['segment_cancel_rate'] = df_full['market_segment'].map(seg_cancel_rate).fillna(df_train['is_canceled'].mean())
    return df_full
