# app.py
"""Streamlit app ported from the Brazilian E-Commerce notebook.
- Loads CSVs from `data/` (falls back to Kaggle download if missing)
- Preprocesses and creates derived columns used in the notebook
- Provides basic dashboard with filters & charts
"""

import os
from pathlib import Path
import zipfile
import warnings

import pandas as pd
import streamlit as st

import seaborn as sns
import matplotlib.pyplot as plt
import json
import logging

# configure basic logger so messages show in Streamlit Cloud logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except Exception:
    KAGGLE_AVAILABLE = False

# Ensure Kaggle credentials are available. If environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY` exist,
# write a `kaggle.json` file and set `KAGGLE_CONFIG_DIR` so the Kaggle API can authenticate.
def ensure_kaggle_config():
    kaggle_username = os.environ.get('KAGGLE_USERNAME')
    kaggle_key = os.environ.get('KAGGLE_KEY')
    # Default to project root if KAGGLE_CONFIG_DIR not set
    kaggle_dir = os.environ.get('KAGGLE_CONFIG_DIR', str(Path.cwd()))

    if kaggle_username and kaggle_key:
        logger.info('Kaggle credentials found in environment, writing kaggle.json to %s', kaggle_dir)
        os.environ['KAGGLE_CONFIG_DIR'] = kaggle_dir
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_json_path = Path(kaggle_dir) / 'kaggle.json'
        if not kaggle_json_path.exists():
            with open(kaggle_json_path, 'w') as f:
                json.dump({"username": kaggle_username, "key": kaggle_key}, f)
            try:
                kaggle_json_path.chmod(0o600)
            except Exception:
                pass
    else:
        # If user already set KAGGLE_CONFIG_DIR to a folder containing kaggle.json, respect it
        kg_dir = os.environ.get('KAGGLE_CONFIG_DIR')
        if kg_dir and (Path(kg_dir) / 'kaggle.json').exists():
            logger.info('Using existing kaggle.json in KAGGLE_CONFIG_DIR: %s', kg_dir)
            os.environ['KAGGLE_CONFIG_DIR'] = kg_dir
        else:
            logger.info('No Kaggle credentials or kaggle.json found; download will not be available without credentials')

# Run at import time so download_dataset() can authenticate
ensure_kaggle_config()

# Helper to safely trigger a rerun; some Streamlit runtime versions may not expose experimental_rerun
def safe_rerun():
    try:
        st.experimental_rerun()
    except Exception as e:
        logger.warning('st.experimental_rerun not available or failed: %s', e)
        # Fall back to instructing the user to manually reload the page
        try:
            st.info('Download finished — please refresh the page (browser reload or press R) to load the new files.')
        except Exception:
            # If Streamlit UI not available (unlikely in runtime), just log
            logger.info('Prompt user to refresh the page to pick up downloaded files.')

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

FILES = {
    'orders': 'olist_orders_dataset.csv',
    'order_items': 'olist_order_items_dataset.csv',
    'payments': 'olist_order_payments_dataset.csv',
    'reviews': 'olist_order_reviews_dataset.csv',
    'customers': 'olist_customers_dataset.csv',
    'products': 'olist_products_dataset.csv',
    'sellers': 'olist_sellers_dataset.csv',
    'category': 'product_category_name_translation.csv'
}


def download_dataset():
    """Try to download and unzip the Olist dataset using Kaggle API."""
    if not KAGGLE_AVAILABLE:
        st.warning("Kaggle API not available. Please place CSVs inside the `data/` folder.")
        logger.error('Kaggle package is not installed in this environment')
        return False

    api = KaggleApi()
    try:
        logger.info('Authenticating with Kaggle API...')
        api.authenticate()
        logger.info('Starting dataset download to %s', DATA_DIR)
        api.dataset_download_files('olistbr/brazilian-ecommerce', path=str(DATA_DIR), unzip=True)
        logger.info('Dataset download complete')
        return True
    except Exception as e:
        logger.exception('Kaggle download failed')
        st.warning(f"Kaggle download failed: {e}")
        return False


@st.cache_data
def load_data():
    # Ensure files exist (try Kaggle if missing)
    missing = [f for f in FILES.values() if not (DATA_DIR / f).exists()]
    if missing:
        success = download_dataset()
        if not success:
            raise FileNotFoundError(f"Missing files: {missing}. Put CSVs in `data/` or enable Kaggle download.")

    dfs = {}
    for key, fn in FILES.items():
        path = DATA_DIR / fn
        dfs[key] = pd.read_csv(path)
    return dfs


def preprocess(dfs: dict) -> pd.DataFrame:
    orders = dfs['orders']
    payments = dfs['payments']
    items = dfs['order_items']
    reviews = dfs['reviews']
    customers = dfs['customers']
    products = dfs['products']
    sellers = dfs['sellers']
    category = dfs['category']

    # Basic merges following the notebook
    df = orders.copy()

    # Review aggregation
    review_agg = reviews.groupby('order_id', as_index=False).agg({'review_score': 'mean'})
    df = df.merge(review_agg, on='order_id', how='left')

    # Payments aggregation
    total_payment = payments.groupby('order_id')['payment_value'].sum().reset_index(name='total_payment')
    installments = payments.groupby('order_id')['payment_installments'].max().reset_index(name='payment_installments')
    payment_type = payments.groupby('order_id')['payment_type'].first().reset_index(name='payment_type')
    payments_agg = total_payment.merge(installments, on='order_id').merge(payment_type, on='order_id')
    df = df.merge(payments_agg, on='order_id', how='left')

    # Product category mapping
    category = products.merge(category, on='product_category_name', how='left') \
        .drop(columns=['product_category_name']) \
        .rename(columns={'product_category_name_english': 'product_category'})

    # Items aggregation
    items_full = items.merge(category[['product_id', 'product_category']], on='product_id', how='left')
    total_price = items_full.groupby('order_id')['price'].sum().reset_index(name='total_price')
    total_freight = items_full.groupby('order_id')['freight_value'].sum().reset_index(name='total_freight')
    num_products = items_full.groupby('order_id')['product_id'].nunique().reset_index(name='num_products')
    product_category = items_full.groupby('order_id')['product_category'].first().reset_index(name='product_category')
    items_agg = total_price.merge(total_freight, on='order_id').merge(num_products, on='order_id').merge(product_category, on='order_id')
    df = df.merge(items_agg, on='order_id', how='left')

    # Seller info
    seller_agg = items.merge(sellers[['seller_id', 'seller_state']], on='seller_id', how='left') \
        .groupby('order_id', as_index=False).agg(seller_state=('seller_state','first'))
    df = df.merge(seller_agg, on='order_id', how='left')

    # Datetime conversions
    date_cols = ['order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce')

    # Derived columns
    df['order_condition'] = df.apply(lambda row: 'canceled' if row['order_status']=='canceled' else
                                     ('not_approved' if pd.isna(row['order_approved_at']) else
                                      ('not_delivered' if pd.isna(row['order_delivered_customer_date']) else 'delivered')
                                     ), axis=1)

    df['review_score'] = df['review_score'].fillna('no_review')
    df = df.dropna(subset=['total_payment', 'payment_installments'])
    df['payment_type'] = df['payment_type'].fillna(df['payment_type'].mode().iloc[0])

    mask_delivered = df['order_status'] == 'delivered'
    df.loc[mask_delivered, ['total_price', 'total_freight', 'num_products']] = df.loc[mask_delivered, ['total_price', 'total_freight', 'num_products']].fillna(0)

    df['product_category'] = df['product_category'].fillna('no_category')
    df['seller_state'] = df['seller_state'].fillna('unknown')

    df['delivery_days'] = (df['order_delivered_customer_date'] - df['order_purchase_timestamp']).dt.days
    df['is_late'] = df['order_delivered_customer_date'] > df['order_estimated_delivery_date']

    return df


def main():
    st.set_page_config(page_title='Olist E-Commerce Dashboard', layout='wide')
    st.title('Olist - Brazilian E-Commerce')

    # Safely read secrets from Streamlit (works on Cloud and local)
    try:
        k_username = st.secrets.get('KAGGLE_USERNAME') if hasattr(st, 'secrets') else None
        k_key = st.secrets.get('KAGGLE_KEY') if hasattr(st, 'secrets') else None
    except Exception:
        # Any parsing issue or missing file — treat as no secrets
        k_username = None
        k_key = None

    if k_username and k_key:
        kg_dir = os.environ.get('KAGGLE_CONFIG_DIR', '/tmp')
        os.makedirs(kg_dir, exist_ok=True)
        kaggle_json_path = Path(kg_dir) / 'kaggle.json'
        if not kaggle_json_path.exists():
            with open(kaggle_json_path, 'w') as f:
                json.dump({"username": k_username, "key": k_key}, f)
            try:
                kaggle_json_path.chmod(0o600)
            except Exception:
                pass
        os.environ['KAGGLE_CONFIG_DIR'] = str(kg_dir)

    # Check which files are present
    missing_files = [v for v in FILES.values() if not (DATA_DIR / v).exists()]

    st.sidebar.header('Dataset & Settings')
    st.sidebar.write('Data directory: ', str(DATA_DIR))

    if missing_files:
        st.warning('Missing dataset files: ' + ', '.join(missing_files))

        # Diagnostics expander
        with st.expander('Diagnostics & Fixes'):
            st.write('Steps to fix:')
            st.write('1. Add Kaggle credentials in **App → Settings → Secrets** (two separate entries).')
            st.write('2. Click **Try Download (Kaggle)** below (requires secrets).')
            st.write('3. Or upload `brazilian-ecommerce.zip` into `data/` and use the Unzip button.')

            st.markdown('**Secrets example (TOML, do not paste JSON):**')
            st.code('''KAGGLE_USERNAME = "your_kaggle_username"
KAGGLE_KEY = "your_kaggle_key"''', language='toml')

            # Show current content of data/ for debugging
            try:
                files_in_data = sorted([p.name for p in DATA_DIR.glob('*')])
            except Exception:
                files_in_data = []
            st.write('Files currently in `data/`:', files_in_data if files_in_data else 'No files found')

            # Manual unzip action
            zips = list(DATA_DIR.glob('*.zip'))
            if zips:
                for z in zips:
                    st.info(f'Found zip: {z.name}')
                if st.button('Unzip dataset files found in data/'):
                    with st.spinner('Unzipping...'):
                        for z in zips:
                            try:
                                with zipfile.ZipFile(z, 'r') as zip_ref:
                                    zip_ref.extractall(DATA_DIR)
                                st.success(f'Unzipped {z.name}')
                            except Exception as e:
                                st.error(f'Failed to unzip {z.name}: {e}')
                        safe_rerun()

            # Try Download using Kaggle (requires Kaggle API and valid secrets)
            st.markdown('---')
            if not KAGGLE_AVAILABLE:
                st.error('Kaggle API package not installed on this instance. Automatic download not available.')
            else:
                st.write('Kaggle package is available.')
                secret_note = st.empty()
                if not (k_username and k_key):
                    secret_note.warning('No Kaggle credentials found. Add them in App → Settings → Secrets as two entries: `KAGGLE_USERNAME` and `KAGGLE_KEY`.')
                if st.button('Try Download (Kaggle)'):
                    if not (k_username and k_key):
                        st.error('Kaggle credentials missing. Add secrets first.')
                    else:
                        out = st.empty()
                        with out.container():
                            st.info('Starting download...')
                            try:
                                ok = download_dataset()
                                if ok:
                                    st.success('Download finished. Re-running the app to pick up files...')
                                    safe_rerun()
                                else:
                                    st.error('Download attempted but reported failure. Check logs below or run locally with kaggle CLI.')
                            except Exception as e:
                                st.error(f'Download failed with error: {e}')
        st.info('Alternative: download dataset manually from Kaggle and place CSVs under `data/`')
        st.stop()

    # All files present — load and continue
    try:
        with st.spinner('Loading data...'):
            dfs = load_data()
            df = preprocess(dfs)
    except Exception as e:
        st.error(f'Failed to load data: {e}')
        st.stop()

    # Sidebar filters
    st.sidebar.header('Filters')
    order_conditions = st.sidebar.multiselect('Order Condition', options=df['order_condition'].unique(), default=df['order_condition'].unique())

    # Date range selector
    min_date = df['order_purchase_timestamp'].min().date() if not df['order_purchase_timestamp'].isna().all() else None
    max_date = df['order_purchase_timestamp'].max().date() if not df['order_purchase_timestamp'].isna().all() else None
    if min_date and max_date:
        date_range = st.sidebar.date_input('Order purchase date range', [min_date, max_date], min_value=min_date, max_value=max_date)
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    else:
        start_dt = pd.to_datetime('1970-01-01')
        end_dt = pd.to_datetime('2099-01-01')

    # Category and seller filters
    categories = st.sidebar.multiselect('Product Category', options=sorted(df['product_category'].dropna().unique()), default=None)
    seller_states = st.sidebar.multiselect('Seller State', options=sorted(df['seller_state'].dropna().unique()), default=None)

    # Top N selector for ranking charts
    top_n = st.sidebar.slider('Top N for rankings', min_value=5, max_value=30, value=10)

    # Apply filters
    mask = df['order_condition'].isin(order_conditions) & (df['order_purchase_timestamp'] >= start_dt) & (df['order_purchase_timestamp'] <= end_dt)
    if categories:
        mask = mask & df['product_category'].isin(categories)
    if seller_states:
        mask = mask & df['seller_state'].isin(seller_states)
    df_filtered = df[mask]

    # Key metrics & insights
    total_orders = df_filtered['order_id'].nunique()
    total_revenue = df_filtered['total_payment'].sum()
    avg_delivery = df_filtered['delivery_days'].dropna().mean()
    late_count = df_filtered['is_late'].astype(bool).sum()
    delivered_count = df_filtered[df_filtered['order_condition']=='delivered'].shape[0]
    late_rate = (late_count / delivered_count) if delivered_count else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric('Orders', f'{total_orders:,}')
    k2.metric('Revenue', f'Rp {total_revenue:,.0f}')
    k3.metric('Avg Delivery Days', f'{avg_delivery:.1f}' if not pd.isna(avg_delivery) else 'N/A')
    k4.metric('Late Rate', f'{late_rate:.1%}')

    # Generate short automated insights based on the filtered data
    ins_col1, ins_col2 = st.columns(2)
    with ins_col1:
        st.subheader('Quick Insights')
        # Top product by count
        if not df_filtered.empty and 'product_category' in df_filtered.columns:
            top_by_count = df_filtered[df_filtered['order_condition']=='delivered']['product_category'].value_counts().idxmax() if not df_filtered[df_filtered['order_condition']=='delivered'].empty else None
            top_by_revenue = df_filtered[df_filtered['order_condition']=='delivered'].groupby('product_category')['total_payment'].sum().idxmax() if not df_filtered[df_filtered['order_condition']=='delivered'].empty else None
            st.write(f'- Top product (by orders): **{top_by_count}**' if top_by_count else '- Top product (by orders): N/A')
            st.write(f'- Top product (by revenue): **{top_by_revenue}**' if top_by_revenue else '- Top product (by revenue): N/A')
        else:
            st.write('No product info available for the selected filters.')

        # Payment type dominance
        if not df_filtered.empty and 'payment_type' in df_filtered.columns:
            dominant_payment = df_filtered['payment_type'].value_counts().idxmax()
            st.write(f'- Dominant payment type: **{dominant_payment}**')

    with ins_col2:
        st.subheader('Delivery & Satisfaction')
        if delivered_count:
            mean_review_on_time = df_filtered[(df_filtered['order_condition']=='delivered') & (df_filtered['is_late']==False) & (df_filtered['review_score']!='no_review')]['review_score'].mean()
            mean_review_late = df_filtered[(df_filtered['order_condition']=='delivered') & (df_filtered['is_late']==True) & (df_filtered['review_score']!='no_review')]['review_score'].mean()
            st.write(f'- Avg review (on time): **{mean_review_on_time:.2f}**' if not pd.isna(mean_review_on_time) else '- Avg review (on time): N/A')
            st.write(f'- Avg review (late): **{mean_review_late:.2f}**' if not pd.isna(mean_review_late) else '- Avg review (late): N/A')
            if not pd.isna(mean_review_on_time) and not pd.isna(mean_review_late):
                if mean_review_late < mean_review_on_time:
                    st.info('Orders delivered late have lower average review scores — consider improving delivery SLAs.')
                else:
                    st.info('No clear negative impact of late delivery on review score in this subset.')
        else:
            st.write('No delivered orders in selection.')

    # Top products (count & revenue)
    st.header('Produk Paling Laris & Revenue Tertinggi')
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Jumlah Penjualan (Top 10)')
        prod_count = df_filtered[df_filtered['order_condition']=='delivered']['product_category'].value_counts().nlargest(10)
        st.bar_chart(prod_count)

    with col2:
        st.subheader('Revenue (Top 10)')
        prod_revenue = df_filtered[df_filtered['order_condition']=='delivered'].groupby('product_category')['total_payment'].sum().nlargest(10)
        st.bar_chart(prod_revenue)

    # Late vs Review
    st.header('Pengaruh Keterlambatan ke Kepuasan Pelanggan')
    if 'is_late' in df_filtered.columns and 'review_score' in df_filtered.columns:
        delivered_orders = df_filtered[(df_filtered['order_condition']=='delivered') & (df_filtered['review_score']!='no_review')]
        if not delivered_orders.empty:
            late_review = delivered_orders.groupby('is_late')['review_score'].mean()
            st.bar_chart(late_review)

    # Payment type
    st.header('Metode Pembayaran')
    pay_count = df_filtered.groupby('payment_type')['order_id'].count()
    pay_revenue = df_filtered.groupby('payment_type')['total_payment'].sum()
    col3, col4 = st.columns(2)
    with col3:
        st.subheader('Jumlah Transaksi')
        st.bar_chart(pay_count)
    with col4:
        st.subheader('Total Payment')
        st.bar_chart(pay_revenue)

    # Seller performance
    st.header('Seller Performance')
    seller_perf = df_filtered[df_filtered['order_condition']=='delivered'].groupby('seller_state').agg({'delivery_days':'mean','order_id':'count'}).rename(columns={'delivery_days':'avg_delivery_days','order_id':'num_orders'}).sort_values('num_orders', ascending=False)
    st.dataframe(seller_perf.head(10))

    # Correlation / boxplot
    st.header('Korelasi & Distribusi')
    fig, ax = plt.subplots(figsize=(10,5))
    sns.boxplot(x='num_products', y='total_payment', data=df_filtered, ax=ax)
    st.pyplot(fig)


if __name__ == '__main__':
    main()
