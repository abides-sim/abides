rm cache/*\.\.log*.pickle
rm -rf visualizations
python asset_returns_stylized_facts.py -r /media/nick/FinData/intraday/1m_ohlc/1m_ohlc_2011 -s ../log/random_fund_value -s ../log/random_fund_diverse -s ../log/hist_fund_value -s ../log/hist_fund_diverse