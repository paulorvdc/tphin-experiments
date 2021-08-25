import pandas as pd
from copy import deepcopy

from tphin_utils import period_year

df = pd.read_parquet('/media/pauloricardo/basement/commodities_usecase/soybean_corn.parquet')

df['DateStr'] = df['Date']
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
df['Week'] = df['Date'].dt.week
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year
df['WeekYear'] = df['Date'].apply(period_year, interval='week')
df['MonthYear'] = df['Date'].apply(period_year, interval='month')
df = df.sort_values(by='Date').reset_index(drop=True)
df = df.reset_index().rename(columns={'index':'EventId'})

from Giveme5W1H.extractor.document import Document
from Giveme5W1H.extractor.extractor import MasterExtractor

extractor = MasterExtractor()

components = {'what': [], 'where': [],'who': [], 'why': [], 'how': []}
for index, row in df.iterrows():
    doc = Document.from_text(row['Headlines'], row['DateStr'])
    doc = extractor.parse(doc)
    for component in components.keys():
      try:
          components[component].append(doc.get_top_answer(component).get_parts_as_text())
      except:
          components[component].append(None)
          
for component in components.keys():
    df[component] = components[component]


def get_label(df, commoditie_value_feature='Corn_Cepea_Dolar', interval_feature='WeekYear', commodity='Corn'):
    new_df = deepcopy(df)
    commodity_group = new_df[[commoditie_value_feature, interval_feature]].groupby(by=interval_feature, as_index=False, sort=False).mean()
    trends = []
    for index, row in commodity_group.iterrows():
        if index == 0:
            trends.append('up')
            continue
        tendency_value = row[commoditie_value_feature] - commodity_group[commoditie_value_feature].iloc[index-1]
        if tendency_value >= 4:
            trends.append('big_up')
        elif tendency_value >= 0:
            trends.append('up')
        elif tendency_value <= -4:
            trends.append('big_down')
        elif tendency_value < 0:
            trends.append('down')
    commodity_group[interval_feature + commodity + 'Trend'] = pd.Series(trends)
    new_df[interval_feature + commodity + 'Trend'] = [None] * len(new_df)
    for index, row in new_df.iterrows():
        new_df[interval_feature + commodity + 'Trend'].loc[index] = commodity_group[interval_feature + commodity + 'Trend'][commodity_group[interval_feature] == row[interval_feature]].iloc[0]
    return new_df

df = get_label(df)
df = get_label(df, commoditie_value_feature='Corn_Cepea_Dolar', interval_feature='MonthYear', commodity='Corn')
df = get_label(df, commoditie_value_feature='Soy_Cepea_Dolar', interval_feature='WeekYear', commodity='Soy')
df = get_label(df, commoditie_value_feature='Soy_Cepea_Dolar', interval_feature='MonthYear', commodity='Soy')

df.to_parquet('/media/pauloricardo/basement/commodities_usecase/soybean_corn_4w1h.parquet')