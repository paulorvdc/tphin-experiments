import pandas as pd
import glob
import os

path = r'/media/pauloricardo/basement/commodities_usecase/sheets/'
all_files = glob.glob(os.path.join(path, "*.xlsx"))

def _read_excel(path):
    return pd.read_excel(path, engine='openpyxl')
    
df_from_each_file = (_read_excel(f) for f in all_files)
df = pd.concat(df_from_each_file, ignore_index=True).dropna()


from sentence_transformers import SentenceTransformer, LoggingHandler
import numpy as np
import logging

#### Just some code to print debug information to stdout
np.set_printoptions(threshold=100)

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

# Load Sentence model (based on BERT) from URL
model = SentenceTransformer('distiluse-base-multilingual-cased')

df['embedding'] = list(model.encode(df['Headlines'].to_list()))
df.to_parquet('/media/pauloricardo/basement/commodities_usecase/soybean_corn.parquet')