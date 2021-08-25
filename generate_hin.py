import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

from tphin_utils import make_hin
from tphin_utils import inner_connections

df = pd.read_parquet('/media/pauloricardo/basement/commodities_usecase/soybean_corn_4w1h.parquet')

G_week_corn = make_hin(df)
G_week_corn = inner_connections(G_week_corn)
nx.write_gpickle(G_week_corn, "/media/pauloricardo/basement/commodities_usecase/corn_week.gpickle")

G_week_soy = make_hin(df, commodities_feature='WeekYearSoyTrend')
G_week_soy = inner_connections(G_week_soy)
nx.write_gpickle(G_week_soy, "/media/pauloricardo/basement/commodities_usecase/soybean_week.gpickle")

G_month_corn = make_hin(df, date_feature='MonthYear', commodities_feature='MonthYearCornTrend')
G_month_corn = inner_connections(G_month_corn)
nx.write_gpickle(G_month_corn, "/media/pauloricardo/basement/commodities_usecase/corn_month.gpickle")

G_month_soy = make_hin(df, date_feature='MonthYear', commodities_feature='MonthYearSoyTrend')
G_month_soy = inner_connections(G_month_soy)
nx.write_gpickle(G_month_soy, "/media/pauloricardo/basement/commodities_usecase/soybean_month.gpickle")

