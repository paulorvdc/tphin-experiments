import pandas as pd
import networkx as nx
import numpy as np
from scipy.spatial.distance import cosine

def make_hin(df, 
             id_feature='EventId', date_feature='WeekYear', date_value_feature='Date', 
             what_feature='what', where_feature='where', who_feature='who', why_feature='why', how_feature='how',
             commodities_feature='WeekYearCornTrend'
             ):
    G = nx.Graph()
    for index,row in df.iterrows():
        node_id = 'Event' + str(row[id_feature])
        date_value = row[date_value_feature]
        node_date = row[date_feature]
        node_what = row[what_feature]
        node_where = row[where_feature]
        node_who = row[who_feature]
        node_why = row[why_feature]
        node_how = row[how_feature]
        # label
        node_commodities = row[commodities_feature]
        
        # event <-> date
        G.add_edge(node_id, node_date, edge_type='event_date', edge_value=date_value)
        G.nodes[node_id]['node_type'] = 'event'
        G.nodes[node_date]['node_type'] = 'date'
        # event <-> what
        if node_what is not None:
            G.add_edge(node_id, node_what, edge_type='event_what')
            G.nodes[node_what]['node_type'] = 'what'
        # event <-> where
        if node_where is not None:
            G.add_edge(node_id, node_where, edge_type='event_where')
            G.nodes[node_where]['node_type'] = 'where'
        # event <-> who
        if node_who is not None:
            G.add_edge(node_id, node_who, edge_type='event_who')
            G.nodes[node_who]['node_type'] = 'who'
        # event <-> why
        if node_why is not None:
            G.add_edge(node_id, node_why, edge_type='event_why')
            G.nodes[node_why]['node_type'] = 'why'
        # event <-> how
        if node_how is not None:
            G.add_edge(node_id, node_how, edge_type='event_how')
            G.nodes[node_how]['node_type'] = 'how'
        # event <-> trend
        if node_commodities is not None:
            G.add_edge(node_id, node_commodities, edge_type='event_trend')
            G.nodes[node_commodities]['node_type'] = 'trend'
        # embedding
        G.nodes[node_id]['embedding'] = row.embedding
    return G

def difference(start, end, interval):
    x = end - start
    r = {
            'week': int(x / np.timedelta64(1, 'W')),
            'fortnight': int(x / np.timedelta64(2, 'W')),
            'month': int(x / np.timedelta64(1, 'M'))
        }
    return r[interval]

def inner_connections(G, interval='week', embedding_feature='embedding', type_feature='edge_type', desired_type_feature='event_date', value_feature='edge_value', return_type_feature='event_event'):
    edges_to_add = []
    for node1, neighbor1 in G.edges:
        if embedding_feature in G.nodes[node1]:
            if G[node1][neighbor1][type_feature] == desired_type_feature:
                for node2, neighbor2 in G.edges:
                    if embedding_feature in G.nodes[node2]:
                        if G[node2][neighbor2][type_feature] == desired_type_feature:
                            temp_cosine = cosine(G.nodes[node1][embedding_feature], G.nodes[node2][embedding_feature])
                            if temp_cosine <= 0.5 and temp_cosine != 0.0:
                                if abs(difference(G[node1][neighbor1][value_feature], G[node2][neighbor2][value_feature], interval)) <= 3:
                                    edges_to_add.append((node1,node2))
    for new_edge in edges_to_add:
        G.add_edge(new_edge[0],new_edge[1],edge_type=return_type_feature)
    return G

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

