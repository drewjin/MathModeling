all_edges = defaultdict(float)
for a in a_team:
    for b in b_team:
        cur_edge = (a,b)
        data = g.get_edge_data(a,b)
        if data is None:
            all_edges[cur_edge] = 0
        else:
            score = data['score']
            if data['weight'] >= 0:
                if score[0] > score[1]:
                    all_edges[cur_edge] = float(score[0] / sum(score))
                elif score[0] == score[1]:
                    all_edges[cur_edge] = float(score[0] / sum(score))
                else:
                    all_edges[cur_edge] = 0
            elif data['weight'] < 0:
                if score[0] < score[1]:
                    all_edges[cur_edge] = float(score[0] / sum(score))
                elif score[0] == score[1]:
                    all_edges[cur_edge] = float(score[0] / sum(score))
                else:
                    all_edges[cur_edge] = 0
for elem in all_edges.items():
    if elem[-1] != 0:
        print(elem)
for edge in g.edges(data=True):
    print(edge)