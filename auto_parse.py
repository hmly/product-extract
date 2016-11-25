"""
Automatically parse a table of item description and identify two groups or more
If a word describes an attribute and the word that it is describing
Adjective and Noun relation
v1      25 February 2015
v2      18 March 2015
v3      06 April 2015
v4      01 May 2015

Products:       52 226
Unique Words:   392 915
Departs:        17 out of 17 (attrib-rich)
"""

import nltk
import networkx as nx
import math
import numpy as np
import re

from collections import Counter
from sklearn import cluster
from regex_patterns import patterns

__author__ = 'Hmly'


# global static variables
PARSE = False  # desc - eg. 1FT -> $DIM1
clusters_info = {}  # all clusters
keywords = []
link = {}  # all node edges
methods = ('overlap', 'entropy')
prod = {}  # all products - id: (desc, dept, ...)
prod_by_attrib = {}  # products organized by attributes
prod_by_dept = {}  # products organized by dept
"""
0-4 NOT normaized and 5-6 normalized
items order of to_incl musst match order of node_features
with additional item to include, increase value of normal index by 1
eg. degree placed at index 2
"""
feat_to_index = {'assort': 0,
                 'clust': 1,
                 'degree': 2,
                 'eigenv': 3,
                 'weigh': 4,
                 'betw': 5,
                 'clos': 6}
to_excl = ('betw', 'clos', 'weigh', 'degree', 'eigenv')
to_incl = ('assort', 'clust')
norm_index = 3


def get_data():
    """
    read saved data from files
    """
    with open('products/updated_product.txt') as f:
        f.readline()  # ignore 1st line of headers
        for line in f:
            l = line.replace('\n', '').split('|')
            prod[int(l[0])] = tuple(l[1:])

    with open('Data/departs_dict.txt') as f:
        for line in f:
            name, ids = line.split('||')
            prod_by_dept[name] = [int(n) for n in ids.split('|')[:-1]]

    with open('Data/attributes_dict.txt') as f:
        for line in f:
            att, ids = line.split('||')
            prod_by_attrib[att] = [int(n) for n in ids.split('|')[:-1]]

    with open('attributes/all_keywords.txt') as f:
        for line in f:
            line = line.replace('\n', '').split()
            for w in line:
                keywords.append(w)

    """lefts = ['KILLER', 'KLLR', 'TRAP', 'ANNUAL', 'PERENNIAL', 'SEASONAL', 'BL', 'BK', 'GR', 'BRWN',
                'GRN', 'GRY', 'ORG', 'WHT', 'YLW', 'GALLONS', 'QUART', 'LIFETIME', 'WARRANTY', 'WARNTY']
    for w in lefts:
        keyword.append(w)"""


def get_graph():
    """
    read from saved graphml file
    :return Graph object
    """
    return nx.read_graphml('data/graphs/products_desc.graphml')


def get_weights():
    """
    read from saved weights file
    builld a frequency distribution
    :return dict fd
    """
    fd = {}
    with open('Data/weights.txt') as f:
        for line in f:
            key, count = line[:-1].split('|')
            fd[key] = int(count)
    return fd


def get_vectors():
    """
    read vector file and exlucde the features in to_excl
    build dict of vectors minus excluded features
    :return dict of word: node features
    """
    excl_indices = [feat_to_index[s] for s in to_excl]  # replace names with int index
    vectors_minus_excl = {}

    with open('data/vectors/vectors.txt') as f:
        for line in f:
            key, features = line.split('|')
            features = [float(n) for n in features[1:-2].split(', ')]  # clean the raw string
            vectors_minus_excl[key] = [features[i] for i in range(len(features)) if i not in excl_indices]

    return vectors_minus_excl


def normalize(data):
    """
    normalize value to range of -1 <= x <= 1
    :param data-collection of values
    :return normalize value
    """
    # value = np.linalg.norm(data)
    max_x = max(data.values())
    min_x = min(data.values())

    return {x: (data[x] - min_x)/(max_x - min_x) for x in data.keys()}


def generate_vector(graph):
    """
    calculate node features
    :return set of dicts
    """
    assrt = nx.assortativity.average_neighbor_degree(graph)
    clst = nx.clustering(graph)
    dg = nx.degree(graph)
    ev = nx.eigenvector_centrality(graph)
    wg = get_weights()
    # bt = nx.betweenness_centrality(graph)
    # cls = nx.closeness_centrality(graph)

    '''return (nx.assortativity.average_neighbor_degree(graph),
            nx.clustering(graph),
            nx.degree(graph),
            nx.eigenvector_centrality(graph),
            get_weights(),
            nx.betweenness_centrality(graph),
            nx.closeness_centrality(graph))'''
    return [assrt, clst, dg, ev, wg]


def display_coef_matrix(coef_matrix):
    """
    display and save correlation coefficient matrix of features
    only in to_incl
    :param coef_matrix: list of correlation between features
    """
    with open('Data/features_correlation.txt', 'w') as f:
        for feat in to_incl:
            print('%13s' % feat, end='')
            f.write('%13s' % feat)
        print()
        f.write('\n')

        for i in range(len(coef_matrix)):
            for coef in coef_matrix[i]:
                f.write('%13.3f' % coef)
                print('%13.3f' % coef, end='')
            print('%15s' % to_incl[i], end='')
            f.write('%15s\n' % to_incl[i])
            print()


def generate_nodes_vectors(graph):
    """
    generate list of dict from graph
    extract and organize features by node - vectors
    :param graph: object
    """
    feats = generate_vector(graph)
    data = {}

    # save raw data of vectors
    raw = {}
    for node in graph.nodes():
        raw[node] = [feats[i][node] for i in range(len(feats))]
    with open('data/vectors/vectors_raw.txt', 'w') as f:
        for node in raw.keys():
            f.write(str(node) + ' ' + str(raw[node])[1:-1].replace(',', '') + '\n')

    # normalize values
    for i in range(norm_index):
        feats[i] = normalize(feats[i])

    # build dict of node name: vector of features
    for node in graph.nodes():
        data[node] = [feats[i][node] for i in range(len(feats))]

    # normalize vectors ending at norm_index
    """for i in range(norm_index):
        values = list(feats[i].values())
        for j in data.keys():
            data[j][i] = normalize(data[j][i], values)"""

    # save to file
    with open('data/vectors/vectors.txt', 'w') as f, open('data/vectors/vectors_excel.txt', 'w') as f2:
        for node in data.keys():
            f2.write(str(node) + ' ' + str(data[node])[1:-1].replace(',', '') + '\n')
            f.write('|'.join([str(node), str(data[node])]) + '\n')


def compute_features(vectors):
    """
    generate correlation coefficient matrix of node features in to_incl
    :param vectors: dict of node to vectors
    """
    n_att = len(feat_to_index) - len(to_excl)
    norm = [[] for _ in range(n_att)]

    # build new lisst organize by feature instead of node
    for i in range(n_att):
        for node in vectors.keys():
            norm[i].append(vectors[node][i])

    # build matrix of features correlation
    coef_matrix = np.corrcoef(norm)
    display_coef_matrix(coef_matrix)


def generate_graph(directed=False):
    """
    create a undirected/directed graph and save data
    :param directed: option to build a directed graph
    """
    if directed:
        graph = nx.DiGraph()
        for relation in link.keys():
            graph.add_edge(relation[0], relation[1], weight=link[relation])
    else:
        graph = nx.Graph()
        for relation in link.keys():
            graph.add_edge(relation[0], relation[1])

    with open('Data/nodes.txt', 'w') as f:
        for key in sorted(link.keys()):
            f.write(str(key) + '\n')

    nx.write_graphml(graph, 'data/graphs/products_desc.graphml')


def filter_desc():
    """
    build list of product desc from selected departments
    calculate weights using freq dist
    """
    get_data()
    parsed_desc = []
    depts_to_incl = ('LUMBER', 'BUILDING MATERIALS', 'PAINT', 'HARDWARE', 'SEASONAL/GARDEN',
                     'MILLWORK', 'WALL/FLOOR COVERING', 'HARDWARE (25H)', 'TOOLS (25T)', 'BATH',
                     'PLUMBING', 'ELECTRICAL', 'LIGHTING', 'GRD/INDOOR', 'GRD/OUTDOOR',
                     'KITCHEN', 'STORAGE')
    # ('BATH', 'BUILDING MATERIALS', 'ELECTRICAL', 'LIGHTING', 'LUMBER', 'PAINT', 'STORAGE')
    # 'MILLWORK', 'STORAGE', 'PAINT', 'TOOLS (25T)')
    stopwords = nltk.corpus.stopwords.words('english')
    items = [prod[i_id][0] for dept in depts_to_incl for i_id in prod_by_dept[dept]]

    for itm in items:
        parsed = [w for w in itm.split() if w not in stopwords and len(w) > 2]
        parsed_desc.append(parsed)

    fd = nltk.FreqDist([w for grp in parsed_desc for w in grp])
    print('products', len(items), '...', 'uniques', len(list(fd.keys())))
    with open('Data/weights.txt', 'w') as f:
        for node in fd.keys():
            f.write('|'.join([str(node), str(fd[node])]) + '\n')

    return parsed_desc


def parse_products():
    """
    arrange words in same item desc as pair eg. (orange, tray), (tray, orange) -> (orange, tray)
    build links of nodes as dict of (pair of words): weight (# of occurrences)
    """
    parsed_desc = filter_desc()

    words_relation = [(desc[i], desc[j], 1 / (j - i)) for desc in parsed_desc
                      for i in range(len(desc))
                      for j in range(i + 1, len(desc))]

    for relation in words_relation:
        if relation in link:
            link[tuple(sorted(relation[:2]))] += relation[2]
        else:
            link[tuple(sorted(relation[:2]))] = relation[2]
    generate_graph()


def find_dist(node, center):
    """
    calculate the distance between two point
    :param node: node within the cluster
    :param center: center point of the cluster
    :return distance: x > 0
    """
    return math.sqrt(sum(math.pow((n - m), 2) for n, m in zip(node, center)))


def find_max_dist(clust, center):
    """
    find distance between two nodes of cluster
    from various tests, the data is skewed to the left thus median is more appropriate
    find the media distance (may not exist in data) using
    find closest value to left and right of median (one with min diff)
    :param clust: cluster list of node names
    :param center: center point of the cluster
    :return tuple of median, center, dist: median (found based on distribution)
    """
    dist = [find_dist(node, center) for node in clust]

    if len(dist) > 0:
        median = np.median(dist)
        lower = max(d for d in dist if d <= median)
        upper = max(d for d in dist if d >= median)

        if abs(lower - median) < (upper - median):
            return (lower, center), dist
        elif abs(lower - median) > (upper - median):
            return (upper, center), dist
        else:
            return (median, center), dist
    return (0, center), dist


def generate_overlap_coef(clusters):
    """
    calculate the overlapping coefficient of two clusters
    repeat procedure for all clusters
    :param clusters: dict of center to list of points
    """
    coef = []
    points = []
    for center in clusters.keys():
        radius, list_of_points = find_max_dist(clusters[center], center)
        coef.append(radius)
        points.append(list_of_points)

    return coef, points


def mean_or_median(points):
    """
    display diff between median vs. mean via histogram distribution
    victor: median due to mostly left skewed data
    """
    for i, cl in enumerate(points):
        data = Counter([int(d * 100) / 100 for d in cl])
        for d in sorted(data.keys()):
            print(d, '*' * data[d])
        print('-' * 10 + str(i) + '-' * 10)


def generate_optimal_k(clusters, centers):
    """
    find the best k via the number of clusters that overlap and size of each cluster
    :param clusters: dict of center to list of points
    :param centers: list of centers for all clusters
    :return cluster_coef: dict of center to overlapping coefficient with other clusters
    """
    groups = list(zip(clusters, centers))
    comms = {}
    clusters_coef = {}

    # Create dict of centers to nodes points groups: [(node points), (center)]
    for com in groups:
        comms[tuple(com[1])] = com[0]

    # Create relation b/w center and community #
    for n, cn in enumerate(centers):
        clusters_info[tuple(cn)] = n

    # Create coefficient table - building table of coefficient for each cluster
    clusters_radius, points = generate_overlap_coef(comms)
    for cl in clusters_radius:
        rad, cn = cl
        clusters_coef[cn] = []
        for other_rad, other_cn in [cl2 for cl2 in clusters_radius if cl2 != cl]:
            dist_bw_centers = find_dist(cn, other_cn)
            coef = (rad + other_rad - dist_bw_centers) / dist_bw_centers
            clusters_coef[cn].append(coef)

    # mean_or_median(points)
    return clusters_coef


def overlapping_coef(clusters, total_exp, n_nodes, n_attribs):
    sig_cl = 0
    sig_attrib = 0

    print('ID# %10s %15s %8s %8s %15s' % ('SIZE', 'ACTUAL-ATTRIB', 'p', 'z', 'SIGNIFICANCE'))
    for satr, c, act, exp, size, feats, com in sorted(clusters, reverse=True):
        if satr > 1.96:
            sig_cl += size
            sig_attrib += act
        # idat = [satr for satr, c, act, exp, size, feats in clusters]
        # if satr == max(idat) or satr == min(idat):
        print('%3d %10d %15d %8.2f %8.2f %15s' % (c, size, act, act / size, satr, satr > 1.96), end=' ')
        for f in feats:
            print('%8.5f' % f[0], end=' ')
        print()
        total_exp.append(exp)
    print('ID#  %9d %15d' % (n_nodes, n_attribs))
    print('%13.2f%% %14.2f%% %8s' % (
        sig_cl / n_nodes * 100, sig_attrib / n_attribs * 100, sig_attrib / n_attribs * 100 > 40))

    raw_data = {}
    with open('data/vectors/vectors_raw.txt') as f:
        for line in f:
            vals = line.split()
            raw_data[vals[0]] = vals[1:]

    with open('data/vectors/vectors_sigf.txt', 'w') as f:
        sigf_words = []
        for satr, c, act, exp, size, feats, com in clusters:
            if satr > 1.96:
                for w in com:
                    sigf_words.append(w)
        for word in raw_data.keys():
            if word in sigf_words:
                f.write(' '.join([word, str(raw_data[word])[1:-1].replace(',', ''), 'True', '\n']))
            else:
                f.write(' '.join([word, str(raw_data[word])[1:-1].replace(',', ''), 'False', '\n']))


def entropy_coef(clusters):
    vals = []
    for satr, c, act, exp, size, asize in clusters:
        vals.append(satr)
    entropy = []
    for sat in vals:
        try:
            entr = sat * math.log10(sat)
        except (ZeroDivisionError, ValueError):
            entr = 0
        finally:
            entropy.append(entr)
    print('%8.3f' % (sum(entropy)))


def generate_stats(community, method, vectors):
    """
    generate statistics of each clusters
    :param community: list of list of all clusters points
    """
    # Generate statistics of # of colors and other attributes for each cluster
    n_nodes = sum(len(cl) for n, cl in community)
    n_attribs = len(set([w for n, cl in community
                         for w in cl
                         for p in patterns.values() if re.findall(p, w)]))
    exp_n_attrib = n_attribs / n_nodes
    p = n_attribs / n_nodes
    total_exp = []
    clusters = []

    for c, com in community:
        act_attrib = len(set([w for w in com for p in patterns.values() if re.findall(p, w)]))
        exp_attrib = int(exp_n_attrib * len(com))
        size = len(com)
        p1 = act_attrib / size
        feats = [[] for _ in range(len(to_incl))]
        for i in range(len(feats)):
            avg = sum(vectors[w][i] for w in com) / size
            feats[i].append(avg)
        clusters.append([(p1 - p) / ((p * (1 - p)) / size) ** (1 / 2), c, act_attrib, exp_attrib, len(com), feats, com])

    if method == 'overlap':
        overlapping_coef(clusters, total_exp, n_nodes, n_attribs)
    elif method == 'entropy':
        entropy_coef(clusters)


# Use clustering algorithms to find set of communities
def clustering(data, k, method):
    features = np.array([data[node] for node in sorted(data.keys())])
    nodes_name = [node for node in sorted(data.keys())]
    rep = 50

    # K-means
    kmeans = cluster.KMeans(n_clusters=k, n_init=rep, max_iter=rep)
    kmeans.fit(features)
    centers, labels, _ = kmeans.cluster_centers_, kmeans.labels_, kmeans.inertia_

    # Mean-shift - blobs in smooth density
    """meanshift = cluster.MeanShift(bin_seeding=False, cluster_all=False)
    meanshift.fit(features)
    centers, labels = meanshift.cluster_centers_, meanshift.labels_
    k = len(centers)"""

    # Spectral Clustering (slow) - only when measure of center & spread of cluster not suitable desc
    """spectralclustering = cluster.SpectralClustering(n_clusters=k)
    spectralclustering.fit(features)
    labels = spectralclustering.labels_"""

    # Aggolomerative Clustering - hierarchical

    """agglo = cluster.AgglomerativeClustering(n_clusters=k)
    agglo.fit(features)
    labels = agglo.labels_"""

    # Match labels to nodes
    nodes = list(zip(nodes_name, labels))

    # Generate communities
    community = [[] for _ in range(k)]
    clusters = [[] for _ in range(k)]
    # keys = sorted(data.keys())[list(data.values()).index(node)]

    for node, label in nodes:
        community[label].append(node)
        clusters[label].append(data[node])

    # Find overlapping communities - best k
    overlap_coef = generate_optimal_k(clusters, centers.tolist())
    # overlap = max(max(x) for x in overlap_coef.values())
    # if max(max(x) for x in overlap_coef.values()) > 0:
    print('k: %2d %8.3f %8s' % (k, max(max(x) for x in overlap_coef.values()), '-'.join([w[:3] for w in to_incl])))
    # ' | n:', len([q for q in set([z for y in [x for x in overlap_coef.values()] for z in y]) if q > threshold]))
    # for n, coef in enumerate(overlap_coef.values()):
    # print('cluster %s %10.5f' % (n, max(coef)), end=' | ')

    com = [(n, cl) for n, cl in enumerate(community)]
    # Saving
    with open('clustering/communities.txt', 'w') as f:
        for n, cl in com:
            f.write('\n\n***Community ' + str(n) + '***\n')
            # f.write(str(sorted(set([w for w in cl if w.isalpha()]))) + '\n')
            c = 0
            # print(n, len(cl), len(sorted(cl)), end=' ')
            for node in sorted(cl):
                check = False
                for p in patterns.values():
                    if re.findall(p, node):
                        f.write('[' + str(node) + '] ')
                        check = True
                        break
                if not check:
                    f.write(str(node) + ' ')
                    c += 1
                if c == 12:
                    f.write('\n')
                    c = 0
    generate_stats(com, method, data)
    # return k, max(max(x) for x in overlap_coef.values())


def top_words(upper=100):
    """
    get the top 100 most frequent words in product desc split by space
    """
    prod_words = []
    stopwords = nltk.corpus.stopwords.words('english')
    for k in prod.keys():
        words = [w for w in prod[k][0].split() if w not in stopwords and
                 len(w) > 2 and
                 w.isalpha()]
        for w in words:
            prod_words.append(w)

    fd = nltk.FreqDist(prod_words)
    with open('Data/top100.txt', 'w') as f:
        for w, n in fd.most_common()[:upper]:
            f.write('%10s %4s \n' % (w, str(n)))


def generate_features_table():
    with open('Data/nodes_features.txt', 'w') as fw, open('data/vectors/vectors.txt') as fr:
        fw.write('|'.join(['assortativity', 'clustering', 'eigenvector', 'betweeness', 'closeness', 'degree', 'weight'])
                 + '\n')
        for line in fr:
            line = line.split('|')[1].split(', ')
            fw.write('|'.join(line))


def main():
    get_data()
    parse_products()
    graph = get_graph()
    generate_nodes_vectors(graph)
    vectors = get_vectors()
    # compute_features(vectors)
    k = 45  # 36 - 45 - 39
    # for k in range(35, 55):
    clustering(vectors, k, methods[0])


if __name__ == '__main__':
    """pairs = ([('assort', 'clust'), ('degree', 'weigh', 'clos', 'betw', 'eigenv')],  # 2
             [('assort', 'clust', 'eigenv', 'betw'), ('degree', 'weigh', 'clos')],  # 1
             [('assort', 'clust', 'eigenv', 'clos'), ('degree', 'weigh', 'betw')],  # 4
             [('assort', 'clust', 'eigenv', 'clos', 'betw'), ('degree', 'weigh')])  # 3

    for pair in pairs:
        to_incl = pair[0]
        to_excl = pair[1]
        norm_index = len([w for w in to_incl if w not in 'betw clos']) - 1
        main()"""
    main()