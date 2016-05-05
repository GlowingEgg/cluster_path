
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import ConvexHull
from sklearn import cluster
from timeit import default_timer as timer

nodes = pd.read_csv("./nodes.csv")
edges = pd.read_csv("./edges.csv")

graph = nx.from_pandas_dataframe(edges, 'node', 'node2', 'weight')


graph = nx.convert.convert_to_undirected(graph)


coordinates = nodes.as_matrix(columns =['x', 'y'])
start = timer()


db = cluster.DBSCAN(eps = .4, min_samples = 1).fit(coordinates)


numClusters = len(set(db.labels_))

clusters = pd.Series([coordinates[db.labels_ == i] for i in xrange(numClusters)])


clusterA = np.array(clusters)


hulls = [ConvexHull(thiscluster) for thiscluster in clusterA]


hullCoordinates = []
for hull in hulls:
    coordinates = []
    for point in hull.vertices:
        coordinates.append(hull.points[point])
    hullCoordinates.append(coordinates)


clusterNav = [[0] * 4] * 4


allConnections = []
for idx, clusterPath in enumerate(clusterNav):
    connector = []
    for i in range(4):
        edge = []
        for coordinateA in hullCoordinates[idx]:
            nameA = nodes.loc[((nodes['x'] == coordinateA[0]) & (nodes['y'] == coordinateA[1]))].name.item()
            for coordinateB in hullCoordinates[i]:
                nameB = nodes.loc[((nodes['x'] == coordinateB[0]) & (nodes['y'] == coordinateB[1]))].name.item()
                if ((edges['node'] == nameA) & (edges['node2'] == nameB)).any():
                    edge.append([nameA, nameB, edges.loc[((edges['node'] == nameA) & (edges['node2'] == nameB))].weight.item()])
        edge.sort(key = lambda x: x[2])
        if edge:
            connector.append(edge[0])
    if connector:
        allConnections.append(connector)

clusNav = []
for idx, clustar in enumerate(allConnections):
    singles = []
    for index, clustars in enumerate(clustar):
        coordinates = [nodes.loc[nodes['name'] == clustars[1]].x.item(), nodes.loc[nodes['name'] == clustars[1]].y.item()]
        for inx, row in np.ndenumerate(clusterA):
            for liz in row:
                if ((coordinates[0] == liz[0].item()) & (coordinates[1] == liz[1].item())):
                    clus = inx
                    break
        singles.append([clus[0], clustars[0], clustars[1], clustars[2]])
    clusNav.append(singles)


bigGraph = nx.Graph()
for index, row in enumerate(clusNav):
    for this in row:
        bigGraph.add_edge(index, this[0], weight = this[3])
bidirectional = []
for index, row in enumerate(clusNav):
    for element in row:
        if element[0] != index:
            inter = [element[0],index, element[2], element[1], element[3]]
            bidirectional.append(inter)

for row in bidirectional:
    clusNav[row[0]].append([row[1], row[2], row[3], row[4]])
for row in clusNav:
    row.sort(key = lambda x: x[0])

clusterNavs = []
for i in range(4):
    inter = []
    for j in range(4):
        path = nx.shortest_path(bigGraph, source=i, target=j)
        inter.append(path)
    clusterNavs.append(inter)

end = timer()
time = end - start
print "Preprocessing took ", time, "seconds"


# ## ask for input, find cluster of each point, query clusterNavs for path, see what's next, 
# ## query clusNav for point within cluster, navigate to it


#sauce = input("Enter a source node, a-x")
#dest = input("Enter a destination node, a-x")
sauce = 'a'
dest = 'x'
finalPath = []
start = timer()
sourceCoord = [nodes.loc[nodes['name'] == sauce].x.item(), nodes.loc[nodes['name'] == sauce].y.item()] 
destCoord = [nodes.loc[nodes['name'] == dest].x.item(), nodes.loc[nodes['name'] == dest].y.item()]
for inx, row in np.ndenumerate(clusterA):
            for liz in row:
                if ((sourceCoord[0] == liz[0].item()) & (sourceCoord[1] == liz[1].item())):
                    sourceCluster = inx[0]
for inx, row in np.ndenumerate(clusterA):
            for liz in row:
                if ((destCoord[0] == liz[0].item()) & (destCoord[1] == liz[1].item())):
                    destCluster = inx[0]

while sourceCluster != destCluster:
    clusterPath = clusterNavs[sourceCluster][destCluster][1:]
    nextHop = clusterPath[0]
    outerNode = clusNav[sourceCluster][nextHop][1]
    innerPath = nx.astar_path(graph, sauce, outerNode)
    finalPath.append(innerPath)
    sauce = clusNav[sourceCluster][nextHop][2]
    sourceCluster = nextHop
    del clusterPath[0]
innerPath = nx.astar_path(graph, sauce, dest)
finalPath.append(innerPath)
end = timer() - start
print "Runtime for my algorithm: ", end, "seconds" 
print "Path is: "
for path in finalPath:
    for item in path:
        print item

start = timer()
sauce = 'a'
dest = 'x'
path = nx.shortest_path(graph, sauce, dest)
end = timer() - start
print "runtime for Djikstra: ", end, "seconds"