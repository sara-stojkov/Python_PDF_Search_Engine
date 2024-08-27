class Graph:
    """ Implementacija usmerenog grafa (potreban za PageRank)"""

    class Vertex:
        __slots__ = '_element', '_data'  

        def __init__(self, x, data=None):
            self._element = x
            self._data = data  

        def element(self):
            return self._element

        def data(self):
            return self._data

        def __hash__(self):
            return hash(id(self))

        def __str__(self):
            return str(self._element)

    class Edge:
        __slots__ = '_origin', '_destination', '_element'

        def __init__(self, origin, destination, element):
            self._origin = origin
            self._destination = destination
            self._element = element

        def endpoints(self):
            return self._origin, self._destination

        def opposite(self, v):
            if not isinstance(v, Graph.Vertex):
                raise TypeError('v mora biti instanca klase Vertex')
            if self._destination == v:
                return self._origin
            elif self._origin == v:
                return self._destination
            raise ValueError('v nije čvor ivice')

        def element(self):
            return self._element

        def __hash__(self):
            return hash((self._origin, self._destination))

        def __str__(self):
            return '({0},{1},{2})'.format(self._origin, self._destination, self._element)

    def __init__(self, directed=True):
        self._outgoing = {}
        self._incoming = {} if directed else self._outgoing

    def _validate_vertex(self, v):
        if not isinstance(v, self.Vertex):
            raise TypeError('Očekivan je objekat klase Vertex')
        if v not in self._outgoing:
            raise ValueError('Vertex ne pripada ovom grafu.')

    def is_directed(self):
        return self._incoming is not self._outgoing

    def vertex_count(self):
        return len(self._outgoing)

    def vertices(self):
        return self._outgoing.keys()

    def edge_count(self):
        total = sum(len(self._outgoing[v]) for v in self._outgoing)
        return total if self.is_directed() else total // 2

    def edges(self):
        result = set()
        for secondary_map in self._outgoing.values():
            result.update(secondary_map.values())
        return result

    def get_edge(self, u, v):
        self._validate_vertex(u)
        self._validate_vertex(v)
        return self._outgoing[u].get(v)

    def degree(self, v, incoming=True):
        self._validate_vertex(v)
        adj = self._incoming if incoming else self._outgoing
        return len(adj[v])

    def incident_edges(self, v, outgoing=True):
        self._validate_vertex(v)
        adj = self._outgoing if outgoing else self._incoming
        for edge in adj[v].values():
            yield edge

    def insert_vertex(self, x=None, data=None): 
        v = self.Vertex(x, data)
        self._outgoing[v] = {}
        if self.is_directed():
            self._incoming[v] = {}
        return v

    def insert_edge(self, u, v, x=None):
        if self.get_edge(u, v) is not None:
            print("Ivica već postoji..")
        e = self.Edge(u, v, x)
        self._outgoing[u][v] = e
        self._incoming[v][u] = e
