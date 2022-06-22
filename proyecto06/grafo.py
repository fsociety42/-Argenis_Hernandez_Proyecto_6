import copy
import random
import sys

from math import log, atan2, cos, sin
import pygame
import random

import heapdict
import numpy as np

from arista import Arista
from nodo import Nodo

class Grafo(object):
    """
    Clase Grafo.
    Generación y manejo de Grafos

    Parametros
    ----------
    id : str
        id o nombre del Grafo
    dirigido : bool
        True si el Grafo es dirigido, de otro modo, False

    Atributos
    ---------
    id : str
        id o nombre del Grafo
    dirigido : bool
        True si el Grafo es dirigido, de otro modo, False
    V : dict
        Diccionario de Nodos o Vertices del Grafo.
        key: Nodo.id
        value: Nodo
    E : dict
        Diccionario de Aristas (edges) del Grafo
        key: Arista.id
        value: Arista

    """
    def __init__(self, id='grafo', dirigido=False):
        self.id =       id
        self.dirigido = dirigido
        self.V =        dict()
        self.E =        dict()
        self.attr =     dict()

    def copy_grafo(self, id=f"copy", dirigido=False):
        """
        Regresa una copia deep del grafo

        Returns
        -------
        other : Grafo
            deep copy de self
        """
        other = Grafo(id, dirigido)
        other.V = copy.deepcopy(self.V)
        other.E = copy.deepcopy(self.E)
        other.attr = copy.deepcopy(self.attr)

        return other

    def __repr__(self):
        """
        Asigna representación repr a los Grafos

        Returns
        -------
        str
            representación en str de los Grafos
        """
        return str("id: " + str(self.id) + '\n'
                   + 'nodos: ' + str(self.V.values()) + '\n'
                   + 'aristas: ' + str(self.E.values()))


    def add_nodo(self, nodo):
        """
        Agrega objeto nodo al grafo

        Parametros
        ----------
        nodo : Nodo
            objeto Nodo que se va a agregar a self.V

        Returns
        -------
        None
        """
        self.V[nodo.id] = nodo


    def add_arista(self, arista):
        """
        Agrega arista al grafo si esta no existe de antemano en dicho grafo.
        Agrega el otro nodo de la arista al parametro connected_to de cada nodo

        Parametros
        ----------
        arista : Arista
            objeto Arista que se va agregar a self.E

        Returns
        -------
        True o False : bool
            True si se agrego la arista, de otro modo, False

        """
        if self.get_arista(arista.id):
            return False

        self.E[arista.id] = arista

        u, v = arista.u.id, arista.v.id
        self.V[u].connected_to.append(v)
        self.V[v].connected_to.append(u)

        return True


    def get_arista(self, arista_id):
        """
        Revisa si la arista ya existe en el grafo

        Parametros
        ----------
        arista_id : Arista.id
            atributo id de un objeto de la clase Arista

        Returns
        -------
        True o False : bool
            True si la arista existe, de otro modo, Falso
        """
        if self.dirigido:
            return arista_id in self.E
        else:
            u, v = arista_id
            return (u, v) in self.E or (v, u) in self.E


    def random_weights(self):
        """
        Asigna un peso random a todas las aristas del nodo
        """
        for arista in self.E.values():
            arista.attrs['weight'] = random.randint(1, 100)

    def costo(self):
        """
        Calcula el costo del grafo. Suma del peso de las aristas

        Returns
        -------
        costo : float
            suma del peso de las aristas del grafo
        """
        _costo = 0
        for edge in self.E.values():
            _costo += edge.attrs['weight']

        return _costo

    def to_graphviz(self, filename):
        """
        Exporta grafo a formato graphvizDOT

        Parametros
        ----------
        filename : file
            Nombre de archivo en el que se va a escribir el grafo

        Returns
        -------
        None
        """
        edge_connector = "--"
        graph_directive = "graph"
        if self.dirigido:
            edge_connector = "->"
            graph_directive = "digraph"

        with open(filename, 'w') as f:
            f.write(f"{graph_directive} {self.id} " + " {\n")
            for nodo in self.V:
                if "Dijkstra" in self.id:
                    f.write(f"\"{nodo} ({self.V[nodo].attrs['dist']})\";\n")
                else:
                    f.write(f"{nodo};\n")
            for arista in self.E.values():
                if "Dijkstra" in self.id:
                    weight = np.abs(self.V[arista.u.id].attrs['dist']
                                    - self.V[arista.v.id].attrs['dist'])
                    f.write(f"\"{arista.u} ({self.V[arista.u.id].attrs['dist']})\""
                            + f" {edge_connector} "
                            + f"\"{arista.v} ({self.V[arista.v.id].attrs['dist']})\""
                            # + f" [weight={weight}];\n")
                            + f";\n")
                else:
                    f.write(f"{arista.u} {edge_connector} {arista.v};\n")
            f.write("}")


def grafoMalla(m, n, dirigido=False):
    """
    Genera grafo de malla
    :param m: número de columnas (> 1)
    :param n: número de filas (> 1)
    :param dirigido: el grafo es dirigido?
    :return: grafo generado
    """
    if m < 2 or n < 2:
        print("Error. m y n, deben ser mayores que 1", file=sys.stderr)
        exit(-1)

    total_nodes = m*n
    last_col = m - 1
    last_row = n - 1
    g = Grafo(id=f"grafoMalla_{m}_{n}", dirigido=dirigido)
    nodos = g.V

    # agregar nodos
    for id in range(total_nodes):
        g.add_nodo(Nodo(id))

    # agregar aristas
    # primera fila
    g.add_arista(Arista(nodos[0], nodos[1]))
    g.add_arista(Arista(nodos[0], nodos[m]))
    for node in range(1, m - 1):
        g.add_arista(Arista(nodos[node], nodos[node - 1]))
        g.add_arista(Arista(nodos[node], nodos[node + 1]))
        g.add_arista(Arista(nodos[node], nodos[node + m]))
    g.add_arista(Arista(nodos[m-1], nodos[m-2]))
    g.add_arista(Arista(nodos[m-1], nodos[m - 1 + m]))

    # filas [1 : n - 2]
    for node in range(m, total_nodes - m):
        col = node % m
        g.add_arista(Arista(nodos[node], nodos[node - m]))
        g.add_arista(Arista(nodos[node], nodos[node + m]))
        if col == 0:
            g.add_arista(Arista(nodos[node], nodos[node + 1]))
        elif col == last_col:
            g.add_arista(Arista(nodos[node], nodos[node - 1]))
        else:
            g.add_arista(Arista(nodos[node], nodos[node + 1]))
            g.add_arista(Arista(nodos[node], nodos[node - 1]))

    # última fila (n - 1)
    col_0 = total_nodes - m
    col_1 = col_0 + 1
    last_node = total_nodes - 1
    g.add_arista(Arista(nodos[col_0], nodos[col_1]))
    g.add_arista(Arista(nodos[col_0], nodos[col_0 - m]))
    for node in range(col_1, last_node):
        g.add_arista(Arista(nodos[node], nodos[node - 1]))
        g.add_arista(Arista(nodos[node], nodos[node + 1]))
        g.add_arista(Arista(nodos[node], nodos[node - m]))
    g.add_arista(Arista(nodos[last_node], nodos[last_node - m]))
    g.add_arista(Arista(nodos[last_node], nodos[last_node - 1]))

    return g

def grafoErdosRenyi(n, m, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo Erdos-Renyi
    :param n: número de nodos (> 0)
    :param m: número de aristas (>= n-1)
    :param dirigido: el grafo es dirigido?
    :param auto: permitir auto-ciclos?
    :return: grafo generado
    """
    if m < n-1 or n < 1:
        print("Error: n > 0 y m >= n - 1", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoErdos_Renyi_{n}_{m}")
    nodos = g.V

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))

    # crear aristas
    rand_node = random.randrange
    for arista in range(m):
        while True:
            u = rand_node(n)
            v = rand_node(n)
            if u == v and not auto:
                continue
            if g.add_arista(Arista(nodos[u], nodos[v])):
                break

    return g

def grafoGilbert(n, p, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo Gilbert
    :param n: número de nodos (> 0)
    :param p: probabilidad de crear una arista (0, 1)
    :param dirigido: el grafo es dirigido?
    :param auto: permitir auto-ciclos?
    :return: grafo generado
    """
    if p > 1 or p < 0 or n < 1:
        print("Error: 0 <= p <= 1 y n > 0", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoGilbert_{n}_{int(p * 100)}", dirigido=dirigido)
    nodos = g.V

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))


    # crear pares de nodos, diferente generador dependiendo del parámetro auto
    if auto:
        pairs = ((u, v) for u in nodos.keys() for v in nodos.keys())
    else:
        pairs = ((u, v) for u in nodos.keys() for v in nodos.keys() if u != v)

    # crear aristas
    for u, v in pairs:
        add_prob = random.random()
        if add_prob <= p:
            g.add_arista(Arista(nodos[u], nodos[v]))

    return g

def grafoGeografico(n, r, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo geográfico simple
    Se situan todos los nodos con coordenadas dentro de un rectangulo unitario
    Se crean aristas de un nodo a todos los que estén a una distancia <= r de
        un nodo en particular
    :param n: número de nodos (> 0)
    :param r: distancia máxima para crear un nodo (0, 1)
    :param dirigido: el grafo es dirigido?
    :param auto: permitir auto-ciclos?
    :return: grafo generado
    """
    if r > 1 or r < 0 or n < 1:
        print("Error: 0 <= r <= 1 y n > 0", file=sys.stderr)
        exit(-1)

    coords = dict()
    g = Grafo(id=f"grafoGeografico_{n}_{int(r * 100)}", dirigido=dirigido)
    nodos = g.V

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))
        x = round(random.random(), 3)
        y = round(random.random(), 3)
        coords[nodo] = (x, y)

    # crear aristas
    r **= 2
    for u in nodos:
        vs = (v for v in nodos if u != v)
        # si auto es true, se agrega la arista del nodo u a sí mismo
        if auto:
            g.add_arista(Arista(nodos[u], nodos[u]))
        # se agregan todos los nodos dentro de la distancia r
        for v in vs:
            dist = (coords[u][0] - coords[v][0]) ** 2 \
                    + (coords[u][1] - coords[v][1]) ** 2
            if dist <= r:
                g.add_arista(Arista(nodos[u], nodos[v]))

    return g

def grafoBarabasiAlbert(n, d, dirigido=False, auto=False):
    """
    Genera grafo aleatorio con el modelo Barabasi-Albert
    :param n: número de nodos (> 0)
    :param d: grado máximo esperado por cada nodo (> 1)
    :param dirigido: el grafo es dirigido?
    :param auto: permitir auto-ciclos?
    :return: grafo generado
    """
    if n < 1 or d < 2:
        print("Error: n > 0 y d > 1", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoBarabasi_{n}_{d}", dirigido=dirigido)
    nodos = g.V
    nodos_deg = dict()

    # crear nodos
    for nodo in range(n):
        g.add_nodo(Nodo(nodo))
        nodos_deg[nodo] = 0

    # agregar aristas al azar, con cierta probabilidad
    for nodo in nodos:
        for v in nodos:
            if nodos_deg[nodo] == d:
                break
            if nodos_deg[v] == d:
                continue
            p = random.random()
            equal_nodes = v == nodo
            if equal_nodes and not auto:
                continue

            if p <= 1 - nodos_deg[v] / d \
               and g.add_arista(Arista(nodos[nodo], nodos[v])):
                nodos_deg[nodo] += 1
                if not equal_nodes:
                        nodos_deg[v] += 1

    return g

def grafoDorogovtsevMendes(n, dirigido=False):
    """
    Genera grafo aleatorio con el modelo Barabasi-Albert
    :param n: número de nodos (≥ 3)
    :param dirigido: el grafo es dirigido?
    :return: grafo generado
    Crear 3 nodos y 3 aristas formando un triángulo. Después, para cada nodo
    adicional, se selecciona una arista al azar y se crean aristas entre el nodo
    nuevo y los extremos de la arista seleccionada.

    """
    if n < 3:
        print("Error: n >= 3", file=sys.stderr)
        exit(-1)

    g = Grafo(id=f"grafoDorogovtsev_{n}", dirigido=dirigido)
    nodos = g.V
    aristas = g.E

    # crear primeros tres nodos y sus correspondientes aristas
    for nodo in range(3):
        g.add_nodo(Nodo(nodo))
    pairs = ((u, v) for u in nodos for v in nodos if u != v)
    for u, v in pairs:
        g.add_arista(Arista(nodos[u], nodos[v]))

    # crear resto de nodos
    for nodo in range(3, n):
        g.add_nodo(Nodo(nodo))
        u, v = random.choice(list(aristas.keys()))
        g.add_arista(Arista(nodos[nodo], nodos[u]))
        g.add_arista(Arista(nodos[nodo], nodos[v]))

    return g



    def BFS(self, s):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo "breadth first
            search"

        Parametros
        ----------
        s : Nodo
            nodo raíz del árbol que se va a generar

        Returns
        -------
        bfs : Grafo
            árbol generado
        """
        if not s.id in self.V:
            print("Error, node not in V", file=sys.stderr)
            exit(-1)

        bfs = Grafo(id=f"BFS_{self.id}", dirigido=self.dirigido)
        discovered = set()
        bfs.add_nodo(s)
        L0 = [s]
        discovered = set()
        added = [s.id]

        while True:
            L1 = []
            for node in L0:
                aristas = [ids_arista for ids_arista in self.E
                            if node.id in ids_arista]

                for arista in aristas:
                    v = arista[1] if node.id == arista[0] else arista[0]

                    if v in discovered:
                        continue

                    bfs.add_nodo(self.V[v])
                    bfs.add_arista(self.E[arista])
                    discovered.add(v)
                    L1.append(self.V[v])

            L0 = L1
            if not L0:
                break

        return bfs


    def DFS_R(self, u):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo "depth first
            search".
        Usa una función recursiva

        Parametros
        ----------
        u : Nodo
            nodo raíz del árbol que se va a generar

        Returns
        -------
        dfs : Grafo
            árbol generado
        """
        dfs = Grafo(id=f"DFS_R_{self.id}", dirigido=self.dirigido)
        discovered = set()
        self.DFS_rec(u, dfs, discovered)

        return dfs


    def DFS_rec(self, u, dfs, discovered):
        """
        Función recursiva para agregar nodos y aristas al árbol DFS

        Parametros
        ----------
        u : Nodo
            nodo actual, en el que se continúa la búsqueda a lo profundo
        dfs : Grafo
            Grafo que contendrá al árbol de búsquedo a lo produndo.
        discovered : set
            nodos que ya han sido descubiertos

        Returns
        -------
        dfs: Grafo
            arbol generado
        """
        dfs.add_nodo(u)
        discovered.add(u.id)
        aristas = (arista for arista in self.E if u.id in arista)

        for arista in aristas:
            v = arista[1]
            if not self.dirigido:
                v = arista[0] if u.id == arista[1] else arista[1]
            if v in discovered:
                continue
            dfs.add_arista(self.E[arista])
            self.DFS_rec(self.V[v], dfs, discovered)


    def DFS_I(self, s):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo "depth first
            search".
        Metodo iterativo

        Parametros
        ----------
        s : Nodo
            nodo raíz del árbol que se va a generar

        Returns
        -------
        dfs : Grafo
            árbol generado
        """
        dfs = Grafo(id=f"DFS_I_{self.id}", dirigido=self.dirigido)
        discovered = {s.id}
        dfs.add_nodo(s)
        u = s.id
        frontera = []
        while True:
            # añadir a frontera todos los nodos con arista a u
            aristas = (arista for arista in self.E if u in arista)
            for arista in aristas:
                v = arista[1] if u == arista[0] else arista[0]
                if v not in discovered:
                    frontera.append((u, v))

            # si la frontera está vacía, salir del loop
            if not frontera:
                break

            # sacar nodo de la frontera
            parent, child = frontera.pop()
            if child not in discovered:
                dfs.add_nodo(self.V[child])
                arista = Arista(self.V[parent], self.V[child])
                dfs.add_arista(arista)
                discovered.add(child)

            u = child

        return dfs


    def Dijkstra(self, s):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Dijkstra,
        que encuentra el grafo del camino más corto entre nodos
        Usa una función recursiva

        Parametros
        ----------
        s : Nodo
            nodo raíz del árbol que se va a generar

        Returns
        -------
        tree : Grafo
            árbol generado
        """
        tree = Grafo(id=f"{self.id}_Dijkstra")
        line = heapdict.heapdict()
        parents = dict()
        in_tree = set()


        """
        asignar valores infinitos a los nodos.
        asignar nodo padre en el arbol a None
        """
        line[s] = 0
        parents[s] = None
        for node in self.V:
            if node == s:
                continue
            line[node] = np.inf
            parents[node] = None

        while line:
            u, u_dist = line.popitem()
            if u_dist == np.inf:
                continue

            self.V[u].attrs['dist'] = u_dist
            tree.add_nodo(self.V[u])
            if parents[u] is not None:
                arista = Arista(self.V[parents[u]], self.V[u])
                tree.add_arista(arista)
            in_tree.add(u)

            # get neighbor nodes
            neigh = []
            for arista in self.E:
                if self.V[u].id in arista:
                    v = arista[0] if self.V[u].id == arista[1] else arista[1]
                    if v not in in_tree:
                        neigh.append(v)

            # actualizar distancias de ser necesario
            for v in neigh:
                arista = (u, v) if (u, v) in self.E else (v, u)
                if line[v] > u_dist + self.E[arista].attrs['weight']:
                    line[v] = u_dist + self.E[arista].attrs['weight']
                    parents[v] = u

        return tree

    def KruskalD(self):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Kruskal
        directo, que encuentra el árbol de expansión mínima

        Returns
        -------
        mst : Grafo
            árbol de expansión mínima (mst)
        """

        mst = Grafo(id=f"{self.id}_KruskalD")

        # sort edges by weight
        edges_sorted = list(self.E.values())
        edges_sorted.sort(key = lambda edge: edge.attrs['weight'])

        # connected component
        connected_comp = dict()
        for nodo in self.V:
            connected_comp[nodo] = nodo

        # add edges, iterating by weight
        for edge in edges_sorted:
            u, v = edge.u, edge.v
            if connected_comp[u.id] != connected_comp[v.id]:
                # add nodes and edge to mst
                mst.add_nodo(u)
                mst.add_nodo(v)
                mst.add_arista(edge)

                # change the connected component of v to be the same as u
                for comp in connected_comp:
                    if connected_comp[comp] == connected_comp[v.id]:
                        other_comp = connected_comp[v.id]
                        connected_comp[comp] = connected_comp[u.id]

                        # if we change the connected comp of one node,
                        # change it for the whole connected comp
                        iterator = (key for key in connected_comp \
                                    if connected_comp[key] == other_comp)
                        for item in iterator:
                            connected_comp[item] = connected_comp[u.id]

        return mst


    def KruskalI(self):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Kruskal
        inverso, que encuentra el árbol de expansión mínima

        Returns
        -------
        mst : Grafo
            árbol de expansión mínima (mst)
        """
        mst = self.copy_grafo(id=f"{self.id}_KruskalI", dirigido=self.dirigido)

        # sort edges by weight
        edges_sorted = list(self.E.values())
        edges_sorted.sort(key = lambda edge: edge.attrs['weight'], reverse=True)

        # start removing edges from mst
        for edge in edges_sorted:
            u, v = edge.u.id, edge.v.id
            key, value = (u, v), edge
            del(mst.E[(u, v)])
            # if graph not connected after removal, put back the edge again
            if len(mst.BFS(edge.u).V) != len(mst.V):
                mst.E[(u, v)] = edge

        return mst


    def Prim(self):
        """
        Crea un nuevo grafo de tipo árbol mediante el algoritmo de Prim,
        que encuentra el árbol de expansión mínima

        Returns
        -------
        mst : Grafo
            árbol de expansión mínima (mst)
        """
        mst = Grafo(id=f"{self.id}_Prim")
        line = heapdict.heapdict()
        parents = dict()
        in_tree = set()

        s = random.choice(list(self.V.values()))

        """
        asignar valores infinitos a los nodos.
        asignar nodo padre en el arbol a None
        """
        line[s.id] = 0
        parents[s.id] = None
        for node in self.V:
            if node == s.id:
                continue
            line[node] = np.inf
            parents[node] = None

        while line:
            u, u_dist = line.popitem()
            if u_dist == np.inf:
                continue

            self.V[u].attrs['dist'] = u_dist
            mst.add_nodo(self.V[u])
            if parents[u] is not None:
                arista = Arista(self.V[parents[u]], self.V[u])
                if (u, parents[u]) in self.E:
                    weight = self.E[(u, parents[u])].attrs['weight']
                else:
                    weight = self.E[(parents[u], u)].attrs['weight']
                arista.attrs['weight'] = weight
                mst.add_arista(arista)
            in_tree.add(u)

            # get neighbor nodes
            neigh = []
            for arista in self.E:
                if self.V[u].id in arista:
                    v = arista[0] if self.V[u].id == arista[1] else arista[1]
                    if v not in in_tree:
                        neigh.append(v)

            # actualizar distancias de ser necesario
            for v in neigh:
                arista = (u, v) if (u, v) in self.E else (v, u)
                if line[v] > self.E[arista].attrs['weight']:
                    line[v] = self.E[arista].attrs['weight']
                    parents[v] = u

        return mst


# create the main surface (or window)
WIDTH, HEIGHT   = 1700, 1000
BORDER          = 15
WIN             = pygame.display.set_mode((WIDTH, HEIGHT))

# colors
BG              = (251, 241, 199)
BLUE            = (69, 133, 136)
BLACK           = (40, 40, 40)
RED             = (157, 0, 6)

# colors black mode
BG              = (40, 40, 40)
BLUE            = (131, 165, 152)
BLACK           = (235, 219, 178) # Actually white
RED             = (251, 73, 52)

# configuration
ITERS           = 1500
FPS             = 40
NODE_RADIUS     = 15
DIST_MIN        = (min(WIDTH, HEIGHT)) // 15
NODE_MIN_WIDTH  = 25
NODE_MIN_HEIGHT = 25
NODE_MAX_WIDTH  = WIDTH - 25
NODE_MAX_HEIGHT = HEIGHT - 25

# spring constants
c1 = 16
c2 = 0.2
c3 = 3.8
c4 = 0.2


def spring(g):
    """
    Muestra una animación del metodo de visualizacion spring de Eades.

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    run = True
    clock = pygame.time.Clock()

    init_nodes(g)
    draw_edges(g)
    draw_nodes(g)


    i = 0
    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        if i > ITERS:
            continue

        WIN.fill(BG)
        update_nodes(g)
        draw_edges(g)
        draw_nodes(g)
        pygame.display.update()
        i += 1

    pygame.quit()
    return


def init_nodes(g):
    """
    Inicializa los nodos del grafo g en posiciones random

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    for node in g.V.values():
        x = random.randrange(NODE_MIN_WIDTH, NODE_MAX_WIDTH)
        y = random.randrange(NODE_MIN_HEIGHT, NODE_MAX_HEIGHT)
        node.attrs['coords'] = [x, y]

    return

def update_nodes(g):
    """
    Aplica la fuerza a los nodos del grafo G para actualizar su poisicion

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    for node in g.V.values():
        x_attraction = 0
        y_attraction = 0
        x_node, y_node = node.attrs['coords']

        for other in node.connected_to:
            x_other, y_other = g.V[other].attrs['coords']
            d = ((x_node - x_other) ** 2 + (y_node - y_other)**2) ** 0.5

            # defining minimum distance
            if d < DIST_MIN:
                continue
            attraction = c1 * log(d / c2)
            angle = atan2(y_other - y_node, x_other - x_node)
            x_attraction += attraction * cos(angle)
            y_attraction += attraction * sin(angle)

        not_connected = (other for other in g.V.values()
                         if (other.id not in node.connected_to and other != node))
        x_repulsion = 0
        y_repulsion = 0
        for other in not_connected:
            x_other, y_other = other.attrs['coords']
            d = ((x_node - x_other) ** 2 + (y_node - y_other)**2) ** 0.5
            if d == 0:
                continue
            repulsion = c3 / d ** 0.5
            angle = atan2(y_other - y_node, x_other - x_node)
            x_repulsion -= repulsion * cos(angle)
            y_repulsion -= repulsion * sin(angle)

        fx = x_attraction + x_repulsion
        fy = y_attraction + y_repulsion
        node.attrs['coords'][0] += c4 * fx
        node.attrs['coords'][1] += c4 * fy

        # Restrict for limits of window
        node.attrs['coords'][0] = max(node.attrs['coords'][0], NODE_MIN_WIDTH)
        node.attrs['coords'][1] = max(node.attrs['coords'][1], NODE_MIN_HEIGHT)
        node.attrs['coords'][0] = min(node.attrs['coords'][0], NODE_MAX_WIDTH)
        node.attrs['coords'][1] = min(node.attrs['coords'][1], NODE_MAX_HEIGHT)

    return


def draw_nodes(g):
    """
    Dibuja los nodos del grafo g

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    for node in g.V.values():
        pygame.draw.circle(WIN, BLUE, node.attrs['coords'], NODE_RADIUS - 3, 0)
        pygame.draw.circle(WIN, RED, node.attrs['coords'], NODE_RADIUS, 3)

    return


def draw_edges(g):
    """
    Dibuja las aristas del grafo g

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    for edge in g.E:
        u, v = edge
        u_pos = g.V[u].attrs['coords']
        v_pos = g.V[v].attrs['coords']

        pygame.draw.line(WIN, BLACK, u_pos, v_pos, 1)

    return



# create the main surface (or window)
WIDTH, HEIGHT   = 1200, 1000
BORDER          = 15
WIN             = pygame.display.set_mode((WIDTH, HEIGHT))

# colors
BG              = (255, 255, 255)
BLUE            = (0, 0, 0)
BLACK           = (0, 40, 40)
RED             = (0, 0, 0)

# colors black mode
BG              = (255, 255, 255)
BLUE            = (0, 0, 0)
BLACK           = (0, 0, 0) # Actually white
RED             = (0, 0, 0)

# configuration
ITERS           = 1500
FPS             = 40
NODE_RADIUS     = 15
DIST_MIN        = (min(WIDTH, HEIGHT)) // 15
NODE_MIN_WIDTH  = 25
NODE_MIN_HEIGHT = 25
NODE_MAX_WIDTH  = WIDTH - 25
NODE_MAX_HEIGHT = HEIGHT - 25

# spring constants
c1 = 1.65
c2 = 0.7
c3 = 4.8
c4 = 0.1

def fruchterman_reginold(g):
    """
    Muestra una animación del metodo de visualizacion de Furchterman y Reginold
    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """
    run = True
    clock = pygame.time.Clock()

    init_nodes(g)
    draw_edges(g)
    draw_nodes(g)

    while run:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

        WIN.fill(BG)
        update_nodes(g)
        draw_edges(g)
        draw_nodes(g)
        pygame.display.update()

    pygame.quit()
    return


def init_nodes(g):
    """
    Inicializa los nodos del grafo g en posiciones random

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    for node in g.V.values():
        x = random.randrange(NODE_MIN_WIDTH, NODE_MAX_WIDTH)
        y = random.randrange(NODE_MIN_HEIGHT, NODE_MAX_HEIGHT)
        node.attrs['coords'] = [x, y]

    return


def update_nodes(g):
    C = 1
    temp = 1
    area = (WIDTH - NODE_MIN_WIDTH) * (HEIGHT - NODE_MIN_HEIGHT)
    k = C * (area / len(g.V)) ** 0.5
    for node in g.V.values():
        fx=0
        fy=0
        for other in g.V.values():
            if node == other:
                continue

            d=((other.attrs['coords'][0] - node.attrs['coords'][0]) ** 2 +
               (other.attrs['coords'][1] - node.attrs['coords'][1])**2) ** 0.9
            if d==0:
                continue

            force= (d / abs(d)) * k**2 / d
            angle = atan2(other.attrs['coords'][1] - node.attrs['coords'][1], other.attrs['coords'][0] - node.attrs['coords'][0])
            fx-= force * cos(angle)
            fy-= force * sin(angle)

            if other.id in node.connected_to:
                #Attraction force - Adjacent nodes
                d = ((other.attrs['coords'][0] - node.attrs['coords'][0]) ** 2
                    + (other.attrs['coords'][1] - other.attrs['coords'][1]) ** 2) ** 0.5

                if d < DIST_MIN: #30
                    continue

                force = d / abs(d) * d**2 / k
                angle = atan2(other.attrs['coords'][1] - node.attrs['coords'][1],
                              other.attrs['coords'][0] - node.attrs['coords'][0])
                fx+= force * cos(angle)
                fy+= force * sin(angle)

        node.attrs['coords'][0] += c4*fx
        node.attrs['coords'][1] += c4*fy

    return


def draw_nodes(g):
    """
    Dibuja los nodos del grafo g

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    for node in g.V.values():
        pygame.draw.circle(WIN, BLUE, node.attrs['coords'], NODE_RADIUS - 3, 0)
        pygame.draw.circle(WIN, RED, node.attrs['coords'], NODE_RADIUS, 3)

    return


def draw_edges(g):
    """
    Dibuja las aristas del grafo g

    Parametros
    ----------
    g : Grafo
        grafo para el cual se realiza la visualizacion
    """

    for edge in g.E:
        u, v = edge
        u_pos = g.V[u].attrs['coords']
        v_pos = g.V[v].attrs['coords']

        pygame.draw.line(WIN, BLACK, u_pos, v_pos, 1)

    return