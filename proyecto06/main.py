import random
from time import perf_counter

from grafo import *
from arista import *
from nodo import *


def main():
    path = "/"

    nodos = 100
    nodos_malla = (4, 4)

    m_erdos = 2000
    p_gilbert = 0.004
    r_geografico = 0.3
    d_barabasi = 5

    print("\nMalla")
    g = grafoMalla(*nodos_malla)
    #spring(g)
    fruchterman_reginold(g)

    #print("\nErdos")
    #g = grafoErdosRenyi(nodos, m_erdos)
    #spring(g)

    # print("\nGilbert")
    # g = grafoGilbert(nodos, p_gilbert, dirigido=False, auto=False)
    # spring(g)

    #print("\nGeo")
    #g = grafoGeografico(nodos, r_geografico)
    #spring(g)

    # print("\nBarabasi")
    # g = grafoBarabasiAlbert(nodos, d_barabasi, auto=False)
    # spring(g)

    # print("\nDorog")
    # g = grafoDorogovtsevMendes(nodos, dirigido=False)
    # spring(g)

if __name__ == "__main__":
    main()
