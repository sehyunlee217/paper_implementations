import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# 1. Create Line 1 (38 stations, path graph)
n_stations = 38
G = nx.path_graph(n_stations)
L = nx.laplacian_matrix(G).toarray()

# 2. Compute Eigenvectors
evals, evecs = np.linalg.eigh(L)

# 3. Plot the first 3 non-trivial eigenvectors
plt.figure(figsize=(12, 6))
stations = np.arange(n_stations)

for i in range(1, 4):
    plt.plot(stations, evecs[:, i], label=f"Eigenvector {i}", marker="o", markersize=4)

plt.title("Laplacian Eigenvectors for TTC Line 1 (Spectral Coordinates)")
plt.xlabel("Station Index (0=Finch, 19=Union, 37=Vaughan)")
plt.ylabel("Eigenvector Value (Spatial Encoding)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
