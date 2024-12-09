import numpy as np
from tqdm import tqdm
import sigcoeff
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch


def frame_generator(num_frames, hold):
    for frame in np.arange(num_frames):
        yield frame
        if frame == num_frames - 1:
            for _ in range(hold):
                yield frame

def animate_1D(X, interval=100, cmap='Reds', save_as=None, hold=10):
    n_nodes, n_steps = X.shape

    norm = Normalize(vmin=np.min(X), vmax=np.max(X)) # Normalize values for colormap

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_xlim(-0.5, n_nodes - 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Create nodes
    nodes = [plt.Circle((i, 0), 0.2, color='white', ec='black') for i in range(n_nodes)]
    for node in nodes:
        ax.add_patch(node)

    # Create arrows between nodes
    arrows = []
    for i in range(n_nodes - 1):
        arrow = FancyArrowPatch((i + 0.2, 0), (i + 0.8, 0), arrowstyle='simple', color='black', mutation_scale=5 )
        ax.add_patch(arrow)
        arrows.append(arrow)

    cmap_func = plt.cm.get_cmap(cmap)
    time_label = ax.text(0, -0.4, "", fontsize=20, ha="center", transform=ax.transAxes, color="black")

    def update(frame):
        # Update node colors
        values = X[:, frame]
        for i, node in enumerate(nodes):
            node.set_facecolor(cmap_func(norm(values[i])))

        # Update time label
        t_scaled = frame / (n_steps - 1)  # Scale time to [0, 1]
        time_label.set_text(f"t = {t_scaled:.2f}")

    ani = FuncAnimation( fig, update, frames=frame_generator(n_steps, hold), interval=interval, repeat = True )

    if save_as:
        ani.save(save_as, writer=PillowWriter(fps=1000 // interval))
    else:
        plt.show()


def animate_2D(X, interval=100, cmap='Reds', save_as=None, hold=10):
    rows, cols, n_steps = X.shape

    norm = Normalize(vmin=np.min(X), vmax=np.max(X))  # Normalize values for colormap

    fig, ax = plt.subplots(figsize=(cols * 2, rows * 2))
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Create nodes for the grid
    nodes = [[plt.Circle((j, rows - i - 1), 0.2, color='white', ec='black') for j in range(cols)] for i in range(rows)]
    for row in nodes:
        for node in row:
            ax.add_patch(node)

    # Create arrows between nodes
    arrows = []
    for i in range(rows):
        for j in range(cols):
            # Horizontal arrows
            if j < cols - 1:
                arrow = FancyArrowPatch((j + 0.2, rows - i - 1), (j + 0.8, rows - i - 1),
                                        arrowstyle='simple', color='black', mutation_scale=15)
                ax.add_patch(arrow)
                arrows.append(arrow)

            # Vertical arrows
            if i < rows - 1:
                arrow = FancyArrowPatch((j, rows - i - 1 - 0.2), (j, rows - i - 1 - 0.8),
                                        arrowstyle='simple', color='black', mutation_scale=15)
                ax.add_patch(arrow)
                arrows.append(arrow)

    cmap_func = plt.cm.get_cmap(cmap)
    time_label = ax.text(0.5, -0.1, "", fontsize=35, ha="center", transform=ax.transAxes, color="black")

    def update(frame):
        # Update node colors
        for i in range(rows):
            for j in range(cols):
                nodes[i][j].set_facecolor(cmap_func(norm(X[i, j, frame])))

        # Update time label
        t_scaled = frame / (n_steps - 1)  # Scale time to [0, 1]
        time_label.set_text(f"t = {t_scaled:.2f}")

    ani = FuncAnimation(fig, update, frames=frame_generator(n_steps, hold), interval=interval, repeat=True)

    if save_as:
        ani.save(save_as, writer=PillowWriter(fps=1000 // interval))
    else:
        plt.show()


class GraphCDE:
    def __init__(self, n, flows, num_steps, N, dyadic_order, M, mode = "single"):
        self.n = n #dimension of y, i.e. number of nodes
        self.flows = flows #{[0,1] : d_1, [1,2] : d_2, .....}, dictionary of size e, mapping edge to dimension of x corresponding to flow
        self.num_steps = num_steps #number of Euler steps to take. len(x) must be divisible by steps.
        self.N = N #N-step Euler scheme

        self.dyadic_order = dyadic_order
        self.M = M
        self.mode = mode

        self.edge_list = {}
        for i in range(self.n):
            self.edge_list[i] = []

        for key in self.flows.keys():
            self.edge_list[key[0]].append(key[1])

        self.walks = [] # walks = all possible walks up to length N, expressed as a list of dimensions of x corresponding to edges
        self.get_walks(self.N)

        self.x = None
        self.y = None
        self.segment_length = None

        self.coeffs = None # dict of (multi_index, coeffs) pairs to avoid recomputing

    def set_x(self, x):
        if x.shape[0] % self.num_steps != 0:
            raise ValueError("len(x) must be divisible by steps")

        if x.device == "cpu" and self.mode != "single":
            raise ValueError("cpu cannot be used with mode = full")

        else:
            self.x = x
            self.segment_length = x.shape[0] // self.num_steps

    def get_walks(self, depth):
        # Recursively generate walks
        if depth == 1:
            self.walks = [list(walk) for walk in self.flows.keys()]
        else:
            self.get_walks(depth - 1)
            new_walks = []
            for walk in self.walks:
                if len(walk) != depth:
                    continue

                nbhd = self.edge_list[walk[-1]]
                for node in nbhd:
                    new_walks.append(walk + [node])
            self.walks += new_walks

    def sparsity(self):
        d = len(self.flows)
        E_0 = len(self.walks) + 1
        return E_0 * (d - 1) / (d**(self.N + 1))

    def walk_to_index(self, walk):
        multi_index = []
        for i in range(len(walk) - 1):
            multi_index.append(self.flows[(walk[i], walk[i+1])])
        return tuple(multi_index)

    def get_coeffs(self, step):
        # compute required coeffs for step i of the Euler scheme
        self.coeffs = {}

        x_segment = self.x[step * self.segment_length : (step + 1) * self.segment_length + 1,:]
        #x_segment = x_segment.to(device = "cuda")

        if self.mode == "single":
            for walk in self.walks:
                multi_index = self.walk_to_index(walk)
                if multi_index not in self.coeffs.keys():
                    self.coeffs[multi_index] = float(sigcoeff.coeff(x_segment, multi_index, self.dyadic_order, self.M))

        else:
            for walk in self.walks[::-1]: #step backwards to fill longest walk first
                multi_index = self.walk_to_index(walk)
                if multi_index not in self.coeffs.keys():
                    # Compute FULL array of sigcoeffs and populate all relevant walks.
                    res = sigcoeff.coeff(x_segment, multi_index, self.dyadic_order, self.M, mode ="full")[-1, :].cpu()
                    for i in range(1, len(multi_index) + 1):
                        self.coeffs[multi_index[:i]] = float(res[i-1])

    def step(self, i):
        #get ith step of scheme
        self.get_coeffs(i-1)
        self.y[i, :] = self.y[i-1,:]
        for walk in self.walks:
            start = walk[0]
            end = walk[-1]
            multi_index = self.walk_to_index(walk)
            self.y[i,end] += self.y[i-1, start] * self.coeffs[multi_index]

    def run(self, y0):
        if self.x is None:
            raise ValueError("x is not set")

        self.y = torch.empty((self.num_steps + 1, self.n))
        self.y[0,:] = y0
        for i in tqdm(range(1, self.num_steps + 1)):
            self.step(i)

        return self.y