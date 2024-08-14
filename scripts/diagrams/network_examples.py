from matplotlib import pyplot as plt
import numpy as np

def blank_fig(f_width, f_height):
    fig = plt.figure()
    ax = fig.add_axes((0,0,1,1)) # x,y coordinates for objects will be btwn 0 and 1

    # Settings
    fig.set_size_inches(f_width, f_height)

    ax.set_xlim(0, f_width)
    ax.set_ylim(0, f_height)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    return fig, ax

FIG_WIDTH = 6
FIG_HEIGHT = 6

# Ring network
fig, ax = blank_fig(FIG_WIDTH, FIG_HEIGHT)

N_VERTEX = 10
VERTEX_RADIUS = 0.25
RING_RADIUS = 2.0

positions = []
edge_pos_right = []
edge_pos_left = []
for i in range(N_VERTEX):
    x = RING_RADIUS * np.cos(i*(2*np.pi)/N_VERTEX) + FIG_WIDTH/2
    y = RING_RADIUS * np.sin(i*(2*np.pi)/N_VERTEX) + FIG_HEIGHT/2

    positions.append((x,y))

    angle_to_perimeter = 180
    edge_left_x = (x - FIG_WIDTH/2) * RING_RADIUS * np.cos(0.1) + FIG_WIDTH/2
    edge_left_y = (y - FIG_HEIGHT/2) * RING_RADIUS * np.sin(0.1) + FIG_HEIGHT/2
    edge_right_x = (x - FIG_WIDTH/2) * RING_RADIUS * np.cos(-0.1) + FIG_WIDTH/2
    edge_right_y = (y - FIG_HEIGHT/2) * RING_RADIUS * np.sin(-0.1) + FIG_HEIGHT/2

    edge_pos_left.append((edge_left_x, edge_left_y))
    edge_pos_right.append((edge_right_x, edge_right_y))

for i,p in enumerate(positions):
    x, y = p
    theta = np.linspace(0, 2 * np.pi, 100)
    ax.plot(
        x + VERTEX_RADIUS * np.cos(theta),
        y + VERTEX_RADIUS * np.sin(theta),
        color="midnightblue",
    )

x1,y1 = edge_pos_left[0]
x2,y2 = edge_pos_right[0]
print(edge_pos_left[0])
print(edge_pos_right[0])
ax.plot([x1,x2], [y1,y2], color="black")

fig.savefig("ring_network.png")
