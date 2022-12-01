# =============================================================================
# Construal Level Theory for Agent-based Planning
#
# C McClurg, AR Wagner, S Rajtmajer
# =============================================================================
# Plotting the maze and paths

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from warnings import warn
import heapq
import textwrap
import random


pInc = 0
goal = 'office'

##############################################################################
class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position
    
    def __repr__(self):
      return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
      return self.f < other.f
    
    # defining greater than for purposes of heap queue
    def __gt__(self, other):
      return self.f > other.f

def return_path(current_node, maze, start, end):
    path = []
    current = current_node
    while current is not None:
        path.append(current.position)
        current = current.parent
    maze[start] = 2
    maze[end] = 2
    for p in path[1:-1]:  maze[p] = 3
    ans = [path[::-1], maze]
    return ans 

def astar(x0, z0, xF, zF, allow_diagonal_movement = False):
    """
    Returns a list of tuples as a path from the given start to the given end in the given maze
    :param maze:
    :param start:
    :param end:
    :return:
    """
    #https://gist.github.com/Nicholas-Swift/003e1932ef2804bebef2710527008f44
    
    #map coordinates
    mapLim = pd.read_excel('map.xlsx')
    mapLim = mapLim.drop(columns = ['set'])
    mapLim = mapLim.set_index('loc')
    xMin = mapLim.iloc[-1, 0]
    xMax = mapLim.iloc[-1, 1]
    zMin = mapLim.iloc[-1, 2]
    zMax = mapLim.iloc[-1, 3]
    del mapLim
    
    #map discretization  
    maze = pd.read_excel('maze.xlsx')
    maze = maze.values
    nx_maze = len(maze[0])
    nz_maze = len(maze)
    dx_maze = round((xMax -xMin) / nx_maze, 2)
    dz_maze = round((zMax -zMin) / nz_maze, 2)
    
    #start node position
    nx0 = int(round((x0 - xMin)/dx_maze -0.5,0))
    nz0 = int(round((z0 - zMin)/dz_maze -0.5,0))
    start = (nz0, nx0)
            
    #end node position
    nxF = int(round((xF - xMin)/dx_maze -0.5,0))
    nzF = int(round((zF - zMin)/dz_maze -0.5,0))
    end = (nzF, nxF)
    
    #correct end if necessary
    if maze[end] == 1:
        old_end = end
        print('changing start location')
        for i in range(5):
            temp = (nzF, nxF+i)
            if maze[temp] == 0: 
                end = temp
                break
            temp = (nzF+i, nxF)
            if maze[temp] == 0: 
                end = temp
                break
        print('old end', old_end)
        print('new end', end)
    
    #create start, end nodes
    start_node = Node(None, start)
    start_node.g = start_node.h = start_node.f = 0
    end_node = Node(None, end)
    end_node.g = end_node.h = end_node.f = 0

    #initialize open, closed lists
    open_list = []
    closed_list = []

    #heapify the open_list and add start node
    heapq.heapify(open_list) 
    heapq.heappush(open_list, start_node)

    #add a stop condition
    outer_iterations = 0
    max_iterations = (len(maze[0]) * len(maze) // 2)

    #neigboring squares searched
    adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)
    if allow_diagonal_movement:
        adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0), (-1, -1), (-1, 1), (1, -1), (1, 1),)

    #find path
    while len(open_list) > 0:
        outer_iterations += 1

        if outer_iterations > max_iterations:
          # if we hit this point return the path such as it is
          # it will not contain the destination
          warn("giving up on pathfinding too many iterations")
          return return_path(current_node, maze, start, end)       
        
        #get current node
        current_node = heapq.heappop(open_list)
        closed_list.append(current_node)

        #found the goal
        if current_node == end_node:
            return return_path(current_node, maze, start, end)

        # Generate children
        children = []
        
        for new_position in adjacent_squares: # Adjacent squares

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

            # Make sure within range
            if node_position[0] > (len(maze) - 1) or node_position[0] < 0 or node_position[1] > (len(maze[len(maze)-1]) -1) or node_position[1] < 0:
                continue

            # Make sure walkable terrain
            if maze[node_position[0]][node_position[1]] != 0:
                continue

            # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:
            # Child is on the closed list
            if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                continue

            # Create the f, g, and h values
            child.g = current_node.g + 1
            child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
            child.f = child.g + child.h

            # Child is already in the open list
            if len([open_node for open_node in open_list if child.position == open_node.position and child.g > open_node.g]) > 0:
                continue

            # Add the child to the open list
            heapq.heappush(open_list, child)

    warn("Couldn't get a path to destination")
    return None

def write_Placement(xTemp, yGround, zTemp, yawTemp):
    return '\t\t<Placement x="' + str(xTemp) + '" y="' + str(yGround) + '" z="' + str(zTemp) + '" yaw="' + str(yawTemp)  + '"/>\n'

def nearest_pos(pos):
    
    #pos
    x = pos[0]
    z = pos[1]
    
    #map limits
    mapLim = pd.read_excel('map.xlsx')
    mapLim = mapLim.drop(columns = ['set'])
    mapLim = mapLim.set_index('loc')
    xmin = mapLim.iloc[-1, 0]
    xmax = mapLim.iloc[-1, 1]
    zmin = mapLim.iloc[-1, 2]
    zmax = mapLim.iloc[-1, 3]
    
    #correct pos if not in map limits
    if x > xmax:    x == xmax
    elif x < xmin:  x == xmin
    
    if z > zmax:    z == zmax
    elif z < zmin:  z == zmin

    #maze details
    maze = pd.read_excel('maze.xlsx')
    maze = maze.values
    nx_maze = len(maze[0])
    nz_maze = len(maze)
    dx_maze = round((xmax -xmin) / nx_maze, 2)
    dz_maze = round((zmax -zmin) / nz_maze, 2)
    
    #convert pos to node
    nx0 = int(round((x - xmin)/dx_maze -0.5,0))
    nz0 = int(round((z - zmin)/dz_maze -0.5,0)) 
    
    #correct node if necessary
    if nx0 > 49:    nx0 = 49
    elif nx0 < 0:   nx0 = 0
    
    if nz0 > 49:    nz0 = 49
    elif nz0 < 0:   nz0 = 0
    
    #search for OK node
    if maze[(nz0, nx0)] == 1:
        search = True
        k = 1
        while search:
            for i in range(-k,k+1):
                for j in range(-k, k+1):
                    nz_temp = nz0 + j
                    nx_temp = nx0 + i
                    
                    if nx_temp > 49:    nx_temp = 49
                    elif nx_temp < 0:   nx_temp = 0
                    
                    if nz_temp > 49:    nz_temp = 49
                    elif nz_temp < 0:   nz_temp = 0
                    
                    node_temp = (nz_temp, nx_temp)
                    if maze[node_temp] == 0:
                        node = node_temp
                        x = round((node[1] +0.5)*dx_maze + xmin, 0)
                        z = round((node[0] +0.5)*dz_maze + zmin, 0)                    
                        search = False      
                    if search is False: break
                if search is False: break
            k+=1
    return (x, z)

def get_Placement(pInc, mapExt, iPos):
    x = iPos[0]
    z = iPos[1]
    
    xMin = mapExt[0]
    xMax = mapExt[1]
    zMin = mapExt[2]
    zMax = mapExt[3]
    
    if x != 0 or z != 0:
        xStart = x
        yStart = 5.5
        zStart = z
        yawStart = 90
    else:
        random.seed(pInc)
        xStart = random.randint(xMin, xMax)+0.5
        zStart = random.randint(zMin, zMax)+0.5
        yStart = 5.5
        yawStart = 90
    print('old start', xStart, zStart)    
    (xStart,zStart) = nearest_pos((xStart,zStart))
    print('new start', xStart, zStart)    
    ans = write_Placement(xStart, yStart, zStart, yawStart) 
    ans = '\t'+ textwrap.dedent(ans)    
    return ans

###############################################################################

#map coordinates
mapLim = pd.read_excel('map.xlsx')
mapLim = mapLim.drop(columns = ['set'])
mapLim = mapLim.set_index('loc')
xMin = mapLim.iloc[-1, 0]
xMax = mapLim.iloc[-1, 1]
zMin = mapLim.iloc[-1, 2]
zMax = mapLim.iloc[-1, 3]
mapExt = [xMin, xMax, zMin, zMax]
mapLim = mapLim[:-1]

#plot map
img = plt.imread("img/map.png")
#plt.rc('font',family='Times New Roman')   
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img, alpha = 1, origin = 'lower', extent = mapExt)
Xmajor_ticks = np.arange(xMin+7, xMax+17, 50)
Xminor_ticks = np.arange(xMin+7, xMax+17, 5)
Zmajor_ticks = np.arange(zMin+13, zMax+23, 20)
Zminor_ticks = np.arange(zMin+13, zMax+23, 5)
ax.set_xticks(Xmajor_ticks)
ax.set_xticks(Xminor_ticks, minor=True)
ax.set_yticks(Zmajor_ticks)
ax.set_yticks(Zminor_ticks, minor=True)
ax.grid(which='minor', alpha=0.1, color ='black')
ax.grid(which='major', alpha=0.2, color ='black')
plt.gca().invert_yaxis()

#start node
temp = get_Placement(pInc, mapExt, (0,0,0)).split('\"')
x0 = float(temp[1])
z0 = float(temp[5])

#end node
xF = mapLim.loc[goal,'xent']
zF = mapLim.loc[goal,'zent']

#find path
[path, maze] = astar(x0, z0, xF, zF)
        
# #plot location boxes
# pLoc = list(mapLim.index.values)
# for i in range(len(mapLim)):
#     loc = mapLim.index.values[i]
#     xmin = mapLim.iloc[i,0]
#     xmax = mapLim.iloc[i,1]
#     zmin = mapLim.iloc[i,2]
#     zmax = mapLim.iloc[i,3]
#     label = mapLim.index.values[i]
#     if loc == 'landfill' or loc == 'farm':
#         ax.add_patch(patches.Rectangle(xy=(xmin, zmin), width=(xmax-xmin), height=(zmax-zmin), color = 'black', linewidth=1, linestyle = 'dotted', fill=False, label = label, zorder = 4))
#     if loc == 'lake':
#         xmax += 20
#         zmin -= 10
#     ax.text(0.5*(xmin+xmax), 0.5*(zmin+zmax), '{}'.format(i+1),
#         horizontalalignment='center',
#         verticalalignment='center',
#         fontsize=7, color='black',
#         fontname = 'Courier',
#         weight = 'bold',
#         rotation =0)
    
#plot nodes
dx_maze = round((xMax -xMin) / 50, 2)
dz_maze = round((zMax -zMin) / 50, 2)
x = xMin + dx_maze/2
z = zMin + dz_maze/2
for row in maze: 
    for val in row:
        if val == 1:    plt.scatter(x, z, color = 'black', marker='.', s = 10, zorder = 3)
        elif val == 2:  
            plt.scatter(x, z, color = 'white', marker='.', s = 10, zorder = 3)
            plt.scatter(x, z, color = 'red', marker='x', s = 10, zorder = 5)
        elif val == 3:  
            plt.scatter(x, z, color = 'white', marker='.', s = 10, zorder = 3)
            plt.scatter(x, z, color = 'blue', marker='.', s = 10, zorder = 4)  
        else:           
            plt.scatter(x, z, color = 'white', marker='.', s = 10, zorder = 3)
        x+=dx_maze
    z +=dz_maze
    x = xMin + dx_maze/2
    
#plot end checkouts
checkouts = [(0,   620), (0,   710), (50, 595), (50,   790),  (120,   595), (190,   595), (200,   790), (220, 635), (220,   710),]
for i in range(len(checkouts)):
    x = checkouts[i][0]
    z = checkouts[i][1]
    plt.scatter(x, z, color = 'green', marker='*', s = 40, zorder = 5)

# #plot location entrances
# pLoc = list(mapLim.index.values)
# for i in range(len(mapLim)):
#     xent = mapLim.iloc[i,4]
#     zent = mapLim.iloc[i,5]
#     xmid = mapLim.iloc[i,6]
#     zmid = mapLim.iloc[i,7]
#     plt.scatter(xent, zent, color = 'orange', marker='s', s = 25, zorder = 2)
#     plt.scatter(xmid, zmid, color = 'purple', marker='s', s = 25, zorder = 2)
    
# #plot end checkouts
# checkouts = [(0,   620), (0,   710), (50, 595), (50,   790), (120,   595), (190,   595), (200,   790), (220, 635), (220,   710)]
# for i in range(len(checkouts)):
#     x = checkouts[i][0]
#     z = checkouts[i][1]
#     plt.scatter(x, z, color = 'y', marker='*', s = 40, zorder = 3)

