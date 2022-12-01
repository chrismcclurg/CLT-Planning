# =============================================================================
# Construal Level Theory for Agent-based Planning
#
# C McClurg, AR Wagner, S Rajtmajer
# =============================================================================

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from malmo import get_xz
import pandas as pd
import random 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#inputs
pSet = 2
pRan = 0  #alpha
pInc = 0  #seed

#similar locations
mapSim = pd.read_excel('truth.xlsx')
id_item = mapSim.name.tolist()
col = 's' + str(pSet) + '_loc'
simLoc = mapSim[col].tolist()

#map coordinates
mapLim = pd.read_excel('map.xlsx')
mapLim = mapLim[mapLim.set < (pSet + 1)]
locSet = list(mapLim.set)[:-1]
mapLim = mapLim.drop(columns = ['set'])
mapLim = mapLim.set_index('loc')
x1 = mapLim.iloc[-1, 0]
x2 = mapLim.iloc[-1, 1]
z1 = mapLim.iloc[-1, 2]
z2 = mapLim.iloc[-1, 3]
mapExt = [x1, x2, z1, z2]
mapLim = mapLim[:-1]
#mapLim = mapLim.sort_index()

random.seed(pInc)  
img = plt.imread("img/map.png") 
plt.rc('font',family='Times New Roman')     
fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(img, alpha = 1, origin = 'lower', extent = mapExt)

# plt.xlim(x1, x2)
# plt.ylim(z1, z2)

import numpy as np
Xmajor_ticks = np.arange(x1+7, x2+17, 50)
Xminor_ticks = np.arange(x1+7, x2+17, 5)
Zmajor_ticks = np.arange(z1+13, z2+23, 20)
Zminor_ticks = np.arange(z1+13, z2+23, 5)
ax.set_xticks(Xmajor_ticks)
ax.set_xticks(Xminor_ticks, minor=True)
ax.set_yticks(Zmajor_ticks)
ax.set_yticks(Zminor_ticks, minor=True)
ax.grid(which='minor', alpha=0.1, color ='black')
ax.grid(which='major', alpha=0.2, color ='black')
plt.xlim(x1, x2)
plt.ylim(z1, z2)

plt.gca().invert_yaxis()

cmap = matplotlib.cm.get_cmap('hsv')
cn = list(range(0,270,15))
pLoc = list(mapLim.index.values)
lineList = ['solid', 'dashdot', 'dotted']
#LOCATION BOXES
for i in range(len(mapLim)):
    loc = mapLim.index[i]
    xmin = mapLim.iloc[i,0]
    xmax = mapLim.iloc[i,1]
    zmin = mapLim.iloc[i,2]
    zmax = mapLim.iloc[i,3]
    xmid = (xmin + xmax) / 2 - 20
    zmid = (zmin + zmax) / 2 + 5
    color = cmap(cn[i])
    loc = loc.split()
    if len(loc) > 1: loc = loc[0] + '\n' + loc[1]
    else: loc = loc[0]
    loc = str(loc)
    hix = locSet[i]
    label = '{} {}'. format(str(i+1), mapLim.index.values[i])
    l = lineList[hix]
    if loc == 'landfill' or loc == 'farm' or loc == 'garden':
        ax.add_patch(patches.Rectangle(xy=(xmin, zmin), width=(xmax-xmin), height=(zmax-zmin), color = 'sienna', linewidth=1, fill=True, label='_nolegend_'))
    elif loc == 'church':
        ax.add_patch(patches.Rectangle(xy=(xmin, zmin), width=(xmax-xmin), height=(zmax-zmin), color = 'grey', linewidth=1, fill=True, label='_nolegend_'))
    ax.add_patch(patches.Rectangle(xy=(xmin, zmin), width=(xmax-xmin), height=(zmax-zmin), color = 'black', linewidth=0, linestyle = l, fill=True, label = '_nolegend_'))
    ax.add_patch(patches.Rectangle(xy=(xmin, zmin), width=(xmax-xmin), height=(zmax-zmin), color = color, linewidth=1, fill=False, label = label))
    ax.text(0.5*(xmin+xmax), 0.5*(zmin+zmax), '{}'.format(i+1),
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=7, color='white',
        weight = 'bold',
        rotation =0, zorder = 7)


#ITEM LOCATIONS

#remove duplicates
id_item2 = list(set(id_item))
id_item2.sort()
simLoc2 = []

for x in id_item2:
    ix = id_item.index(x)
    temp = simLoc[ix]
    simLoc2.append(temp)
        
        
random.seed(0)
cmap = matplotlib.cm.get_cmap('hsv')
cn = list(range(0,270,15))
pLoc = list(mapLim.index)
for i in range(len(id_item2)):
    xz = get_xz(simLoc2[i], mapLim, pRan, mapExt)
    x = xz[0]
    z = xz[1]
    a = xz[2]
    simLoc[i]
    temp = simLoc2[i]
    ix = pLoc.index(temp)
    color = cmap(cn[ix])
    plt.scatter(x, z, s = 3, color = color, marker = '.', zorder = 7)

# #AGENT LOCATION
# #random.seed(pInc)
# for i in range(10):
#     random.seed(i)
#     x = random.randint(x1, x2)
#     z = random.randint(z1, z2)  
#     #plt.scatter(x, z, s = 30, color = 'black', marker = 'x')
#     plt.scatter(x, z, s = 10, color = 'black', zorder = 3)
#     #plt.arrow(x,z,-10,0, width = 2, color = 'black')
#     #plt.annotate(str(i), (x+2,z-7))

#TITLE AND LEGENDS
#ax.legend(loc=(1.04,0), ncol=1)
#plt.xlabel("x-coordinate")
#plt.ylabel("z-coordinate")
#plt.title('map (set = {}, \u03B1 = {}, seed = {})'.format(pSet, pRan, pInc))

plt.show()
