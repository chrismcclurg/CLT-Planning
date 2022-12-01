# =============================================================================
# Construal Level Theory for Agent-based Planning
#
# C McClurg, AR Wagner, S Rajtmajer
# =============================================================================
# MALMO-SPECIFIC FILE
# Purpose: collection of functions to define execution of action (in Malmo)
# =============================================================================
# Copyright (c) 2016 Microsoft Corporation
# 
# Permission is hereby granted, free of charge, to any person obtaining a  copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the  rights 
# to use, copy, modify, merge, publish,  distribute,  sublicense,  and/or  sell 
# copies of the Software, and  to  permit  persons  to  whom  the  Software  is 
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall  be  included  in 
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY  KIND,  EXPRESS  OR 
# IMPLIED, INCLUDING BUT NOT LIMITED  TO  THE  WARRANTIES  OF  MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT  SHALL  THE 
# AUTHORS OR COPYRIGHT HOLDERS BE  LIABLE  FOR  ANY  CLAIM,  DAMAGES  OR  OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
# =============================================================================

from __future__ import print_function
from __future__ import division
from builtins import range
from warnings import warn
import heapq
from models import MalmoPython as MP
import os
import sys
import time
import json
import numpy as np
import random
import pandas as pd
import textwrap



if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)
    
# =============================================================================
# FUNCTIONS

#subfunction of many functions (return coordinates based on map constraints)
def get_xz(sLoc, mapLim, pRan, mapExt):
    
    # determine amount of random noise
    if   pRan == 0.33: alist = [1,0,0]
    elif pRan == 0.67: alist = [1,1,0]
    elif pRan == 1.00: alist = [1,1,1]
    else:               alist = [0,0,0]

    # determine limits of most similar
    xmin = mapLim.loc[sLoc,'xmin']
    xmax = mapLim.loc[sLoc,'xmax']
    zmin = mapLim.loc[sLoc,'zmin']
    zmax = mapLim.loc[sLoc,'zmax']
    
    #number of locations in set
    nmap = len(mapLim)
    
    locSet = list(mapLim.index.values)
    nsim = locSet.index(sLoc)

    #random
    offset = 3 #avoid boundaries
    xsim = random.randint(xmin+offset, xmax-offset)
    zsim = random.randint(zmin+offset, zmax-offset)
    nrand = random.randint(0, nmap-1)
    
    #determine limits of random
    xminR = mapLim.iloc[nrand,0]
    xmaxR = mapLim.iloc[nrand,1]
    zminR = mapLim.iloc[nrand,2]
    zmaxR = mapLim.iloc[nrand,3]
    
    #random
    xrand = random.randint(xminR+offset, xmaxR-offset)
    zrand = random.randint(zminR+offset, zmaxR-offset)
    arand = random.choice(alist)
    
    same_as_sim = True
    # assign coordinate values
    if arand == 1:
        if nsim != nrand: same_as_sim = False
        x = xrand
        z = zrand
    else:
        x = xsim
        z = zsim
        
    return [x, z, same_as_sim]

#subfunction of get_missionXML (return script for drawing item)
def write_DrawItem(xTemp, yGround, zTemp, itemTemp):
    return '\t\t\t\t<DrawItem x="' + str(xTemp) + '" y="' + str(yGround) + '" z="' + str(zTemp) + '" type="' + itemTemp + '"/>\n'

#subfunction of get_missionXML (return script for drawing item)
def get_DrawItem(id_item, sLoc_item, mapLim, pRan, mapExt):   
    xz = get_xz(sLoc_item[0], mapLim, pRan, mapExt)
    xTemp = xz[0]
    yTemp = 4
    zTemp = xz[1]
    ans = write_DrawItem(xTemp, yTemp, zTemp, id_item[0]) 
    ans = '\t'+ textwrap.dedent(ans)
    
    for i in range(1,len(id_item)):
        itemTemp = id_item[i]
        xz = get_xz(sLoc_item[i], mapLim, pRan, mapExt)
        xTemp = xz[0]
        yTemp = 5
        zTemp = xz[1]
        ans += write_DrawItem(xTemp, yTemp, zTemp, itemTemp)
    return ans

#subfunction of get_missionXML (return script for drawing block)
def write_DrawBlock(xTemp, yGround, zTemp, blockTemp):
    return '\t\t\t\t<DrawBlock x="' + str(xTemp) + '" y="' + str(yGround) + '" z="' + str(zTemp) + '" type="' + blockTemp + '"/>\n'

#subfunction of get_missionXML (return script for drawing block)
def get_DrawBlock(id_block, sLoc_block, mapLim, pRan, mapExt, problemList):   
    xz = get_xz(sLoc_block[0], mapLim, pRan, mapExt)
    xTemp = xz[0]
    yTemp = 4
    zTemp = xz[1]
    
    if id_block[0] in problemList: id_block = 'air'
    
    ans = write_DrawBlock(xTemp, yTemp, zTemp, id_block[0]) 
    ans = '\t'+ textwrap.dedent(ans)
    
    for i in range(1,len(id_block)):
        blockTemp = id_block[i]
        if blockTemp not in problemList: 
            xz = get_xz(sLoc_block[i], mapLim, pRan, mapExt)
            xTemp = xz[0]
            yTemp = 5
            zTemp = xz[1]
            ans += write_DrawBlock(xTemp, yTemp, zTemp, blockTemp)
    return ans

#subfunction of get_missionXML (return script for changing ground within location boundaries, ie stone -> grass)
def change_ground(id_block, sLoc, mapLim, pRan, mapExt, y = 3):   
    ans = ''
    if sLoc in mapLim.index.values:
        # determine limits of location
        xmin = mapLim.loc[sLoc,'xmin']
        xmax = mapLim.loc[sLoc,'xmax']
        zmin = mapLim.loc[sLoc,'zmin']
        zmax = mapLim.loc[sLoc,'zmax']
        for i in range(xmin+3, xmax-3):
            for k in range(zmin+3, zmax-3):
                ans += write_DrawBlock(i, y, k, id_block) 
    return ans

#subfunction of get_missionXML (return script for changing specified volume, useful for map re-construction)
def change_section(id_block, xmin, xmax, ymin, ymax, zmin, zmax):   
    ans = ''
    for i in range(xmin, xmax+1):
        for j in range(ymin, ymax+1):
            for k in range(zmin, zmax+1):
                ans += write_DrawBlock(i, j, k, id_block) 
    return ans

#subfunction of get_missionXML (return script for placing agent at start)
def write_Placement(xTemp, yGround, zTemp, yawTemp):
    return '\t\t<Placement x="' + str(xTemp) + '" y="' + str(yGround) + '" z="' + str(zTemp) + '" yaw="' + str(yawTemp)  + '"/>\n'

#subfunction of many functions (return a "good" location, corrected if location is not in maze or map)
def nearest_pos(pos):
    
    #pos
    x = pos[0]
    z = pos[1]
    
    #map limits
    mapLim = pd.read_excel('utils/map.xlsx')
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
    maze = pd.read_excel('utils/maze.xlsx')
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

#subfunction of get_missionXML (return script for agent start location)
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
        
    (xStart,zStart) = nearest_pos((xStart,zStart))
    ans = write_Placement(xStart, yStart, zStart, yawStart) 
    ans = '\t'+ textwrap.dedent(ans)    
    return ans

#subfunction of get_missionXML (return script for writing to screen)
def write_Summary(iTrial):
    return 'TCRISP Trial no. {})\n'.format(iTrial)

#function (return mission specs for malmo input)
def get_missionXML(pSet, pRan, pInc, iPos, iTrial):
    
    #item locations
    mapSim = pd.read_excel('utils/truth.xlsx')
    
    mapSim_item  = mapSim[mapSim.type =='ItemType']
    mapSim_block = mapSim[mapSim.type =='BlockType']
    
    id_item  = mapSim_item.name.tolist()
    id_block = mapSim_block.name.tolist()
    
    col = 's' + str(pSet) + '_loc'
    sLoc_item = mapSim_item[col].tolist()
    sLoc_block = mapSim_block[col].tolist()
    
    #remove duplicates
    id_item2 = list(set(id_item))
    id_item2.sort()
    id_block2 = list(set(id_block))
    id_block2.sort()
    
    sLoc_item2 = []
    sLoc_block2 = []
    
    for x in id_item2:
        ix = id_item.index(x)
        temp = sLoc_item[ix]
        sLoc_item2.append(temp)
        
    for x in id_block2:
        ix = id_block.index(x)
        temp = sLoc_block[ix]
        sLoc_block2.append(temp)
    
    del mapSim, mapSim_item, mapSim_block

    #map coordinates
    mapLim = pd.read_excel('utils/map.xlsx')
    mapLim = mapLim[mapLim.set < (pSet + 1)]
    mapLim = mapLim.drop(columns = ['set'])
    mapLim = mapLim.set_index('loc')
    xmin = mapLim.iloc[-1, 0]
    xmax = mapLim.iloc[-1, 1]
    zmin = mapLim.iloc[-1, 2]
    zmax = mapLim.iloc[-1, 3]
    mapExt = [xmin, xmax, zmin, zmax]
    mapLim = mapLim[:-1]

    #problematic items (agent gets stuck)
    problemList = ['water', 'lava', 'web', 'gray_shulker_box', 'green_shulker_box', 'gray_shulker_box', 'light_blue_shulker_box', 
                   'lime_shulker_box', 'magenta_shulker_box', 'orange_shulker_box', 'pink_shulker_box', 'purple_shulker_box', 
                   'red_shulker_box', 'silver_shulker_box', 'white_shulker_box', 'yellow_shulker_box', 'blue_shulker_box', 
                   'brown_shulker_box', 'cyan_shulker_box']

    random.seed(pInc) 
    
    ans ='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
                <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
                
                  <About>
                    <Summary>
                    ''' + write_Summary(iTrial) + '''
                    </Summary>
                  </About>
                  
                <ServerSection>
                  <ServerInitialConditions>
                    <Time>
                        <StartTime>6000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime>
                    </Time>
                    <Weather>clear</Weather>
                  </ServerInitialConditions>
                  <ServerHandlers>
                      <FileWorldGenerator src="C:/Users/chris/malmo/Minecraft/run/saves/small_town" />                      
                      <DrawingDecorator>     
                          ''' + change_ground('glowstone', 'landfill', mapLim, pRan, mapExt, 3) + '''    
                          ''' + change_ground('dirt', 'garden', mapLim, pRan, mapExt, 3) + '''                           
                          ''' + change_ground('red_flower', 'garden', mapLim, pRan, mapExt, 4) + '''    
                          ''' + change_ground('dirt', 'farm', mapLim, pRan, mapExt, 3) + ''' 
                          ''' + change_section('fence',109,112,4,4,781,781) + '''  
                          ''' + change_section('iron_bars',166,169,4,6,719,719) + '''  
                          ''' + change_section('iron_bars',183,183,4,6,732,736) + '''  
                          ''' + change_section('iron_bars',107,110,4,5,721,721) + '''  
                          ''' + change_section('iron_bars',110,110,4,5,721,725) + '''
                          ''' + change_section('dirt',115,122,3,3,725,725) + '''  
                          ''' + change_section('grass',117,177,3,3,722,722) + '''  
                          ''' + change_section('stone',115,115,3,3,727,740) + '''  
                          ''' + change_section('stone',116,122,3,3,726,726) + '''  
                          ''' + change_section('stone',111,114,3,3,734,734) + '''  
                          ''' + change_section('double_stone_slab',107,115,3,3,741,748) + '''  
                          ''' + change_section('double_stone_slab',756,764,3,3,722,722) + '''  
                          ''' + change_section('stone',143,154,3,3,756,764) + '''  
                          ''' + change_section('air',107,114,4,4,742,747) + '''  
                          ''' + change_section('red_flower',115,122,4,4,725,725) + '''  
                          ''' + change_section('air',115,127,4,4,717,723) + '''  
                          ''' + change_section('air',115,125,4,5,722,722) + '''  
                          ''' + change_section('air',125,125,4,5,723,723) + '''  
                          ''' + change_section('air',126,126,4,5,724,724) + ''' 
                          ''' + change_section('air',127,127,4,5,725,725) + '''  
                          ''' + change_section('iron_bars',124,127,4,5,726,726) + '''  
                          ''' + change_section('fence',107,109,4,4,741,741) + '''  
                          ''' + change_section('fence',115,115,4,4,742,747) + '''  
                          ''' + change_section('lime_shulker_box',119,119,4,5,741,741) + '''  
                          ''' + change_section('lime_shulker_box',123,123,4,5,735,735) + '''  
                          ''' + change_section('iron_bars',103,103,4,5,721,725) + '''  
                          ''' + change_section('iron_bars',96,96,4,5,722,724) + '''  
                          ''' + change_section('fence',65,67,4,4,733,733) + '''  
                          ''' + change_section('fence',65,67,4,4,740,740) + '''  
                          ''' + change_section('fence',64,70,4,4,770,770) + '''  
                          ''' + change_section('fence',64,70,4,4,767,767) + '''  
                          ''' + change_section('fence_gate',98,99,4,4,781,781) + '''  
                          ''' + change_section('fence_gate',119,122,4,4,781,781) + '''  
                          ''' + change_section('iron_bars',152,154,4,5,764,764) + '''  
                          ''' + change_section('iron_bars',143,143,4,5,761,763) + '''  
                          ''' + change_section('iron_bars',137,140,4,5,696,696) + '''  
                          ''' + change_section('air',167,167,4,7,700,700) + '''  
                          ''' + change_section('air',170,170,4,7,700,700) + '''  
                          ''' + change_section('sandstone',170,170,4,6,699,703) + '''  
                          ''' + change_section('air',170,170,6,6,702,703) + '''                            
                          ''' + change_section('sandstone',167,167,4,6,699,703) + '''  
                          ''' + change_section('air',167,167,6,6,702,703) + '''                            
                          ''' + change_section('fence',131,147,4,4,658,658) + '''  
                          ''' + change_section('fence_gate',142,144,4,4,658,658) + '''  
                          ''' + change_section('fence_gate',134,135,4,4,658,658) + '''  
                          ''' + change_section('fence',131,140,4,4,654,654) + '''  
                          ''' + change_section('fence',140,140,4,4,652,654) + '''        
                          ''' + change_section('planks',90,102,3,3,725,735) + ''' 
                          ''' + change_section('air',133,133,4,5,720,726) + '''                              
                          ''' + change_section('iron_bars',133,135,4,5,727,727) + '''        
                          ''' + change_section('iron_bars',153,155,4,5,728,728) + '''                                      
                          ''' + change_section('air',141,142,4,5,651,651) + '''                              
                          ''' + change_section('stone',99,99,3,3,723,723) + '''    
                          ''' + change_section('fence_gate',171,171,4,4,670,670) + '''
                          ''' + change_section('planks',168,173,4,5,674,674) + '''    
                          ''' + change_section('planks',177,177,4,5,670,670) + '''    
                          ''' + change_section('stone',168,173,3,3,671,673) + '''                              
                          ''' + change_section('air',138,139,4,5,692,692) + '''    
                          ''' + change_section('air',69,69,5,6,661,661) + '''    
                          ''' + change_section('air',71,71,5,6,736,737) + '''    
                          ''' + change_section('air',71,71,4,5,768,769) + '''    
                          ''' + change_section('air',141,142,4,5,651,651) + '''    
                          ''' + change_section('air',148,148,4,6,646,650) + '''    
                          ''' + change_section('air',149,154,4,6,646,646) + '''    
                          ''' + change_section('air',149,154,4,7,657,657) + '''    
                          ''' + change_section('air',97,98,4,6,726,734) + '''         
                          ''' + change_section('brick_block',98,102,4,10,725,725) + '''       
                          ''' + change_section('brick_block',102,102,4,10,726,734) + '''       
                          ''' + change_section('wooden_slab',99,102,11,11,724,736) + '''      
                          ''' + change_section('air',99,101,4,5,725,725) + '''        
                          ''' + change_section('air',99,101,4,6,733,733) + '''      
                          ''' + change_section('air',99,101,6,6,728,732) + '''                                                                                                                                                                                                                                                                                                                                                                                                                              
                          ''' + change_section('air',29,31,5,7,752,754) + '''    
                          ''' + change_section('air',45,45,5,7,755,757) + '''    
                          ''' + change_section('brick_block',19,24,5,9,750,750) + '''    
                          ''' + change_section('brick_block',98,102,4,10,735,735) + '''  
                          ''' + change_section('fence',200,225,4,4,745,745) + '''  
                          ''' + change_section('fence',225,225,4,4,745,775) + '''    
                          ''' + change_section('fence',200,225,4,4,775,775) + '''    
                          ''' + change_section('fence',200,200,4,4,765,775) + '''    
                          ''' + change_section('fence',200,200,4,4,745,755) + '''    
                          ''' + change_section('sandstone',28,28,5,6,752,753) + '''     
                          ''' + change_section('planks',25,42,3,3,679,696) + '''   
                          ''' + change_section('yellow_flower',10,44,4,4,660,665) + ''' 
                          ''' + change_section('yellow_flower',10,44,4,4,650,655) + '''  
                          ''' + change_section('yellow_flower',10,44,4,4,640,645) + '''   
                          ''' + change_section('red_flower',10,44,4,4,655,660) + '''   
                          ''' + change_section('red_flower',10,44,4,4,645,650) + '''   
                          ''' + change_section('red_flower',10,44,4,4,635,640) + '''        
                          ''' + change_section('air',71,71,5,7,734,739) + '''   
                          ''' + change_section('air',68,70,5,7,739,739) + '''   
                          ''' + change_section('air',68,70,5,7,734,734) + '''   
                          ''' + change_section('brick_block',68,68,5,7,734,739) + '''   
                          ''' + change_section('air',68,68,5,6,736,737) + '''   
                          ''' + change_section('wooden_slab',68,70,8,8,735,738) + '''                         
                          ''' + change_section('blue_shulker_box',155,155,3,3,615,615) + '''    
                          ''' + change_section('blue_shulker_box',159,159,3,3,613,613) + '''    
                          ''' + change_section('blue_shulker_box',159,159,3,3,615,615) + '''                             
                          ''' + change_section('blue_shulker_box',147,147,3,3,609,610) + '''    
                          ''' + change_section('blue_shulker_box',147,147,3,3,615,615) + '''    
                          ''' + change_section('blue_shulker_box',147,147,3,3,617,619) + '''    
                          ''' + change_section('blue_shulker_box',148,148,3,3,609,619) + '''    
                          ''' + change_section('blue_shulker_box',149,149,3,3,608,611) + '''    
                          ''' + change_section('blue_shulker_box',149,149,3,3,613,620) + '''                             
                          ''' + change_section('blue_shulker_box',150,154,3,3,608,621) + '''    
                          ''' + change_section('blue_shulker_box',157,162,3,3,624,624) + '''    
                          ''' + change_section('blue_shulker_box',157,164,3,3,623,623) + '''    
                          ''' + change_section('blue_shulker_box',156,164,3,3,622,622) + '''    
                          ''' + change_section('blue_shulker_box',153,154,3,3,622,622) + '''    
                          ''' + change_section('blue_shulker_box',155,165,3,3,621,621) + '''    
                          ''' + change_section('blue_shulker_box',150,159,3,3,608,608) + '''    
                          ''' + change_section('blue_shulker_box',161,165,3,3,608,608) + '''    
                          ''' + change_section('blue_shulker_box',161,165,3,3,607,607) + '''    
                          ''' + change_section('blue_shulker_box',163,165,3,3,606,606) + '''    
                          ''' + change_section('blue_shulker_box',156,159,3,3,607,607) + '''    
                          ''' + change_section('blue_shulker_box',151,154,3,3,607,607) + '''    
                          ''' + change_section('blue_shulker_box',166,166,3,3,613,619) + '''    
                          ''' + change_section('blue_shulker_box',160,165,3,3,609,620) + '''    
                          ''' + change_section('blue_shulker_box',155,159,3,3,609,612) + '''    
                          ''' + change_section('blue_shulker_box',155,159,3,3,616,620) + '''    
                          ''' + change_section('air',72,80,5,8,724,749) + '''    
                          ''' + change_section('air',72,74,4,6,764,768) + '''    
                          ''' + change_section('air',163,175,5,8,682,697) + '''    
                          ''' + change_section('air',168,169,5,6,698,698) + '''    
                          ''' + change_section('green_shulker_box',110,110,4,7,726,733) + '''    
                          ''' + change_section('green_shulker_box',111,111,4,8,726,726) + '''    
                          ''' + change_section('air',115,115,4,6,727,740) + '''    
                          ''' + change_section('air',112,114,4,6,741,741) + '''    
                          ''' + change_section('air',111,114,4,6,734,734) + '''    
                          ''' + change_section('air',144,154,4,5,756,760) + '''    
                          ''' + change_section('air',137,150,4,6,730,737) + '''    
                          ''' + change_section('air',146,149,5,7,764,764) + '''    
                          ''' + change_section('air',144,144,5,8,765,769) + '''   
                          ''' + change_section('air',139,139,5,8,765,769) + '''    
                          ''' + change_section('air',139,144,5,8,770,770) + '''    
                          ''' + change_section('planks',139,144,4,4,765,770) + '''    
                          ''' + change_section('brick_block',140,143,4,8,764,764) + '''  
                          ''' + change_section('brick_block',140,143,4,8,764,764) + '''  
                          ''' + change_section('glass_pane',148,149,4,5,738,738) + '''  
                          ''' + change_section('glass_pane',138,139,4,5,738,738) + '''  
                          ''' + change_section('glass_pane',145,150,4,6,729,729) + '''  
                          ''' + change_section('grass',146,149,3,3,724,728) + ''' 
                          ''' + change_section('air',25,29,5,9,752,754) + '''   
                          ''' + change_section('glass_pane',26,27,5,7,761,761) + '''  
                          ''' + change_section('glass_pane',137,143,8,10,728,728) + '''  
                          ''' + change_section('glass_pane',137,150,4,6,738,738) + '''  
                          ''' + change_section('air',136,143,7,7,726,726) + '''  
                          ''' + change_section('fence',185,185,4,4,656,656) + '''  
                          ''' + change_section('air',115,115,7,11,727,740) + '''  
                          ''' + change_section('air',111,114,7,8,734,734) + '''  
                          ''' + change_section('air',112,114,9,9,734,734) + '''  
                          ''' + change_section('glass_pane',116,122,4,7,726,726) + '''  
                          ''' + change_section('green_shulker_box',111,114,7,8,726,726) + '''  
                          ''' + change_section('green_shulker_box',112,114,9,9,726,726) + ''' 
                          ''' + change_section('glass_pane',116,122,4,7,741,741) + '''   
                          ''' + change_section('air',63,63,4,4,766,771) + '''     
                          ''' + change_section('air',64,64,4,5,771,780) + '''
                          ''' + change_section('fence',64,64,4,4,771,780) + '''
                          ''' + change_section('air',64,64,4,5,757,766) + '''
                          ''' + change_section('fence',64,64,4,4,757,766) + '''
                          ''' + change_section('planks',140,143,9,9,760,769) + '''
                          ''' + change_section('planks',132,144,9,9,760,763) + '''
                          ''' + change_section('fence',132,144,10,10,760,760) + '''
                          ''' + change_section('fence',144,144,10,10,760,763) + '''
                          ''' + change_section('fence',132,132,10,10,760,763) + '''
                          ''' + change_section('grass',87,89,3,3,757,783) + '''
                          ''' + change_section('planks',87,89,4,5,756,756) + '''
                          ''' + change_section('planks',87,89,4,5,781,781) + '''
                          ''' + change_section('air',86,86,4,5,757,780) + '''
                          ''' + change_section('air',69,83,10,12,762,775) + '''
                          ''' + change_section('air',91,92,4,6,728,730) + '''
                          ''' + change_section('air',149,150,5,8,768,772) + '''
                          ''' + change_section('air',71,81,7,7,763,773) + '''
                          ''' + change_section('air',81,86,4,11,763,773) + '''
                          ''' + change_section('planks',82,88,3,3,763,773) + '''
                          ''' + change_section('sandstone',81,88,4,8,763,763) + '''
                          ''' + change_section('sandstone',81,88,4,8,774,774) + '''
                          ''' + change_section('sandstone',71,71,7,8,764,773) + '''
                          ''' + change_section('sandstone',72,80,7,8,763,763) + '''
                          ''' + change_section('sandstone',70,70,7,7,767,770) + '''
                          ''' + change_section('air',70,71,6,6,768,769) + '''
                          ''' + change_section('stone',69,88,9,9,761,776) + '''
                          ''' + change_section('glass_pane',88,88,4,8,764,773) + '''
                          ''' + change_section('sandstone',69,69,8,8,762,775) + '''
                          ''' + change_section('red_shulker_box',85,87,3,3,764,773) + '''
                          ''' + change_section('red_shulker_box',71,84,3,3,768,769) + '''
                          ''' + change_section('fence',143,143,4,4,652,656) + '''
                          ''' + change_section('fence',144,144,4,4,656,656) + '''
                          ''' + change_section('fence',144,144,4,4,657,657) + '''
                          ''' + change_section('air',139,140,4,6,648,648) + '''
                          ''' + change_section('oak_stairs',139,139,4,4,648,648) + '''
                          ''' + change_section('glass',138,138,5,6,648,648) + '''
                          ''' + change_section('air',138,141,4,6,728,728) + '''
                          ''' + change_section('stonebrick',138,141,3,3,727,727) + '''
                          ''' + change_section('stonebrick',137,150,3,3,728,738) + '''
                          ''' + change_section('grass',63,65,3,3,632,634) + '''
                          ''' + get_DrawItem(id_item2, sLoc_item2, mapLim, pRan, mapExt) + '''
                          ''' + get_DrawBlock(id_block2, sLoc_block2, mapLim, pRan, mapExt, problemList) + '''
                      </DrawingDecorator>
                      <ServerQuitFromTimeUp timeLimitMs="600000"/>
                      <ServerQuitWhenAnyAgentFinishes/>
                    </ServerHandlers>
                  </ServerSection>
                  <AgentSection mode="Creative">
                    <Name>ROBOT</Name>
                    <AgentStart>
                        ''' + get_Placement(pInc, mapExt, iPos) + '''
                    </AgentStart>
                    <AgentHandlers>
                      <ObservationFromFullStats/>
                      <ObservationFromNearbyEntities>
                          <Range name="item_obs" xrange="3" yrange="1" zrange="3"/>
                      </ObservationFromNearbyEntities> 
                      <ObservationFromGrid>
                          <Grid name="block_obs">
                              <min x="-3" y="-1" z="-3"/>
                              <max x="3" y="1" z="3"/>
                          </Grid>
                      </ObservationFromGrid>                      
                      <ContinuousMovementCommands turnSpeedDegs="480"/>
                      <AgentQuitFromReachingPosition>
                          <Marker tolerance="5" x="0"   y="4" z="620"/>      
                          <Marker tolerance="5" x="0"   y="4" z="710"/>   
                          <Marker tolerance="5" x="50"  y="4" z="595"/>                      
                          <Marker tolerance="5" x="50"  y="4" z="790"/>    
                          <Marker tolerance="5" x="120" y="4" z="595"/>      
                          <Marker tolerance="5" x="190" y="4" z="595"/>      
                          <Marker tolerance="5" x="200" y="4" z="790"/>   
                          <Marker tolerance="5" x="220" y="4" z="635"/>                      
                          <Marker tolerance="5" x="220" y="4" z="710"/>  
                      </AgentQuitFromReachingPosition>
                    </AgentHandlers>
                  </AgentSection>
                </Mission>'''
    return ans   

# astar path planning
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


def makePath(x0, z0, xF, zF, allow_diagonal_movement = False):
    
    #map coordinates
    mapLim = pd.read_excel('utils/map.xlsx')
    mapLim = mapLim.drop(columns = ['set'])
    mapLim = mapLim.set_index('loc')
    xMin = mapLim.iloc[-1, 0]
    xMax = mapLim.iloc[-1, 1]
    zMin = mapLim.iloc[-1, 2]
    zMax = mapLim.iloc[-1, 3]
    del mapLim
    
    #map discretization  
    maze = pd.read_excel('utils/maze.xlsx')
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
    return [],[end]
    
#function (run plan per trial conditions)    
def run(iEnv, iPlan):
        
    #READ PACK
    xObserve    = iEnv[0]      #observations within any given location
    xNoFind     = iEnv[1]      #failed searches
    xPossess    = iEnv[2]      #items in possession (not used)
    iSet        = iEnv[3]      #location set
    iRan        = iEnv[4]      #amount of randomness (not used)
    iInc        = iEnv[5]      #seed
    iStats      = iEnv[6]      #stats from current trial (plans attempted, time, dist, etc.)
    iResult     = iEnv[7]      #result from current plan (0 if failed, 1 if successful)
    iPos        = iEnv[8]      #current position
    iTrial      = iEnv[9]      #trial number
    
    #READ PLAN
    total_item = list(iPlan)
    total_loc = list(iPlan.index)
    plan_loc = []
    plan_item = []
    for i in range(len(total_loc)):
        if len(total_item[i]) > 0: 
            plan_loc.append(total_loc[i])
            plan_item.append(total_item[i])
           
    #READ MAP SPECS
    missionXML = get_missionXML(iSet, iRan, iInc, iPos, iTrial) 
    my_mission = MP.MissionSpec(missionXML, True)
    my_mission_record = MP.MissionRecordSpec()
    
    #DEFAULT MALMO OBJECTS
    agent_host = MP.AgentHost()
    try:
        agent_host.parse( sys.argv )
    except RuntimeError as e:
        print('ERROR:',e)
        print(agent_host.getUsage())
        exit(1)
    if agent_host.receivedArgument("help"):
        print(agent_host.getUsage())
        exit(0)
    
    my_client_pool = MP.ClientPool()
    for i in range(10000, 11000): 
        my_client_pool.add(MP.ClientInfo("127.0.0.1", i))
    
    #TRY TO CONNECT TO SERVER
    max_retries = 12
    for retry in range(max_retries):
        try:
            agent_host.startMission(my_mission, my_client_pool, my_mission_record, 0, "")
            break
        except RuntimeError as e:
            if retry == max_retries - 1:
                print("Error starting mission:",e)
                exit(1)
            else:
                time.sleep(2)
    
    #LOAD MAP
    world_state = agent_host.getWorldState()
    while not world_state.has_mission_begun:
        #print(".", end="")
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
        for error in world_state.errors:
            print("Error:",error.text)
    
    def dYaw(iPos,xT,zT):
        tol = 1    
        x0 = round(iPos[0], 1)
        z0 = round(iPos[1], 1)
        yaw0 = round(iPos[2], 1)
        
        dx = xT - x0
        dz = zT - z0
        
        if dx<0 and dz>0:   yaw = np.arctan(-dx/dz) / np.pi * 180
        elif dx<0 and dz<0: yaw = np.arctan(dx/dz) / np.pi * 180 + 90
        elif dx>0 and dz>0: yaw = np.arctan(-dx/dz) / np.pi * 180 + 360
        elif dx>0 and dz<0: yaw = np.arctan(-dx/dz) / np.pi * 180 + 180
        elif dx==0 and dz>0: yaw = 0.0
        elif dx==0 and dz<0: yaw = 180.0
        elif dx>0 and dz==0: yaw = 270.0
        elif dx<0 and dz==0: yaw = 90.0
        else: yaw = yaw0
        
        yaw0 = np.round(yaw0,1)
        yaw = np.round(yaw,1)
        if yaw0 == -0.0: yaw0 = 0.0
        if yaw == -0.0: yaw = 0.0 
        
        ans = yaw - yaw0 
        ans = round(ans, 1)
        
        if ans > 180 : ans = ans - 360
        if ans < -180: ans = ans + 360
        if np.absolute(ans) < tol: ans = 0.0
        #print(ans)
        return ans
    
    def dPos(iPos,xT,zT):
        tol = 1
        x0 = round(iPos[0], 1)
        z0 = round(iPos[1], 1)
        xT = round(xT, 1)
        zT = round(zT, 1)
        dx = xT - x0
        dz = zT - z0  
        ans = np.sqrt(dx**2+dz**2) 
        ans = round(ans,1)
        if np.absolute(ans) < tol: ans = 0.0
        return ans
    
    def rotate(deg):
        speed = 480
        agent_host.sendCommand("jump 0")
        if deg > 0: 
            agent_host.sendCommand("turn 1")
            time.sleep(deg/speed)
        elif deg < 0:
            agent_host.sendCommand("turn -1")
            time.sleep(-deg/speed) 
        agent_host.sendCommand("turn 0")
        return
        
    def translate(dist):
        agent_host.sendCommand("jump 0")
        est_speed = 4.5 
        if dist > 0:
            agent_host.sendCommand("move 1")
            time.sleep(dist/est_speed)
        if dist < 0:
            agent_host.sendCommand("move -1")
            time.sleep(-dist/est_speed)
        agent_host.sendCommand("move 0")
        return
    
    def slide(dist):
        agent_host.sendCommand("jump 0")
        est_speed = 4.5 
        if dist > 0: 
            agent_host.sendCommand("strafe 1")
            time.sleep(dist/est_speed)
        elif dist < 0:
            agent_host.sendCommand("strafe -1")
            time.sleep(-dist/est_speed) 
        agent_host.sendCommand("strafe 0")
        return
        
    def rest(t):
        agent_host.sendCommand("jump 0")
        agent_host.sendCommand("move 0")
        time.sleep(t)   
        return
    
    def juke():
        a = random.choice([-1,1])
        rotate(a*135) 
        translate(1)
        rotate(a*90) 
        translate(1)
        rotate(a*-90)
        translate(1)
        rotate(a*-90) 
        return
        
    def move(iPos, xT, zT, state):     
        if not state.is_mission_running: return state
        else: 
            deg = dYaw(iPos,xT,zT)
            dist = dPos(iPos,xT,zT)
            rotate(deg)
            translate(dist) 
            return   
    
    def predictor(iPos, path, maze_info, state):    
        if not state.is_mission_running: return [(iPos[0],iPos[1]), state]
        else:
            dx = maze_info[0]
            dz = maze_info[1]
            xmin = maze_info[2]
            zmin = maze_info[3]    
            x0 = round((path[0][1] +0.5)*dx + xmin, 0)
            z0 = round((path[0][0] +0.5)*dz + zmin, 0)            
            xT = round((path[1][1] +0.5)*dx + xmin, 0)
            zT = round((path[1][0] +0.5)*dz + zmin, 0)
            iPos = (x0, z0, iPos[2])
            move(iPos, xT, zT, state)
            xzT = (xT, zT)
            xF = round((path[-1][1] +0.5)*dx + xmin, 0)
            zF = round((path[-1][0] +0.5)*dz + zmin, 0) 
            xzF = (xF, zF)
            return xzT, xzF
    
    def corrector(iPos, xzT, state):        
        if not state.is_mission_running: return False
        else:
            tol = 2
            xT  = np.round(xzT[0],0)
            zT  = np.round(xzT[1],0)
            errX = np.round(iPos[0],1) - xT
            errZ = np.round(iPos[1],1) - zT
            delta = dYaw(iPos,xT,zT)        
            if np.absolute(errX)<tol and np.absolute(errZ)<tol: fCorrect = False    #do not correct (small error)
            elif np.absolute(delta)>90: fCorrect = False                            #do not correct (unreasonable angle)
            else: 
                move(iPos, xT, zT, state)                
                fCorrect = True
        return fCorrect
     
    def followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state): 
        #start timer
        searchTime  = 0.0
        maxTime     = 60.0     
        initTime    = np.round(time.time(), 1)
        while len(path)>1:                                                          #loop until path is complete
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state)
            if not state.is_mission_running: break
            xzT, xzF = predictor(iPos, path, maze_info, state)                           #move to target node
            path = path[1:]                                                         #remove first node                               
            fCorrect = True                                                         #correction flag
            nCorrect = 0                                                            #correction count
            while fCorrect:                                                         #loop until at target node
                fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state)
                if not state.is_mission_running: break
                fCorrect = corrector(iPos, xzT, state)                              #move to target node
                nCorrect +=1
                if nCorrect > 2:                                                    #if too many corrections (obstacle)
                    fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state)
                    if not state.is_mission_running: break                    
                    juke()                                                          #juke
                    fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state)
                    if not state.is_mission_running: break
                    try:    
                        path = makePath(iPos[0], iPos[1], xzF[0], xzF[1])[0]        #make new path
                    except:
                        newPos = nearest_pos(iPos)
                        path = makePath(newPos[0], newPos[1], xzF[0], xzF[1])[0] 
                    break 
            if not state.is_mission_running: break
            
            #check timer
            currTime = np.round(time.time(), 1)
            searchTime = np.round((currTime - initTime),1)
            if searchTime > maxTime: break                          #end plan (fail)        
    
        return fSearch, iPos, iStats, xObserve, xNoFind, state
    
    def get_route(iPos, iPlan, mapLim, state):
        if not state.is_mission_running: return [[], []]
        else:
                   
            #filter locations by set
            route = ['church', 'restaurant', 'store', 'hospital', 'library', 'museum',
                         'farm', 'landfill', 'lake', 'bedroom', 'kitchen', 'gas station', 
                         'house', 'school', 'post office', 'office',  'garden', 'brewery']
            possLoc =  list(iPlan.index.values)
            route = [x for x in route if x in possLoc]
            
            #re-order locations (CW around the map)
            iPlan = iPlan.loc[route]
            sItemList = list(iPlan)
            
            #filter locations (items > 0)
            ix = [i for i, x in enumerate(sItemList) if len(x) >0] 
            sLocList = [route[i] for i in ix] 
            sItemList = [sItemList[i] for i in ix] 
            
            #convert items to "block" names
            df_item = pd.read_excel('utils/truth.xlsx', keep_default_na=False)
            id_item = df_item.id_item.tolist()
            id_block = df_item.id_block.tolist()
            sTempList = []
            for sItem in sItemList:
                sTemp = []
                for x in sItem:
                    ix = id_item.index(x)
                    temp = id_block[ix]
                    sTemp.append(temp)
                sTempList.append(sTemp)
            sItemList = sTempList
                    
            #make the closest location first
            min_d = 10000
            ix = 0
            for i in range(len(sLocList)):
                sLoc = sLocList[i]
                xT = mapLim.loc[sLoc,'xent']                            
                zT = mapLim.loc[sLoc,'zent']
                temp_d = dPos(iPos,xT,zT)
                if temp_d < min_d:
                    min_d = temp_d
                    ix = i
            sLocList = sLocList[ix:] + sLocList[:ix]    
            sItemList = sItemList[ix:] + sItemList[:ix]   
            return sLocList, sItemList
    
    def get_loc(iPos, mapLim):
        x = round(iPos[0], 1)
        z = round(iPos[1], 1)        
        ans = 'N/A'
        tol = 3 #probably should be the same as nearby "range" in xml
        for i in range(len(mapLim)):
            sLoc = mapLim.index[i]
            xmin = mapLim.loc[sLoc,'xmin'] - tol
            xmax = mapLim.loc[sLoc,'xmax'] + tol
            zmin = mapLim.loc[sLoc,'zmin'] - tol
            zmax = mapLim.loc[sLoc,'zmax'] + tol
            if x >= xmin and x<=xmax and z >= zmin and z<=zmax: ans = sLoc   
        return ans
    
    def process(fSearch, xObserve, xNoFind, obsEnt, obsLoc, sItem, sLoc):
        #list of items
        df = pd.read_excel('utils/truth.xlsx')
        id_block = list(df.id_block)
        
        if obsLoc != 'N/A':
            #record observations at item indicies
            for ent in obsEnt:
                ix = [i for i in range(len(id_block)) if id_block[i] == ent]
                for ixx in ix:
                    xObserve[ixx].append(obsLoc)                
                    xObserve[ixx] = list(set(xObserve[ixx]))
              
            #record failed searches at item indicies
            if obsLoc == sLoc:
                fSearch = False                           
                for item in sItem:
                    ix = [i for i in range(len(id_block)) if id_block[i] == item]
                    for ixx in ix:
                        if not sLoc in xObserve[ixx]: 
                            xNoFind[ixx].append(obsLoc)
                            xNoFind[ixx] = list(set(xNoFind[ixx]))  
                            fSearch = True
                            
        return fSearch, xObserve, xNoFind
    
    def observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state):
        new_state = agent_host.getWorldState()
        if new_state.is_mission_running:
            if new_state.number_of_observations_since_last_state > 0: state = new_state
            msg = state.observations[-1].text
            obs = json.loads(msg)   

            #update position (iPos)
            x = np.round(obs.get(u'XPos', 0))
            z = np.round(obs.get(u'ZPos', 0))
            yaw = np.round(obs.get(u'Yaw', 0)) 
            iPos = (x, z, yaw)
            
            #update stats (iStats)
            if fSearch:
                t = obs.get(u'TimeAlive', 0) 
                d = obs.get(u'DistanceTravelled', 0) 
                iStats = (t,d)
            
            #update item feedback (xObserve, xNoFind)
            if fSearch:
                block_obs = obs.get(u'block_obs', 0)
                tempList = obs.get(u'item_obs', 0)  
                item_obs = []
                if len(tempList)>1 :
                    for i in range(1,len(tempList)):
                        temp = tempList[i]['name']
                        item_obs.append(temp)
                obsEnt = list(set(item_obs + block_obs))
                obsLoc = get_loc(iPos, mapLim)
                fSearch, xObserve, xNoFind = process(fSearch, xObserve, xNoFind, obsEnt, obsLoc, sItem, sLoc)    
        else: state = new_state
        return fSearch, iPos, iStats, xObserve, xNoFind, state
    
    def get_endpoint(iPos):
        options = [(0,   620), (0,   710), (50, 595), (50,   790), (120,   595), (190,   595), (200,   790), (220, 635), (220,   710)]
        min_d = 10000
        for i in range(len(options)):
            xT = options[i][0]
            zT = options[i][1]
            temp_d = dPos(iPos,xT,zT)
            if temp_d < min_d: 
                min_d = temp_d
                ix = i
        ans = options[ix]
        return ans
    
    #map     
    mapLim = pd.read_excel('utils/map.xlsx')
    mapLim = mapLim.drop(columns = ['set'])
    mapLim = mapLim.set_index('loc')
    mapLim = mapLim[:-1]
    
    #maze
    maze_info = [4.72, 4.2, -7, 587]
    
    max_search_time = 60
    
    #loading test
    state = agent_host.getWorldState()
    while not state.has_mission_begun:
        state = agent_host.getWorldState()
        for error in state.errors:
            print("Error:",error.text)
    
    #running test    
    fSearch = True
    while state.is_mission_running:  
        while len(state.observations) == 0: 
                rest(0.1)
                state = agent_host.getWorldState()        
        fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, [], [], state)
        sLocList, sItemList = get_route(iPos, iPlan, mapLim, state) 
        if not state.is_mission_running: break
        print('locations:', sLocList)
        print('items:', sItemList)
        
        for i in range(len(sLocList)):  
            sLoc    = sLocList[i]
            sItem   = sItemList[i]       
            fSearch = True
            offset  = 5
            
            #location entrance
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            print('target: {}'.format(sLoc))                
            xT = mapLim.loc[sLoc,'xent']                            
            zT = mapLim.loc[sLoc,'zent']  
            path = makePath(iPos[0], iPos[1], xT, zT)[0] 
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            fSearch, iPos, iStats, xObserve, xNoFind, state = followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state)            
            if not state.is_mission_running: break
            
            if not fSearch: continue
            
            #start timer
            searchTime = 0.00
            search_t0 = np.round(time.time(), 2)
        
            #location C
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            print('searching C')                
            xT = mapLim.loc[sLoc,'xmid']                            
            zT = mapLim.loc[sLoc,'zmid']    
            path = makePath(iPos[0], iPos[1], xT, zT)[0] 
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            fSearch, iPos, iStats, xObserve, xNoFind, state = followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state)            
            if not state.is_mission_running: break
        
            if not fSearch: continue #end search for current item
            
            #check timer
            search_t1 = np.round(time.time(), 2)
            searchTime = np.round((search_t1 - search_t0),2)
            if searchTime > max_search_time: break           #end plan (fail)

            #location N
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            print('searching N')                
            xT = mapLim.loc[sLoc,'xmid']                            
            zT = mapLim.loc[sLoc,'zmin']  + offset   
            path = makePath(iPos[0], iPos[1], xT, zT)[0] 
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            fSearch, iPos, iStats, xObserve, xNoFind, state = followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state)            
            if not state.is_mission_running: break
            
            if not fSearch: continue #end search for current item
            
            #check timer
            search_t1 = np.round(time.time(), 2)
            searchTime = np.round((search_t1 - search_t0),2)
            if searchTime > max_search_time: break           #end plan (fail)

            #location E
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            print('searching E')                
            xT = mapLim.loc[sLoc,'xmax']  - offset                          
            zT = mapLim.loc[sLoc,'zmid']   
            path = makePath(iPos[0], iPos[1], xT, zT)[0] 
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            fSearch, iPos, iStats, xObserve, xNoFind, state = followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state)            
            if not state.is_mission_running: break
            
            if not fSearch: continue #end search for current item
            
            #check timer
            search_t1 = np.round(time.time(), 2)
            searchTime = np.round((search_t1 - search_t0),2)
            if searchTime > max_search_time: break           #end plan (fail)

            #location S
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            print('searching S')                
            xT = mapLim.loc[sLoc,'xmid']                            
            zT = mapLim.loc[sLoc,'zmax']  - offset   
            path = makePath(iPos[0], iPos[1], xT, zT)[0] 
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            fSearch, iPos, iStats, xObserve, xNoFind, state = followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state)            
            if not state.is_mission_running: break
            
            if not fSearch: continue #end search for current item
            
            #check timer
            search_t1 = np.round(time.time(), 2)
            searchTime = np.round((search_t1 - search_t0),2)
            if searchTime > max_search_time: break           #end plan (fail)

            #location W
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            print('searching W')                
            xT = mapLim.loc[sLoc,'xmin'] + offset                         
            zT = mapLim.loc[sLoc,'zmid'] 
            path = makePath(iPos[0], iPos[1], xT, zT)[0] 
            fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) #location entrance
            if not state.is_mission_running: break
            fSearch, iPos, iStats, xObserve, xNoFind, state = followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state)            
            if not state.is_mission_running: break
            
            if not fSearch: continue    #end search for current item   
            if fSearch: break           #end plan 
            
            #check timer
            search_t1 = np.round(time.time(), 2)
            searchTime = np.round((search_t1 - search_t0),2)
            if searchTime > max_search_time: break           #end plan (fail)
            print(search_t0, search_t1, searchTime)

        #update result
        if fSearch:                     #fail (still searching for item)
            iResult = -1
            print('PLAN FAIL!')
        if not fSearch:                 #success (all items found in search)
            iResult = 1             
            print('PLAN SUCCESS!')
        
        #leave map
        fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) 
        iPos_end = iPos
        if not state.is_mission_running: break
        print('target: checkout')                
        xT, zT = get_endpoint(iPos)
        path = makePath(iPos[0], iPos[1], xT, zT)[0] 
        fSearch, iPos, iStats, xObserve, xNoFind, state = observe(fSearch, iPos, iStats, xObserve, xNoFind, mapLim, sItem, sLoc, state) 
        if not state.is_mission_running: break
        fSearch, iPos, iStats, xObserve, xNoFind, state = followPath(fSearch, iPos, iStats, path, mapLim, xObserve, xNoFind, sItem, sLoc, state)            
        iPos = iPos_end
        if not state.is_mission_running: break
            
    #update pack    
    iEnv = [xObserve, xNoFind, xPossess, iSet, iRan, iInc, iStats, iResult, iPos, iTrial]

    return iEnv

# =============================================================================








