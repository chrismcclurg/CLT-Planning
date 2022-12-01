# =============================================================================
# Construal Level Theory for Agent-based Planning
#
# C McClurg, AR Wagner, S Rajtmajer
# =============================================================================
# CONSTRUAL-SPECIFIC FILE
# Purpose: collection of functions to define execution of construal process
# =============================================================================

import pandas as pd
import itertools
import numpy as np
import itertools as it
import random
from scipy.spatial import distance

#subfunction of many functions (return distance between vectors)
def get_vecDistance(x1, x2, distance_metric):
    if distance_metric   == 'euclidean': 
        return np.linalg.norm(x1-x2)
    elif distance_metric == 'euclidean_squared': 
        return np.square(np.linalg.norm(x1-x2))
    elif distance_metric == 'cosine': 
        return distance.cosine(x1,x2)

#function (return matrix representation of items and constituents)    
def get_recipeMatrix(id_item, df_recipe):
    nitem = len(df_recipe)
    ans = np.zeros((nitem, nitem))
    id = [x for x in df_recipe.index.values]
    cols = ['c0', 'c1', 'c2', 'c3', 'c4']
    counts = ['n0', 'n1', 'n2', 'n3', 'n4']
    for i in range(len(cols)):
        col = cols[i]
        count = counts[i]
        val = [x for x in df_recipe[col].values]
        countVal = [x for x in df_recipe[count].values]
        for j in range(nitem):
            tempID = id[j]
            tempVal = val[j]
            tempCount = countVal[j]
            if tempVal in id_item: 
                ix = id_item.index(tempID)
                iy = id_item.index(tempVal)
                ans[ix,iy] = tempCount
    return ans

#function (return matrix representation of items and features)    
def get_featureMatrix(id_uri, df_embed): 
    ans = [[] for x in range(len(id_uri))]
    for i in range(len(id_uri)):
        uri = id_uri[i]
        vec = df_embed.loc[[uri]].values
        ans[i] = vec[0]
    ans = np.matrix(ans)
    return ans    

#function (return matrix representation of distance between items)
def get_cosineMatrix(matFeature):
    nitem = len(matFeature)
    ans = np.zeros((nitem, nitem))
    for i in range(nitem):
        for j in range(nitem):
            if i < j:
                a = matFeature[i]
                b = matFeature[j]
                c = get_vecDistance(a,b, 'cosine')
                ans[i,j] = np.round(c,3)
            else:
                ans[i,j] = None
    return ans        

#function (return matrix representation of items and believed locations)   
def get_locationMatrix(yBelief, pLoc):
    nitem = len(yBelief)
    nloc = len(pLoc)
    ans = np.zeros((nitem, nloc))
    for i in range(nitem):
        ntemp = len(yBelief[i])
        for j in range(ntemp):
            temploc = yBelief[i][j]
            iloc = pLoc.index(temploc)
            ans[i][iloc] = 1
    return ans         

#subfunction of abstract_similar (return matrix of distances with max distance filter applied) 
def get_filterMatrix(matCosine, maxDist):
    ans = np.copy(matCosine)
    for i in range(len(ans)):
        for j in range(len(ans)):
            temp = ans[i,j]
            if i == j: ans[i,j] = 1
            elif np.isnan(temp) == True: ans[i,j] = 0
            elif temp > maxDist: ans[i,j] = 0
            else: ans[i,j] = 1
    return ans

#subfunction of abstract_cluster (returns centroids per all classes)
def get_centroids(matLocation, matFeature, maxDist): 
    data = matFeature.copy()
    nloc = len(matLocation[0])
    totalCentroids = [[] for x in range(nloc)]
    for k in range(nloc):
        ix = np.argwhere(matLocation[:,k] == 1)
        ix = [int(x) for x in ix]
        x_train = [data[i] for i in ix]
        if len(x_train)>0:
            centroids = [[0 for x in range(len(x_train[0]))]]
            centroids[0] = x_train[0]
            total_num = [1]
            for i in range(1,len(x_train)):
                distances=[]
                indices = []
                for j in range(0,len(centroids)):
                    d = get_vecDistance(x_train[i], centroids[j], 'cosine')
                    if d < maxDist:
                        distances.append(d)
                        indices.append(j)
                if len(distances)==0:
                    centroids.append(x_train[i])
                    total_num.append(1)
                else:
                    min_d = np.argmin(distances)
                    centroids[indices[min_d]] = np.add(np.multiply(total_num[indices[min_d]],centroids[indices[min_d]]),x_train[i])
                    total_num[indices[min_d]]+=1
                    centroids[indices[min_d]] = np.divide(centroids[indices[min_d]],(total_num[indices[min_d]]))
        else:
                centroids = []
        
        totalCentroids[k] = centroids
    
    return totalCentroids
 
#function (return predicted locations based on similar items)   
def abstract_similar(yBelief, matFeature, maxDist):
    len_net = len(yBelief)
    old_net = [x[:] for x in yBelief] #deepcopy
    ans  = [[] for x in range(len_net)]
    matCosine   = get_cosineMatrix(matFeature)
    filt = get_filterMatrix(matCosine, maxDist)   
    for i in range(len_net):
        for j in range(len_net):  
            if filt[i,j]==1: 
                ans[i].extend(old_net[j])
    ans = [list(set(x)) for x in ans]
    return ans

#function (return predicted locations based on similar centroids)   
def abstract_cluster(yBelief, matFeature, maxDist, pLoc):    
    matLocation = get_locationMatrix(yBelief, pLoc)
    centroids = get_centroids(matLocation, matFeature, maxDist)
    nitem = len(yBelief)
    nloc = len(centroids)
    ans  = [[] for x in range(nitem)]
    data = matFeature.copy()    
    for i in range(nitem):
        for j in range(nloc):
            ncent = len(centroids[j])
            for k in range(ncent):  
                d = get_vecDistance(data[i], centroids[j][k], 'cosine')
                if d<=maxDist:
                    loc = pLoc[j]
                    ans[i].append(loc)
    ans = [list(set(x)) for x in ans]
    return ans

#function (return predicted locations based on KG-related items)   
def abstract_scaling(pMod, pSrc, iSet):
    abstract = pd.read_excel(pSrc, keep_default_na=False)
    col = 'set' + str(iSet) + '_lev' + str(pMod)
    ans = abstract[col].apply(eval)
    # if pMod == 0:
    #     ans = abstract.L0.apply(eval)
    #     ans = list(ans)
    # elif pMod == 1:
    #     ans = abstract.L1.apply(eval)
    #     ans = list(ans)
    # elif pMod == 2:
    #     ans = abstract.L2.apply(eval)
    #     ans = list(ans)
    return ans

#subfunction of plan (return item names based on their index)
def map_items(a, id_item):
    if type(a) is int: ans = a
    elif len(a)>0: ans = [id_item[x] for x in a]
    else: ans = []
    return ans

#subfunction of plan (return location names based on their index)
def map_location(a, pLoc):
    ans = []
    for i in range(len(a)):
        if type(a[i]) is int: ans.append(pLoc[a[i]])
        else: ans.append(a[i])
    return ans

#subfunction of plan (return locations from items using matrix math)
def L(mLocTest, xTest):
    ix = np.argwhere(xTest == 1)
    y = [] 
    for i in range(len(ix)):
        xVec = np.zeros(len(xTest))
        xVec[ix[i]] = 1
        temp = np.matmul(mLocTest, xVec)
        iy = np.argwhere(temp == 1)
        y.append(iy)
    y = [list(x) for x in y]
    try:
        y = [np.concatenate(x) for x in y]
        a = list(it.product(*y))
        ans = np.zeros((len(mLocTest), len(a)))  
        ans = pd.DataFrame(ans)
        
        for i in range(len(ans.columns)):
            ans[i] = np.empty((len(ans), 0)).tolist()
       
        for i in range(len(a)):
            for j in range(len(a[0])):
                id = ix[j][0]
                loc = a[i][j]
                ans.iloc[loc, i].append(id)
                
        for i in range(len(ans.columns)):
            ans[i] = ans[i].apply(lambda x: np.array(x))
    except:
        #dataframe is empty (not enough location data)
        ans = [[] for x in range(len(mLocTest))]
        ans = pd.DataFrame(zip(ans))
    
    return ans

#subfunction of plan (return recipe items from items using matrix math)
def R(mRecTest, xTest):
    
    #use the matrix to find the ingredients
    ans = np.matmul(mRecTest, xTest)
    ans = [1 if x>0 else 0 for x in ans]
    ans = np.array(ans)  
    
    #get the indicies of the new vector
    ix = np.argwhere(ans >0)
    ix = [x[0] for x in ix]
    
    #as long as something is returned, check out process 
    if len(ix)>0:
        checkIn = np.argwhere(xTest >0)     #initial items check in
        checkIn = [x[0] for x in checkIn]
        leftOver = checkIn.copy()           #initial items not accounted for 
        checkOut = []                       #initial items accounted for
        iy = np.argwhere(mRecTest[ix] >0)
        iy = [x[1] for x in iy]
        for i in range(len(iy)):
            if iy[i] in leftOver:
                leftOver.remove(iy[i])
                checkOut.append(iy[i])
        if len(leftOver)>0:                 #if there are items unaccounted for
            for i in range(len(leftOver)):
                temp_ix = leftOver[i]
                ans[temp_ix] = 1            #add the unaccounted for items to the plan
                
    else: ans = np.zeros(len(xTest))        
            
    return ans

#subfunction of plan (return whether plan function should continue onto next level, i.e. L(items), L(R(items))...)
def check_continue(pX):
    ans = True
    nX = len(pX.columns)
    if nX == 1:
        temp = [x for x in pX[0] if len(x)>0]
        nTemp = len(temp)
        if nTemp == 0: ans = False
    return ans

#subfunction of plan (return location count in a given plan, used for scoring)
def count_loc(pX, pLoc):   
    nX = len(pX.columns)
    nLoc = len(pLoc)-1
    ans = []
    for i in range(nX):
        n = 0
        x = pX[i][0:nLoc]
        for tempx in x: 
            if len(tempx)>0: n+=1
        ans.append(n)
    return ans

#subfunction of plan (return plan minus any items currently possessed)
def check_possession(pX, xPossess, pLoc):  
    ixPoss = np.argwhere(np.array(xPossess) >0)
    ixPoss = [x[0] for x in ixPoss]
    ans = pX.copy(deep = True)
    nX = len(ans.columns)
    nLoc = len(pLoc)-1
    for i in range(nX):
        for j in range(nLoc):
            temp = list(ans[i][j])
            for k in range(len(temp)):
                if temp[k] in ixPoss: 
                    ans[i][j] = ans[i][j][ans[i][j] != temp[k]]                    
                    ans[i][nLoc] = np.append(ans[i][nLoc], temp[k])
        ans[i][nLoc] = [int(x) for x in ans[i][nLoc]]
    return ans

#subfunction of plan (return scores of plan options)
def score(pen_make, pen_loc, pen_mod, pLoc):
    if pen_loc == 0: ans = 1.000
    else: 
        ans = (1-pen_make/4) * (1-pen_loc/(18)) * (1-pen_mod/3)
        ans = round(ans,3)
    return ans

#function (return plans given item info and predictions)
def plan(yBelief, xPossess, mRecipe, id_item, pLoc, pTask, pMod):
    
    mLocation = get_locationMatrix(yBelief, pLoc)
    
    mLocTest = mLocation.copy().transpose()
    mRecTest = mRecipe.copy().transpose()  
    
    xTest = np.zeros(len(mRecTest))
    ix = id_item.index(pTask) 
    xTest[ix] = 1
    
    p0  = L(mLocTest, xTest)
    p0  = check_possession(p0, xPossess, pLoc)
    con = check_continue(p0)
    n1 = n2 = n3 = n4 = 0 
    n0 = len(p0.columns)
    
    # 'make' penalty per each plan
    temp = [0 for x in range(n0)]
    temp = pd.Series(temp, index = p0.columns)
    p0 = p0.append(temp, ignore_index=True)
    
    # 'loc' penalty per each plan
    temp = count_loc(p0, pLoc)
    temp = pd.Series(temp, index = p0.columns)
    p0 = p0.append(temp, ignore_index=True) 
    
    
    if con: 
        p1 = L(mLocTest, R(mRecTest, xTest))
        p1  = check_possession(p1, xPossess, pLoc)
        con = check_continue(p1) 
        
        # 'make' penalty per each plan
        n1 = len(p1.columns)      
        temp = [0 for x in range(n1)]
        temp = pd.Series(temp, index = p1.columns)
        p1 = p1.append(temp, ignore_index=True)
        
        # 'loc' penalty per each plan
        temp = count_loc(p1, pLoc)
        temp = pd.Series(temp, index = p1.columns)
        p1 = p1.append(temp, ignore_index=True)   
        print('p1', n1)

    if con: 
        p2 = L(mLocTest, R(mRecTest, R(mRecTest, xTest)))
        p2  = check_possession(p2, xPossess, pLoc)
        con = check_continue(p2) 
        
        # 'make' penalty per each plan
        n2 = len(p2.columns)      
        temp = [0 for x in range(n2)]
        temp = pd.Series(temp, index = p2.columns)
        p2 = p2.append(temp, ignore_index=True)
        
        # 'loc' penalty per each plan
        temp = count_loc(p2, pLoc)
        temp = pd.Series(temp, index = p2.columns)
        p2 = p2.append(temp, ignore_index=True)  
        print('p2', n2)

        
    if con: 
        p3 = L(mLocTest, R(mRecTest, R(mRecTest, R(mRecTest, xTest))))
        p3  = check_possession(p3, xPossess, pLoc)       
        con = check_continue(p3) 
        
        # 'make' penalty per each plan
        n3 = len(p3.columns)      
        temp = [0 for x in range(n3)]
        temp = pd.Series(temp, index = p3.columns)
        p3 = p3.append(temp, ignore_index=True)
        
        # 'loc' penalty per each plan
        temp = count_loc(p3, pLoc)
        temp = pd.Series(temp, index = p3.columns)
        p3 = p3.append(temp, ignore_index=True) 
        print('p3', n3)

        
    if con: 
        p4 = L(mLocTest, R(mRecTest, R(mRecTest, R(mRecTest, R(mRecTest, xTest)))))
        p4 = check_possession(p4, xPossess, pLoc)
        
        # 'make' penalty per each plan
        n4 = len(p4.columns)      
        temp = [0 for x in range(n4)]
        temp = pd.Series(temp, index = p4.columns)
        p4 = p4.append(temp, ignore_index=True)
        
        # 'loc' penalty per each plan
        temp = count_loc(p4, pLoc)
        temp = pd.Series(temp, index = p4.columns)
        p4 = p4.append(temp, ignore_index=True)    
        print('p4', n4)

    n = len(p0.columns)
    p = p0.copy()
        
    if n1 > 0: 
        p1 = p1.rename(columns=lambda x: x+n)
        n += n1
        p = p.join(p1)
    if n2 > 0: 
        p2 = p2.rename(columns=lambda x: x+n)
        n += n2
        p = p.join(p2)
    if n3 > 0: 
        p3 = p3.rename(columns=lambda x: x+n)
        n += n3
        p = p.join(p3)
    if n4 > 0: 
        p4 = p4.rename(columns=lambda x: x+n)
        n += n4
        p = p.join(p4)
    
    for i in range(len(p.columns)):
        p[i] = p[i].apply(lambda x: map_items(x, id_item))
        
        
    # 'mod' penalty per each plan
    temp = [pMod for x in range(n)]
    temp = pd.Series(temp, index = p.columns)
    p = p.append(temp, ignore_index=True)
    
    #total score per each plan
    nLoc = len(pLoc)
    pT = p.T
    pT[nLoc+3] = pT.apply(lambda x: score(x[nLoc], x[nLoc+1], x[nLoc+2], pLoc), axis=1)
    p = pT.T
      
    old_index = p.index.values.tolist()
    old_index[nLoc]   = 'penalty_make'
    old_index[nLoc+1] = 'penalty_loc'
    old_index[nLoc+2] = 'penalty_mod'
    old_index[nLoc+3] = 'score'

    new_index = map_location(old_index, pLoc)
    new_index = pd.Series(new_index)
    p = p.set_index(new_index)
    p = p.rename(columns=lambda x: 'P'+ str(x))
    p = p.iloc[:, :-1]
    
    return p    

#function (return sparse initial item locations)
def guess(id_item, id_uri, pLoc, pSrc, iSet):
    abstract = pd.read_excel(pSrc, keep_default_na=False)
    col = 'set' + str(iSet) + '_lev0'
    ans = abstract[col].apply(eval)
    ans = list(ans)
    return ans

#function (return item locations per abstraction type and level)
def reason(yBelief, matFeature, pLoc, pAbs, pSrc, pMod, iSet):
    if pAbs == 0:   ans = abstract_scaling(pMod, pSrc, iSet)  
    elif pAbs == 1: ans = abstract_similar(yBelief, matFeature, pMod)
    elif pAbs == 2: ans = abstract_cluster(yBelief, matFeature, pMod, pLoc)
    elif pAbs is None:  ans = yBelief
    else:
        ans = []
        print('please input appropriate abstraction type.')
    return ans    

#function (return unique plans, sorted by scores)
def review(yTemp, yPlan, pLoc, pInc, pTop = None):
    if len(yTemp.columns)>0:
        ix = yTemp.index.values.tolist()
        ix = pd.Series(ix)
        if yPlan is None: 
            yPlan = yTemp.copy(deep=True)
            planP = list(yPlan.T.values)
            planP = [list(x) for x in planP]
                
        else:
            nLoc = len(pLoc)
            yPlan = yPlan.T.reset_index(drop=True).T
            yTemp = yTemp.T.reset_index(drop=True).T
            
            planP = list(yPlan.T.values)
            planP = [list(x) for x in planP]
            planNP = [x[0:nLoc] for x in planP]
            
            for i in range(len(yTemp.columns)):
                tempP = list(yTemp[i])
                tempNP = tempP[0:nLoc]
                if tempNP not in planNP: planP.append(tempP)
        planP.sort(key=lambda x: x[-1], reverse = True)
        high_score = planP[0][-1]
        ix_high = [i for i in range(0,len(planP)) if planP[i][-1]==high_score]
        ix_high = max(ix_high) + 1
        planP_low = planP[ix_high:]
        planP_high = planP[:ix_high]
        random.seed(pInc)
        random.shuffle(planP_high)
        planP = planP_high + planP_low
        if pTop is not None: planP = planP[0:pTop]
        ans = pd.DataFrame(planP)   
        ans = ans.T
        ans = ans.set_index(ix)
        ans = ans.rename(columns=lambda x: 'P'+ str(x))
    elif yPlan is not None: 
        ans = yPlan
    else:  
        ans = None
    return ans

#function (return item predictions, minus observations)
def recall(yReason, xNoFind):
    ans = [x[:] for x in yReason]
    for i in range(len(xNoFind)):
        for j in range(len(xNoFind[i])):
            if xNoFind[i][j] in ans[i]: ans[i].remove(xNoFind[i][j])
            
    return ans

#function (return item belief, updated with observations)
def revise(yBelief, xObserve, xNoFind, pLoc):
    
    #add observations
    ans = [x[:] for x in yBelief]
    for i in range(len(ans)): ans[i].extend(xObserve[i])
    ans = [list(set(x)) for x in ans]        

    #remove noFind 
    for i in range(len(xNoFind)):
        for j in range(len(xNoFind[i])):
            if xNoFind[i][j] in ans[i]: ans[i].remove(xNoFind[i][j])
            
    #remove observations at unknown locations
    for i in range(len(ans)):
        temp = ans[i]
        tempFilt = [x for x in temp if x in pLoc]
        ans[i] = tempFilt  
    return ans

# =============================================================================

