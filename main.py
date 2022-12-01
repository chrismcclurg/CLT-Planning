# =============================================================================
# Construal Level Theory for Agent-based Planning
#
# C McClurg, AR Wagner, S Rajtmajer
# =============================================================================

from multiprocess import Process, Queue
from models.construal    import get_recipeMatrix
from models.construal    import get_featureMatrix
from models.construal    import guess, recall, plan, review, revise, reason
from models.conceptnet   import get_uri
from models.malmo        import run
import time, datetime
import pandas as pd
import numpy as  np
import random

# =============================================================================

def trial(q, iInput):             
    nRun_max = 20
       
    #unpack inputs 
    iTrial      = iInput[0]
    iTask       = iInput[1]
    iAbs        = iInput[2]
    iSet        = iInput[3]
    iPPR        = iInput[4]
    iInc        = iInput[5]
    id_item     = iInput[6]
    id_uri      = iInput[7]
    mRecipe     = iInput[8]    
    mFeature    = iInput[9] 
    
    #initialize observation vars
    xObserve = [[] for x in range(len(id_item))]        #true locations of items (observed during trial)
    xNoFind  = [[] for x in range(len(id_item))]        #false locations of items (observed during trial)
    xPossess = [0 for x in range(len(id_item))]
    
    #initizalize environment vars
    iContinue = True 
    iStats = (0,0)
    iResult = 0
    iPos = (0,0,90)
    iRan = 0
    
    #initialize trial output
    planTime    = 0.0
    runTime     = 0.0
    totTime     = 0.0
    mcDist      = 0
    mcTime      = 0
    nRun        = 0
    nObs        = 0
    nPlanner    = 0
    nNoPlan     =  0
    startTime   = ''
    endTime     = ''
    lastPlan    = ''
    
    #initialize environment input
    iEnv = [xObserve, xNoFind, xPossess, iSet, iRan, iInc, iStats, iResult, iPos, iTrial]
    
    #initialize planning 
    pSrc    = 'utils/conceptnet.xlsx'
    pLoc    = pd.read_excel(pSrc, keep_default_na=False)['set'+str(iSet)+'_pLoc'].values
    pLoc    = [x for x in pLoc if len(x)>0]    
    pLoc.append('self') 
    yBelief = guess(id_item, id_uri, pLoc, pSrc, iSet) #belief with minor abstraction
    yReason = [[] for x in range(len(id_item))]        #belief with abstraction   
    
    #initialize output
    iOut = [iTrial, iTask, iAbs, iSet, iPPR, iInc, startTime, endTime, planTime/60, runTime/60, totTime/60, mcTime, mcDist, nObs, nPlanner, nNoPlan, nRun, iResult, lastPlan]
    
    while iContinue:
        
        yPlan    = None                                     #plans formed by agent

        #start plan timer
        plan_t0     = np.round(time.time(), 2)
        startTime   = datetime.datetime.now()
        startTime   = startTime.strftime("%Y-%m-%d %H:%M:%S")
        
        #make plans
        nPlanner += 1
        if iAbs == 0:   pModList = [0,1,2]      
        else:           pModList = [0, 0.8, 1.0]   
        for pMod in pModList:
            ix  = pModList.index(pMod) + 1    
            print("\nPLANNING level {} of 3".format(ix))           
            print('\treason')
            yReason = reason(yBelief, mFeature, pLoc, iAbs, pSrc, pMod, iSet)       #reason locations with abstraction
            print('\trecall')  
            yReason = recall(yReason, xNoFind)                                      #update locations with feedback
            print('\tplan')       
            yTemp   = plan(yReason, xPossess, mRecipe, id_item, pLoc, iTask, pMod)  #make plans and score
            print('\treview') 
            yPlan   = review(yTemp, yPlan, pLoc, iInc)                              #add plans and sort 

        #check plans
        if yPlan is not None: 
            temp = yPlan.P0                                                  
            temp = temp.score                                             
            if temp == 1: iContinue = False
            del temp
            count = len(yPlan.columns)
            if count > iPPR: yPlan = yPlan.iloc[:(len(pLoc)-1),:iPPR]
            else: yPlan = yPlan.iloc[:(len(pLoc)-1),:count] 
            
        else: 
            yPlan= pd.DataFrame(index = pLoc, columns = ['P0'])
            ix = np.random.randint(0, len(pLoc)-1)
            temp = [[] for x in range(len(pLoc))]
            temp[ix] = [iTask]
            yPlan.P0 = temp
            yPlan = yPlan.iloc[:(len(pLoc)-1)]
            nNoPlan += 1
            
        #end plan timer
        plan_t1 = np.round(time.time(), 2)
        planTime += (plan_t1 - plan_t0)
        planTime = np.round(planTime, 2)
        
        #start run timer
        run_t0 = np.round(time.time(), 2)
           
        #run plans
        for col in yPlan:
            iPlan = yPlan[col]
            print('running trial {}'.format(iTrial))
            iEnv = run(iEnv, iPlan)  
            iResult = iEnv[7]
            nRun +=1
            print('nRun', nRun)
            if iResult == 1: 
                iContinue = False
                break
            
        #end run timer
        run_t1 = np.round(time.time(), 2)
        runTime += (run_t1 - run_t0)
        runTime = np.round(runTime, 2)

        #update pack
        xObserve    = iEnv[0]
        xNoFind     = iEnv[1]
        xPossess    = iEnv[2]
        iStats      = iEnv[6]
        iResult     = iEnv[7]
        iPos        = iEnv[8]
        
        mcTime += iStats[0]              #cummulative minecraft time
        mcDist += iStats[1]              #cummulative minecraft distance
        
        if nRun == nRun_max: iContinue = False
        
        #prepare for re-plan OR write stats to file
        if iContinue is True:  yBelief = revise(yBelief, xObserve, xNoFind, pLoc)                            
        else: 
            lastLoc = list(iPlan.index.values)
            lastItem = list(iPlan)
            lastPlan = []
            for ix in range(len(lastLoc)): 
                temp = lastItem[ix]
                if len(temp)>0: lastPlan.append((lastLoc[ix], lastItem[ix])) 
                
            endTime = datetime.datetime.now()
            endTime = endTime.strftime("%Y-%m-%d %H:%M:%S")
            
            for ix in range(len(xObserve)):
                temp = len(xObserve[ix])
                nObs += temp      
                
            totTime = np.round(planTime + runTime, 2)
            
            iOut = [iTrial, iTask, iAbs, iSet, iPPR, iInc, startTime, endTime, planTime/60, runTime/60, totTime/60, mcTime, mcDist, nObs, nPlanner, nNoPlan, nRun, iResult, lastPlan]
            print('completed trial {}'.format(iTrial))
            q.put(iOut)
            time.sleep(0.1)

# =============================================================================
# EXECUTION

if __name__ == '__main__':
            
    #input parameters
    pInc   = 10                 #iterations per condition  
    pAbs   = [0,1,2,None]       #type of abstraction (0=Sc, 1=Si, 2=Cl, None)
    pPPR   = [1,3,5]            #plans per replan
    pSet   = [0,1,2]            #location set number 

    print('Agent loading inputs....')
    print('\t COMPLETE parameters from user.')
    
    #read word vectors (Numberbatch)
    df_embed = pd.read_hdf('utils/numberbatch.h5')
    df_embed = df_embed.loc[df_embed.index.str.startswith('/c/en', na=False)]
    uri_check = df_embed.index.values.tolist()
    print('\t COMPLETE vectors from numberbatch.')
    
    #read items (Minecraft v1.11.2) 
    df_item = pd.read_excel('utils/truth.xlsx', keep_default_na=False)
    id_item = df_item.id_item.tolist()
    id_block = df_item.id_block.tolist()
    id_uri = [get_uri(x, uri_check) for x in id_block]
    del uri_check, df_item
    print('\t COMPLETE items from minecraft.')
    
    #get random task item
    random.seed(1)
    pTask = random.sample(id_item, 10)
    
    #read recipes (Minecraft v1.11.2) 
    df_recipe = pd.read_excel('utils/truth.xlsx', keep_default_na=False)
    df_recipe = df_recipe.set_index('id_item')
    print('\t COMPLETE recipes from minecraft.')
    
    #compute descriptive matricies
    mRecipe   = get_recipeMatrix(id_item, df_recipe)
    mFeature  = get_featureMatrix(id_uri, df_embed)
    del df_recipe, df_embed
    print('\t COMPLETE descriptive matricies.')
    
    #pack for multiprocessing
    iTrial   = 0                  #trial no.
    iStats   = (0,0)              #stats per trial
    iResult  = 0                  #plan result (-1,0,1 for fail, IP, success)
    iPos     = (0,0,90)           #start position (needed for next attempt within trial)
    iRan     = 0                  #degree of randomness on map
    testPack = []
    for iTask in pTask:                         #SWEEP (task item)
        for iAbs in pAbs:                       #SWEEP (abstraction type)
            for iSet in pSet:                   #SWEEP (location set)     
                for iPPR in pPPR:               #SWEEP (plans per replan)
                    for iInc in range(pInc):    #SWEEP (increment, random seed) 
                        iInput = [iTrial, iTask, iAbs, iSet, iPPR, iInc, id_item, id_uri, mRecipe, mFeature]
                        testPack.append(iInput)
                        iTrial+=1            
    print('\t COMPLETE compiling test pack.')
    
    # trial(testPack[0])
    
    now = datetime.datetime.now()
    d = now.strftime('%Y-%m%d')
    filename = './output/clt-planning_{}.xlsx'.format(d)
     
    #exectute test
    totalResult = []
    q = Queue()
    nProcs= 1
    pHandle = []
    
    for i in range(nProcs):
        pHandle.append(Process(target=trial, args=(q,testPack[0])))
        testPack.pop(0)
        pHandle[-1].start()
    
    while len(pHandle):
        pHandle = [x for x in pHandle if x.is_alive()]        
        s = nProcs - len(pHandle)
        
        for i in range(s):
            if len(testPack):
                pHandle.append(Process(target=trial, args=(q,testPack[0])))
                testPack.pop(0)
                pHandle[-1].start()
        
        while q.qsize()> 0:
            singleResult = q.get()
            totalResult.append(singleResult)
            df = pd.DataFrame(totalResult, columns = ['trial', 'task', 'abstraction type', 'item set', 'plans per replan', 'seed', 'tsStart', 'tsEnd','plan time (min)', 'run time (min)', 'total time (min)', 'run time (minecraft ticks)', 'distance (minecraft units)', 'observations', 'nPS', 'nRunNP', 'nRun', 'result', 'lastPlan'])
            df.to_excel(filename) 

# =============================================================================

    