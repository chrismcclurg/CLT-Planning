# =============================================================================
# Construal Level Theory for Agent-based Planning
#
# C McClurg, AR Wagner, S Rajtmajer
# =============================================================================
# CONCEPTNET-SPECIFIC FILE
# Purpose: direct execute to read/ write levels of data from ConceptNet w/ API.
# Otherwise: collection of functions to define interactions with ConceptNet
# =============================================================================

import time
import pandas as pd
import requests
import re
import wordfreq
from datetime import datetime

# =============================================================================
#INPUT 

#items
df_item = pd.read_excel('utils/truth.xlsx', keep_default_na=False)
id_item = df_item.id_item.tolist()
id_block = df_item.id_block.tolist()

#locations
pLoc   = ['church', 'house', 'market' ]

#filename to write
pSave  = 'conceptnet00.xlsx'

# =============================================================================
#FUNCTIONS
 
STOPWORDS = ['the', 'a', 'an']
DROP_FIRST = ['to']
DOUBLE_DIGIT_RE = re.compile(r'[0-9][0-9]')
DIGIT_RE = re.compile(r'[0-9]')

def standardized_uri(language, term):
    """
    Get a URI that is suitable to label a row of a vector space, by making sure
    that both ConceptNet's and word2vec's normalizations are applied to it.
    'language' should be a BCP 47 language code, such as 'en' for English.
    If the term already looks like a ConceptNet URI, it will only have its
    sequences of digits replaced by #. Otherwise, it will be turned into a
    ConceptNet URI in the given language, and then have its sequences of digits
    replaced.
    """
    if not (term.startswith('/') and term.count('/') >= 2):
        term = _standardized_concept_uri(language, term)
    return replace_numbers(term)


def english_filter(tokens):
    """
    Given a list of tokens, remove a small list of English stopwords. This
    helps to work with previous versions of ConceptNet, which often provided
    phrases such as 'an apple' and assumed they would be standardized to
 	'apple'.
    """
    non_stopwords = [token for token in tokens if token not in STOPWORDS]
    while non_stopwords and non_stopwords[0] in DROP_FIRST:
        non_stopwords = non_stopwords[1:]
    if non_stopwords:
        return non_stopwords
    else:
        return tokens


def replace_numbers(s):
    """
    Replace digits with # in any term where a sequence of two digits appears.
    This operation is applied to text that passes through word2vec, so we
    should match it.
    """
    if DOUBLE_DIGIT_RE.search(s):
        return DIGIT_RE.sub('#', s)
    else:
        return s


def _standardized_concept_uri(language, term):
    if language == 'en':
        token_filter = english_filter
    else:
        token_filter = None
    language = language.lower()
    norm_text = _standardized_text(term, token_filter)
    return '/c/{}/{}'.format(language, norm_text)


def _standardized_text(text, token_filter):
    tokens = simple_tokenize(text.replace('_', ' '))
    if token_filter is not None:
        tokens = token_filter(tokens)
    return '_'.join(tokens)


def simple_tokenize(text):
    """
    Tokenize text using the default wordfreq rules.
    """
    return wordfreq.tokenize(text, 'xx')

#function to consistently treat OOV in conceptnet numberbatch
def get_uri(x, id_embed):
    x0 = standardized_uri('en', x) 
    n0 = len(x0)
    reverse_flag = 0
    if x0 in id_embed: ans = x0
    else:
        n = 0
        while n < (n0-6): 
            temp = '/c/en/' + x0[(6+n):n0]
            if temp in id_embed: 
                ans = temp
                n = n0 -6
            n += 1    
            if len(temp) == 7: reverse_flag = 1
        if reverse_flag ==1:
            n = (n0-1)
            while n > 6: 
                temp = x0[0:n]
                if temp in id_embed: 
                    ans = temp
                    n = 6
                n -= 1    
                if len(temp) == 7: ans = ''
    return ans   

#subfunction of many conceptnet search functions (returns a word without determiners)
def remove_determiner(x):
    det_list = ['a', 'an', 'the', 'this', 'that', 'these', 'those', 'my', 'your', 'his', 'her', 'its', 'our', 'their']
    x_list = x.split(' ')
    if len(x_list) > 0:
        if x_list[0] in det_list: x_list = x_list[1:]
    xnew = ' '.join(x_list) 
    return xnew 

#subfunction of get_abstractCN and get_initial (returns known locations from conceptnet)
def cnL(start, pLoc):
    ans = []
    rel     = '/r/AtLocation'
    link    = 'http://api.conceptnet.io/query?start='+start+'&rel='+rel
    edges   = requests.get(link).json()['edges'] 
    for edge in edges:
        end = edge['end']['label']
        end = remove_determiner(end)
        if end in pLoc: ans.append(end)
    return ans

#subfunction of get_abstractCN and get_initial (returns all related items from conceptnet)
def cnR(start):
    ans = []
    rel     = '/r/RelatedTo'
    other = '/c/en'
    link    = 'http://api.conceptnet.io/query?start=' +start +'&rel=' +rel +'&other=' +other
    edges   = requests.get(link).json()['edges'] 
    for edge in edges:
        tail = edge['end']['label']
        tail = remove_determiner(tail)    
        ans.append(tail)
    ans = list(set(ans))
    #ans = [standardized_uri('en', x) for x in ans] 
    ans = [get_uri(x, uri_check) for x in ans]

    return ans

#search the conceptnet and count the queries as you go
def search(uri, pMod, pLoc, n):    
    
    ans = cnL(uri, pLoc)
    n+=1
    
    xLoc = []
    xNoLoc = []
    yLoc = []
    
    if pMod > 0:
        print('.')
        relList = cnR(uri)
        n+=1
        for j in range(len(relList)):
            start = relList[j] 
            if start in xNoLoc: temp = []
            elif start in xLoc:
                ix = xLoc.index(start)
                temp = yLoc[ix]
            else:
                temp = cnL(start, pLoc)
                n+=1
                if len(temp)<1: xNoLoc.append(start)
                else: 
                    xLoc.append(start)
                    yLoc.append(temp)
                    
            ans.extend(temp)

    if pMod > 1:  
        print('.')
        for j in range(len(relList)):
            start = relList[j]
            newRelList = cnR(start)
            n+=1
            for k in range(len(newRelList)):
                newStart = newRelList[k]
                if newStart in xNoLoc: temp = []
                elif newStart in xLoc:
                    ix = xLoc.index(newStart)
                    temp = yLoc[ix]
                else:
                    temp = cnL(newStart, pLoc)
                    n+=1
                    if len(temp)<1: xNoLoc.append(newStart)
                    else: 
                        xLoc.append(newStart)
                        yLoc.append(temp)
                ans.extend(temp)
    return (ans,n)
# =============================================================================
#EXECUTION

if __name__ == "__main__":
    #read numberbatch word embeddings
    print('Reading conceptnet word embeddings...')
    df_embed = pd.read_hdf('utils/numberbatch.h5')
    df_embed = df_embed.loc[df_embed.index.str.startswith('/c/en', na=False)]
    uri_check = df_embed.index.values.tolist()
    print('Complete.\n')
    
    #read minecraft items (v1.11.2) 
    print('Reading items and obtaining URIs...')
    id_uri = [get_uri(x, uri_check) for x in id_block]
    del df_item
    print('Complete.\n')
    
    #unique_uri
    set_uri = list(set(id_uri))
    n = 0
    
    #get level 0
    L0_short = []
    for i in range(len(set_uri)):
        print('L0, item: ' + str(i) + ', requests: ' + str(n))
        temp_uri = set_uri[i]
        ans = search(temp_uri, 0, pLoc, n)
        temp = ans[0]
        L0_short.append(temp)
        n = ans[1]
        if n > 5000: 
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            print('SLEEP AT ' + current_time + ' TO RESET COUNT')
            time.sleep(3600)
            n=0
            
    #get level 1
    L1_short = []
    for i in range(len(set_uri)):
        print('L1, item: ' + str(i) + ', requests: ' + str(n))
        temp_uri = set_uri[i]
        ans = search(temp_uri, 1, pLoc, n)
        temp = ans[0]
        L1_short.append(temp)
        n = ans[1]
        if n > 5000: 
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            print('SLEEP AT ' + current_time + ' TO RESET COUNT')
            time.sleep(3600)
            n=0
            
    #get level 2
    L2_short = []
    for i in range(len(set_uri)):
        print('L2, item: ' + str(i) + ', requests: ' + str(n))
        temp_uri = set_uri[i]
        ans = search(temp_uri, 2, pLoc, n)
        temp = ans[0]
        L2_short.append(temp)
        n = ans[1]
        if n > 5000: 
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            print('SLEEP AT ' + current_time + ' TO RESET COUNT')
            time.sleep(3600)
            n=0
    
    #remove duplicates    
    L0_short = [list(set(x)) for x in L0_short]   
    L1_short = [list(set(x)) for x in L1_short]   
    L2_short = [list(set(x)) for x in L2_short]   
    
    #form larger lists
    L0 = []
    L1 = []
    L2 = []
    for uri in id_uri:
        ix = set_uri.index(uri)
        L0.append(L0_short[ix])
        L1.append(L1_short[ix])
        L2.append(L2_short[ix])
        
    #make pLoc appropriate length
    nadd = len(L2) - len(pLoc)
    xadd = ['']*nadd
    pLoc.extend(xadd)
    
    #create dataframe
    df = pd.DataFrame(zip(pLoc,L0,L1,L2), columns = ['pLoc', 'L0', 'L1', 'L2'])
    
    #write dataframe to file
    df.to_excel(pSave)

# =============================================================================
    
    
    
        
        
