import pandas as pd
import numpy as np
import tqdm
#import tqdm.notebook
import os
import json
import itertools
import dask.distributed
from .fileop import *
def chunk_formatter(cut):
    return f'chunk_{cut}.csv'
def locations_formatter():
    return 'locations.csv'
def save_locations(path,locations):
    with open(path+locations_formatter(),'w') as f:
        json.dump(locations,f)
def check_path(path):
    if path!='/':
        return path+'/'
    else:
        return path
def load_locations(path):
    try:
        with open(path+locations_formatter(),'r') as f:
            loc= json.load(f)
            N = len(loc.keys())
    except FileNotFoundError:
        loc = None
        N = None
    finally:
        return loc, N
def chunking(dataset,chunksize, path,desc='chunking', scheduler = None):
    if isinstance(dataset,str):
            #ds = pd.read_csv(dataset,index_col=0).shape[0]
        with open(dataset,'r') as f:
            for size, _ in enumerate(f):
                pass
    else:
        with open(dataset,'r') as f:
            for size, _ in enumerate(f):
                pass
    #ind = pd.unique(ds.index)   
    #l   = np.arange(0,np.size(ind),chunksize)
    l = np.arange(0, size,chunksize)
    if size>l[-1]:
        l = np.append(l,size)
    locations = {}
    iterator = [[i,slice(l[i],l[i+1]),path,dataset] for i in range(len(l)-1)]
    print('Number of files', len(l)-1)
    def chunker(inputs):
        cut,_slice, path, dataset = inputs
        
        loc = path+f'chunk_{cut}.csv'
        if isinstance(dataset,str):
            dataset = next(itertools.islice(pd.read_csv(dataset,index_col=0,chunksize=_slice.stop-_slice.start),cut,cut+1))
            
        #if cut<len(l):
        #    #dataset.loc[ind[l[cut-1]]:ind[l[cut]-1]].to_csv(loc)
        #    dataset.iloc[cut-1:cut].to_csv(loc)
        #else:
        #    print(ind[-1],l[cut-1],dataset.shape)
        #    dataset.loc[ind[l[cut-1]]:].to_csv(loc)
        dataset.to_csv(loc)
        return loc
    if scheduler is None:
        for cut in tqdm.notebook.tqdm(iterator,desc=desc):
            #locations[cut] = chunker([cut,path,dataset])
            locations[cut[0]] = chunker(cut)
            #dataset.loc[ind[l[cut-1]]:ind[l[cut]-1]].to_csv(path+f'chunk_{cut}.csv')
            #locations[cut] = path+f'chunk_{cut}.csv'
    else:
        
        with dask.distributed.Client(scheduler) as client:
            #big_future = client.scatter(big_data)  
            locations = client.gather(client.map(chunker,iterator))
            
            locations = dict(zip(range(1,len(l)+1),locations))
    save_locations(path,locations)
    return locations

class Chunk:
    def __init__(self,path,chunksize, counter=True,scheduler=None):
        self.counter = counter
        self.chunksize= chunksize
        self.scheduler = scheduler
        self.path = check_path(path)
        self.locations,self.N = load_locations(self.path)
        
    def to_chunk(self,dataset):
        self.locations = chunking(dataset,self.chunksize, make_file(self.path),scheduler=self.scheduler)
        self.N = len(self.locations.keys())
        return self
    def __str__(self):
        return f'<Chunk object divided in {self.N} chunks of size {self.chunksize} at {hex(id(self))}>'
    def Files(self,desc=''):
        if self.counter:
            yield from tqdm.notebook.tqdm(self.locations.items(), total=self.N,desc=desc)
        else:
            yield from self.locations.items()
    def join(self, *args,**kwargs):
        for key, path in self.Files('join'):
            pd.read_csv(path,index_col=0)\
                .join(*args,**kwargs)\
                .to_csv(path)
        return self
    def to_csv(self, path_new,*args,**kwargs):
        locations = {}
        path_new = make_file(path_new)
        def _to_csv(cut,path):
            loc = path_new+chunk_formatter(cut)
            pd.read_csv(path,index_col=0)\
                .to_csv(loc,*args,**kwargs)
            return loc
        for cut, path in self.Files('to_csv'):
            locations[cut] = _to_csv(cut,path)
        save_locations(path_new,locations)
        return Chunk(path_new,self.chunksize,self.counter,self.scheduler)
    def __setitem__(self, key,val):
        for _, path in self.Files('__setitem__'):
            p = pd.read_csv(path,index_col=0)
            p[key] = val
            p.to_csv(path)
    def __getitem__(self, key):
        p=[]
        for _, path in self.Files('__getitem__'):
            if not isinstance(key,list):
                key = [key]
            p.append(pd.read_csv(path,index_col=0,usecols=key))
        return pd.concat(p)
    def set_index(self, *args,**kwargs):
        for key, path in self.Files('set_index'):
            pd.read_csv(path,index_col=0)\
                .set_index(*args,**kwargs)\
                .to_csv(path)
        return self
    def reset_index(self, *args,**kwargs):
        for key, path in self.Files('reset_index'):
            pd.read_csv(path,index_col=0)\
                .reset_index(*args,**kwargs)\
                .to_csv(path)
        return self
    def drop_duplicates(self, *args,**kwargs):
        for key, path in self.Files('drop_duplicates'):
            pd.read_csv(path,index_col=0)\
                .drop_duplicates(*args,**kwargs)\
                .to_csv(path)
        return self
    def applyfunc(self, func,*args,desc='apply',**kwargs):
        for key, path in self.Files(desc):
            func(pd.read_csv(path,index_col=0),*args,**kwargs)\
                .to_csv(path)
        return self
    def iterchunks(self):
        for key, path in self.Files('iterating'):
            yield key,path, pd.read_csv(path,index_col=0)
    
        
            
            
