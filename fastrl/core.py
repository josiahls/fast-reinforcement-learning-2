# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/00_core.ipynb (unless otherwise specified).

__all__ = ['isnone', 'map_dict_ex', 'batch_sz', 'add_batch', 'D']

# Cell
# Python native modules
import os,warnings
# Third party libs
from fastcore.all import *
from fastai.torch_core import *
from fastai.basics import *
import pandas as pd
# Local modules

# Cell
def isnone(v): return v is None

# Cell
def map_dict_ex(d,f,*args,gen=False,wise=None,**kwargs):
    "Like `map`, but for dicts and uses `bind`, and supports `str` and indexing"
    g = (bind(f,*args,**kwargs) if callable(f)
         else f.format if isinstance(f,str)
         else f.__getitem__)

    if wise is None:  return map(g,d.items())
    return ((k,g(v)) if wise=='value' else (g(k),v) for k,v in d.items())

# Cell
def batch_sz(arr):
    if isinstance(arr,np.ndarray): return arr.shape[0]
    elif isinstance(arr,Tensor):   return arr.size()[0]
    elif isinstance(arr,(list,L)):     return len(arr)
    return None

@typedispatch
def stack(a,b): return L(a)+L(b)
@typedispatch
def stack(a:L,b): return a+L(b)
@typedispatch
def stack(a,b:L): return L(a)+b
@typedispatch
def stack(a:L,b:L): return a+b
@typedispatch
def stack(a:Tensor,b:Tensor): return torch.vstack((a,b))
@typedispatch
def stack(a:(np.array,np.ndarray),b:(np.array,np.ndarray)): return np.vstack((a,b))

class UnCollatable(Exception):
    def __init__(self,data,reasons:str=None,msg=''):
        store_attr()
        self.reasons=reasons.split(',')
        if 'nones' in reasons: self.msg+=f'Some values are not listy: {self.data}'
        if 'mismatch' in reasons: self.msg+=f'Some bs do not match {self.data}'

    def __str__(self): return self.msg


# Cell
_error_msg='Found idxs: %s have values more than %s e.g.: %s'

def add_batch(a,indexes):
    if hasattr(a,'expand_dims') and len(indexes)==1 and len(indexes)!=a.shape[0]:
        return np.expand_dims(a,0)
    return a

class D(dict):
    "Improved version of `dict` with array handling abilities"
    def __init__(self,*args,**kwargs):
        if isinstance(args,(tuple,list,L)):
            if len(args)==1 and isinstance(args[0],(tuple,list,L)):
                args=args[0]
                if all([type(v)==dict for v in args]):
                    args=L(args).map(D)
                    args=(sum(args[1:],args[0]),)
        super().__init__(*args,**kwargs)

    def __add__(self,o:'D')->'D':
        if not self.eq_k(o): ValueError(f'Key Mismatch: self:{self.keys()} o:{self.keys()}')
        d=deepcopy(self)
        for k in self: d[k]=stack(d[k],o[k])
        return d

    def eq_k(self,o:'D'): return set(o.keys())==set(self.keys())
    def eq_types(self,o:'D'): return set(map(type,o.values()))==set(map(type,o.values()))
    def _new(self,*args,**kwargs): return type(self)(*args,**kwargs)
    def argwhere(self,k,f,*args,**kwargs): return f(self[k],*args,**kwargs)
    def filter(self,k=None,f=None,*args,indexes=None,**kwargs):
        if indexes is None: indexes=f(self[k],*args,**kwargs)
        bs=self.bs()
        if max(indexes)>=bs: raise IndexError(_error_msg%(indexes,bs,max(idxs)))
        return self.subset(indexes)

    def subset(self,indexes):
        return type(self)({k:add_batch(self[k][indexes],indexes) for k in self})

    def map(self,f,*args,gen=False,**kwargs):
        return (self._new,noop)[gen](map_dict_ex(self,f,*args,**kwargs))
    def mapk(self,f,*args,gen=False,wise='key',**kwargs):
        return self.map(f,*args,gen=gen,wise=wise,**kwargs)
    def mapv(self,f,*args,gen=False,wise='value',**kwargs):
        return self.map(f,*args,gen=gen,wise=wise,**kwargs)

    def bs(self,validate=True):
        bs_map=self.mapv(batch_sz)
        if validate:
            if any(list(self.mapv(isnone).values())):
                raise UnCollatable(bs_map,'nones')
            if max(bs_map.values())!=min(bs_map.values()):
                raise UnCollatable(bs_map,'mismatch')
        return max(bs_map.values())

    @delegates(pd.DataFrame)
    def pandas(self,**kwargs):
        d=deepcopy(self)
        items=list(d.items())
        for k,v in items:
            bs=d.bs()
            if hasattr(v,'mean'): d[f'{k}_mu']=v.reshape(bs,-1).mean(axis=1)
            if isinstance(v,np.ndarray):
                d[k]=[str(v.shape)]*bs
            if isinstance(v,Tensor):
                d[k]=[str(v.size())]*bs
        return pd.DataFrame(d,**kwargs)