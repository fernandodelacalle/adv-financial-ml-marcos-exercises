import numpy as np
import pandas as pd
import multiprocessing as mp
import datetime as dt

def linParts(numAtoms,
             numThreads):
    """SNIPPET 20.5 THE linParts FUNCTION
    partition of atoms with a single loop
    """
    parts=np.linspace(0,numAtoms,min(numThreads,numAtoms)+1)
    parts=np.ceil(parts).astype(int)
    return parts

def nestedParts(numAtoms,
                numThreads,
                upperTriang=False):
    """SNIPPET 20.6 THE nestedParts FUNCTION
    Partition of atoms with an inner loop
    """ 
    parts,numThreads_=[0],min(numThreads,numAtoms)
    for num in range(numThreads_):
        part=1+4*(parts[-1]**2+parts[-1]+numAtoms*(numAtoms+1.)/numThreads_)
        part=(-1+part**.5)/2.
        parts.append(part)
    parts=np.round(parts).astype(int)
    if upperTriang: # the first rows are heaviest
        parts=np.cumsum(np.diff(parts)[::-1])
        parts=np.append(np.array([0]),parts)
    return parts

def mpPandasObj(func,
                pdObj,
                numThreads=24,
                mpBatches=1,
                linMols=True,
                **kargs):
    """SNIPPET 20.7 THE mpPandasObj, USED AT VARIOUS POINTS IN THE BOOK
    Parallelize jobs, return a dataframe or series
    + func: function to be parallelized. Returns a DataFrame
    + pdObj[0]: Name of argument used to pass the molecule
    + pdObj[1]: List of atoms that will be grouped into molecules
    + kwds: any other argument needed by func
    
    Example: df1=mpPandasObj(func,('molecule',df0.index),24,**kwds)
     """
    import pandas as pd
    #if linMols:parts=linParts(len(argList[1]),numThreads*mpBatches)
    #else:parts=nestedParts(len(argList[1]),numThreads*mpBatches)
    if linMols:parts=linParts(len(pdObj[1]),numThreads*mpBatches)
    else:parts=nestedParts(len(pdObj[1]),numThreads*mpBatches)
    
    jobs=[]
    for i in range(1,len(parts)):
        job={pdObj[0]:pdObj[1][parts[i-1]:parts[i]],'func':func}
        job.update(kargs)
        jobs.append(job)
    if numThreads==1:out=processJobs_(jobs)
    else: out=processJobs(jobs,numThreads=numThreads)
    if isinstance(out[0],pd.DataFrame):df0=pd.DataFrame()
    elif isinstance(out[0],pd.Series):df0=pd.Series()
    else:return out
    for i in out:df0=df0.append(i)
    df0=df0.sort_index()
    return df0

def processJobs_(jobs):
    """SNIPPET 20.8 SINGLE-THREAD EXECUTION, FOR DEBUGGING
    Run jobs sequentially, for debugging
    """
    out=[]
    for job in jobs:
        out_=expandCall(job)
        out.append(out_)
    return out

def reportProgress(jobNum,
                   numJobs,
                   time0,
                   task):
    """SNIPPET 20.9 EXAMPLE OF ASYNCHRONOUS CALL TO PYTHON’S MULTIPROCESSING LIBRARY
    Report progress as asynch jobs are completed
    """ 
    msg=[float(jobNum)/numJobs, (time.time()-time0)/60.]
    msg.append(msg[1]*(1/msg[0]-1))
    timeStamp=str(dt.datetime.fromtimestamp(time.time()))
    msg=timeStamp+' '+str(round(msg[0]*100,2))+'% '+task+' done after '+ \
        str(round(msg[1],2))+' minutes. Remaining '+str(round(msg[2],2))+' minutes.'
    if jobNum<numJobs:sys.stderr.write(msg+'\r')
    else:sys.stderr.write(msg+'\n')
    return

def processJobs(jobs,
                task=None,
                numThreads=24):
    """SNIPPET 20.9 EXAMPLE OF ASYNCHRONOUS CALL TO PYTHON’S MULTIPROCESSING LIBRARY
    Run in parallel.
    jobs must contain a 'func' callback, for expandCall
    """ 
    if task is None:task=jobs[0]['func'].__name__
    pool=mp.Pool(processes=numThreads)
    outputs,out,time0=pool.imap_unordered(expandCall,jobs),[],time.time()
    # Process asyn output, report progress
    for i,out_ in enumerate(outputs,1):
        out.append(out_)
        reportProgress(i,len(jobs),time0,task)
    pool.close();pool.join() # this is needed to prevent memory leaks
    return out

def expandCall(kargs):
    """SNIPPET 20.10 PASSING THE JOB (MOLECULE) TO THE CALLBACK FUNCTION
    Expand the arguments of a callback function, kargs['func']
    """
    func=kargs['func']
    del kargs['func']
    out=func(**kargs)
    return out

def _pickle_method(method):
    """SNIPPET 20.11 PLACE THIS CODE AT THE BEGINNING OF YOUR ENGINE"""
    func_name=method.im_func.__name__
    obj=method.im_self
    cls=method.im_class
    return _unpickle_method, (func_name,obj,cls)

def _unpickle_method(func_name,
                     obj,
                     cls):
    """SNIPPET 20.11 PLACE THIS CODE AT THE BEGINNING OF YOUR ENGINE"""
    for cls in cls.mro():
        try:func=cls.__dict__[func_name]
        except KeyError:pass
        else:break
    return func.__get__(obj,cls)

#SNIPPET 20.11 PLACE THIS CODE AT THE BEGINNING OF YOUR ENGINE
import copyreg,types, multiprocessing as mp
copyreg.pickle(types.MethodType,_pickle_method,_unpickle_method)