import torch as t
import workers

# conda activate ~/Developer/mlab/env && cd ~/Developer/mlab/days/w3d1/refactor && python test_workers.py

def func(worker):
    
    dtype = t.float64
    
    if worker.first:
        shape = t.randint(1, 10, (5,)).tolist()
        data = t.randn(shape, dtype=dtype)
        
        print(worker.rank, data.shape, data.dtype)
        
        worker.send(data, worker.ranks[-1])
        worker.send(data, worker.next)
        
    if worker.middle:
        data = worker.recv(worker.prev, dtype)
        print(worker.rank, data.shape, data.dtype)
        worker.send(data, worker.next)
        
    if worker.last:
        data_0 = worker.recv(0, dtype)
        data_1 = worker.recv(worker.prev, dtype)
        
        assert data_0.equal(data_1)
        print('Test passed!')
        

if __name__=="__main__":
    workers.create_workers(func, n_workers = 8, use_gpu = False)
