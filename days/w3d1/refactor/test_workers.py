import torch as t
import workers

# conda activate ~/Developer/mlab/env && cd ~/Developer/mlab/days/w3d1/refactor && python test_workers.py

n_workers = 4
def func(worker):
    print(f'I am worker {worker.rank}. First: {worker.first}, middle: {worker.middle}, last {worker.last}')
    
    dtype = t.float64
    
    if worker.rank == 0:
        shape = t.randint(1, 10, (5,)).tolist()
        data = t.randn(shape, dtype=dtype)
        
        print(worker.rank, data.shape, data.dtype)
        
        worker.send(data, 1)
        worker.send(data, 2)
        
    if worker.rank == 1:
        data = worker.recv(0, dtype)
        print(worker.rank, data.shape, data.dtype)
        worker.send(data, 3)
        
    if worker.rank == 2:
        data = worker.recv(0, dtype)
        print(worker.rank, data.shape, data.dtype)
        worker.send(data, 3)
        
    if worker.rank == 3:
        data_1 = worker.recv(1, dtype)
        data_2 = worker.recv(2, dtype)
        
        assert data_1.equal(data_2)
        print('Test passed!')
        
        
    
if __name__=="__main__":
    workers.create_workers(4, func, False)
