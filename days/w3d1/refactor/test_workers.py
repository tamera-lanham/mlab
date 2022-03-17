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
        print('test passed!')

        
def func2(worker):
    
    if worker.first:
        data = t.tensor(64, dtype=t.int32)
        worker.send_to_all(data)
    
    else:
        data = worker.recv(0, t.int32)
        print(data)
        assert data == 64
        
    if worker.last:
        print('test passed!')
        
def func3(worker):
    tensor_list = [t.arange(i, i+10) for i in range(5)]
    dtype=tensor_list[0].dtype
    
    if worker.first:
        worker.send_multiple(tensor_list, worker.ranks[-1])
        
    if worker.last:
        tensor_list_received = worker.recv_multiple(0, dtype)
        
        for a, b in zip(tensor_list, tensor_list_received):
            assert a.equal(b)
        
        print('test passed!')


if __name__=="__main__":
    workers.create_workers(func, n_workers = 8, use_gpu = False)
    #workers.create_workers(func2, n_workers = 8, use_gpu = False)
    #workers.create_workers(func3, n_workers = 8, use_gpu = False)
    
    
    
