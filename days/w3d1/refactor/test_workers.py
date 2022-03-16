import workers

n_workers = 4
def func(worker):
    print(f'I am worker {worker.rank}, and this is my next group: {worker.next}')
    
if __name__=="__main__":
    workers.create_workers(4, func)
