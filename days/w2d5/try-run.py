from run import *

devices = [t.cuda.device(i) for i in range(t.cuda.device_count())]

size = len(devices)

processes = []
mp.set_start_method('spawn', force=True)
for rank in range(size):
    p = mp.Process(target=init_processes, args=(rank, size, run))
    print("I am starting!", rank, size)
    p.start()
    processes.append(p)
for p in processes:
    p.join()  