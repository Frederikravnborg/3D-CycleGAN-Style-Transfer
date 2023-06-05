
# import torch
# from torch.utils.data import DataLoader
# from data import FaceDataset
# from config import *


# from time import time
# import multiprocessing as mp
# for num_workers in range(2, mp.cpu_count(), 2):  
#     train_loader = DataLoader(train_reader,shuffle=True,num_workers=num_workers,batch_size=64,pin_memory=True)
# start = time()
# for epoch in range(1, 3):
#     for i, data in enumerate(train_loader, 0):
#         pass
# end = time()
# print("Finish with:{} second, num_workers={}".format(end - start, num_workers))


import time
pin_memory = True
print('pin_memory is', pin_memory)
 
for num_workers in range(0, 20, 1): 
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
    sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    start = time.time()
    for epoch in range(1, 5):
        for i, data in enumerate(train_loader):
            pass
    end = time.time()
    print("Finish with:{} second, num_workers={}".format(end - start, num_workers))