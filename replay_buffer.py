import numpy as np
import random

class replay_buffer:
    def __init__(self, size):
        '''parameters: size, buffer size, when overflow, overwrite'''
        self.size=size
        self.storage=[]
        self.next_idx=0
    
    def __len__(self):
        return len(self.storage)
    
    def add_data(self,seq,action,reward,next_seq,done):
        buf_data=(seq,action,reward,next_seq,done)
        if self.next_idx>=len(self.storage):
            self.storage.append(buf_data)
        else: 
            self.storage[self.next_idx]=buf_data
        self.next_idx=(self.next_idx+1)%self.size
        
    def sample(self,idxes):
        seq_s,actions,rewards,next_seq_s,dones=[],[],[],[],[]
        for i in idxes:
            data=self.storage[i]
            
            seq,action,reward,next_seq,done=data
            '''
            seq_s.append(seq)
            actions.append(action)
            rewards.append(reward)
            next_seq_s.append(next_seq)
            dones.append(done)
        return seq_s,actions,rewards,next_seq_s,dones
            '''
            
            seq_s.append(np.array(seq, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_seq_s.append(np.array(next_seq, copy=False))
            dones.append(done)
        return np.array(seq_s), np.array(actions), np.array(rewards), np.array(next_seq_s), np.array(dones)
        
    
    def sampling(self,batch_size):
        idxes = [random.randint(0, len(self.storage) - 1) for i in range(batch_size)]
        #print(idxes,len(self.storage),batch_size) #print idxes, buf_size and batch_size
        return self.sample(idxes)