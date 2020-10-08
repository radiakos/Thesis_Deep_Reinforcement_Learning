import numpy as np
import random

class Prioritized_replay_buffer:
    def __init__(self, size, alpha=0.6):
        self.alpha = alpha
        self.size   = size
        self.storage     = []
        self.next_idx   = 0
        self.priorities = np.zeros((size,), dtype=np.float32)
        self.eps = 1e-5
        
    def __len__(self):
        return len(self.storage)

    def add_data(self,seq,action,reward,next_seq,done):
        buf_data=(seq,action,reward,next_seq,done)
        max_prio = self.priorities.max() if self.storage else 1.0
        if self.next_idx>=len(self.storage):
            self.storage.append(buf_data)
        else: 
            self.storage[self.next_idx]=buf_data
        self.priorities[self.next_idx] = max_prio
        self.next_idx=(self.next_idx+1)%self.size
        

    def sample(self, idxes,probs, beta=0.4):
        seq_s,actions,rewards,next_seq_s,dones,idxes_s,weights_s=[],[],[],[],[],[],[]
        total    = len(self.storage)
        for i in idxes:
            data=self.storage[i]
            seq,action,reward,next_seq,done=data
            weights  = (total * self.priorities[i]) ** (-beta)
            weights /= weights.max()
            idxes_s.append(idxes)
            weights_s.append(np.array(weights, dtype=np.float32))
            seq_s.append(np.array(seq, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_seq_s.append(np.array(next_seq, copy=False))
            dones.append(done)
        
        return np.array(seq_s), np.array(actions), np.array(rewards), np.array(next_seq_s), np.array(dones), np.array(idxes),np.array(weights, dtype=np.float32)
        
    def sampling(self,batch_size,beta):
        if len(self.storage) == self.size:
            prios = self.priorities
        else:
            prios = self.priorities[:self.next_idx]
        probs  = prios ** self.alpha
        probs /= probs.sum()
        idxes = np.random.choice(len(self.storage), batch_size, p=probs)

        return self.sample(idxes,probs,beta)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

