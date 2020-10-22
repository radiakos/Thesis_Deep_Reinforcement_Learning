import numpy as np
import random

class Replay_buffer:
    """Store agent's experience in an experience replay buffer"""

    def __init__(self, size):
        """Initialize an Replay buffer object.
        Params
        ======
            size(int): replay buffer size
            if overflow, overwrite accordint to FIFO
        """
        self.size=size
        self.storage=[]
        self.next_idx=0
    
    def __len__(self):
        """returns the length of storaged experience"""
        return len(self.storage)
    
    def add_data(self,seq,action,reward,next_seq,done):
        """Adds experiences to memory
        Params
        ======
        seq(numpy array):sequence of frames at time t-1
        action(int):agent's action at time t-1
        reward(float):agent's reward at time t-1
        next_seq(numpy array):sequence of frames at time t
        done(boolean):end of epeisode signal
        """
        buf_data=(seq,action,reward,next_seq,done)
        if self.next_idx>=len(self.storage):
            self.storage.append(buf_data)
        else: 
            self.storage[self.next_idx]=buf_data
        self.next_idx=(self.next_idx+1)%self.size
        
    def sample(self,idxes):
        """Samples experiences from memory
        Params
        ======
        idxes(array):array which shows the selected samples's position in buffer
        """
        seq_s,actions,rewards,next_seq_s,dones=[],[],[],[],[]
        for i in idxes:
            data=self.storage[i]
            seq,action,reward,next_seq,done=data
            seq_s.append(np.array(seq, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            next_seq_s.append(np.array(next_seq, copy=False))
            dones.append(done)

        return np.array(seq_s), np.array(actions), np.array(rewards), np.array(next_seq_s), np.array(dones)
        
    
    def sampling(self,batch_size):
        """Samples memory's positions
        Params
        ======
        batch_size(int):how many experience will be sampled from buffer
        """
        idxes = [random.randint(0, len(self.storage) - 1) for i in range(batch_size)]

        return self.sample(idxes)


class Prioritized_replay_buffer:
    """Store agent's experience in an prioritized experience replay buffer"""
        
    def __init__(self, size, alpha=0.6):
        """Initialize an Replay buffer object.
        Params
        ======
            size(int): replay buffer size
            if overflow, overwrite accordint to FIFO
        """
        self.alpha=alpha
        self.size=size
        self.storage=[]
        self.next_idx=0
        self.priorities=np.zeros((size,),dtype=np.float32)
        self.eps=1e-5
        
    def __len__(self):
        """returns the length of storaged experience"""
        return len(self.storage)

    def add_data(self,seq,action,reward,next_seq,done):
        """Adds experiences to memory
        Params
        ======
        seq(numpy array):sequence of frames at time t-1
        action(int):agent's action at time t-1
        reward(float):agent's reward at time t-1
        next_seq(numpy array):sequence of frames at time t
        done(boolean):end of epeisode signal
        """
        buf_data=(seq,action,reward,next_seq,done)
        max_prio = self.priorities.max() if self.storage else 1.0
        if self.next_idx>=len(self.storage):
            self.storage.append(buf_data)
        else: 
            self.storage[self.next_idx]=buf_data
        self.priorities[self.next_idx] = max_prio
        self.next_idx=(self.next_idx+1)%self.size
        

    def sample(self, idxes,probs, beta=0.4):
        """Samples experiences and their weights from memory
        Params
        ======
        idxes(array):array which shows the selected samples's position in buffer
        probs(array):array of probanilities of each sampled experience
        beta(float):factor of importance for probabilties 
        """
        seq_s,actions,rewards,next_seq_s,dones,idxes_s,weights_s=[],[],[],[],[],[],[]
        total=len(self.storage)
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
        """Samples experiences's positions from memory according to their priorities
        Params
        ======
        batch_size(int):how many experience will be sampled from buffer
        beta(float):factor of importance for priorities 
        """
        if len(self.storage) == self.size:
            prios=self.priorities
        else:
            prios=self.priorities[:self.next_idx]
        probs=prios**self.alpha
        probs/=probs.sum()
        idxes=np.random.choice(len(self.storage), batch_size, p=probs)

        return self.sample(idxes,probs,beta)

    def update_priorities(self, batch_indices, batch_priorities):
        """Update sampled experiences's priorities 
        Params
        ======
        batch_indices(array):array of the selected samples's position in buffer
        batch_priorities(array):array of the selected samples's priorities
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

