from dataclasses import dataclass
import numpy as np
import torch
from collections import deque
import random
from compressive_transformer_pytorch import CompressiveTransformer
from collections import namedtuple
Memory = namedtuple('Memory', ['mem', 'compressed_mem'])

import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction

def top_k(logits, thres = 0.8):
	k = int((1 - thres) * logits.shape[-1])
	val, ind = torch.topk(logits, k)
	probs = torch.full_like(logits, float('-inf'))
	probs.scatter_(1, ind, val)
	return probs

def default(x, val):
	if x is not None:
		return x
	return val if not isfunction(val) else val()

def to(t):
	return {'dtype': t.dtype, 'device': t.device}

class Net(nn.Module):
	def __init__(self, **kwargs):
		super().__init__()
		self.num_tokens = kwargs.get('num_tokens')
		self.online = CompressiveTransformer(**kwargs)
		self.target = CompressiveTransformer(**kwargs)

	def empty_mem(self, x):
		x = self.online.token_emb(x)
		x = self.online.to_model_dim(x)
		b, t, d = x.shape

		assert t <= self.online.seq_len, f'input contains a sequence length {t} that is greater than the designated maximum sequence length {self.online.seq_len}'

		memories = (None, None)
		mem, cmem = memories

		num_memory_layers = len(self.online.memory_layers)
		init_empty_mem = lambda: torch.empty(num_memory_layers, b, 0, d, **to(x))
		mem = default(mem, init_empty_mem)
		cmem = default(cmem, init_empty_mem)

		# total_len = mem.shape[2] + cmem.shape[2] + self.online.seq_len
		# pos_emb = self.online.pos_emb[:, (self.online.seq_len - t):total_len]

		# next_mem = []
		# next_cmem = []
		# aux_loss = torch.tensor(0., requires_grad = True, **to(x))

		if self.online.enhanced_recurrence:
			mem = torch.roll(mem, -1, 0)
			cmem = torch.roll(cmem, -1, 0)

		return Memory(mem=mem, compressed_mem=cmem)

	def forward(self, model, inp, cmem = None, reinforce = True):
		if model == "online":
			out, mem, aux_loss = self.online(inp, cmem)
		elif model == "target":
			out, mem, aux_loss = self.target(inp, cmem)
		if not reinforce:
			out = F.softmax(out)
		return out, mem, aux_loss
	

class Policy:
	def __init__(self, net: Net, use_cuda=False) -> None:
		self.net: Net = net
		self.exploration_rate = 1
		self.exploration_rate_decay = 0.99999975
		self.exploration_rate_min = 0.1

		self.memory = deque(maxlen=10)
		self.batch_size = 4
		self.gamma = 0.9

		self.optimizer = torch.optim.Adam(self.net.online.parameters(), lr=0.00025)
		self.loss_fn = torch.nn.SmoothL1Loss()
		self.burnin = 1e4  # min. experiences before training
		self.learn_every = 3  # no. of experiences between updates to Q_online
		self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync
		self.use_cuda = use_cuda
		self.curr_step = 0

	def act(self, inp, cmem):
		if np.random.rand() < self.exploration_rate:
			action, mem = np.random.randint(0, self.net.num_tokens), None
		else:
			with torch.no_grad():
				logits, mem, _ = self.net('target', inp, cmem)
				filtered_logits = top_k(logits, thres = 0.8)
				action = torch.multinomial(F.softmax(filtered_logits, dim=-1), 1)
		return action, mem

	def cache(self, state, next_state, action, reward, done):
		"""
		Store the experience to self.memory (replay buffer)

		Inputs:
		state (LazyFrame),
		next_state (LazyFrame),
		action (int),
		reward (float),
		done(bool))
		"""

		state = state
		next_state = next_state
		action = action
		reward = reward
		done = done

		self.memory.append((state, next_state, action, reward, done))

	def recall(self):
		"""
		Retrieve a batch of experiences from memory
		"""
		batch = random.sample(self.memory, self.batch_size)
		state, next_state, action, reward, done = zip(*batch)
		return state, next_state, action, reward, done

	def td_estimate(self, state, action):
		states, mems = zip(*state)
		max_state_len = max(len(s) for s in states)
		states = torch.from_numpy(np.array([[0] * (max_state_len - len(s)) + list(s) for s in states]))
		mems = self.net.empty_mem(states)
		logits, mem, aux_loss = self.net("online", states, mems)
		current_Q = logits[
			np.arange(0, self.batch_size), -1, action
		]  # Q_online(s,a)
		return current_Q, aux_loss

	@torch.no_grad()
	def td_target(self, reward, next_state, done):
		states, mems = zip(*next_state)
		max_state_len = max(len(s) for s in states)
		states = torch.from_numpy(np.array([[0] * (max_state_len - len(s)) + list(s) for s in states]))
		mems = self.net.empty_mem(states)
		next_state_Q = self.net("online", states, mems)[0][:, -1]
		best_action = torch.argmax(next_state_Q, axis=1)
		next_Q = self.net("target", states, mems)[0][
			np.arange(0, self.batch_size), -1, best_action
		]
		return (reward + (1 - done) * self.gamma * next_Q).float()

	def update_Q_online(self, td_estimate, td_target, aux_loss):
		loss = self.loss_fn(td_estimate, td_target) + aux_loss
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		return loss.item()

	def sync_Q_target(self):
		self.net.target.load_state_dict(self.net.online.state_dict())

	def save(self):
		save_path = (
			self.save_dir / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
		)
		torch.save(
			dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
			save_path,
		)
		print(f"MarioNet saved to {save_path} at step {self.curr_step}")
	
	def learn(self):
		if self.curr_step % self.sync_every == 0:
			self.sync_Q_target()

		# if self.curr_step % self.save_every == 0:
		# 	self.save()

		if self.curr_step < self.burnin:
			return None, None

		if self.curr_step % self.learn_every != 0:
			return None, None

		# Sample from memory
		state, next_state, action, reward, done = self.recall()

		# Get TD Estimate
		td_est, aux_loss = self.td_estimate(state, action)

		# Get TD Target
		td_tgt = self.td_target(reward, next_state, done)

		# Backpropagate loss through Q_online
		loss = self.update_Q_online(td_est, td_tgt, aux_loss)

		return (td_est.mean().item(), loss)

@dataclass
class Step:
	state: deque
	action: int
	reward: float
	next_state: deque
	done: bool
	
	def __getitem__(self, key):
		return (self.state, self.action, self.reward, self.next_state, self.done)[key]
	

class Environment:
	def __init__(self, reward_func, done_func, state=None):
		self.reward_func = reward_func
		self.done_func = done_func
		self.state = deque(maxlen=10) if state is None else state
		self.done = False
	
	def step(self, action) -> Step:
		if self.done:
			return Step(self.state, action, 0, self.state, True)
		prev_state = self.state.copy()
		self.state.append(action)
		reward = self.reward_func(self.state)
		self.done = self.done_func(self.state)
		return Step(prev_state, action, reward, self.state, self.done)

	def reset(self):
		self.state.clear()
		self.done = False
		return self.state


def loop():
	use_cuda = torch.cuda.is_available()
	print(f"Using CUDA: {use_cuda}")

	policy = Policy(net=Net(
		num_tokens=2,
		dim=10,
		heads=2,
		depth=2,
		seq_len=10,
		mem_len=10,
		cmem_len=2,
		cmem_ratio=5,
	))
	
	env = Environment(reward_func=lambda state: sum(state), done_func=lambda state: sum(state) == 10)

	episodes = 10
	for e in range(episodes):

		state = env.reset()

		cmem = None
		# Play the game!
		while True:

			# Run agent on the state
			action, mem = policy.act(state, cmem)

			# Agent performs action
			state, action, reward, next_state, done = env.step(action)

			# Remember
			policy.cache((state, cmem), (next_state, mem), action, reward, done)
			cmem = mem

			policy.curr_step += 1
			# Learn
			q, loss = policy.learn()
			print(q, loss)

			# Update state
			state = next_state

			# Check if end of game
			if done:
				break

loop()