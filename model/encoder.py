import torch
import numpy as np

from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, transforms
from torch.autograd import Variable

class Encoder(nn.Module):
	def __init__(self, input_dim, embedding_dim):
		super().__init__()
		self.input_dim = input_dim
		self.embedding_dim = embedding_dim
		self.state_embed = nn.Sequential(nn.Linear(input_dim, embedding_dim), nn.LeakyReLU())
		self.key_encoder = nn.Linear(embedding_dim, embedding_dim, bias = False)
		self.query_encoder = nn.Linear(embedding_dim, embedding_dim, bias = False)
		self.value_encoder = nn.Linear(embedding_dim, embedding_dim, bias = False)

	def forward(self, ego_state, other_state):
		ego_embedding = self.state_embed(ego_state)
		other_embedding = self.state_embed(other_state)
		ego_query = self.query_encoder(ego_embedding) #1 x embedding_dim
		other_keys = self.key_encoder(other_embedding) #n x embedding_dim
		attention_dot = torch.dot(ego_query, torch.transpose(other_keys, -2, -1))/np.sqrt(self.embedding_dim)
		other_values = self.value_encoder(other_state)
		return attention_dot * other_values
