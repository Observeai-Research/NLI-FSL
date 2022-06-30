import sys

sys.path.append('..')
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class MLMAN(nn.Module):

    def __init__(self, word_vec_mat, max_length, word_embedding_dim=50, pos_embedding_dim=5, args=None,
                 hidden_size=100, drop=True):
        nn.Module.__init__(self)
        self.word_embedding_dim = word_embedding_dim + 2 * pos_embedding_dim
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.embedding = embedding.Embedding(word_vec_mat, max_length, word_embedding_dim, pos_embedding_dim)
        self.args = args
        self.conv = nn.Conv2d(1, self.hidden_size*2, kernel_size=(3, self.word_embedding_dim), padding=(1, 0))
        self.proj = nn.Linear(self.hidden_size*8, self.hidden_size)
        self.lstm_enhance = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)

        self.multilayer = nn.Sequential(nn.Linear(self.hidden_size*8, self.hidden_size),
                                            nn.ReLU(),
                                            nn.Linear(self.hidden_size, 1))
        self.drop = drop
        self.dropout = nn.Dropout(0.2)
        self.cost = nn.CrossEntropyLoss()
        self.apply(self.weights_init)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            init_linear(m)
        elif classname.find('LSTM') != -1:
            init_lstm(m)
        # elif classname.find('Conv') != -1:
        #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #     m.weight.data.normal_(0, np.sqrt(2. / n))
        #     if m.bias is not None:
        #         m.bias.data.zero_()

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).float())

    def context_encoder(self, input):
        input_mask = (input['mask'] != 0).float()
        max_length = input_mask.long().sum(1).max().item()
        input_mask = input_mask[:, :max_length].contiguous()
        embedding = self.embedding(input)
        embedding_ = embedding[:, :max_length].contiguous()

        if self.drop:
            embedding_ = self.dropout(embedding_)

        conv_out = self.conv(embedding_.unsqueeze(1)).squeeze(3)
        conv_out = conv_out * input_mask.unsqueeze(1)

        return conv_out.transpose(1,2).contiguous(), input_mask, max_length

    def lstm_encoder(self, input, mask, lstm):
        if self.drop:
            input = self.dropout(input)
        mask = mask.squeeze(2)
        sequence_lengths = mask.long().sum(1)
        sorted_inputs, sorted_sequence_lengths, restoration_indices, _ = sort_batch_by_length(input, sequence_lengths)

        packed_sequence_input = pack_padded_sequence(sorted_inputs,
                                                     sorted_sequence_lengths,
                                                     batch_first=True)
        lstmout, _ = lstm(packed_sequence_input)
        unpacked_sequence_tensor, _ = pad_packed_sequence(lstmout, batch_first=True)
        unpacked_sequence_tensor = unpacked_sequence_tensor.index_select(0, restoration_indices)

        return unpacked_sequence_tensor


    def CoAttention(self, support, query, support_mask, query_mask):

        att = support @ query.transpose(1, 2)
        att = att + support_mask * query_mask.transpose(1, 2) * 100
        support_ = F.softmax(att, 2) @ query * support_mask
        query_ = F.softmax(att.transpose(1,2), 2) @ support * query_mask
        return support_, query_

    def local_matching(self, support, query, support_mask, query_mask):

        support_, query_ = self.CoAttention(support, query, support_mask, query_mask)
        enhance_query = self.fuse(query, query_, 2)
        enhance_support = self.fuse(support, support_, 2)

        return enhance_support, enhance_query

    def fuse(self, m1, m2, dim):
        return torch.cat([m1, m2, torch.abs(m1 - m2), m1 * m2], dim)

    def local_aggregation(self, enhance_support, enhance_query, support_mask, query_mask, K):

        max_enhance_query, _ = torch.max(enhance_query, 1)
        mean_enhance_query = torch.sum(enhance_query, 1) / torch.sum(query_mask, 1)
        enhance_query = torch.cat([max_enhance_query, mean_enhance_query], 1)

        enhance_support = enhance_support.view(enhance_support.size(0) // K, K, -1, self.hidden_size * 2)
        support_mask = support_mask.view(enhance_support.size(0), K, -1, 1)

        max_enhance_support, _ = torch.max(enhance_support, 2)
        mean_enhance_support = torch.sum(enhance_support, 2) / torch.sum(support_mask, 2)
        enhance_support = torch.cat([max_enhance_support, mean_enhance_support], 2)

        return enhance_support, enhance_query

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        '''

        support, support_mask, support_len = self.context_encoder(support)
        query, query_mask, query_len = self.context_encoder(query)

        batch = support.size(0)//(N*K)

        # concate S_k operation
        support = support.view(batch, 1, N, K, support_len, self.hidden_size*2).expand(batch, N*Q, N, K, support_len, self.hidden_size*2).contiguous().view(batch*N*Q*N, K*support_len, self.hidden_size*2)
        support_mask = support_mask.view(batch, 1, N, K, support_len).expand(batch, N*Q, N, K, support_len).contiguous().view(-1, K*support_len, 1)
        query = query.view(batch, N*Q, 1, query_len, self.hidden_size*2).expand(batch, N*Q, N, query_len, self.hidden_size*2).contiguous().view(batch*N*Q*N, query_len, self.hidden_size*2)
        query_mask = query_mask.view(batch, N*Q, 1, query_len).expand(batch, N*Q, N, query_len).contiguous().view(-1, query_len, 1)

        enhance_support, enhance_query = self.local_matching(support, query, support_mask, query_mask)

        # reduce dimensionality
        enhance_support = self.proj(enhance_support)
        enhance_query = self.proj(enhance_query)
        enhance_support = torch.relu(enhance_support)
        enhance_query = torch.relu(enhance_query)

        # split operation
        enhance_support = enhance_support.view(batch*N*Q*N*K, support_len, self.hidden_size)
        support_mask = support_mask.view(batch*N*Q*N*K, support_len, 1)

        # LSTM
        enhance_support = self.lstm_encoder(enhance_support, support_mask, self.lstm_enhance)
        enhance_query = self.lstm_encoder(enhance_query, query_mask, self.lstm_enhance)

        # Local aggregation

        enhance_support, enhance_query = self.local_aggregation(enhance_support, enhance_query, support_mask, query_mask, K)

        tmp_query = enhance_query.unsqueeze(1).repeat(1, K, 1)
        cat_seq = torch.cat([tmp_query, enhance_support], 2)
        beta = self.multilayer(cat_seq)
        one_enhance_support = (enhance_support.transpose(1, 2) @ F.softmax(beta, 1)).squeeze(2)

        J_incon = torch.sum((one_enhance_support.unsqueeze(1) - enhance_support) ** 2, 2).mean()

        cat_seq = torch.cat([enhance_query, one_enhance_support], 1)
        logits = self.multilayer(cat_seq)

        logits = logits.view(batch*N*Q, N)
        _, pred = torch.max(logits, 1)

        return logits, pred, J_incon


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import warnings




class Encoder(nn.Module):
	"""
			Multi-head attention encoder
	"""

	def __init__(self, config, embedding,input_channels=1):
		super(Encoder, self).__init__()

		# Generic parameters
		self.device = config['device']

		self.drop = nn.Dropout(0.2)
		emb_layer,num_emb, emb_dim = create_emb_layer(embedding)
		self.embed = emb_layer
		self.hidden_size=config['hidden_size']
		self.r = config['r']

		self.d_a = config['d_a']
		self.bilstm = nn.LSTM(emb_dim, self.hidden_size, bidirectional=True, batch_first=True)
		self.ws1 = nn.Linear(self.hidden_size * 2, self.d_a, bias=False)
		self.ws2 = nn.Linear(self.d_a, self.r, bias=False)
		self.tanh = nn.Tanh()
		self.softmax = nn.Softmax(dim=1)
	
	def forward(self,input, len, mask,segmented_input = None, segment_id=None, segmented_len = None, segmented_mask = None):
		warnings.filterwarnings('ignore')
		
		#Pack padded sequence process
		sorted_inputs, sorted_sequence_lengths, restoration_indices,permut_index = sort_batch_by_length(input, len, self.device)
		embedded_inputs = self.embed(sorted_inputs)

		sorted_mask = mask.index_select(0, permut_index)
		packed_emb = pack_padded_sequence(embedded_inputs, sorted_sequence_lengths, batch_first=True)

		#Initialize hidden states
		h_0 = Variable(torch.zeros(2, input.shape[0], self.hidden_size))
		c_0 = Variable(torch.zeros(2, input.shape[0], self.hidden_size))
		h_0 = h_0.to(self.device)
		c_0 = c_0.to(self.device)
		
		outp = self.bilstm(packed_emb, (h_0, c_0))[0] ## [bsz, len, d_h * 2]
		mod_outp = pad_packed_sequence(outp)[0].transpose(0,1).contiguous()
		mod_outp = mod_outp.index_select(0, restoration_indices)
		mod_outp = mod_outp.contiguous()
		noattn_rep = self.drop(mod_outp) #bsz, len, d_h

		# bsz, #seg, len, d_h	
		size = mod_outp.size()
		compressed_embeddings = mod_outp.view(-1, size[2])	# [bsz * len, d_h * 2]
		compressed_embeddings = self.drop(compressed_embeddings)
		hbar = self.tanh(self.ws2(self.ws1(compressed_embeddings)))
		alphas = hbar.view(size[0], size[1], -1)			# [bsz, len, hop]
		attention = torch.transpose(alphas, 1, 2)
		attention = attention.contiguous ()  # [bsz, hop, len]
		
		current_mask = torch.narrow(mask, dim=1, start=0, length=attention.shape[-1])
		repeated_mask = current_mask.unsqueeze(1).repeat(1,attention.shape[1], 1)
		masked_attention = masked_softmax(attention, repeated_mask,2) #[bsz, hop, len]
		
		multihead_rep = torch.bmm(masked_attention, mod_outp) #[bsz, hop,d_h*2]
		sentence_embedding = torch.sum(multihead_rep,1)/ self.r
		
		return sentence_embedding,masked_attention,multihead_rep,noattn_rep


#---Helper function -----#
def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor, device):
	"""
	Sort a batch first tensor by some specified lengths.
	Parameters
	----------
	tensor : torch.FloatTensor, required.
A batch first Pytorch tensor.
	sequence_lengths : torch.LongTensor, required.
A tensor representing the lengths of some dimension of the tensor which
we want to sort by.
	Returns
	------ 
	sorted_tensor : torch.FloatTensor
The original tensor sorted along the batch dimension with respect to sequence_lengths.
	sorted_sequence_lengths : torch.LongTensor
The original sequence_lengths sorted by decreasing size.
	restoration_indices : torch.LongTensor
Indices into the sorted_tensor such that
``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
	permuation_index : torch.LongTensor
The indices used to sort the tensor. This is useful if you want to sort many
tensors using the same ordering.
	"""

	sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)

	permutation_index = permutation_index.to(device)
	sorted_tensor = tensor.index_select(0, permutation_index)
	index_range = Variable(torch.arange(0, len(sequence_lengths)).long())
	index_range = index_range.to(device)
	# This is the equivalent of zipping with index, sorting by the original
	# sequence lengths and returning the now sorted indices.
	#_, reverse_mapping = permutation_index.sort(0, descending=False)
	
	_, reverse_mapping = permutation_index.sort(0, descending=False)
	restoration_indices = index_range.index_select(0, reverse_mapping)

	return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index

def create_emb_layer(weights_matrix, non_trainable=False):
	weights_matrix = torch.from_numpy(weights_matrix)
	num_embeddings, embedding_dim = weights_matrix.size()
	emb_layer = nn.Embedding(num_embeddings, embedding_dim)
	emb_layer.load_state_dict({'weight': weights_matrix})
	if non_trainable:
					emb_layer.weight.requires_grad = False

	return emb_layer, num_embeddings, embedding_dim


def masked_softmax(vec, mask, dim=1):
	masked_vec = vec * mask.float()
	max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
	exps = torch.exp(masked_vec-max_vec)
	masked_exps = exps * mask.float()
	masked_sums = masked_exps.sum(dim, keepdim=True)
	zeros=(masked_sums == 0)
	masked_sums += zeros.float()
	return masked_exps/masked_sums



