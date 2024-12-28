import math
import torch
from torch import nn


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))
        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))
        return y


class GLU(nn.Module):
    def __init__(self, input_size):
        super(GLU, self).__init__()
        self.fc1 = nn.Linear(input_size, input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_state_size, output_size, dropout, hidden_context_size=None, batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size = hidden_state_size
        self.dropout_val = dropout

        if self.input_size != self.output_size:
            self.skip_layer = TimeDistributed(
                nn.Linear(self.input_size, self.output_size), batch_first=batch_first)
        else:
            self.skip_layer = None

        self.fc1 = TimeDistributed(
            nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()

        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(
                self.hidden_context_size, self.hidden_state_size), batch_first=batch_first)
        else:
            self.context = None

        self.fc2 = TimeDistributed(
            nn.Linear(self.hidden_state_size,  self.output_size), batch_first=batch_first)
        self.elu2 = nn.ELU()

        self.dropout = nn.Dropout(self.dropout_val)
        self.bn = TimeDistributed(nn.BatchNorm1d(
            self.output_size), batch_first=batch_first)
        self.gate = TimeDistributed(
            GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.fc1(x)
        if context is not None and self.context is not None:
            context = self.context(context)
            x = x+context
        x = self.elu1(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x+residual
        x = self.bn(x)

        return x


class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context_size=None):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_inputs = num_inputs
        self.dropout_val = dropout
        self.context_size = context_size

        # Flattened GRN to get weights
        self.flattened_grn = GatedResidualNetwork(
            self.num_inputs*self.input_size,
            self.hidden_size,
            self.num_inputs,
            self.dropout_val,
            hidden_context_size=self.context_size
        )

        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(
                GatedResidualNetwork(
                    self.input_size, self.hidden_size, self.hidden_size, self.dropout_val)
            )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embedding, context=None):
        # embedding: (B,T,num_inputs*input_size)
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)

        # sparse weights: (B,T,num_inputs)
        sparse_weights = self.softmax(
            sparse_weights).unsqueeze(3)  # (B,T,num_inputs,1)

        var_outputs = []
        for i in range(self.num_inputs):
            start_idx = i*self.input_size
            end_idx = (i+1)*self.input_size
            var_i = embedding[:, :, start_idx:end_idx]  # (B,T,input_size)
            var_out = self.single_variable_grns[i](var_i)
            var_outputs.append(var_out)

        # (B,T,num_inputs,hidden_size)
        var_outputs = torch.stack(var_outputs, axis=2)

        outputs = var_outputs*sparse_weights  # (B,T,num_inputs,hidden_size)
        outputs = outputs.sum(axis=2)  # (B,T,hidden_size)

        return outputs, sparse_weights


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1,max_seq_len,d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B,T,d_model)
        seq_len = x.size(1)
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :seq_len, :]
        return x

#############################
# Modèle TFT adapté au problème de classification
#############################


class TFTClassifier(nn.Module):
    def __init__(self,
                 cat_dims,       # liste donnant le nb de catégories par variable catégorielle
                 num_numerical,  # nb de features numériques
                 num_classes=24,
                 embedding_size=8,   # taille des embeddings catégoriels
                 hidden_size=128,
                 lstm_layers=1,
                 dropout=0.1,
                 attn_heads=4,
                 max_seq_len=100):
        super().__init__()

        self.cat_dims = cat_dims
        self.num_cats = len(cat_dims)
        self.num_numerical = num_numerical
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers
        self.dropout_val = dropout
        self.attn_heads = attn_heads
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes

        # Embeddings pour les variables catégorielles
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(cat_dim, self.embedding_size) for cat_dim in cat_dims
        ])

        # Linear pour les numériques (on transforme chaque variable numérique en embedding_size)
        self.num_linear = nn.ModuleList([
            nn.Linear(1, self.embedding_size) for _ in range(num_numerical)
        ])

        # Dimension totale après concat des embeddings
        # total input per timestep = num_cats * embedding_size + num_numerical * embedding_size
        self.input_dim = self.num_cats*self.embedding_size + \
            self.num_numerical*self.embedding_size

        # Variable Selection Network
        # On applique le VSN à l'ensemble des variables à chaque timestep
        self.vsn = VariableSelectionNetwork(input_size=self.embedding_size,
                                            num_inputs=(
                                                self.num_cats+self.num_numerical),
                                            hidden_size=self.hidden_size,
                                            dropout=self.dropout_val)

        # LSTM Encoder
        self.lstm = nn.LSTM(input_size=self.hidden_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_layers,
                            dropout=(
                                self.dropout_val if self.lstm_layers > 1 else 0),
                            batch_first=True,
                            bidirectional=True)
        self.lstm_reduce = nn.Linear(self.hidden_size*2, self.hidden_size)

        self.post_lstm_gate = TimeDistributed(
            GLU(self.hidden_size), batch_first=True)
        self.post_lstm_norm = TimeDistributed(
            nn.BatchNorm1d(self.hidden_size), batch_first=True)

        self.static_enrichment = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, self.dropout_val, batch_first=True)

        # Positional encoding
        self.position_encoding = PositionalEncoder(
            d_model=self.hidden_size, max_seq_len=self.max_seq_len)

        # Multihead attention
        self.multihead_attn = nn.MultiheadAttention(
            self.hidden_size, self.attn_heads, dropout=self.dropout_val, batch_first=True)
        self.post_attn_gate = TimeDistributed(
            GLU(self.hidden_size), batch_first=True)
        self.post_attn_norm = TimeDistributed(
            nn.BatchNorm1d(self.hidden_size), batch_first=True)

        # Feed Forward
        self.pos_wise_ff = GatedResidualNetwork(
            self.hidden_size, self.hidden_size, self.hidden_size, self.dropout_val, batch_first=True)

        # Sortie
        self.pre_output_norm = TimeDistributed(
            nn.BatchNorm1d(self.hidden_size), batch_first=True)
        self.pre_output_gate = TimeDistributed(
            GLU(self.hidden_size), batch_first=True)
        self.output_layer = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        # x : (B,T,F) avec F = nb cat + nb num
        # Séparer cat et num
        B, T, F = x.size()

        cat_part = x[:, :, :self.num_cats].long()  # (B,T,num_cats)
        num_part = x[:, :, self.num_cats:].float()  # (B,T,num_num)

        # Embeddings catégoriels
        cat_emb_list = []
        for i, emb in enumerate(self.cat_embeddings):
            cat_emb_list.append(emb(cat_part[:, :, i]))
        cat_emb = torch.cat(cat_emb_list, dim=-1)  # (B,T,num_cats*emb_size)

        # Embeddings numériques
        num_emb_list = []
        for i, lin in enumerate(self.num_linear):
            # extraire la i-ème feature numérique et la passer dans lin
            feat = num_part[:, :, i].unsqueeze(-1)  # (B,T,1)
            num_emb_list.append(lin(feat))
        num_emb = torch.cat(num_emb_list, dim=-1)  # (B,T,num_num*emb_size)

        # Concat cat_emb et num_emb
        # (B,T,(num_cats+num_num)*emb_size)
        full_emb = torch.cat([cat_emb, num_emb], dim=-1)

        # VSN
        # Note : VSN attend (B,T,(num_inputs*input_size))
        # Ici, input_size = embedding_size
        # On a num_inputs = num_cats+num_num
        # On a déjà full_emb de dim (B,T,(num_cats+num_num)*emb_size), c'est correct.
        vsn_out, sparse_weights = self.vsn(full_emb)  # (B,T,hidden_size)

        # Ajouter encodage positionnel
        vsn_out = self.position_encoding(vsn_out)  # (B,T,hidden_size)

        # LSTM
        lstm_out, _ = self.lstm(vsn_out)  # (B,T,hidden_size*2)
        lstm_out = self.lstm_reduce(lstm_out)  # (B,T,hidden_size)

        # Skip-connection + normalisation
        lstm_out = self.post_lstm_gate(lstm_out + vsn_out)
        lstm_out = self.post_lstm_norm(lstm_out)

        # Pas de static embedding ici, on peut passer context=None
        attn_in = self.static_enrichment(lstm_out)  # (B,T,hidden_size)

        # Attention
        # multihead_attn batch_first=True => attn_in : (B,T,H)
        attn_out, attn_weights = self.multihead_attn(attn_in, attn_in, attn_in)
        attn_out = self.post_attn_gate(attn_out) + attn_in
        attn_out = self.post_attn_norm(attn_out)

        # Position-wise FF
        ff_out = self.pos_wise_ff(attn_out)
        ff_out = self.pre_output_gate(ff_out) + attn_out
        ff_out = self.pre_output_norm(ff_out)

        # Pooling sur la dimension temporelle (moyenne) pour la classification
        pooled = ff_out.mean(dim=1)  # (B,H)

        # Couche de sortie
        logits = self.output_layer(pooled)  # (B,num_classes)
        return logits
