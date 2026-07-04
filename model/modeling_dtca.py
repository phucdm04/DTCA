import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
import torchvision
import numpy as np
import os
from transformers import RobertaModel,BertModel,AlbertModel,ElectraModel,ViTModel,SwinModel,DeiTModel,ConvNextModel

# from torchcrf import CRF
from typing import List, Optional
class CRF(nn.Module):
    def __init__(self, num_tags: int, batch_first: bool = False) -> None:
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags: {num_tags}')
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize the transition parameters.

        The parameters will be initialized randomly from a uniform distribution
        between -0.1 and 0.1.
        """
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)
        nn.init.uniform_(self.transitions, -0.1, 0.1)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_tags={self.num_tags})'

    def forward(
            self,
            emissions: torch.Tensor,
            tags: torch.LongTensor,
            mask: Optional[torch.ByteTensor] = None,
            reduction: str = 'sum',
    ) -> torch.Tensor:
        self._validate(emissions, tags=tags, mask=mask)
        if reduction not in ('none', 'sum', 'mean', 'token_mean'):
            raise ValueError(f'invalid reduction: {reduction}')
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)

        # shape: (batch_size,)
        numerator = self._compute_score(emissions, tags, mask)
        # shape: (batch_size,)
        denominator = self._compute_normalizer(emissions, mask)
        # shape: (batch_size,)
        llh = numerator - denominator

        if reduction == 'none':
            return llh
        if reduction == 'sum':
            return llh.sum()
        if reduction == 'mean':
            return llh.mean()
        assert reduction == 'token_mean'
        return llh.sum() / mask.float().sum()

    def decode(self, emissions: torch.Tensor,
               mask: Optional[torch.ByteTensor] = None) -> List[List[int]]:
        self._validate(emissions, mask=mask)
        if mask is None:
            mask = emissions.new_ones(emissions.shape[:2], dtype=torch.uint8)

        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)

        return self._viterbi_decode(emissions, mask)

    def _validate(
            self,
            emissions: torch.Tensor,
            tags: Optional[torch.LongTensor] = None,
            mask: Optional[torch.ByteTensor] = None) -> None:
        if emissions.dim() != 3:
            raise ValueError(f'emissions must have dimension of 3, got {emissions.dim()}')
        if emissions.size(2) != self.num_tags:
            raise ValueError(
                f'expected last dimension of emissions is {self.num_tags}, '
                f'got {emissions.size(2)}')

        if tags is not None:
            if emissions.shape[:2] != tags.shape:
                raise ValueError(
                    'the first two dimensions of emissions and tags must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(tags.shape)}')

        if mask is not None:
            if emissions.shape[:2] != mask.shape:
                raise ValueError(
                    'the first two dimensions of emissions and mask must match, '
                    f'got {tuple(emissions.shape[:2])} and {tuple(mask.shape)}')
            no_empty_seq = not self.batch_first and mask[0].all()
            no_empty_seq_bf = self.batch_first and mask[:, 0].all()
            if not no_empty_seq and not no_empty_seq_bf:
                raise ValueError('mask of the first timestep must all be on')

    def _compute_score(
            self, emissions: torch.Tensor, tags: torch.LongTensor,
            mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # tags: (seq_length, batch_size)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and tags.dim() == 2
        assert emissions.shape[:2] == tags.shape
        assert emissions.size(2) == self.num_tags
        assert mask.shape == tags.shape
        assert mask[0].all()

        seq_length, batch_size = tags.shape
        mask = mask.float()

        # Start transition score and first emission
        # shape: (batch_size,)
        score = self.start_transitions[tags[0]]
        score += emissions[0, torch.arange(batch_size), tags[0]]

        for i in range(1, seq_length):
            # Transition score to next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += self.transitions[tags[i - 1], tags[i]] * mask[i]

            # Emission score for next tag, only added if next timestep is valid (mask == 1)
            # shape: (batch_size,)
            score += emissions[i, torch.arange(batch_size), tags[i]] * mask[i]

        # End transition score
        # shape: (batch_size,)
        seq_ends = mask.long().sum(dim=0) - 1
        # shape: (batch_size,)
        last_tags = tags[seq_ends, torch.arange(batch_size)]
        # shape: (batch_size,)
        score += self.end_transitions[last_tags]

        return score

    def _compute_normalizer(
            self, emissions: torch.Tensor, mask: torch.ByteTensor) -> torch.Tensor:
        # emissions: (seq_length, batch_size, num_tags)
        # mask: (seq_length, batch_size)
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length = emissions.size(0)

        # Start transition score and first emission; score has size of
        # (batch_size, num_tags) where for each batch, the j-th column stores
        # the score that the first timestep has tag j
        # shape: (batch_size, num_tags)
        score = self.start_transitions + emissions[0]

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            next_score = torch.logsumexp(next_score, dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)

        score += self.end_transitions
        return torch.logsumexp(score, dim=1)

    def _viterbi_decode(self, emissions: torch.FloatTensor,
                        mask: torch.ByteTensor) -> List[List[int]]:
        assert emissions.dim() == 3 and mask.dim() == 2
        assert emissions.shape[:2] == mask.shape
        assert emissions.size(2) == self.num_tags
        assert mask[0].all()

        seq_length, batch_size = mask.shape
        score = self.start_transitions + emissions[0]
        history = []

        for i in range(1, seq_length):
            broadcast_score = score.unsqueeze(2)
            broadcast_emission = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emission
            next_score, indices = next_score.max(dim=1)
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
            history.append(indices)

        score += self.end_transitions
        seq_ends = mask.long().sum(dim=0) - 1
        best_tags_list = []

        for idx in range(batch_size):
            _, best_last_tag = score[idx].max(dim=0)
            best_tags = [best_last_tag.item()]

            for hist in reversed(history[:seq_ends[idx]]):
                best_last_tag = hist[idx][best_tags[-1]]
                best_tags.append(best_last_tag.item())

            # Reverse the order because we start from the last timestep
            best_tags.reverse()
            best_tags_list.append(best_tags)

        return best_tags_list



class DTCAModel(nn.Module):
    def __init__(self,config1,config2,text_num_labels,alpha,beta,text_model_name="roberta",image_model_name='vit'):
        super().__init__()
        if text_model_name == 'roberta':
            self.roberta = RobertaModel(config1,add_pooling_layer=False)
        elif text_model_name == 'bert':
            self.bert = BertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'albert':
            self.albert = AlbertModel(config1, add_pooling_layer=False)
        elif text_model_name == 'electra':
            self.electra = ElectraModel(config1)
        if image_model_name == 'vit':
            self.vit = ViTModel(config2)
        elif image_model_name == 'swin':
            self.swin = SwinModel(config2)
        elif image_model_name == 'deit':
            self.deit = DeiTModel(config2)
        elif image_model_name == 'convnext':
            self.convnext = ConvNextModel(config2)
        self.alpha = alpha
        self.beta = beta
        self.text_model_name=text_model_name
        self.image_model_name=image_model_name
        self.config1 = config1
        self.config2 = config2
        self.text_num_labels = text_num_labels
        self.image_text_cross = MultiHeadAttention(8,config1.hidden_size,config1.hidden_size,config1.hidden_size)
        self.dropout = nn.Dropout(config1.hidden_dropout_prob)
        self.loss_fct = CrossEntropyLoss()
        self.classifier1 = nn.Linear(config1.hidden_size, self.text_num_labels)
        self.classifier0= nn.Linear(config1.hidden_size,self.text_num_labels)
        self.CRF = CRF(self.text_num_labels,batch_first=True)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                pixel_values=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                image_labels=None,
                head_mask=None,
                cross_labels=None,
                return_dict=None):
        return_dict = return_dict if return_dict is not None else self.config1.use_return_dict
        if self.text_model_name == 'bert':
            text_outputs = self.bert(input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict)
        elif self.text_model_name == 'roberta':
            text_outputs = self.roberta(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'albert':
            text_outputs = self.albert(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        elif self.text_model_name == 'electra':
            text_outputs = self.electra(input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            text_outputs=None
        if self.image_model_name == 'vit':
            image_outputs = self.vit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'swin':
            image_outputs = self.swin(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'deit':
            image_outputs = self.deit(pixel_values,head_mask=head_mask)
        elif self.image_model_name == 'convnext':
            image_outputs = self.convnext(pixel_values)
        else:
            image_outputs=None

        text_last_hidden_states = text_outputs["last_hidden_state"]
        image_last_hidden_states = image_outputs["last_hidden_state"]

        # cross_crf_loss
        image_text_cross_attention, _ = self.image_text_cross(text_last_hidden_states, image_last_hidden_states,
                                                              image_last_hidden_states)
        cross_logits = self.classifier0(image_text_cross_attention)
        mask = (labels != -100)
        mask[:,0] = 1
        # print(cross_logits.shape, cross_labels.shape)
        cross_crf_loss =  -self.CRF(cross_logits,cross_labels,mask=mask) / 10

        # word patch align
        batch_size, image_len, _ = image_last_hidden_states.shape
        _,text_len,_ = text_last_hidden_states.shape
        text_pad = (attention_mask == 1).clone().detach()
        image_pad = torch.zeros(batch_size, image_len,dtype=torch.bool, device=attention_mask.device)
        ot_dist =  optimal_transport_dist(text_last_hidden_states,image_last_hidden_states,text_pad,image_pad)
        word_region_align_loss = ot_dist.mean()

        # text_loss
        sequence_output1 = self.dropout(text_last_hidden_states)
        text_token_logits = self.classifier1(sequence_output1)

        # word_region_align_loss = ot_dist.masked_select(targets == 0)
        # getTextLoss: CrossEntropy
        text_loss = self.loss_fct(text_token_logits.view(-1, self.text_num_labels), labels.view(-1))
        loss =  cross_crf_loss + self.beta * word_region_align_loss + self.alpha * text_loss
        # loss = text_loss

        # end train
        return {"loss":loss,
            "logits":text_token_logits,
            "cross_logits": cross_logits,
                }


class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, dropout2=False, attn_type='softmax'):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        if dropout2:
            # self.dropout2 = nn.Dropout(dropout2)
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout2)
        else:
            self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_type=attn_type,
                                                       dropout=dropout)

        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(d_model)

        if n_head > 1:
            self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
            nn.init.xavier_normal_(self.fc.weight)

    def forward(self, q, k, v, attn_mask=None, dec_self=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        if hasattr(self, 'dropout2'):
            q = self.dropout2(q)

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        if attn_mask is not None:
            attn_mask = attn_mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output, attn = self.attention(q, k, v, attn_mask=attn_mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        if hasattr(self, 'fc'):
            output = self.fc(output)

        if hasattr(self, 'dropout'):
            output = self.dropout(output)

        if dec_self:
            output = self.layer_norm(output + residual)
        else:
            output = self.layer_norm(output + residual)

        return output, attn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1, attn_type='softmax'):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)
        if attn_type == 'softmax':
            self.attn_type = nn.Softmax(dim=2)
            # self.softmax = BottleSoftmax()
        else:
            self.attn_type = nn.Sigmoid()

    def forward(self, q, k, v, attn_mask=None, stop_sig=False):
        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if attn_mask is not None:
            # attn = attn.masked_fill(attn_mask, -np.inf)
            attn = attn.masked_fill(attn_mask, -1e6)

        if stop_sig:
            print('**')
            stop()

        attn = self.attn_type(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


def position_encoding_init(n_position, d_pos_vec):
    ''' Init the sinusoid position encoding table '''

    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos_vec) for j in range(d_pos_vec)]
        if pos != 0 else np.zeros(d_pos_vec) for pos in range(n_position)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


def distant_cross_entropy(logits, positions, mask=None):
    '''
    :param logits: [N, L]
    :param positions: [N, L]
    :param mask: [N]
    '''
    log_softmax = nn.LogSoftmax(dim=-1)
    log_probs = log_softmax(logits)
    if mask is not None:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               (torch.sum(positions.to(dtype=log_probs.dtype), dim=-1) + mask.to(dtype=log_probs.dtype)))
    else:
        loss = -1 * torch.mean(torch.sum(positions.to(dtype=log_probs.dtype) * log_probs, dim=-1) /
                               torch.sum(positions.to(dtype=log_probs.dtype), dim=-1))
    return loss



def cost_matrix_cosine(x, y, eps=1e-5):
    """ Compute cosine distnace across every pairs of x, y (batched)
    [B, L_x, D] [B, L_y, D] -> [B, Lx, Ly]"""
    assert x.dim() == y.dim()
    assert x.size(0) == y.size(0)
    assert x.size(2) == y.size(2)
    x_norm = F.normalize(x, p=2, dim=-1, eps=eps)
    y_norm = F.normalize(y, p=2, dim=-1, eps=eps)
    cosine_sim = x_norm.matmul(y_norm.transpose(1, 2))
    cosine_dist = 1 - cosine_sim
    return cosine_dist


def trace(x):
    """ compute trace of input tensor (batched) """
    b, m, n = x.size()
    assert m == n
    mask = torch.eye(n, dtype=torch.bool, device=x.device
                     ).unsqueeze(0).expand_as(x)
    trace = x.masked_select(mask).contiguous().view(
        b, n).sum(dim=-1, keepdim=False)
    return trace


@torch.no_grad()
def ipot(C, x_len, x_pad, y_len, y_pad, joint_pad, beta, iteration, k):
    """ [B, M, N], [B], [B, M], [B], [B, N], [B, M, N]"""
    b, m, n = C.size()
    sigma = torch.ones(b, m, dtype=C.dtype, device=C.device
                       ) / x_len.unsqueeze(1)
    T = torch.ones(b, n, m, dtype=C.dtype, device=C.device)
    A = torch.exp(-C.transpose(1, 2)/beta)

    # mask padded positions
    sigma.masked_fill_(x_pad, 0)
    joint_pad = joint_pad.transpose(1, 2)
    T.masked_fill_(joint_pad, 0)
    A.masked_fill_(joint_pad, 0)

    # broadcastable lengths
    x_len = x_len.unsqueeze(1).unsqueeze(2)
    y_len = y_len.unsqueeze(1).unsqueeze(2)

    # mask to zero out padding in delta and sigma
    x_mask = (x_pad.to(C.dtype) * 1e4).unsqueeze(1)
    y_mask = (y_pad.to(C.dtype) * 1e4).unsqueeze(1)

    for _ in range(iteration):
        Q = A * T  # bs * n * m
        sigma = sigma.view(b, m, 1)
        for _ in range(k):
            delta = 1 / (y_len * Q.matmul(sigma).view(b, 1, n) + y_mask)
            sigma = 1 / (x_len * delta.matmul(Q) + x_mask)
        T = delta.view(b, n, 1) * Q * sigma
    T.masked_fill_(joint_pad, 0)
    return T


def optimal_transport_dist(txt_emb, img_emb, txt_pad, img_pad,
                           beta=0.5, iteration=50, k=1):
    """ [B, M, D], [B, N, D], [B, M], [B, N]"""
    cost = cost_matrix_cosine(txt_emb, img_emb)
    # mask the padded inputs
    joint_pad = txt_pad.unsqueeze(-1) | img_pad.unsqueeze(-2)
    cost.masked_fill_(joint_pad, 0)

    txt_len = (txt_pad.size(1) - txt_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)
    img_len = (img_pad.size(1) - img_pad.sum(dim=1, keepdim=False)
               ).to(dtype=cost.dtype)

    T = ipot(cost.detach(), txt_len, txt_pad, img_len, img_pad, joint_pad,
             beta, iteration, k)
    distance = trace(cost.matmul(T.detach()))
    return distance





