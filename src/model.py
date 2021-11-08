import numpy as np
import torch
import torch.cuda
from torch import nn

from utils import Embeddings, BeatPositionalEncoding

from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask

D_MODEL = 512
N_LAYER_ENCODER = 12
N_HEAD = 8

ATTN_DECODER = "causal-linear"


################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    logits -= np.max(logits)
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= (sum(probs) + 1e-10)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    try:
        word = np.random.choice(sorted_index, size=1, p=sorted_probs)[0]
    except:
        word = sorted_index[0]
    return word


# -- nucleus -- #
def nucleus(probs, p):
    probs /= (sum(probs) + 1e-5)
    sorted_probs = np.sort(probs)[::-1]
    sorted_index = np.argsort(probs)[::-1]
    cusum_sorted_probs = np.cumsum(sorted_probs)
    after_threshold = cusum_sorted_probs > p
    if sum(after_threshold) > 0:
        last_index = np.where(after_threshold)[0][0] + 1
        candi_index = sorted_index[:last_index]
    else:
        candi_index = sorted_index[:]
    candi_probs = [probs[i] for i in candi_index]
    candi_probs /= sum(candi_probs)
    word = np.random.choice(candi_index, size=1, p=candi_probs)[0]
    return word


def sampling(logit, p=None, t=1.0):
    logit = logit.squeeze().cpu().numpy()
    probs = softmax_with_temperature(logits=logit, temperature=t)

    if p is not None:
        cur_word = nucleus(probs, p=p)
    else:
        cur_word = weighted_sampling(probs)
    return cur_word


'''
last dimension of input data | attribute:
0: bar/beat
1: type
2: density
3: pitch
4: duration
5: instr
6: strength onset_density
7: time_encoding
'''


class CMT(nn.Module):
    def __init__(self, n_token, init_n_token, is_training=True):
        super(CMT, self).__init__()

        print("D_MODEL", D_MODEL, " N_LAYER", N_LAYER_ENCODER, " N_HEAD", N_HEAD, "DECODER ATTN", ATTN_DECODER)

        # --- params config --- #
        self.n_token = n_token
        self.d_model = D_MODEL
        self.n_layer_encoder = N_LAYER_ENCODER  #
        # self.n_layer_decoder = N_LAYER_DECODER
        self.dropout = 0.1
        self.n_head = N_HEAD  #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        # self.emb_sizes = [64, 32, 512, 128, 32]
        self.emb_sizes = [64, 32, 64, 512, 128, 32, 64]

        self.init_n_token = init_n_token  # genre, key, instrument
        self.init_emb_sizes = [64, 64, 64]
        self.time_encoding_size = 256
        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)

        self.init_emb_genre = Embeddings(self.init_n_token[0], self.init_emb_sizes[0])
        self.init_emb_key = Embeddings(self.init_n_token[1], self.init_emb_sizes[1])
        self.init_emb_instrument = Embeddings(self.init_n_token[2], self.init_emb_sizes[2])
        self.init_in_linear = nn.Linear(int(np.sum(self.init_emb_sizes)), self.d_model)

        self.encoder_emb_barbeat = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.encoder_emb_type = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.encoder_emb_beat_density = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.encoder_emb_pitch = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.encoder_emb_duration = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.encoder_emb_instr = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.encoder_emb_onset_density = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.encoder_emb_time_encoding = Embeddings(self.n_token[7], self.time_encoding_size)
        self.encoder_pos_emb = BeatPositionalEncoding(self.d_model, self.dropout)

        # # linear
        self.encoder_in_linear = nn.Linear(int(np.sum(self.emb_sizes)), self.d_model)
        self.encoder_time_linear = nn.Linear(int(self.time_encoding_size), self.d_model)

        self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
            n_layers=self.n_layer_encoder,
            n_heads=self.n_head,
            query_dimensions=self.d_model // self.n_head,
            value_dimensions=self.d_model // self.n_head,
            feed_forward_dimensions=2048,
            activation='gelu',
            dropout=0.1,
            attention_type="causal-linear",
        ).get()

        # blend with type
        self.project_concat_type = nn.Linear(self.d_model + 32, self.d_model)

        # individual output
        self.proj_barbeat = nn.Linear(self.d_model, self.n_token[0])
        self.proj_type = nn.Linear(self.d_model, self.n_token[1])
        self.proj_beat_density = nn.Linear(self.d_model, self.n_token[2])
        self.proj_pitch = nn.Linear(self.d_model, self.n_token[3])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[4])
        self.proj_instr = nn.Linear(self.d_model, self.n_token[5])
        self.proj_onset_density = nn.Linear(self.d_model, self.n_token[6])

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    def forward_init_token_vis(self, x, memory=None, is_training=True):
        emb_genre = self.init_emb_genre(x[..., 0])
        emb_key = self.init_emb_key(x[..., 1])
        emb_instrument = self.init_emb_instrument(x[..., 2])
        return emb_genre, emb_key, emb_instrument

    def forward_init_token(self, x, memory=None, is_training=True):
        emb_genre = self.init_emb_genre(x[..., 0])
        emb_key = self.init_emb_key(x[..., 1])
        emb_instrument = self.init_emb_instrument(x[..., 2])
        embs = torch.cat(
            [
                emb_genre,
                emb_key,
                emb_instrument,
            ], dim=-1)
        encoder_emb_linear = self.init_in_linear(embs)
        if is_training:
            return encoder_emb_linear
        else:
            pos_emb = encoder_emb_linear.squeeze(0)
            h, memory = self.transformer_encoder(pos_emb, memory=memory)
            y_type = self.proj_type(h)
            return h, y_type, memory

    def forward_hidden(self, x, memory=None, is_training=True, init_token=None):
        # linear transformer: b, s, f   x.shape=(bs, nf)

        # embeddings
        emb_barbeat = self.encoder_emb_barbeat(x[..., 0])
        emb_type = self.encoder_emb_type(x[..., 1])
        emb_beat_density = self.encoder_emb_beat_density(x[..., 2])
        emb_pitch = self.encoder_emb_pitch(x[..., 3])
        emb_duration = self.encoder_emb_duration(x[..., 4])
        emb_instr = self.encoder_emb_instr(x[..., 5])
        emb_onset_density = self.encoder_emb_onset_density(x[..., 6])
        emb_time_encoding = self.encoder_emb_time_encoding(x[..., 7])

        embs = torch.cat(
            [
                emb_barbeat,
                emb_type,
                emb_beat_density,
                emb_pitch,
                emb_duration,
                emb_instr,
                emb_onset_density
            ], dim=-1)

        encoder_emb_linear = self.encoder_in_linear(embs)
        # import ipdb;ipdb.set_trace()
        encoder_emb_time_linear = self.encoder_time_linear(emb_time_encoding)
        encoder_emb_linear = encoder_emb_linear + encoder_emb_time_linear
        encoder_pos_emb = self.encoder_pos_emb(encoder_emb_linear, x[:, :, 8])

        if is_training:
            assert init_token is not None
            init_emb_linear = self.forward_init_token(init_token)
            encoder_pos_emb = torch.cat([init_emb_linear, encoder_pos_emb], dim=1)
        else:
            assert init_token is not None
            init_emb_linear = self.forward_init_token(init_token)
            encoder_pos_emb = torch.cat([init_emb_linear, encoder_pos_emb], dim=1)
        # transformer
        if is_training:
            attn_mask = TriangularCausalMask(encoder_pos_emb.size(1), device=x.device)
            encoder_hidden = self.transformer_encoder(encoder_pos_emb, attn_mask)
            # print("forward decoder done")
            y_type = self.proj_type(encoder_hidden[:, 7:, :])
            return encoder_hidden, y_type

        else:
            encoder_mask = TriangularCausalMask(encoder_pos_emb.size(1), device=x.device)
            h = self.transformer_encoder(encoder_pos_emb, encoder_mask)  # y: s x d_model
            h = h[:, -1:, :]
            h = h.squeeze(0)
            y_type = self.proj_type(h)

            return h, y_type

    def forward_output(self, h, y):
        # for training
        tf_skip_type = self.encoder_emb_type(y[..., 1])
        h = h[:, 7:, :]
        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        y_barbeat = self.proj_barbeat(y_)
        y_beat_density = self.proj_beat_density(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_instr = self.proj_instr(y_)
        y_onset_density = self.proj_onset_density(y_)
        # import ipdb;ipdb.set_trace()

        return y_barbeat, y_pitch, y_duration, y_instr, y_onset_density, y_beat_density

    def forward_output_sampling(self, h, y_type, recurrent=True):
        '''
        for inference
        '''
        y_type_logit = y_type[0, :]  # dont know wtf
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(
            np.array([cur_word_type])).long().unsqueeze(0)

        if torch.cuda.is_available():
            type_word_t = type_word_t.cuda()

        tf_skip_type = self.encoder_emb_type(type_word_t).squeeze(0)

        # concat
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        # project other
        y_barbeat = self.proj_barbeat(y_)

        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_instr = self.proj_instr(y_)
        y_onset_density = self.proj_onset_density(y_)
        y_beat_density = self.proj_beat_density(y_)

        # sampling gen_cond
        cur_word_barbeat = sampling(y_barbeat, t=1.2)
        cur_word_pitch = sampling(y_pitch, p=0.9)
        cur_word_duration = sampling(y_duration, t=2, p=0.9)
        cur_word_instr = sampling(y_instr, p=0.90)
        cur_word_onset_density = sampling(y_onset_density, p=0.90)
        cur_word_beat_density = sampling(y_beat_density, p=0.90)

        # collect
        next_arr = np.array([
            cur_word_barbeat,
            cur_word_type,
            cur_word_beat_density,
            cur_word_pitch,
            cur_word_duration,
            cur_word_instr,
            cur_word_onset_density,
        ])
        return next_arr

    def inference_from_scratch(self, **kwargs):
        vlog = kwargs['vlog']
        C = kwargs['C']

        def get_p_beat(cur_bar, cur_beat, n_beat):
            all_beat = cur_bar * 16 + cur_beat - 1
            p_beat = round(all_beat / n_beat * 100) + 1
            return p_beat

        dictionary = {'bar': 17}
        strength_track_list = [1, 2, 3]

        pre_init = np.array([
            [5, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 0, 3],
            [0, 0, 4],
            [0, 0, 5],
        ])
        init = np.array([
            [17, 1, vlog[0][1], 0, 0, 0, 0, 1, 0],  # bar
        ])

        with torch.no_grad():
            final_res = []
            h = None

            init_t = torch.from_numpy(init).long()
            pre_init = torch.from_numpy(pre_init).long().unsqueeze(0)
            if torch.cuda.is_available():
                pre_init = pre_init.cuda()
                init_t = init_t.cuda()

            print('------ initiate ------')
            for step in range(init.shape[0]):
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                print(input_)
                final_res.append(init[step, :][None, ...])
                h, y_type = self.forward_hidden(input_, is_training=False, init_token=pre_init)

            print('------- condition -------')
            assert vlog is not None
            n_beat = vlog[0][4]
            len_vlog = len(vlog)
            cur_vlog = 1
            cur_track = 0
            idx = 0
            acc_beat_num = vlog[0][1]
            beat_num = {}
            acc_note_num = 0
            note_num = 0
            err_note_number_list = []
            err_beat_number_list = []
            p_beat = 1
            cur_bar = 0
            while (True):
                # sample others
                print(idx, end="\r")
                idx += 1
                next_arr = self.forward_output_sampling(h, y_type)
                if next_arr[1] == 1:
                    replace = False
                if next_arr[1] == 2 and next_arr[5] == 0:
                    next_arr[5] = 1
                    print("warning note with instrument 0 detected, replaced by drum###################")
                if cur_vlog >= len_vlog:
                    print("exceed vlog len")
                    break
                vlog_i = vlog[cur_vlog]
                if vlog_i[0] == dictionary['bar'] and next_arr[0] == dictionary['bar']:
                    err_beat_number = np.abs(len(beat_num.keys()) - acc_beat_num)
                    err_beat_number_list.append(err_beat_number)
                    flag = (np.random.rand() < C)
                    print("replace beat density-----", vlog_i, next_arr)
                    if flag:
                        next_arr = np.array([17, 1, vlog_i[1], 0, 0, 0, 0])
                        print("replace beat density-----", next_arr)
                        beat_num = {}
                        acc_beat_num = vlog_i[1]
                        replace = True
                        cur_vlog += 1
                    else:
                        print("replace denied----")
                        cur_vlog += 1
                elif vlog_i[0] < dictionary['bar'] and next_arr[0] >= vlog_i[0]:
                    err_note_number = np.abs(acc_note_num - note_num)
                    err_note_number_list.append(err_note_number)
                    print("replace onset density----", vlog_i, next_arr)
                    if cur_track == 0:
                        cur_density = next_arr[2]
                        flag = (np.random.rand() < C)
                        if next_arr[0] == dictionary['bar']:
                            cur_density = 1

                    next_arr = np.array(
                        [vlog_i[0], 1, cur_density, 0, 0, strength_track_list[cur_track], vlog_i[2] + 0])
                    replace = True
                    acc_note_num = vlog_i[2] + 0
                    note_num = 0
                    cur_track += 1
                    if cur_track >= len(strength_track_list):
                        cur_track = 0
                        cur_vlog += 1

                if next_arr[1] == 1:
                    beat_num[next_arr[0]] = 1
                elif next_arr[1] == 2 and replace == True:
                    note_num += 1

                if next_arr[0] == dictionary['bar']:
                    cur_bar += 1
                if next_arr[1] == 1:
                    if next_arr[0] == 17:
                        cur_beat = 1
                    else:
                        cur_beat = next_arr[0]
                    p_beat = get_p_beat(cur_bar, cur_beat, n_beat)
                if p_beat >= 102:
                    print("exceed max p_beat----")
                    break
                next_arr = np.concatenate([next_arr, [p_beat], [cur_bar * 16 + cur_beat - 1]])
                final_res.append(next_arr[None, ...])
                print(next_arr)
                # forward
                input_cur = torch.from_numpy(next_arr).long().unsqueeze(0).unsqueeze(0)
                if torch.cuda.is_available():
                    input_cur = input_cur.cuda()
                input_ = torch.cat((input_, input_cur), dim=1)
                if replace:
                    h, y_type = self.forward_hidden(input_, is_training=False, init_token=pre_init)
                else:
                    h, y_type = self.forward_hidden(input_, is_training=False, init_token=pre_init)
                if next_arr[1] == 0:
                    print("EOS predicted")
                    break

        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res, err_note_number_list, err_beat_number_list


    def train_forward(self, **kwargs):
        x = kwargs['x']
        target = kwargs['target']
        loss_mask = kwargs['loss_mask']
        init_token = kwargs['init_token']
        h, y_type = self.forward_hidden(x, memory=None, is_training=True, init_token=init_token)
        y_barbeat, y_pitch, y_duration, y_instr, y_onset_density, y_beat_density = self.forward_output(h, target)

        # reshape (b, s, f) -> (b, f, s)
        y_barbeat = y_barbeat[:, ...].permute(0, 2, 1)
        y_type = y_type[:, ...].permute(0, 2, 1)
        y_pitch = y_pitch[:, ...].permute(0, 2, 1)
        y_duration = y_duration[:, ...].permute(0, 2, 1)
        y_instr = y_instr[:, ...].permute(0, 2, 1)
        y_onset_density = y_onset_density[:, ...].permute(0, 2, 1)
        y_beat_density = y_beat_density[:, ...].permute(0, 2, 1)

        # loss
        loss_barbeat = self.compute_loss(
            y_barbeat, target[..., 0], loss_mask)
        loss_type = self.compute_loss(
            y_type, target[..., 1], loss_mask)
        loss_beat_density = self.compute_loss(
            y_beat_density, target[..., 2], loss_mask)
        loss_pitch = self.compute_loss(
            y_pitch, target[..., 3], loss_mask)
        loss_duration = self.compute_loss(
            y_duration, target[..., 4], loss_mask)
        loss_instr = self.compute_loss(
            y_instr, target[..., 5], loss_mask)
        loss_onset_density = self.compute_loss(
            y_onset_density, target[..., 6], loss_mask)

        return loss_barbeat, loss_type, loss_pitch, loss_duration, loss_instr, loss_onset_density, loss_beat_density

    def forward(self, **kwargs):
        if kwargs['is_train']:
            return self.train_forward(**kwargs)
        return self.inference_from_scratch(**kwargs)