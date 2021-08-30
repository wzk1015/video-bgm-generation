from utils import *

from fast_transformers.builders import TransformerEncoderBuilder, TransformerDecoderBuilder
from fast_transformers.masking import TriangularCausalMask, FullMask
from fast_transformers.builders import RecurrentEncoderBuilder

# D_MODEL = 256
# N_LAYER_ENCODER = 4
# N_LAYER_DECODER = 8
# N_HEAD = 8

D_MODEL = 512
N_LAYER_ENCODER = 12
N_HEAD = 8

ATTN_DECODER = "causal-linear"

################################################################################
# Sampling
################################################################################
# -- temperature -- #
def softmax_with_temperature(logits, temperature):
    probs = np.exp(logits / temperature) / np.sum(np.exp(logits / temperature))
    return probs


def weighted_sampling(probs):
    probs /= (sum(probs)+ 1e-10)
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



class BaseModel(nn.Module):
    def __init__(self, n_token, is_training=True):
        super(BaseModel, self).__init__()

        print("D_MODEL", D_MODEL, " N_LAYER", N_LAYER_ENCODER, " N_HEAD", N_HEAD, "DECODER ATTN", ATTN_DECODER)

        # --- params config --- #
        self.n_token = n_token
        self.d_model = D_MODEL
        self.n_layer_encoder = N_LAYER_ENCODER  #
        self.dropout = 0.1
        self.n_head = N_HEAD  #
        self.d_head = D_MODEL // N_HEAD
        self.d_inner = 2048
        self.loss_func = nn.CrossEntropyLoss(reduction='none')
        self.emb_sizes = [64, 32, 64, 512, 128, 32, 64]

        # --- modules config --- #
        # embeddings
        print('>>>>>:', self.n_token)

        self.encoder_emb_barbeat = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.encoder_emb_type = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.encoder_emb_beat_density = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.encoder_emb_pitch = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.encoder_emb_duration = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.encoder_emb_instr = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.encoder_emb_onset_density = Embeddings(self.n_token[6], self.emb_sizes[6])
        self.encoder_pos_emb = PositionalEncoding(self.d_model, self.dropout)

        # # linear
        self.encoder_in_linear = nn.Linear(int(np.sum(self.emb_sizes)), self.d_model)


        print(' [o] using RNN backend.')
        self.transformer_encoder = self.get_encoder_builder().from_kwargs(
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

    def get_encoder_builder(self):
        raise NotImplementedError

    def get_decoder_builder(self):
        raise NotImplementedError

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss


    def forward_hidden(self, x, memory=None, is_training=True):
        # linear transformer: b, s, f   x.shape=(bs, nf)

        # embeddings
        emb_barbeat = self.encoder_emb_barbeat(x[..., 0])
        emb_type = self.encoder_emb_type(x[..., 1])
        emb_beat_density = self.encoder_emb_beat_density(x[..., 2])
        emb_pitch = self.encoder_emb_pitch(x[..., 3])
        emb_duration = self.encoder_emb_duration(x[..., 4])
        emb_instr = self.encoder_emb_instr(x[..., 5])
        emb_onset_density = self.encoder_emb_onset_density(x[..., 6])

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
        encoder_pos_emb = self.encoder_pos_emb(encoder_emb_linear)

        # transformer
        if is_training:
            attn_mask = TriangularCausalMask(encoder_pos_emb.size(1), device=x.device)
            encoder_hidden = self.transformer_encoder(encoder_pos_emb, attn_mask)
            # print("forward decoder done")
            y_type = self.proj_type(encoder_hidden)
            return encoder_hidden, y_type

        else:
            pos_emb = encoder_pos_emb.squeeze(0)
            h, memory = self.transformer_encoder(pos_emb, memory=memory)
            y_type = self.proj_type(h)
            return h, y_type, memory

    def forward_output(self, h, y):
        # for training
        tf_skip_type = self.encoder_emb_type(y[..., 1])

        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_ = self.project_concat_type(y_concat_type)

        y_barbeat = self.proj_barbeat(y_)
        y_beat_density = self.proj_beat_density(y_)
        y_pitch = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_instr = self.proj_instr(y_)
        y_onset_density = self.proj_onset_density(y_)

        return y_barbeat, y_pitch, y_duration, y_instr, y_onset_density, y_beat_density


    def forward_output_sampling(self, h, y_type, recurrent=True):
        '''
        for inference
        '''
        # sample type
        # print(y_type.shape)
        # exit(0)
        y_type_logit = y_type[0, :] # dont know wtf
        cur_word_type = sampling(y_type_logit, p=0.90)

        type_word_t = torch.from_numpy(
            np.array([cur_word_type])).long().unsqueeze(0)
        
        if torch.cuda.is_available():
            type_word_t = type_word_t.cuda()

        tf_skip_type = self.encoder_emb_type(type_word_t).squeeze(0)

        # print("y_type", y_type.shape, "h", h.shape, "tf_skip", tf_skip_type.shape,
        #       "cur word type", cur_word_type.shape, "type word t", type_word_t.shape)
        # exit(0)
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


    def inference_from_scratch(self, dictionary=None, condition=False, vlog=None, o_den_track_list=[1,2,3]):
        classes = ['bar-beat', 'type', 'pitch', 'duration', 'instr_type']
        import copy
        dictionary = {'bar':17}
        if vlog is None:
            init = np.array([
                [17, 1, 5, 0, 0, 0, 0],  # bar
            ])
        else:
            init = np.array([
                [17, 1, vlog[0][1], 0, 0, 0, 0],  # bar
            ])

        cnt_token = len(init)
        with torch.no_grad():
            final_res = []
            memory = None
            h = None

            cnt_bar = 1
            init_t = torch.from_numpy(init).long()
            if torch.cuda.is_available():
                init_t = init_t.cuda()
            print('------ initiate ------')
            for step in range(init.shape[0]):
                input_ = init_t[step, :].unsqueeze(0).unsqueeze(0)
                final_res.append(init[step, :][None, ...])
                print(input_)
                h, y_type, memory = self.forward_hidden(
                    input_, memory, is_training=False)

            if condition:
                print('------- condition -------')
                assert vlog is not None
                len_vlog = len(vlog)
                cur_vlog = 1
                cur_track = 0
                idx = 0
                while (True):
                    replace = False
                    # sample others
                    print(idx, end="\r")
                    idx += 1
                    next_arr = self.forward_output_sampling(h, y_type)
                    if next_arr[1] == 2 and next_arr[5] == 0:
                        next_arr[5] = 1
                        print("warning note with instrument 0 detected, replaced by drum###################")
                    if cur_vlog >= len_vlog:
                        print("exceed vlog len")
                        break
                    vlog_i = vlog[cur_vlog]
                    if vlog_i[0] == dictionary['bar'] and next_arr[0] == dictionary['bar']:
                        next_arr = np.array([17, 1, vlog_i[1], 0, 0, 0, 0])
                        print("replace beat density!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", next_arr)
                        cur_vlog += 1
                    elif vlog_i[0] < dictionary['bar'] and next_arr[0] >= vlog_i[0]:
                        print("replace onset density!!!!", vlog_i, next_arr)
                        if cur_track == 0:
                            cur_beat = next_arr[2]
                        next_arr = np.array([vlog_i[0], 1, cur_beat, 0, 0, o_den_track_list[cur_track], vlog_i[2]+1])
                        replace = True
                        cur_track += 1
                        if cur_track >= len(o_den_track_list):
                            cur_track = 0
                            cur_vlog += 1
                    final_res.append(next_arr[None, ...])
                    print(next_arr)
                    # forward
                    input_ = torch.from_numpy(next_arr).long().unsqueeze(0).unsqueeze(0)
                    if torch.cuda.is_available():
                        input_ = input_.cuda()
                    if replace:
                        h, y_type, memory = self.forward_hidden(input_, memory, is_training=False)
                    else:
                        h, y_type, memory = self.forward_hidden(input_, memory, is_training=False)
                    if next_arr[1] == 0:
                        print("EOS predicted")
                        break
                    # print(next_arr)
                    if len(final_res) > 15000:
                        print("exceed max len")
                        break
            else:
                print('------ generate ------')
                idx=0
                while (True):
                    # sample others
                    print(idx, end="\r")
                    idx += 1
                    next_arr = self.forward_output_sampling(h, y_type)
                    final_res.append(next_arr[None, ...])
                    print(next_arr)
                    # forward
                    input_ = torch.from_numpy(next_arr).long().unsqueeze(0).unsqueeze(0)
                    if torch.cuda.is_available():
                        input_ = input_.cuda()
                    h, y_type, memory = self.forward_hidden(input_, memory, is_training=False)
                    if next_arr[1] == 0:
                        print("EOS predicted")
                        break
                    if len(final_res) > 5000:
                        print("exceed max len")
                        break


        print('\n--------[Done]--------')
        final_res = np.concatenate(final_res)
        print(final_res.shape)
        return final_res

    # def inference_from_scratch(self, dictionary):


class ModelForTraining(BaseModel):
    def get_encoder_builder(self):
        return TransformerEncoderBuilder

    def get_decoder_builder(self):
        return TransformerDecoderBuilder

    def forward(self, x, target, loss_mask):
        # encoder_hidden = self.forward_encoder(x)
        # h, y_type = self.forward_decoder(x, memory=encoder_hidden, is_training=True)
        h, y_type = self.forward_hidden(x, memory=None, is_training=True)
        y_barbeat, y_pitch, y_duration, y_instr,  y_onset_density, y_beat_density = self.forward_output(h, target)

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


class ModelForInference(BaseModel):
    def get_encoder_builder(self):
        # return TransformerEncoderBuilder
        return RecurrentEncoderBuilder

    def get_decoder_builder(self):
        return TransformerDecoderBuilder
        # return RecurrentDecoderBuilder

    def forward(self, dic, condition=False, vlog=None, o_den_track_list=[1,2,3]):
        return self.inference_from_scratch(dic, condition=condition, vlog=vlog, o_den_track_list=o_den_track_list)

