import torch
import torch.nn.functional as F


def GreedyDecode(model, inputs, input_lengths):

    assert inputs.dim() == 3
    # f = [batch_size, time_step, feature_dim]
    f, _ = model.encoder(inputs, input_lengths)

    zero_token = torch.LongTensor([[3]])
    if inputs.is_cuda:
        zero_token = zero_token.cuda()
    results = []
    batch_size = inputs.size(0)

    def decode(inputs, lengths):
        token_list = []
        gu, hidden = model.decoder(zero_token)
        for t in range(lengths):
            h = model.joint(inputs[t].view(-1), gu.view(-1))
            out = h
            pred = torch.argmax(out, dim=-1)
            pred = int(pred)
            #print(pred == 0)
            if pred != 0 or True:
                token_list.append(pred)
                token = torch.LongTensor([[pred]])
                if zero_token.is_cuda:
                    token = token.cuda()
                gu, hidden = model.decoder(token, hidden_states=hidden)

        return token_list

    for i in range(batch_size):
        decoded_seq = decode(f[i], input_lengths[i])
        results.append(decoded_seq)

    return results