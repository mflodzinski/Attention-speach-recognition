from pathlib import Path
from model import Model
from tokenizer import CharTokenizer
from utils import get_formated_date, load_stat_dict
from torch.optim import Optimizer
from data import DataLoader
from typing import Callable, Union
from torch.nn import Module
from functools import wraps
from hprams import hprams
from tqdm import tqdm
import torch
import os
from search import GreedyDecode
from warprnnt_pytorch import RNNTLoss

OPT = {"sgd": torch.optim.SGD}


def save_checkpoint(func) -> Callable:
    """Save a checkpoint after each iteration"""

    @wraps(func)
    def wrapper(obj, *args, **kwargs):
        result = func(obj, *args, **kwargs)
        if not os.path.exists(hprams.training.checkpoints_dir):
            os.mkdir(hprams.training.checkpoints_dir)
        timestamp = get_formated_date()
        model_path = os.path.join(hprams.training.checkpoints_dir, timestamp + ".pt")
        torch.save(obj.model.state_dict(), model_path)
        print(f"checkpoint saved to {model_path}")
        return result

    return wrapper


class Trainer:
    __train_loss_key = "train_loss"
    __test_loss_key = "test_loss"

    def __init__(
        self,
        criterion: Module,
        tokenizer: CharTokenizer,
        optimizer: Optimizer,
        model: Module,
        device: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        length_multiplier: float,
    ) -> None:
        self.criterion = criterion
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.epochs = epochs
        self.step_history = dict()
        self.history = dict()
        self.length_multiplier = length_multiplier

    def fit(self):
        """The main training loop that train the model on the training
        data then test it on the test set and then log the results
        """
        for _ in range(self.epochs):
            self.train()
            #self.test()
            self.print_results()

    def set_train_mode(self) -> None:
        """Set the models on the training mood"""
        self.model = self.model.train()

    def set_test_mode(self) -> None:
        """Set the models on the testing mood"""
        self.model = self.model.eval()

    def print_results(self):
        """Prints the results after each epoch"""
        result = ""
        for key, value in self.history.items():
            result += f"{key}: {str(value[-1])}, "
        print(result[:-2])

    def test(self):
        """Iterate over the whole test data and test the models
        for a single epoch
        """
        total_loss = 0
        self.set_test_mode()
        for x, x_len, y, ylen in tqdm(self.train_loader):
            x = x.to(self.device); x_len = x_len.to(self.device)
            #print(self.tokenizer.ids2tokens(y.tolist()))
            search = GreedyDecode(self.model, x, x_len) 
            search = self.tokenizer.ids2tokens(search)
            words = ["".join(a) for a in search]
            for i in words:
                print(i)
            # y = y.to(self.device)
            # max_len = int(x.shape[1] * self.length_multiplier)
            # x = torch.squeeze(x, dim=1)
            # result = self.model(x, max_len)
            # # result = result.reshape(-1, result.shape[-1])
            # # y = y.reshape(-1)
            # # y = torch.squeeze(y)
            # print(self.tokenizer.ids2tokens(result[0].argmax(dim=-1).tolist()))
            #loss = self.criterion(result, y, lengths)
            # total_loss += loss.item()
        total_loss /= len(self.test_loader)
        if self.__test_loss_key in self.history:
            self.history[self.__test_loss_key].append(total_loss)
        else:
            self.history[self.__test_loss_key] = [total_loss]

    @save_checkpoint
    def train(self):
        """Iterates over the entire training data and trains the model for a single epoch."""
        total_loss = 0
        self.set_train_mode()

        for inputs, inputs_len, targets, targets_len in tqdm(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            inputs_len, targets_len = inputs_len.to(self.device), targets_len.to(self.device)
            inputs = torch.squeeze(inputs, dim=1)
            self.optimizer.zero_grad()
            
            output_probs = self.model(inputs, inputs_len, targets, targets_len)            
            loss = self.criterion(
                output_probs, targets, inputs_len, targets_len
            )
            print(loss)
            print(loss.mean())
            for idx, l in enumerate(loss):
                if l < 0:
                    print(idx)
                    # import numpy as np
                    # ip = 'inputs'
                    # ip_len = "inputs_len"
                    # t = 'targets'
                    # t_len = "targets_len"
                    # np.save(ip, inputs.cpu().numpy())
                    # np.save(ip_len, inputs_len.cpu().numpy())
                    # np.save(t, targets.cpu().numpy())
                    # np.save(t_len, targets_len.cpu().numpy())
                    # return 


            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100)

            self.optimizer.step()
            total_loss += loss.mean().item()

        total_loss /= len(self.train_loader)

        if self.__train_loss_key in self.history:
            self.history[self.__train_loss_key].append(total_loss)
        else:
            self.history[self.__train_loss_key] = [total_loss]


def get_model_args(vocab_size: int, pad_idx: int, phi_idx: int, sos_idx: int) -> dict:
    device = hprams.device
    decoder_params = dict(
        **hprams.model.decoder, vocab_size=vocab_size, pad_idx=pad_idx
    )
    encoder_params = dict(**hprams.model.encoder)
    joint_params = dict(**hprams.model.joint, vocab_size=vocab_size)
    return {
        "decoder_params": decoder_params,
        "encoder_params": encoder_params,
        "joint_params": joint_params,
        "device": device,
        "phi_idx": phi_idx,
        "pad_idx": pad_idx,
        "sos_idx": sos_idx,
    }


def load_model(vocab_size: int, *args, **kwargs) -> Module:
    model = Model(**get_model_args(vocab_size, *args, **kwargs))
    if hprams.checkpoint is not None:
        load_stat_dict(model, hprams.checkpoint)
    return model


def get_tokenizer():
    tokenizer = CharTokenizer()
    tokenizer = tokenizer.load_tokenizer(hprams.tokenizer.tokenizer_file)
    return tokenizer


def get_data_loader(file_path: Union[str, Path], tokenizer):
    batch_size = hprams.training.batch_size
    max_len = hprams.data.max_str_len
    data_loader = DataLoader(file_path, tokenizer, batch_size, max_len)
    return data_loader


def get_trainer():
    tokenizer = get_tokenizer()
    phi_idx = tokenizer.special_tokens[tokenizer._phi_key][1]
    pad_idx = tokenizer.special_tokens[tokenizer._pad_key][1]
    sos_idx = tokenizer.special_tokens[tokenizer._sos_key][1]
    vocab_size = tokenizer.vocab_size()
    train_loader = get_data_loader(hprams.data.training_file, tokenizer)
    test_loader = get_data_loader(hprams.data.testing_file, tokenizer)
    criterion = RNNTLoss(reduction='none', blank=phi_idx)
    model = load_model(vocab_size, pad_idx=pad_idx, phi_idx=phi_idx, sos_idx=sos_idx)
    optimizer = OPT[hprams.training.optimizer](
        model.parameters(),
        lr=hprams.training.optim.learning_rate,
        momentum=hprams.training.optim.momentum,
    )
    return Trainer(
        criterion=criterion,
        tokenizer=tokenizer,
        optimizer=optimizer,
        model=model,
        device=hprams.device,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=hprams.training.epochs,
        length_multiplier=hprams.length_multiplier,
    )


if __name__ == "__main__":
    trainer = get_trainer()
    trainer.fit()
#     import numpy as np

#     from pathlib import Path
#     from model import Model
#     from tokenizer import CharTokenizer
#     from utils import get_formated_date, load_stat_dict
#     from torch.optim import Optimizer
#     from data import DataLoader
#     from typing import Callable, Union
#     from torch.nn import Module
#     from functools import wraps
#     from hprams import hprams
#     from tqdm import tqdm
#     import torch
#     import os
#     from search import GreedyDecode
#     from warprnnt_pytorch import RNNTLoss

#     inputs = torch.from_numpy(np.load('inputs.npy')).to('cuda')
#     inputs_len = torch.from_numpy(np.load('inputs_len.npy')).to('cuda')
#     targets = torch.from_numpy(np.load('targets.npy')).to('cuda')
#     targets_len = torch.from_numpy(np.load('targets_len.npy')).to('cuda')
#     output_probs = trainer.model(inputs, inputs_len, targets, targets_len)
#     loss_fun = RNNTLoss(reduction='none')
#     loss = loss_fun(
#     output_probs, targets, inputs_len, targets_len
# )
#     print(loss)
#     print(output_probs)
