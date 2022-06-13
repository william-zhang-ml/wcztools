"""
Training script template to make future research easier.
This template trains a linear regression model against 2x + 1 data.
"""
import os
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from loader import SpoofDataset


THIS_DIR = Path(os.path.dirname(os.path.abspath(__file__)))


# pylint: disable=too-few-public-methods
class Config:
    """ Script configuration. """
    seed = 0

    # data loader
    batch_size = 32
    num_workers = 4

    # train loop
    n_epoch = 90
    learn_rate = 0.1
    momentum = 0.9
    sched_step_size = 30
# pylint: enable=too-few-public-methods


def get_model() -> nn.Module:
    """ Returns: initialized by untrained model. """
    return nn.Linear(1, 1)


def train() -> nn.Module:
    """ Returns: trained model. """
    # setup training variables
    loader = DataLoader(
        SpoofDataset(train=True),
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )
    model = get_model()
    optimizer = optim.SGD(
        model.parameters(),
        lr=Config.learn_rate,
        momentum=Config.momentum
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=Config.sched_step_size
    )
    loss_func = nn.MSELoss()

    # train loop
    pbar = tqdm(range(Config.n_epoch))
    for _ in pbar:
        for batch, (inp, targ) in enumerate(loader):
            # (batch_size, ) -> (batch_size, 1)
            inp, targ = inp.unsqueeze(-1), targ.unsqueeze(-1)

            # forward
            pred = model(inp)
            loss = loss_func(pred, targ)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # user feedback and logging
            pbar.set_postfix({
                'batch': f'{batch + 1}/{len(loader)}',
                'loss': f'{loss.detach().numpy(): .03F}'
            })
        scheduler.step()
    return model


def validate(model: nn.Module) -> None:
    """
    Run test set through a model and save outputs, metrics, and charts.

    Args:
        model: model to use
    """
    # setup validation variables
    test_set = SpoofDataset(train=False)
    loader = DataLoader(
        test_set,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )

    # inference loop
    pred_all = []
    for inp, targ in tqdm(loader):
        # (batch_size, ) -> (batch_size, 1)
        inp, targ = inp.unsqueeze(-1), targ.unsqueeze(-1)

        # forward and store results
        with torch.no_grad():
            pred = model(inp)
            pred_all.extend(pred.squeeze().tolist())

    # save inference outputs and/or metrics
    table = test_set.data.copy()  # DataFrame
    table['pred'] = pred_all
    table.to_csv(THIS_DIR / 'data' / 'pred.csv', index=False)

    # plot
    fig, axes = plt.subplots()
    axes.plot(table.x, table.y, 'g.', markersize=1, label='targ')
    axes.plot(table.x, table.pred, 'r.', markersize=1, label='pred')
    axes.legend()
    axes.set_xlabel('input')
    axes.set_ylabel('output')
    axes.grid()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    torch.manual_seed(Config.seed)
    trained_model = train().eval()
    validate(trained_model)
