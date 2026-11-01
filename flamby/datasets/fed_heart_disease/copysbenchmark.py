import argparse

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader as dl
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from opacus.accountants.utils import get_noise_multiplier #COPYS IMPORTS
from torch.func import functional_call, vmap, grad
from opacus.data_loader import DPDataLoader
from opacus.accountants import RDPAccountant
import json, os   # add to the imports at the top


from flamby.datasets.fed_heart_disease import (
    BATCH_SIZE,
    LR,
    NUM_EPOCHS_POOLED,
    Baseline,
    BaselineLoss,
    FedHeartDisease,
    metric,
)
from flamby.utils import evaluate_model_on_tests

def main(num_workers_torch, log=False, log_period=10, debug=False, cpu_only=False):
    """Function to execute the benchmark on Heart Disease.

    Parameters
    ----------
    debug : bool
        Whether or not to use the dataset obtained in debug mode. Default to False.
        Used for consistency with other datasets' APIs.
    """

    metrics_dict = {"AUC": metric}
    use_gpu = torch.cuda.is_available() and not (cpu_only)

    training_dl = dl(
        FedHeartDisease(train=True, pooled=True, debug=debug),
        num_workers=num_workers_torch,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_dl = dl(
        FedHeartDisease(train=False, pooled=True, debug=debug),
        num_workers=num_workers_torch,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    print(f"The training set pooled contains {len(training_dl.dataset)} records")
    print(f"The test set pooled contains {len(test_dl.dataset)} records")

    if log:
        # We compute the number of batches per epoch
        num_local_steps_per_epoch = len(training_dl.dataset) // BATCH_SIZE
        num_local_steps_per_epoch += int(
            (len(training_dl.dataset) - num_local_steps_per_epoch * BATCH_SIZE) > 0
        )

    results = []
    seeds = np.arange(42, 47).tolist()

    # COPYS ADDITION BELOW

    TARGET_EPS = 1
    TARGET_DELTA = 1e-15        # must be > 0

    N = len(training_dl.dataset)
    sample_rate = BATCH_SIZE / N

    SIGMA = get_noise_multiplier(
        target_epsilon=TARGET_EPS,
        target_delta=TARGET_DELTA,
        sample_rate=sample_rate,
        epochs=NUM_EPOCHS_POOLED,
        accountant="rdp",        # or "prv"
    )

    print(f"Noise multiplier sigma = {SIGMA:.3f}  (noise std = sigma * C)")

    training_dl = DPDataLoader.from_data_loader(training_dl)

    print(f"BATCH_SIZE={BATCH_SIZE}, steps/epoch={len(training_dl)}")

    # COPYS ADDITION ABOVE

    for seed in seeds:
        # At each new seed we re-initialize the model
        # and training_dl is shuffled as well
        torch.manual_seed(seed)
        m = Baseline()
        # We put the model on GPU whenever it is possible
        if use_gpu:
            m = m.cuda()
        loss = BaselineLoss()
        optimizer = optim.Adam(m.parameters(), lr=LR)
        if log:
            # We create one summarywriter for each seed in order to overlay the plots
            writer = SummaryWriter(log_dir=f"./runs/seed{seed}")

        # Really not important
        C = 1.0   # clipping norm (tune for accuracy, doesn't affect ε)

        # once per seed, after creating m, loss, optimizer: #COPYS ADDITIONS
        def sample_loss(params, x, target):
            pred = functional_call(m, params, (x.unsqueeze(0),)) # predictions
            return loss(pred, target.unsqueeze(0)) #calculate how far off

        # RETURNS A DICTIONARY RETURNING A GRADIENT FOR EACH SAMPLE IN THE BATCH
        per_sample_grad_fn = vmap(grad(sample_loss), in_dims=(None, 0, 0)) #COPYS ADDITIONS 

        # inside the seed loop, before the epoch loop:
        # Each step stores a small bit of the privacy budget, will be useful for noting how much
        steps_taken = 0
        acc = RDPAccountant() # will help us store the budget spent through each epoch

        for e in tqdm(range(NUM_EPOCHS_POOLED)):
            if log:
                # At each epoch we look at the histograms of all the network's parameters
                for name, p in m.named_parameters():
                    writer.add_histogram(f"client_0/{name}", p, e)
            for s, (X, y) in enumerate(training_dl):
                # traditional training loop with optional GPU transfer
                if use_gpu:
                    X = X.cuda()
                    y = y.cuda()

                #COPYS ADDITIONS BELOW

                # at the top of the batch loop body (before the empty-batch check):
                steps_taken += 1
                # record this noisy step with the accountant (including empty-batch steps)
                acc.step(noise_multiplier=SIGMA, sample_rate=1 / len(training_dl))

                if len(X) == 0:
                    # Poisson sampling gave an empty batch: take a noise-only step
                    # privacy accounting, so we still take an update, using noise only
                    for p in m.parameters():
                        p.grad = torch.randn_like(p) * SIGMA * C / BATCH_SIZE
                    optimizer.step()
                    continue

                # per sample gradients replace lm.backward()
                # lm.backward() is an averaged gradient over the whole batch
                # this gets the gradient for each sample
                params = {k: v.detach() for k, v in m.named_parameters()}
                g = per_sample_grad_fn(params, X, y)       # name -> [B, *param_shape]

                # 1) per-sample L2 norm over ALL parameters together
                # Each sample's weight and bias gradients are treated as one long vector.
                # flatten(1) -> [B, numel], square and sum per sample, add across parameters, sqrt
                # Result: one number per sample = how large that record's gradient is
                norms = torch.sqrt(sum(v.flatten(1).pow(2).sum(1) for v in g.values()))  # [B]

                # 2) clip: scale = min(1, C / norm)
                # caps out influence.
                scale = (C / (norms + 1e-6)).clamp(max=1.0)

                # 3) sum clipped grads, add noise N(0, (sigma*C)^2), divide by expected batch size
                #COPYS ADDITIONS ABOVE
                optimizer.zero_grad() #clear old gradients in the optimizer

                for name, p in m.named_parameters(): #THIS LOOP IS A COPY ADDITION
                    # Methodology: the clipped per-sample gradients over the batch, then add noise once
                    gi = g[name]
                    clipped_sum = (gi * scale.view(-1, *([1] * (gi.dim() - 1)))).sum(0)
                    noise = torch.randn_like(clipped_sum) * SIGMA * C
                    # p.grad = (sum of clipped sample gradients + one noise draw) / batch size
                    p.grad = (clipped_sum + noise) / BATCH_SIZE
                optimizer.step() # Modify and update model weights

                #y_pred = m(X)

                #lm = loss(y_pred, y)

                with torch.no_grad(): # COPY ADDITION
                    # just for monitoring
                    y_pred = m(X)
                    lm = loss(y_pred, y)

                #lm.backward()
                
                #optimizer.step()

                if log:
                    eps = acc.get_epsilon(delta=TARGET_DELTA) #writing the epsilon cost at the end of each epoch
                    writer.add_scalar("epsilon", eps, e)

                    current_step = s + num_local_steps_per_epoch * e
                    if (current_step % log_period) == 0:
                        writer.add_scalar(
                            "Loss/train/client",
                            lm.item(),
                            s + num_local_steps_per_epoch * e,
                        )
                        for k, v in metrics_dict.items():
                            train_batch_metric = v(
                                y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
                            )
                            writer.add_scalar(
                                f"{k}/train/client",
                                train_batch_metric,
                                s + num_local_steps_per_epoch * e,
                            )

        # COPY ADDITIONS BELOW
        # after the epoch loop, per seed:
        print(f"seed {seed}: epsilon = {acc.get_epsilon(delta=TARGET_DELTA):.3f}") #reports final lots

        # NEW: save model and privacy recipe
        os.makedirs("saved", exist_ok=True)
        torch.save(m.state_dict(), f"saved/model_seed{seed}.pt")
        with open(f"saved/config_seed{seed}.json", "w") as f:
            json.dump({
                "seed": seed, "noise_multiplier": SIGMA, "max_grad_norm": C,
                "sample_rate": 1 / len(training_dl), "steps": steps_taken,
                "delta": TARGET_DELTA,
            }, f, indent=2)

        #COPY ADDITIONS ABOVE

        current_results_dict = evaluate_model_on_tests(
            m, [test_dl], metric, use_gpu=use_gpu
        )

        results.append(current_results_dict["client_test_0"])
        print(current_results_dict)

    results = np.array(results)

    if log:
        for i in range(results.shape[0]):
            writer = SummaryWriter(log_dir=f"./runs/tests_seed{seeds[i]}")
            writer.add_scalar("AUC-test", results[i], 0)

    print("Benchmark Results on Heart Disease pooled:")
    print(f"mAUC on 5 runs: {results.mean(): .2%} \\pm {results.std(): .2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-workers-torch",
        type=int,
        help="How many workers to use for the batching.",
        default=0,
    )
    parser.add_argument(
        "--log",
        action="store_true",
        default=True,
        help="Whether to activate tensorboard logging or not default to no logging",
    )
    parser.add_argument(
        "--log-period",
        type=int,
        help="The period in batches for the logging of metric and loss",
        default=20,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to use the dataset obtained in debug mode.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Deactivate the GPU to perform all computations on CPU only.",
    )

    args = parser.parse_args()
    main(args.num_workers_torch, args.log, args.log_period, args.debug, args.cpu_only)
