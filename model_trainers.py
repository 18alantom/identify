import torch

from time import time
from copy import deepcopy


class DistLoss:
    def __init__(self, sim_slink=4.3, dis_thresh=1.9, eps=0.8,
                 dist_func=torch.nn.PairwiseDistance(p=2, eps=1e-8)):
        """
        sim_slink: how exp the sim_loss grows after 1
        dis_thresh: after this dis_thresh, dis_loss < 0
        eps: factor to raise the dissimilar term by.
        dis_func: function to calculate distance between two vectors.
        """
        self.sim_slink = sim_slink
        self.dis_thresh = dis_thresh
        self.eps = eps
        self.dist_func = dist_func

    def dist_calculator(self, x1, x2):
        """
        Function that calculates the distances between
        all the vectors in x1 and x2 which are of the
        shape (m1,n) and (m2,n).

        n: dimensions in vector
        m1, m2: co-ordinates in x1 and x2 

        returns: (m2,m1) shape distance matrix
        """
        dists = []
        for vec in x1:
            dists.append(self.dist_func(x2, vec))
        return torch.stack(dists)

    def __call__(self, y_vectors: "vectors from the model", y: "labels"):
        """
        Loss function to be used if the output of the model is 
        an image embedding and the class of the image is known.

        Calculates the distance between all the y_vectors.
        If the classes match the distance > sim_thresh increases the loss 
        If the classes don't match the distance > 1 + dis_offset decreases loss
        """
        """
        When the dis_thresh is crossed by the dissimilar mean the loss contributed by 
        that term will be < 0
        """
        def t(n): return torch.tensor(n, requires_grad=True)
        rsm = 1e-8

        dist_calculator = self.dist_calculator
        sim_slink = self.sim_slink
        dis_thresh = self.dis_thresh
        eps = self.eps

        l = y_vectors.size(0)

        # Distance between all the y_vectors shape:(l,l)
        dist_matrix = dist_calculator(y_vectors, y_vectors)

        # Creating a shape:(l,l) mask using labels
        y_cross = y.repeat(l).reshape(l, l)
        y_mask = torch.eq(y_cross, y_cross.T)

        # mean tensor of dissimilar classes distances
        dis_mean = dist_matrix[~y_mask].mean()

        # Negating the diagonal cause that dist will be (almost) 0
        temp_tensor = torch.arange(l)
        y_mask[temp_tensor, temp_tensor] = False

        # mean tensor of similar classes distances
        sim_mean = dist_matrix[y_mask].mean()

        # possible nan if batch has all similar or dissimilar classes
        if torch.isnan(sim_mean):
            sim_mean = t(rsm)
        if torch.isnan(dis_mean):
            dis_mean = t(rsm)

        # Loss contributed by similar and dissimilar classes
        sim_loss = sim_mean ** t(sim_slink)
        dis_loss = torch.max(t(dis_thresh)/dis_mean - t(1.), t(rsm)) ** t(eps)

        loss = torch.log(sim_loss + dis_loss + t(1.))
        return loss


def dist_fit(model, optim, train_dl, valid_dl, device, data_count, loss_func=DistLoss(), epochs=25 ):
    start = time()
    def time_st(x): return f"{x//60:0.0f} m {x%60:0.3f} s"
    model = model.to(device)
    losses_tr = []
    losses_va = []

    # Define datasets dict
    tr = 'train'
    va = 'valid'

    sets = [tr, va]
    data = {tr: train_dl, va: valid_dl}

    least_loss = torch.tensor(float('inf'))
    best_model_state_dict = deepcopy(model.state_dict())

    print(f"train samples: {data_count[tr]}, valid samples: {data_count[va]}")
    # Add timer
    for epoch in range(epochs):
        e_start = time()

        print(
            f"\nEPOCH: ({epoch + 1}/{epochs})\t{e_start - start:0.3f} s\n", "-"*20)
        for phase in sets:
            p_start = time()

            is_tr = phase == tr
            if is_tr:
                model.train()
            else:
                model.eval()

            running_loss = 0.

            for batch in data[phase]:
                X, y = batch
                X = X.to(device)
                y = y.to(device)

                optim.zero_grad()

                with torch.set_grad_enabled(is_tr):
                    y_vectors = model(X)

                    loss = loss_func(y_vectors, y)

                    if is_tr:
                        loss.backward()
                        optim.step()

                samp_loss = loss * len(y)
                if is_tr:
                    losses_tr.append(samp_loss)
                else:
                    losses_va.append(samp_loss)
                running_loss += samp_loss

            p_time = time() - p_start
            epoch_loss = running_loss / data_count[phase]
            print(f"{phase}: loss {epoch_loss:0.3f}, time {time_st(p_time)}")

            if (not is_tr) and (least_loss > epoch_loss):
                least_loss = epoch_loss
                best_model_state_dict = deepcopy(model.state_dict())

    tot_time = time() - start
    print(f"\nTime taken: {time_st(tot_time)}")
    return model.load_state_dict(best_model_state_dict), losses_tr, losses_va


def fit(model, optim, train_dl, valid_dl, device, data_count, loss_func=DistLoss(), epochs=25, ):
    pass


def std_fit(model, optim, train_dl, valid_dl, device, data_count, loss_func, epochs=25):
    start = time()
    def time_st(x): return f"{x//60:0.0f} m {x%60:0.3f} s"
    model = model.to(device)
    losses_tr = []
    losses_va = []

    tr = 'train'
    va = 'valid'
    data = {tr: train_dl, va: valid_dl}

    print(f"train samples: {data_count[tr]}, valid samples: {data_count[va]}")
    best_accu = 0.0
    least_loss = 20
    best_model_state_dict = model.state_dict()

    # Add timer
    for epoch in range(epochs):
        e_start = time()

        print(
            f"\nEPOCH: ({epoch + 1}/{epochs})\t{e_start - start:0.3f} s\n", "-"*20)
        for phase in [tr, va]:
            p_start = time()

            is_tr = phase == tr
            is_va = phase == va

            if is_tr:
                model.train()
            else:
                model.eval()

            """
            Loss and accuracy calculated 
            during a single epoch.
            """
            running_loss = 0.
            running_accu = 0

            for batch in data[phase]:
                X, y = batch
                X = X.to(device)
                y = y.to(device)

                optim.zero_grad()

                with torch.set_grad_enabled(is_tr):
                    y_val = model(X)
                    y_cls = torch.argmax(y_val, dim=1)

                    loss = loss_func(y_val, y)

                    if is_tr:
                        loss.backward()
                        optim.step()

                """
                Running Loss:
                    Loss calculated over the entire dataset,
                    for one epoch. Loss for an epoch will be 
                    running loss divided by the total number
                    of samples (not batches).
                Running Accuracy:
                    Number of samples the model got right.
                    
                """
                samp_loss = loss.item() * len(y)
                if is_tr:
                    losses_tr.append(samp_loss)
                else:
                    losses_va.append(samp_loss)

                running_loss += samp_loss
                running_accu += torch.sum(y_cls == y).item()

            p_time = time() - p_start
            epoch_loss = running_loss / data_count[phase]
            epoch_accu = running_accu / data_count[phase]
            print(
                f"{phase}: loss {epoch_loss:0.3f}, accu {epoch_accu:0.3f}, time {time_st(p_time)}")

            if is_va and (epoch_accu > best_accu) or (epoch_accu == best_accu and least_loss > epoch_loss):
                best_accu = epoch_accu
                least_loss = epoch_loss
                best_model_state_dict = deepcopy(model.state_dict())
            elif is_va and least_loss > epoch_loss:
                least_loss = epoch_loss

    tot_time = time() - start
    print(
        f"\nTime taken: {time_st(tot_time)}, Best accuracy: {best_accu:0.3f}")
    return model.load_state_dict(best_model_state_dict), losses_tr, losses_va
