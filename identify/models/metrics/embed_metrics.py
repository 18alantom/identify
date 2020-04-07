import torch


def get_cross_dist(embeds_1, embeds_2):
    """
    Calculates the (L2) distance between all the embeddings in all the embeds.
    embed_1 shape (m, v)
    embed_2 shape (n, v)

    return shape(m,n)
    """
    embeds_cross_dist = []
    for embed in embeds_1:
        dists = torch.norm(embeds_2 - embed, dim=1)
        embeds_cross_dist.append(dists)
    return torch.stack(embeds_cross_dist)


def show_min_max(embeds_cross_dist, labels_1, labels_2, show_sim):
    # I am aware that there maybe better non for loopy way of doing this.
    if show_sim:
        print("SHOWING SIMILAR")
    else:
        print("SHOWING DISSIMILAR")
    max_dist = []
    min_dist = []
    log_dist = []
    for i, vec in enumerate(embeds_cross_dist):
        dists = []
        for j, dist in enumerate(vec):
            if labels_1[i] == labels_2[j] and i != j and show_sim:
                log_dist.append(dist)
                dists.append(dist)
            elif not show_sim and labels_1[i] != labels_2[j]:
                log_dist.append(dist)
                dists.append(dist)

        if len(dists) > 0:
            mx = max(dists)
            mn = min(dists)
            max_dist.append(mx)
            min_dist.append(mn)

    overall = torch.tensor(log_dist).mean()
    mean_min = torch.tensor(min_dist).mean()
    print('---')
    print(f"alltime max(max) = {max(max_dist)}")
    print(f"alltime min(max) = {min(max_dist)}")
    print(f"mean of max      = {torch.tensor(max_dist).mean()}")
    print()
    print(f"alltime min(min) = {min(min_dist)}")
    print(f"alltime max(min) = {max(min_dist)}")
    print(f"mean of min      = {mean_min}")
    print()
    print(f"overall mean     = {overall}")
    print()

    if show_sim:
        return torch.tensor(max_dist).mean(), overall, mean_min
    else:
        return torch.tensor(min_dist).mean(), overall, None


def show_embed_metrics(embeds, labels):
    """
    Calls the min max function, shows stats on embeddings generated from a model 
    Scores returned can be used to calculate threshold
    """
    print(f'Showing embedding distance metrics, {len(labels)} embeds: ')
    dist_matr = get_cross_dist(embeds, embeds)
    _, _, mean_min = show_min_max(dist_matr, labels, labels, show_sim=True)
    print('---')
    _, _, _ = show_min_max(dist_matr, labels, labels, show_sim=False)
    print('---')
    return mean_min
