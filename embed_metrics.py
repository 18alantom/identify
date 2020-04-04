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


def show_min_max(embeds_cross_dist, labels_1, labels_2, show_sim=True, show_per_class=False, show_scores=False):
    # I am aware that there is a better non for loopy way of doing this.
    if show_sim:
        print("SHOWING SIMILAR")
    else:
        print("SHOWING DISSIMILAR")
    max_dist = []
    min_dist = []
    log_dist = []
    for i, vec in enumerate(embeds_cross_dist):
        if show_per_class:
            print(f"class {labels_1[i]}:", end="\t")

        dists = []
        for j, dist in enumerate(vec):
            if show_scores and show_per_class:
                print(f"{dist:0.1f} ", end=" ")
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
            if show_per_class:
                d = ""
                if show_scores:
                    d = "\n\t\t"
                print(f"{d}max: {mx:0.2f}", end="\t")
                print(f"min: {mn:0.2f}{d}")

    print('---')
    print(f"alltime max(max) = {max(max_dist)}")
    print(f"alltime min(max) = {min(max_dist)}")
    print(f"mean of max      = {torch.tensor(max_dist).mean()}")
    print()
    print(f"alltime min(min) = {min(min_dist)}")
    print(f"alltime max(min) = {max(min_dist)}")
    print(f"mean of min      = {torch.tensor(min_dist).mean()}")
    print()
    print(f"overall mean     = {torch.tensor(log_dist).mean()}")
    print()

    if show_sim:
        return torch.tensor(max_dist).mean()
    else:
        return torch.tensor(min_dist).mean()


def show_embed_metrics(embeds, labels, show_per_class=False, show_scores=False):
    """
    Calls the min max function, shows stats on embeddings generated from a model 
    Scores returned can be used to calculate threshold
    """
    print(f'Showing embedding distance metrics, {len(labels)} embeds: ')
    dist_matr = get_cross_dist(embeds, embeds)
    sim_max_mean = show_min_max(dist_matr, labels, labels, show_sim=True,
                                show_per_class=show_per_class, show_scores=show_scores)
    print('---')
    dis_min_mean = show_min_max(dist_matr, labels, labels, show_sim=False,
                                show_per_class=show_per_class, show_scores=show_scores)
    print('---')
    return sim_max_mean, dis_min_mean
