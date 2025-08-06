
def acc_metric(predb, yb):
    return (predb.argmax(dim=1) == yb).float().mean()
