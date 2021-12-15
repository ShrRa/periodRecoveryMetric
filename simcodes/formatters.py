def plot_name(*args,sep='_',ext='png',**kwargs):
    return sep.join(f"{k}" for k in args)+sep+sep.join(f"{k}_{v}" for k,v in kwargs.items())+'.'+ext