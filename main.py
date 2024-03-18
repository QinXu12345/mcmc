from mh import Ising
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.style.use('ggplot')
    fig,ax = plt.subplots()
    betas = [0.01,1,100]
    for i,beta in zip(range(len(betas)),betas): 
        ising = Ising()
        ising.mh(epoch=500000,beta=beta)
        configs = np.array(ising.tolist())
        exp = np.mean(configs,axis=0)
        x = np.linspace(0, configs.shape[1],num=60)
        # sns.lineplot(x = x, y = configs[-1,:])
        sns.lineplot(x = x, y = exp,ax=ax,label=f"$\\beta = {beta}$")
        
    ax.set_title(f"$\\beta$ at {betas}")
    ax.set_xlabel("Spins")
    ax.set_ylabel("Time average")
    plt.show()