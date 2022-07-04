#License BSD
import os
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt

def softmax (x):
    exps = [np.exp(i) for i in x]
    sum_exp = sum(exps)

    sm_x = [e/sum_exp for e in exps]
    return sm_x

def plot_line(ax, idx, label, item, mapper):
    plot = False
    for i in item:
        if i > 0.04:
            plot = True
    if plot:
        x = np.arange(len(item))
        ax.plot(x, item, label=label, color =mapper.to_rgba(idx))
if __name__ == "__main__":

    labels = ['accordion', 'airplanes', 'anchor', 'ant', 'background_google',
    'barrel', 'bass', 'beaver', 'binocular', 'bonsai', 'brain', 'brontosaurus',
    'buddha', 'butterfly', 'camera', 'cannon', 'car_side', 'ceiling_fan',
    'cellphone', 'chair', 'chandelier', 'cougar_body', 'cougar_face', 'crab',
    'crayfish', 'crocodile', 'crocodile_head', 'cup', 'dalmatian',
    'dollar_bill', 'dolphin', 'dragonfly', 'electric_guitar', 'elephant', 'emu',
    'euphonium', 'ewer', 'faces', 'ferry', 'flamingo', 'flamingo_head',
    'garfield', 'gerenuk', 'gramophone', 'grand_piano', 'hawksbill',
    'headphone', 'hedgehog', 'helicopter', 'ibis', 'inline_skate',
    'joshua_tree', 'kangaroo', 'ketch', 'lamp', 'laptop', 'leopards', 'llama',
    'lobster', 'lotus', 'mandolin', 'mayfly', 'menorah', 'metronome', 'minaret',
    'motorbikes', 'nautilus', 'octopus', 'okapi', 'pagoda', 'panda', 'pigeon',
    'pizza', 'platypus', 'pyramid', 'revolver', 'rhino', 'rooster', 'saxophone',
    'schooner', 'scissors', 'scorpion', 'sea_horse', 'snoopy', 'soccer_ball',
    'stapler', 'starfish', 'stegosaurus', 'stop_sign', 'strawberry',
    'sunflower', 'tick', 'trilobite', 'umbrella', 'watch', 'water_lilly',
    'wheelchair', 'wild_cat', 'windsor_chair', 'wrench', 'yin_yang']


    car_tensor = np.load("car_sips_sequence.npy")
    print ("car tensor size: ", car_tensor.shape) 

    # convert logits into softmax
    car_softmax = np.array([softmax(x) for x in car_tensor])

    # Transpose data
    print (car_softmax.shape)

    # Ploting the values
    fig = plt.figure()
    norm = matplotlib.colors.Normalize(vmin=0, vmax=100, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.rainbow)
    ax = fig.add_subplot(1,1,1)
    [plot_line(ax,idx,labels[idx],it, mapper) for idx, it in enumerate(car_softmax.T)]
    ax.legend()
    ax.set_title('Line plot - Classes')
    ax.set_yticklabels([])
    plt.show()
