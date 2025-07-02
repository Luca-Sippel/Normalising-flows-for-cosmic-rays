import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#source= https://www.tensorflow.org/tutorials/keras/classification

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']    #0=T-shirt,...,9=ankle boot

train_images=train_images/255
test_images=test_images/255

neurons=np.array([16,64,256,512])
layers=np.array([1,4,6,8])
batch_size=np.array([16,32,64,128,1024])
activation=['relu','sigmoid','softplus']   

results={}
for n in neurons:
    for l in layers:
        for b in batch_size:
            for act in activation:
                name = f"n{n}_{act}_adam_l{l}_b{b}"
                print(f"Training modell: {name}")
                model_layers = [tf.keras.layers.Flatten(input_shape=(28, 28))]
                for _ in range(l):
                   model_layers.append(tf.keras.layers.Dense(n, activation=act))
                model_layers.append(tf.keras.layers.Dense(10))
                model = tf.keras.Sequential(model_layers)
                model.compile(
                    optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
                history=model.fit(train_images,train_labels, epochs=50,batch_size=b,validation_data=(test_images,test_labels),verbose=1)
                results[name]={"train_acc":history.history['accuracy'],
                              "val_acc":history.history['val_accuracy']}
                
mean_val={}
for name, data in results.items():
    val_acc=data["val_acc"]
    mean_val[name]=np.mean(val_acc[19:])
for name, val in sorted(mean_val.items(), key=lambda x: x[1]):
    print(f"{name:35}: {val:.4f}")

fig, axs = plt.subplots(len(activation), len(batch_size), sharex=True, sharey=True,gridspec_kw={'hspace':0.05})
for i, act in enumerate(activation):
    for j, batch in enumerate(batch_size):
        heatmap = np.full((len(layers),len(neurons)),np.nan)
        for k, layer in enumerate(layers):
            for l, neuron in enumerate(neurons):
                name=f"n{neuron}_{act}_adam_l{layer}_b{batch}"
                if name in results:
                    heatmap[k,l]=mean_val[name]
        ax = axs[i,j]
        im = ax.imshow(heatmap, cmap="viridis",vmin=0.7,vmax=0.9)
        ax.set_title(f"{act},b={batch}",fontsize=8)
        ax.set_xticks(range(len(neurons)))
        ax.set_yticks(range(len(layers)))
        ax.tick_params(axis='x',labelsize=6)
        ax.tick_params(axis='y',labelsize=7)
        ax.set_xticklabels(neurons,fontsize=8)
        ax.set_yticklabels(layers,fontsize=8)
        if j == 0:
            ax.set_ylabel("Layers")
        if i == len(activation) - 1:
            ax.set_xlabel("Neurons")
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.87,0.15,0.015,0.7]) 
fig.colorbar(im, cax=cbar_ax, label='Validation Accuracy')
plt.tight_layout(rect=[0,0,0.8,1]) 
plt.show()

