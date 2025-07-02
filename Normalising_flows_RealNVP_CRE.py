import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions  
tfb = tfp.bijectors
import h5py

#Monte Carlo Simulation from Frediani et al., 2025 (https://arxiv.org/abs/2501.06011)  (data available at https://zenodo.org/records/14623915) 
file_path=r"/home/au400420/BA/batches_regime_Klein-Nishina_SNrate_2.0e+04_dist_spiral_mode_full_causality_True_Ecut_1.0e+04_seeds_0-9999.hdf5"
with h5py.File(file_path, 'r') as f:
    raw_data= np.array(f['fluxes'])   
raw_data=tf.convert_to_tensor(raw_data,dtype=tf.float32)
raw_data=tf.transpose(raw_data)[:,14:33]    
Ens=tf.math.pow(10.0, tf.linspace(1.0,5.0,41))[14:33]
fluxes=tf.math.log(Ens**3*raw_data*0.06)/tf.math.log(10.0)

# a plot function for visualisation
def plot_fluxes(Ens: tf.Tensor, fluxes: tf.Tensor, ymin: float = 0, ymax: float = 5, n_samples: int = 8):
    fluxes=tf.convert_to_tensor(fluxes)
    Ens=tf.convert_to_tensor(Ens)
    idx=tf.random.shuffle(tf.range(tf.shape(fluxes)[0]))[:n_samples]
    for i in idx:
        plt.plot(Ens.numpy(),fluxes[i].numpy(),alpha=0.5)
    median=tfp.stats.percentile(fluxes,50.0,axis=0)
    q05=tfp.stats.percentile(fluxes,5.0,axis=0)
    q95=tfp.stats.percentile(fluxes,95.0,axis=0)
    q16=tfp.stats.percentile(fluxes,16.0,axis=0)
    q84=tfp.stats.percentile(fluxes,84.0,axis=0)
    plt.plot(Ens.numpy(),median.numpy(),color='black',label='Median')
    plt.fill_between(Ens.numpy(),q05.numpy(),q95.numpy(),color='gainsboro',label='90% Quantil')
    plt.fill_between(Ens.numpy(),q16.numpy(),q84.numpy(),color='darkgray',label='68% Quantil')
    plt.legend()
    plt.xscale('log')
    plt.xlabel(r"$E\;\mathrm{[GeV]}$")
    plt.ylabel(r"$\log_{10}(E^3 \phi\;[\mathrm{GeV}^2\,\mathrm{m}^{-2}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}])$")
    plt.ylim(ymin,ymax)
    plt.xlim(Ens[0].numpy(),Ens[-1].numpy())

# structure of a coupling layer with two neural networks for scale and translation
def Coupling(input_shape,layers,activation,neuron,reg=0.01):
    input = tf.keras.layers.Input(shape=(input_shape,))
    s_layers=[input]
    t_layers=[input]
    for i in range(layers):
        s_layers.append(tf.keras.layers.Dense(neuron,activation=activation,kernel_regularizer=tf.keras.regularizers.l2(reg))(s_layers[-1]))
        t_layers.append(tf.keras.layers.Dense(neuron,activation=activation,kernel_regularizer=tf.keras.regularizers.l2(reg))(t_layers[-1]))
    s_output_layer=tf.keras.layers.Dense(input_shape,activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(reg))(s_layers[-1])  
    t_output_layer=tf.keras.layers.Dense(input_shape,activation='linear',kernel_regularizer=tf.keras.regularizers.l2(reg))(t_layers[-1])
    return tf.keras.Model(inputs=input, outputs=[s_output_layer,t_output_layer])

# 5 different masks i developed and tested
def mask(input_shape,layers):  # alternating mask
    mask_list=[]
    for i in range(layers):
        mask=np.zeros(input_shape)
        if i % 2 == 0:
            mask[::2]=1
        else:
            mask[1::2]=1
        mask_list.append(mask)
    return np.stack(mask_list)

def mask1(input_shape,layers): # half-split mask
    mask_list=[]
    for i in range(layers):
        mask=np.zeros(input_shape)
        if i % 2 == 0:
            mask[:9]=1
        else:
            mask[9:]=1
        mask_list.append(mask)
    return np.stack(mask_list)

def mask2(input_shape,layers): # alternating quarter mask
    mask_list=[]
    for i in range(layers):
        mask=np.zeros(input_shape)
        if i % 2 == 0:
            mask[:5]=1
            mask[9:14]=1
        else:
            mask[5:9]=1
            mask[14:19]=1
        mask_list.append(mask)
    return np.stack(mask_list)

def mask3(input_shape,layers):  # rolling block mask
    mask_list=[]
    mask=np.array([1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0,1])
    for i in range(layers):
        mask_roll=np.roll(mask,i)
        mask_list.append(mask_roll)
    return np.stack(mask_list)

def mask4(input_shape,layers): # random mask
    mask_list=[]
    for i in range(layers):
        if i % 2 == 0:
            mask=np.random.randint(0,2,size=input_shape)
        else:
            mask=1-mask_list[-1]
        mask_list.append(mask)
    return np.stack(mask_list)  

class RealNVP(tf.keras.Model):
    def __init__(self,n_coupling_layers,n_layers,neuron,activation,reg,input_shape):
        super().__init__()
        self.n_coupling_layers=n_coupling_layers
        self.base_dist=tfd.MultivariateNormalDiag(loc=tf.zeros(input_shape),scale_diag=tf.ones(input_shape))  
        self.masks=mask(input_shape,n_coupling_layers)    # here you can change the masks by writing the number behind mask like in the definitions above (e.g. mask4)
        self.loss_history=tf.keras.metrics.Mean(name="loss")
        self.layers_list=[Coupling(input_shape,n_layers,activation,neuron,reg) for i in range(n_coupling_layers)]
        
    @property 
    def metrics(self):
        return [self.loss_history]
        
    def call(self,x,training=True):
        log_det_inv = 0
        direction = 1
        if training:
            direction = -1
        for i in range(self.n_coupling_layers)[::direction]:
            x_masked = x * self.masks[i]
            reversed_mask = 1 - self.masks[i]
            s, t = self.layers_list[i](x_masked)
            s *= reversed_mask
            t *= reversed_mask
            gate = (direction - 1) / 2
            x = (reversed_mask*(x*tf.exp(direction*s)+direction*t*tf.exp(gate * s))+x_masked)
            #Training (-1): x=x2*exp(-s)-t*exp(-s)+x1
            # Sampling (1): x=x2*exp(s)+t*1+x1
            log_det_inv += gate * tf.reduce_sum(s, [1])
        return x,log_det_inv
                    
    def log_loss(self,x):
        z,log_det_jacobian=self(x)
        #log[p(x)]=log[p(z)]+log|det(J(x))|
        log_prob=self.base_dist.log_prob(z)+log_det_jacobian
        return -tf.reduce_mean(log_prob)  #NNL
            
    def train_step(self,data):
        with tf.GradientTape() as tape:
            loss=self.log_loss(data)
            grads=tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_history.update_state(loss)
        return {"loss":self.loss_history.result()}
        
    def test_step(self,data):
        loss=self.log_loss(data)
        self.loss_history.update_state(loss)
        return {"loss":self.loss_history.result()}
    
#Parameters:
neurons=512          # number of neurons
n_layer=10           # number of hidden layers
activation_function='softplus'
n_coupling_layer=8   # number of coupling layers
batch=64             # batch size 
epochs=100

#Model
model=RealNVP(n_coupling_layers=n_coupling_layer,n_layers=n_layer,neuron=neurons,activation=activation_function,input_shape=19)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5))
history=model.fit(fluxes, batch_size=batch, epochs=epochs, verbose=2)
result=history.history['loss']

#Loss curve
plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.title('Loss Verlauf')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.show()

#Sampling
x, _=model(model.base_dist.sample(100000),training=False)   #Sampling:training=False
x_tensor=tf.convert_to_tensor(x,dtype=tf.float32)

#Comparison
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plot_fluxes(Ens,fluxes,-1,3)
plt.subplot(1,2,2)
plot_fluxes(Ens,x_tensor,-1,3)
plt.show()

#Analysis
model_analysis=[]
quantil_data=np.percentile(fluxes.numpy(),[5,16,50,84,95],axis=0)
quantil_model=np.percentile(x,[5,16,50,84,95],axis=0)
rel_diff=np.abs((quantil_data-quantil_model)/quantil_data)
std_data=quantil_data[3]-quantil_data[1]
std_model=quantil_model[3]-quantil_model[1]
model_analysis.append(np.mean(np.abs(quantil_data[2]-quantil_model[2])))
model_analysis.append(np.mean(rel_diff[2]))
model_analysis.append(np.mean(std_model))
model_analysis.append(np.mean(std_model)/np.mean(std_data))
print("abs. Diff. Median = ",model_analysis[0])
print("rel. Diff. Median = ",model_analysis[1])
print("model std = ",model_analysis[2])
print("data std = ",np.mean(std_data))
print("std ratio (Model/Data)= ",model_analysis[3])

fig, ax = plt.subplots(3, 1, figsize=(12,8), sharex=True, gridspec_kw={'height_ratios': [5,2,2]})
ax[0].plot(Ens.numpy(),quantil_data[2],color='blue',label='data')
ax[0].fill_between(Ens.numpy(),quantil_data[1],quantil_data[3],color='lightblue',alpha=0.25)
ax[0].plot(Ens.numpy(),quantil_data[1],color='lightblue',linestyle='--')
ax[0].plot(Ens.numpy(),quantil_data[3],color='lightblue',linestyle='--')
ax[0].plot(Ens.numpy(),quantil_model[2],color='red',label='model')
ax[0].fill_between(Ens.numpy(),quantil_model[1],quantil_model[3],color='orange',alpha=0.25)
ax[0].plot(Ens.numpy(),quantil_model[1],color='orange',linestyle='--')
ax[0].plot(Ens.numpy(),quantil_model[3],color='orange',linestyle='--')
ax[0].legend()
ax[2].set_xlabel(r"$E\;\mathrm{[GeV]}$")
ax[0].set_ylabel(r"$\log_{10}(E^3 \phi\;[\mathrm{GeV}^2\,\mathrm{m}^{-2}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}])$")
ax[0].set_xlim(Ens[0].numpy(),Ens[-1].numpy())
ax[1].set_ylabel('rel. diff. median')
ax[1].scatter(Ens.numpy(),(-quantil_data[2]+quantil_model[2])/quantil_model[2])
ax[1].axhline(y=0., color='black', linestyle='--')
ax[1].set_xlim(Ens[0].numpy(),Ens[-1].numpy())
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[1].grid()
ax[2].grid()
ax[1].text(0.02, 0.95, f"mean={round(model_analysis[1],5)}", transform=ax[1].transAxes, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
ax[2].scatter(Ens.numpy(),std_model/std_data)
ax[2].set_ylabel("std ratio (Modell/Data)")
ax[2].axhline(y=1, color='black',linestyle='--')
ax[2].text(0.02, 0.05, f"mean={round(model_analysis[3],4)}", transform=ax[2].transAxes, ha='left', va='bottom', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
plt.show()

#histogramm
bins_plot_number=[0,3,7,10,14,18]
bins_number=100
plt.figure()
for i, bin in enumerate(bins_plot_number):
    plt.subplot(3,2,i+1)
    plt.hist(fluxes.numpy()[:,bin],bins=bins_number,density=True,alpha=0.5,label="data",color='blue')
    plt.hist(x[:,bin],bins=bins_number,density=True,alpha=0.5,label="model",color='red')
    plt.xlabel(r"$\log_{10}(E^3 \phi)$")
    plt.yscale('log')
    plt.legend()
    plt.title(f"Bin {bin}")
plt.show()










