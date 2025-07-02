import numpy as np
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
import tensorflow as tf
tfd = tfp.distributions  
tfb = tfp.bijectors
#%%
N=3000
from sklearn.datasets import make_moons
#data=make_moons(N, noise=0.05)[0].astype("float32")

def halbmond(n):
    x2=np.random.normal(0,4,size=n)
    x1=np.random.normal((x2/2)**2,1)
    return np.stack([x1,x2],axis=1)
#data=halbmond(N)

def checkerboard(n):
    x1=np.random.rand(n) * 4 - 2
    x2=np.random.rand(n)-np.floor(x1) % 2
    data=np.stack([x1, x2*2-1],axis=1).astype("float32")
    return data * 2
data=checkerboard(N)

norm=tf.keras.layers.Normalization()
norm.adapt(data)
normalized_data=norm(data)
#%%
neuron=256
reg=0.01  #loss_ges=loss_train + reg*Σ(gewichte der einzelnen schicht)^2 bei regularizers.l2

def Coupling(input_shape,layers):
    input = tf.keras.layers.Input(shape=(input_shape,))
    s_layers=[input]
    t_layers=[input]
    for i in range(layers):
        s_layers.append(tf.keras.layers.Dense(neuron,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg))(s_layers[-1]))
        t_layers.append(tf.keras.layers.Dense(neuron,activation='relu',kernel_regularizer=tf.keras.regularizers.l2(reg))(t_layers[-1]))
    s_output_layer=tf.keras.layers.Dense(input_shape,activation='tanh',kernel_regularizer=tf.keras.regularizers.l2(reg))(s_layers[-1])  
    t_output_layer=tf.keras.layers.Dense(input_shape,activation='linear',kernel_regularizer=tf.keras.regularizers.l2(reg))(t_layers[-1])
    return tf.keras.Model(inputs=input, outputs=[s_output_layer,t_output_layer])

class RealNVP(tf.keras.Model):
    def __init__(self,n_coupling_layers):
        super().__init__()
        self.n_coupling_layers=n_coupling_layers
        self.base_dist=tfd.MultivariateNormalDiag(loc=[0,0],scale_diag=[1,1])
        self.masks=np.array([[0,1],[1,0]]*(n_coupling_layers//2),dtype="float32")
        self.loss_history=tf.keras.metrics.Mean(name="loss")
        self.layers_list=[Coupling(2,6) for i in range(n_coupling_layers)]
        
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
        return -tf.reduce_mean(log_prob)
            
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
            
#%%        
model=RealNVP(n_coupling_layers=8)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
history=model.fit(normalized_data, batch_size=512, epochs=300, verbose=2,validation_split=0.2)

#%%
plt.figure(figsize=(12,5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Verlauf')
plt.xlabel('Epoche')
plt.ylabel('Loss')
plt.legend()
plt.show()

z, _=model(normalized_data)
x, _=model(model.base_dist.sample(3000),training=False) 

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Zielverteilung")
plt.scatter(normalized_data[:,0],normalized_data[:,1])
plt.axis("equal")
plt.subplot(1,2,2)
plt.title("generierte Verteilung")
plt.scatter(x[:,0],x[:,1])
plt.axis("equal")
plt.show()
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.title("Basisverteilung")
plt.scatter(model.base_dist.sample(3000)[:,0],model.base_dist.sample(3000)[:,1])
plt.axis("equal")
plt.subplot(1,2,2)
plt.title("latenter Raum")
plt.scatter(z[:,0],z[:,1])
plt.axis("equal")
plt.show()
