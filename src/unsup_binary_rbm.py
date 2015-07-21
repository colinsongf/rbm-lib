import numpy as np
import numpy.random as R
import theano
import theano.tensor as T
import time, math, pickle
import PIL

class RBM(object):
    
    #parameters for normal distribution used to initialize weight matrix
    weights_mean = 0
    weights_std_dev = .01
    
    def __init__(self, vis_input=None, vis_num=100, hid_num=100, weights=None, hid_bias=None, vis_bias=None, theano_rng=None):
        
        if weights is None:
            weights_init = np.asarray(R.normal(loc=self.weights_mean, scale=self.weights_std_dev, size=(vis_num, hid_num)), dtype=theano.config.floatX)
            
            #low_bound = -4 * math.sqrt(6.0 / (vis_num + hid_num))
            #high_bound = -low_bound
            #weights_init = np.asarray(R.uniform(low=low_bound, high=high_bound, size=(vis_num, hid_num)), dtype=theano.config.floatX)
            
            weights = theano.shared(value=weights_init, name='weights')
            
        if hid_bias is None:
            hid_bias = theano.shared(value=np.zeros(hid_num, dtype=theano.config.floatX), name='hid_bias')
            
        if vis_bias is None:
            vis_bias = theano.shared(value=np.zeros(vis_num, dtype=theano.config.floatX), name='vis_bias')
        
        if theano_rng is None:
            numpy_rng = R.RandomState(3142)
            theano_rng = T.shared_randomstreams.RandomStreams(numpy_rng.randint(2**31))
        
        self.vis_input = T.dmatrix('vis_input') if vis_input is None else vis_input
        
        self.vis_num = vis_num
        self.hid_num = hid_num
        
        self.weights = weights
        self.hid_bias = hid_bias
        self.vis_bias = vis_bias
        
        self.theano_rng = theano_rng
        
        self.params = [self.weights, self.hid_bias, self.vis_bias]
        
    def propup(self, vis_activation):
        
        hid_activation = T.dot(vis_activation, self.weights) + self.hid_bias
        hid_sigmoid = T.nnet.sigmoid(hid_activation)
        
        return [hid_activation, hid_sigmoid]
    
    def sample_hid_given_vis(self, vis_activation):
        
        hid_activation, hid_sigmoid = self.propup(vis_activation)
        hid_sample = self.theano_rng.binomial(size=hid_sigmoid.shape, n=1, p=hid_sigmoid, dtype=theano.config.floatX)
        
        return [hid_activation, hid_sigmoid, hid_sample]
    
    def propdown(self, hid_activation):
        
        vis_activation = T.dot(hid_activation, self.weights.T) + self.vis_bias
        vis_sigmoid = T.nnet.sigmoid(vis_activation)
        
        return [vis_activation, vis_sigmoid]
    
    def sample_vis_given_hid(self, hid_activation):
        
        vis_activation, vis_sigmoid = self.propdown(hid_activation)
        vis_sample = self.theano_rng.binomial(size=vis_sigmoid.shape, n=1, p=vis_sigmoid, dtype=theano.config.floatX)
        
        return [vis_activation, vis_sigmoid, vis_sample]
    
    def gibbs_hid_vis_hid(self, hid_sample_0):
        
        vis_activation_0, vis_sigmoid_0, vis_sample_0 = self.sample_vis_given_hid(hid_sample_0)
        hid_activation_1, hid_sigmoid_1, hid_sample_1 = self.sample_hid_given_vis(vis_sample_0)
        
        return [vis_activation_0, vis_sigmoid_0, vis_sample_0, hid_activation_1, hid_sigmoid_1, hid_sample_1]
    
    def gibbs_vis_hid_vis(self, vis_sample_0):
        
        hid_activation_0, hid_sigmoid_0, hid_sample_0 = self.sample_hid_given_vis(vis_sample_0)
        vis_activation_1, vis_sigmoid_1, vis_sample_1 = self.sample_vis_given_hid(hid_sample_0)
        
        return [hid_activation_0, hid_sigmoid_0, hid_sample_0, vis_activation_1, vis_sigmoid_0, vis_sample_1]
    
    def unsup_free_energy(self, vis_sample):
        
        hid_activation = T.dot(vis_sample, self.weights) + self.hid_bias
        vis_bias_term = T.dot(vis_sample, self.vis_bias)
        hid_activation_term = T.sum(T.log(1 + T.exp(hid_activation)), axis=1)
        
        free_energy = -vis_bias_term - hid_activation_term
        
        return free_energy
    
    def unsup_cost_updates(self, learning_rate=.1, persistent=None, k=1):
        
        if persistent is None:
            hid_activation_0, hid_sigmoid_0, hid_sample_0 = self.sample_hid_given_vis(self.vis_input)
            
            chain_start = hid_sample_0
        else:
            chain_start = persistent
            
        [vis_activations, vis_sigmoids, vis_samples, hid_activations, hid_sigmoids, hid_samples], updates = theano.scan(self.gibbs_hid_vis_hid, outputs_info=[None, None, None, None, None, chain_start], n_steps=k)
        
        chain_end = vis_samples[-1]
        
        pos_term = T.mean(self.unsup_free_energy(self.vis_input))
        neg_term = T.mean(self.unsup_free_energy(chain_end))
        
        cost = pos_term - neg_term
        
        grad_params = T.grad(cost, self.params, consider_constant=[chain_end])
        
        for grad_param, param in zip(grad_params, self.params):
            updates[param] = param - grad_param * T.cast(learning_rate, dtype=theano.config.floatX)
        
        if persistent:
            #chain_start for next sampling from persistent chain
            chain_start = hid_samples[-1]
            updates[persistent] = chain_start
            
            val_error = self.unsup_cost_pseudo_likelihood(updates)
        else:
            vis_reconstruction = vis_sigmoids[-1]
            
            val_error = self.unsup_reconstruction_cost(vis_reconstruction)
            
        return [val_error, updates]
        
    def unsup_cost_pseudo_likelihood(self, updates):
        
        index_bit_i = theano.shared(value=0, name='index_bit_i')
        
        #visible inputs rounded to the nearest bit
        vi_bit_i = T.iround(self.vis_input)
        #unsupervised free energy for rounded visible inputs
        vi_free_energy = self.unsup_free_energy(vi_bit_i)
        
        #visible input with bit i flipped (i.e. 0->1, 1->0)
        vi_flipped_bit_i = T.set_subtensor(vi_bit_i[:, index_bit_i], 1 - vi_bit_i[:, index_bit_i])
        #unsupervised free energy for visible inputs with bit i flipped
        vi_flipped_free_energy = self.unsup_free_energy(vi_flipped_bit_i)
        
        cost = T.mean(self.vis_num * T.log(T.nnet.sigmoid(vi_flipped_free_energy - vi_free_energy)))
        
        updates[index_bit_i] = (index_bit_i + 1) % self.vis_num
        
        return cost
    
    def unsup_reconstruction_cost(self, vis_reconstruction):
        
        cross_entropy = T.sum(-self.vis_input * T.log(vis_reconstruction) - (1 - self.vis_input) * T.log(1 - vis_reconstruction))
        
        cost = T.mean(cross_entropy)
        
        return cost
    
    def train(self, training_set, num_epoch=15, batch_size=20, learning_rate=.1, is_persistent=False, k=1, filter_display=False):
        
        if is_persistent:
            persistent_init = np.asarray(R.binomial(size=(batch_size, self.hid_num), n=1, p=.5), dtype=theano.config.floatX)
            persistent = theano.shared(value=persistent_init, name='persistent')
        else:
            persistent = None
        
        num_batches = math.ceil(training_set.get_value(borrow=True).shape[0] / float(batch_size))
        
        unsup_cost, updates = self.unsup_cost_updates(learning_rate, persistent, k)        
        
        index = T.lscalar()
        
        train_rbm = theano.function([index], unsup_cost, updates=updates, givens={self.vis_input: training_set[index * batch_size:(index + 1) * batch_size]})
        
        print('Starting training!')
        
        train_start = time.clock()
        plot_time = 0.0
        
        for epoch in range(num_epoch):            
            val_error = [train_rbm(batch_index) for batch_index in range(num_batches)]
            mean_val_error = np.mean(val_error)
            
            print('\tTraining epoch {!s}.  Cost is {!s}'.format(epoch, mean_val_error))
            
            if filter_display:
                plot_start = time.clock()
                
                save_weights_image(weights, 'epoch_{!s}_filters.png'.format(epoch), (28, 28), (10, 10))
                
                plot_end = time.clock()
                plot_time += plot_end - plot_start
        
        train_end = time.clock()
        train_time = (train_end - train_start) - plot_time
        
        print('Training on {!s} epochs completed in {!s} seconds!'.format(num_epoch, train_time))

def save_weights_image(weights, name, weights_shape, hid_num_shape):
    
    return PIL.Image.fromarray(utils.tile_raster_images(X=weights.get_value(borrow=True).T, image_shape=weights_shape, tile_shape=hid_num_shape, tile_spacing=(1,1)))

if __name__=='__main__':
    
    training_set_init = utils.read_mnist()
    training_set = theano.shared(value=np.asarray(training_set_init, dtype=theano.config.floatX), name='training_set', borrow=True)
    
    rbm = RBM(vis_num=784, hid_num=100)
    
    save_weights_image(rbm.weights, 'filters_init.png', (28, 28), (10, 10))
    
    rbm.train(training_set, learning_rate=.01, is_persistent=False, k=1, filter_display=True)