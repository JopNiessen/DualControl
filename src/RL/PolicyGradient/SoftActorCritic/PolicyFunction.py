"""
Soft Actor-Critic: Policy function
"""

# import local libraries
from src.NeuralNetwork.Equinox import SimpleNetwork

# import global libraries
import jax.numpy as jnp
import jax.random as jrandom
import equinox as eqx
import optax


class SoftPolicyFunction:
    """
    Policy function
    """
    def __init__(self, dim, key, eta=1e-3):
        """
        Initialize network
        :param dim: network dimensions (n_inputs, n_hidden, n_output)
        :param key: PRNGKey
        :param eta: learning rate
        """
        n_states, n_controls = dim
        self.model = SimpleNetwork((n_states, 32, n_controls*2), key)
        self.optimizer = optax.sgd(eta)
        self.opt_state = self.optimizer.init(self.model)

        # create jit function
        self.grad = eqx.filter_value_and_grad
        #self.get_control = eqx.filter_jit(self._get_control)
        #self.sample_control = eqx.filter_jit(self._sample_control)
   
    def sample_control(self, mu, sigma, key):
        """
        Sample control
        :input mu: optimal control (network output)
        :input sigma: standard deviation (network output)
        :input key: PRNGKey
        :return: sampled control
        :return: log probability
        """
        control = mu + jrandom.normal(key, (1,)) * sigma
        log_prob = -.5 * ((control - mu) / sigma)**2 - jnp.log(sigma) + jnp.log(2*jnp.pi)/2
        return control, log_prob

    def loss_fn(self, model, state, q_func, key):
        """
        Calculate loss
        :input model: policy network
        :input state: state
        :input q_func: Q-function [function]
        :input key: PRNGKey
        :return: loss
        """
        (mu, sigma) = model(state)
        control, log_probs = self.sample_control(mu, sigma, key)

        q_value = q_func(state, control)
        
        loss = jnp.mean(log_probs - q_value)
        #entropy = None

        # Adding regularization
        #l2_loss = 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_util.tree_leaves(params))
        return loss #+ l2_loss

    def take_step(self, state, q_func, key):
        """
        Update policy network parameters
        :input state: state
        :input q_func: Q-function [function]
        :input key: PRNGKey
        :return: loss
        """
        loss, grads = self.grad(self.loss_fn)(self.model, state, q_func, key)
        updates, self.opt_state = self.optimizer.update(grads, self.opt_state)
        self.model = eqx.apply_updates(self.model, updates)
        return loss
    
    def log_prob(self, state, control):
        """
        Log probability given control Log p(control|state)
        :input state: state
        :input control: control
        :return: log probability [float]
        """
        (mu, sigma) = self.model(state)
        return -.5 * ((control - mu) / sigma)**2 - jnp.log(sigma) + jnp.log(2*jnp.pi)/2
    
    def get_control(self, state, key):
        """
        Fetch control
        :input state: state
        :input key: PRNGKey
        :return: sampled control
        :return: optimal control
        """
        mu, sigma = self.model(state)
        control = mu + jrandom.normal(key, (1,))*sigma
        #control = np.random.normal(mu, max(0, sigma))
        return control, mu


"""
    def predict(self, state):
        return jnp.dot(self.params, state)
    
    def get_control(self, state):
        u_star = self.predict(state)
        xi = np.random.normal()
        u = u_star + xi * self.stdev
        return u, u_star
    
    def objective(self, params, D, q_func):
        obj = 0
        N_samples = 5
        N = len(D)
        for s0_a, s0_b, _, _, _, _ in D:
            s0 = jnp.array([s0_a, s0_b])
            mu = jnp.dot(params, s0)
            exp_value = 0
            for i in range(N_samples):
                u = mu + np.random.normal(0, self.stdev)
                log_pi = -.5 * ((u - mu) / self.stdev)**2 - jnp.log(self.stdev) + jnp.log(2*jnp.pi)/2
                Q = q_func(s0, u)
                exp_value += (log_pi - Q)
            obj += exp_value / N_samples
            return jnp.mean(obj / N)

    def grad_phi(self, state, control, q_func):
        params = self.params
        grad_phi_log_pi = jax.grad(self.log_pi)(params, state, control)
        grad_u_log_pi = jax.grad(self.log_pi, argnums=2)(params, state, control)
        grad_Q = eqx.filter_grad(q_func)(control, state, output_value=True, reverse_order=True)
        return grad_phi_log_pi + (grad_u_log_pi - grad_Q)*state
    
    def update(self, D, q_func, key): #state, control, q_func):
        #grads = self.grad_phi(state, control, q_func)
        #grads = jax.grad(self.objective)(self.params, D, q_func)
        grads = jax.grad(self.loss_fn)(self.params, D, q_func, key)
        param_update = self.eta*grads
        norm = jnp.linalg.norm(param_update)
        if norm > self.gradient_clipping:
            param_update = param_update / norm
        self.params += param_update

    def log_pi(self, params, state, control):
        mu = jnp.dot(stop_gradient(params), state)
        prob = -.5 * ((stop_gradient(control) - mu) / self.stdev)**2 - jnp.log(self.stdev) + jnp.log(2*jnp.pi)/2
        return prob[0]

"""



