# TODO: your agent here!

import random
from collections import namedtuple, deque
    
from keras import layers, models, optimizers
from keras import backend as K

import numpy as np
import copy


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


    
class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, SRS_DropoutFactor, SRS_NnDensity):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here
        self.DropoutFactor = SRS_DropoutFactor
        self.NnDensity = int(SRS_NnDensity)
        
        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions.
        
        
        Note that the raw actions produced by the output layer are in a [0.0, 1.0] range (using a sigmoid activation function). So, we 
        add another layer that scales each output to the desired range for each action dimension. This produces a deterministic action 
        for any given state vector. A noise will be added later to this action to produce some exploratory behavior.
        
        Another thing to note is how the loss function is defined using action value (Q value) gradients:
        """
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        
        # SRS: Introduce ReinforcementFactor and Dropout
        net = layers.Dense(units=32*self.NnDensity, activation='relu')(states)
        net = layers.Dropout(self.DropoutFactor)(net)
        net = layers.Dense(units=64*self.NnDensity, activation='relu')(net)
        net = layers.Dropout(self.DropoutFactor)(net)
        net = layers.Dense(units=32*self.NnDensity, activation='relu')(net)


        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        
        
        """
        These (Q value) gradients will need to be computed using the critic model, and fed in while training. 
        Hence it is specified as part of the "inputs" used in the training function:
        """
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

        
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, SRS_DropoutFactor, SRS_NnDensity):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here
        
        self.DropoutFactor = SRS_DropoutFactor
        self.NnDensity = int(SRS_NnDensity)

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values.
        
        It is simpler than the actor model in some ways, but there some things worth noting. Firstly, while the actor model is meant 
        to map states to actions, the critic model needs to map (state, action) pairs to their Q-values. This is reflected in the 
        input layers.
        """
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        # SRS: Introduce ReinforcementFactor and Dropout
        net_states = layers.Dense(units=32*self.NnDensity, activation='relu')(states)
        net_states = layers.Dropout(self.DropoutFactor)(net_states)
        net_states = layers.Dense(units=64*self.NnDensity, activation='relu')(net_states)

        # Add hidden layer(s) for action pathway
        # SRS: Introduce ReinforcementFactor and Dropout
        
        net_actions = layers.Dense(units=32*self.NnDensity, activation='relu')(actions)
        net_actions = layers.Dropout(self.DropoutFactor)(net_actions)
        net_actions = layers.Dense(units=64*self.NnDensity, activation='relu')(net_actions)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.
        
        """
        These two layers can first be processed via separate "pathways" (mini sub-networks), but eventually need to be combined. 
        This can be achieved, for instance, using the Add layer type in Keras (see Merge Layers https://keras.io/layers/merge/):
        """
        
        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        
        """
        The final output of this model is the Q-value for any given (state, action) pair. However, we also need to compute the 
        gradient of this Q-value with respect to the corresponding action vector, needed for training the actor model. This step needs 
        to be performed explicitly, and a separate function needs to be defined to provide access to these gradients:
        """

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
        
class DDPG():
    """Reinforcement Learning agent that learns using DDPG.
    
    We are now ready to put together the actor and policy models to build our DDPG agent. Note that we will need two copies of each 
    model - one local and one target. This is an extension of the "Fixed Q Targets" technique from Deep Q-Learning, and is used to 
    decouple the parameters being updated from the ones that are producing target values.
    
    Notice that after training over a batch of experiences, we could just copy our newly learned weights (from the local model) 
    to the target model. However, individual batches can introduce a lot of variance into the process, so it's better to perform a 
    soft update, controlled by the parameter tau.
    """
    def __init__(self, task, SRS_Parameters):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        
        
        """ Parameters from trainer """
        # Noise process
        SRS_exploration_mu = SRS_Parameters[0]
        #print(SRS_exploration_mu)
        
        SRS_exploration_theta = SRS_Parameters[1]
        #print(SRS_exploration_theta)
        
        SRS_exploration_sigma = SRS_Parameters[2]
        #print(SRS_exploration_sigma)

        # Replay memory
        SRS_buffer_size = int(SRS_Parameters[3])
        #print(SRS_buffer_size)
        
        SRS_batch_size = int(SRS_Parameters[4])
        #print(SRS_batch_size)

        # Algorithm parameters
        SRS_gamma = SRS_Parameters[5]
        #print(SRS_gamma)
        
        SRS_tau = SRS_Parameters[6]
        #print(SRS_tau)
        
        SRS_DropoutFactor = SRS_Parameters[7]
        
        SRS_NnDensity = SRS_Parameters[8]
        """ Parameters from trainer """
        
        
        
        # Actor (Policy) Model
        self.actor_local = Actor(self.state_size, self.action_size, self.action_low, self.action_high, SRS_DropoutFactor, SRS_NnDensity)
        self.actor_target = Actor(self.state_size, self.action_size, self.action_low, self.action_high, SRS_DropoutFactor, SRS_NnDensity)

        # Critic (Value) Model
        self.critic_local = Critic(self.state_size, self.action_size, SRS_DropoutFactor, SRS_NnDensity)
        self.critic_target = Critic(self.state_size, self.action_size, SRS_DropoutFactor, SRS_NnDensity)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = SRS_exploration_mu #0
        self.exploration_theta = SRS_exploration_theta #0.15
        self.exploration_sigma = SRS_exploration_sigma #0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = SRS_buffer_size #100000
        self.batch_size = SRS_batch_size #64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = SRS_gamma #0.99  # discount factor
        self.tau = SRS_tau #0.01  # for soft update of target parameters
        

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor_local.model.predict(state)[0]
        return list(action + self.noise.sample())  # add some noise for exploration

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)


class OUNoise:
    """Ornstein-Uhlenbeck process.
    
    We'll use a specific noise process that has some desired properties, called the Ornstein–Uhlenbeck process. 
    It essentially generates random samples from a Gaussian (Normal) distribution, but each sample affects the next one such that two 
    consecutive samples are more likely to be closer together than further apart. In this sense, the process in Markovian in nature.

    Why is this relevant to us? We could just sample from Gaussian distribution, couldn't we? Yes, but remember that we want to use 
    this process to add some noise to our actions, in order to encourage exploratory behavior. And since our actions translate to 
    force and torque being applied to a quadcopter, we want consecutive actions to not vary wildly. Otherwise, we may not actually get 
    anywhere! Imagine flicking a controller up-down, left-right randomly!

    Besides the temporally correlated nature of samples, the other nice thing about the OU process is that it tends to settle down 
    close to the specified mean over time. When used to generate noise, we can specify a mean of zero, and that will have the effect 
    of reducing exploration as we make progress on learning the task.
    """

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state