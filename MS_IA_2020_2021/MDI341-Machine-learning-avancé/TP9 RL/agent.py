#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 26, 2021
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np
from copy import deepcopy


class Agent:
    """Agent. Default policy is purely random."""
    def __init__(self, environment, policy=None):
        if policy is not None:
            self.policy = policy
        else:
            self.policy = self.random_policy
        self.environment = environment
            
    def random_policy(self, state):
        actions = self.environment.get_actions(state)
        probs = np.ones(len(actions)) / len(actions)
        return probs, actions

    def get_action(self, state):
        action = None
        probs, actions = self.policy(state)
        if len(actions):
            i = np.random.choice(len(actions), p=probs)
            action = actions[i]
        return action


class OnlinePolicyEvaluation:
    """Online evaluation of a policy."""
    
    def __init__(self, environment, policy, gamma=1, alpha=0.5, eps=0.1, n_steps=1000, init_value=0):
        self.environment = environment
        self.policy = policy # exploration policy
        self.agent = Agent(environment, policy)
        self.gamma = gamma # discount rate
        self.alpha = alpha # learning rate
        self.eps = eps # exploration factor
        self.n_steps = n_steps
        self.init_value = init_value
        self.get_states()
        self.get_rewards()
        self.values = self.init_values()
        
    def get_states(self):
        self.states = self.environment.get_states()
        self.n_states = len(self.states)
        self.state_id = {self.environment.encode(state): i for i, state in enumerate(self.states)}

    def get_rewards(self):
        self.rewards = np.array([self.environment.get_reward(state) for state in self.states])
    
    def init_values(self):
       return self.init_value * np.ones(self.n_states)
        
    def get_episode(self):
        """Get the states and rewards for an episode."""
        self.environment.init_state()
        states = []
        rewards = []
        for t in range(self.n_steps):
            state = deepcopy(self.environment.state)
            states.append(state)
            action = self.agent.get_action(state)
            reward, stop = self.environment.step(action)
            rewards.append(reward)
            if stop:
                break
        return stop, states, rewards

    def improve_policy(self):
        """Improve policy using the current estimation of values."""
        best_action_id = np.zeros(len(self.states), dtype=int)
        for state in self.states:
            i = self.state_id[self.environment.encode(state)]
            actions = self.environment.get_actions(state)
            values_actions = []
            for action in actions:
                probs, states = self.environment.get_transition(state, action)
                indices = np.array([self.state_id[self.environment.encode(s)] for s in states])
                value = np.sum(np.array(probs) * (self.rewards + self.gamma * self.values)[indices])
                values_actions.append(value)
            best_action_id[i] = np.argmax(values_actions)
        # randomized policy for exploration
        def policy(state):
            actions = self.environment.get_actions(state)
            if len(actions) == 1:
                return [1], actions
            else:
                i = self.state_id[self.environment.encode(state)]
                probs = self.eps * np.ones(len(actions)) / len(actions)
                probs[best_action_id[i]] += 1 - self.eps
                return probs, actions
        self.agent = Agent(self.environment, policy)
        self.values = self.init_values()
        # greedy policy for exploitation
        def policy(state):
            actions = self.environment.get_actions(state)
            i = self.state_id[self.environment.encode(state)]
            action = actions[best_action_id[i]]
            return [1], [action]
        return policy    
        

class OnlineControl:
    """Online control."""
    
    def __init__(self, environment, gamma=1, alpha=0.1, eps=0.1, n_steps=10000, init_value=0):
        self.environment = environment
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.n_steps = n_steps
        self.init_value = init_value
        self.get_states()
        self.get_actions()
        self.get_rewards()
        self.init_action_values()
        if not len(self.states):
            print("Warning: Method 'get_states' not available in this environment.")
        
    def get_states(self):
        self.states = self.environment.get_states()
        self.state_id = {self.environment.encode(state): i for i, state in enumerate(self.states)}

    def get_actions(self):
        self.actions = self.environment.get_actions()
        self.action_id = {action: i for i, action in enumerate(self.actions)}
                
    def get_rewards(self):
        self.rewards = np.array([self.environment.get_reward(state) for state in self.states])
    
    def init_action_values(self):
        self.action_values = -np.inf * np.ones((len(self.states), len(self.actions)))
        for i, state in enumerate(self.states):
            index = np.array([self.action_id[action] for action in self.environment.get_actions(state)])
            self.action_values[i, index] = self.init_value

    def get_best_action(self, state):
        state_id = self.state_id[self.environment.encode(state)]
        best_value = np.max(self.action_values[state_id])
        best_action_ids = np.argwhere(self.action_values[state_id] == best_value).ravel()
        return self.actions[np.random.choice(best_action_ids)]
            
    def get_best_action_randomized(self, state):
        if np.random.random() < self.eps:
            actions = self.environment.get_actions(state)
            return actions[np.random.choice(len(actions))]
        else:
            return self.get_best_action(state)

    def get_policy(self):
        def policy(state):
            return [1], [self.get_best_action(state)]
        return policy

