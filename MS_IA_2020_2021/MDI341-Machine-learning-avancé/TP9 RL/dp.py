#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on March 26, 2021
@author: Thomas Bonald <bonald@enst.fr>
"""
import numpy as np


class PolicyEvaluation:
    """Evaluation of a policy."""
    
    def __init__(self, environment, policy, gamma=1, n_iter=100, init_value=0, n_states_max=10000):
        self.environment = environment
        self.policy = policy
        self.gamma = gamma
        self.n_iter = n_iter
        self.init_value = init_value
        self.get_states()
        if self.n_states < n_states_max:
            self.get_rewards()
            self.get_transition_matrix()
            self.get_values()
        else:
            print("The state space is too large for this basic implementation of Policy Evaluation.")
        
    def get_states(self):
        self.states = self.environment.get_states()
        self.n_states = len(self.states)
        self.state_id = {self.environment.encode(state): i for i, state in enumerate(self.states)}

    def get_rewards(self):
        rewards = np.zeros(self.n_states)
        for state in self.states:    
            i = self.state_id[self.environment.encode(state)]  
            rewards[i] = self.environment.get_reward(state)
        self.rewards = rewards
    
    def get_transition_matrix(self):
        transition_matrix = np.zeros((self.n_states, self.n_states))
        for state in self.states:    
            i = self.state_id[self.environment.encode(state)]
            if not self.environment.is_terminal(state):
                for prob, action in zip(*self.policy(state)):
                    probs, states = self.environment.get_transition(state, action)
                    indices = np.array([self.state_id[self.environment.encode(s)] for s in states])
                    transition_matrix[i, indices] += prob * np.array(probs)
        self.transition_matrix = transition_matrix

    def init_values(self):
       return self.init_value * np.ones(self.n_states)
        
    def get_values(self):
        values = self.init_values()
        for t in range(self.n_iter):
            values = self.transition_matrix.dot(self.rewards + self.gamma * values)
        self.values = values
        
    def improve_policy(self):
        best_actions = dict()
        for state in self.states: 
            i = self.state_id[self.environment.encode(state)]
            actions = self.environment.get_actions(state)
            values_actions = []
            for action in actions:
                probs, states = self.environment.get_transition(state, action)
                indices = np.array([self.state_id[self.environment.encode(s)] for s in states])
                value = np.sum(np.array(probs) * (self.rewards + self.gamma * self.values)[indices])
                values_actions.append(value)
            best_actions[i] = actions[np.argmax(values_actions)]
        policy = lambda state: [[1], [best_actions[self.state_id[self.environment.encode(state)]]]]
        return policy
