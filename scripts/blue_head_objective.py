"""Strictly convex frozen-feature blue-logit objective and deterministic Newton solve."""
import numpy as np


class Objective:
    def __init__(self,old,weights,new,new_logits,negative_weight,ridge):
        self.old,self.new = np.asarray(old,np.float64),np.asarray(new,np.float64)
        self.weights,self.logits = np.asarray(weights,np.float64),np.asarray(new_logits,np.float64)
        if (self.old.ndim != 2 or self.new.ndim != 2 or self.old.shape[1] != self.new.shape[1]
                or not len(self.old) or not len(self.new) or self.weights.shape != (len(self.old),)
                or self.logits.shape != (len(self.new),) or not np.isfinite(negative_weight) or negative_weight < 0
                or not np.isfinite(ridge) or ridge <= 0 or np.any(self.weights <= 0)
                or any(not np.isfinite(v).all() for v in (self.old,self.new,self.weights,self.logits))):
            raise ValueError('Invalid convex objective inputs')
        self.negative_weight = float(negative_weight)
        self.quadratic = (self.old.T*self.weights)@self.old/len(self.old)+ridge*np.eye(self.old.shape[1])

    def evaluate(self,delta):
        delta = np.asarray(delta,np.float64)
        if delta.shape != (self.old.shape[1],) or not np.isfinite(delta).all():
            raise ValueError('Invalid parameter vector')
        z = self.logits+self.new@delta
        p = np.exp(-np.logaddexp(0.,-z))
        value = .5*delta@self.quadratic@delta+self.negative_weight*np.logaddexp(0.,z).mean()
        gradient = self.quadratic@delta+self.negative_weight*(self.new.T@p)/len(self.new)
        hessian = self.quadratic+self.negative_weight*(self.new.T*(p*(1-p)))@self.new/len(self.new)
        return float(value),gradient,hessian


def solve(objective,max_iterations=100,tolerance=1e-8):
    if type(max_iterations) is not int or max_iterations < 1 or not 0 < tolerance < 1:
        raise ValueError('Invalid solver settings')
    delta = np.zeros(objective.old.shape[1],np.float64)
    trace = []
    for iteration in range(max_iterations+1):
        value,gradient,hessian = objective.evaluate(delta)
        norm = float(np.max(np.abs(gradient)))
        trace.append(dict(iteration=iteration,loss=value,gradient_inf=norm))
        if not np.isfinite(value) or not np.isfinite(hessian).all():
            raise ValueError('Nonfinite objective')
        if norm <= tolerance:
            return delta,trace
        if iteration == max_iterations:
            break
        direction = np.linalg.solve(hessian,-gradient)
        directional = float(gradient@direction)
        if directional >= 0:
            raise ValueError('Newton direction is not descending')
        step = 1.
        while step >= 2.**-40:
            candidate = delta+step*direction
            if objective.evaluate(candidate)[0] <= value+1e-4*step*directional:
                delta = candidate
                break
            step *= .5
        else:
            raise ValueError('Backtracking failed')
    raise ValueError('Newton solver did not converge')
