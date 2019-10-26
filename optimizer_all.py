from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer
import tensorflow as tf
import numpy as np


class Grad(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, use_locking=False, name="myGrad"):
        super(Grad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._lr_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")

    def _create_slots(self, var_list):
        pass

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        var_update = state_ops.assign_sub(var, lr_t * grad)

        return control_flow_ops.group(*[var_update])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")


class Mom(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, bias_correction=True, use_locking=False, name="myMom"):
        super(Mom, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self.bias_correction = bias_correction

        self._lr_t = None
        self._beta1_t = None

        self._beta1_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1.0 - beta1_power if self.bias_correction else 1.0

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


class Adam(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAdam"):
        super(Adam, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)

        beta1_fix = 1 - beta1_power
        beta2_fix = 1 - beta2_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, beta2_t * v + (grad * grad) * (1 - beta2_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t / beta2_fix) + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AMSGrad(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="myAMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = tf.Variable(self._beta2, name="beta2_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "g", self._name)
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "h", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)

        beta1_fix = 1 - beta1_power
        beta2_fix = 1 - beta2_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        h = self.get_slot(var, "h")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, beta2_t * v + (grad * grad) * (1 - beta2_t), use_locking=self._use_locking)
        h_t = state_ops.assign(h, tf.maximum(h, v_t), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(h_t / beta2_fix) + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        grad_t = self.get_slot(var, "g")
        grad_t = state_ops.assign_sub(grad_t, tf.zeros_like(grad_t), use_locking=self._use_locking)
        grad_t = state_ops.scatter_add(grad_t, grad.indices, grad.values, use_locking=self._use_locking)
        return self._apply_dense(grad_t, var)

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(self._beta2_power * self._beta2_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2], name=name_scope)


class AdamMax(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-10, use_locking=False, name="AdamMax"):
        super(AdamMax, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and tf.contrib.eager.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = tf.Variable(self._beta1, name="beta1_power", trainable=False)

        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1 - beta1_power

        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")

        m_t = state_ops.assign(m, beta1_t * m + grad * (1 - beta1_t), use_locking=self._use_locking)
        v_t = state_ops.assign(v, tf.maximum(beta2_t * v, grad * grad), use_locking=self._use_locking)

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t) + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta1_fix = 1 - beta1_power

        m = self.get_slot(var, "m")
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, grad * (1 - beta1_t))

        v = self.get_slot(var, "v")
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, (grad * grad) * (1 - beta2_t))

        var_update = state_ops.assign_sub(var, lr_t * (m_t / beta1_fix) / (math_ops.sqrt(v_t) + epsilon_t), use_locking=self._use_locking)

        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(self._beta1_power * self._beta1_t, use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1], name=name_scope)


class AdaShift(optimizer.Optimizer):

    def __init__(self, learning_rate=0.001, keep_num=10, beta1=0.9, beta2=0.999, epsilon=1e-10, pred_g_op='max', use_locking=False, name="AdaShift"):
        super(AdaShift, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._keep_num = keep_num
        self._beta2 = beta2
        self._beta1 = beta1
        self._epsilon = epsilon
        self._pred_g_op = pred_g_op

        s = np.asarray([(self._beta1**(self._keep_num-i-1)) for i in range(self._keep_num)])
        self.s = s / np.sum(s) 
        
        self._lr_t = None
        self._beta2_t = None
        self._epsilon_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")
        self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon")

    def _create_slots(self, var_list):

        self.first_var = min(var_list, key=lambda x: x.name)

        for v in var_list:
            for i in range(self._keep_num+1):
                self._zeros_slot(v, "g%d" % i, self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "z", self._name)
            self._zeros_slot(v, "b2p", self._name)

    def _apply_dense(self, grad, var):

        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        g = [self.get_slot(var, "g%d" % i) for i in range(self._keep_num+1)]
        
        v = self.get_slot(var, "v")
        z = self.get_slot(var, "z")
        b2p = self.get_slot(var, "b2p")

        if self._pred_g_op == 'none':
            v_t = state_ops.assign(v, v * beta2_t + tf.square(g[0]) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'max':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_max(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        elif self._pred_g_op == 'mean':
            v_t = state_ops.assign(v, v * beta2_t + tf.reduce_mean(tf.square(g[0])) * (1 - beta2_t), use_locking=self._use_locking)
        else:
            assert False

        with ops.control_dependencies([v_t]):
            g_t = state_ops.assign(g[-1], grad, use_locking=self._use_locking)
            for i in range(self._keep_num):
                with ops.control_dependencies([g_t]):
                    g_t = state_ops.assign(g[i], g[i + 1], use_locking=self._use_locking)

        with ops.control_dependencies([g_t]):
            m_t = tf.reduce_sum([g[i]*self.s[i] for i in range(self._keep_num)], axis=0)
            # m_t = tf.reduce_mean(g[:self._keep_num], axis=0)

        with ops.control_dependencies([v_t]):
            z_t = state_ops.assign(z, tf.cast(tf.logical_or(v_t > 0.0, z > 0.0), tf.float32))

        b2p_t = state_ops.assign(b2p, b2p * beta2_t * tf.sign(z_t) + (1.0 - tf.sign(z_t)), use_locking=self._use_locking)
        b2_fix = tf.maximum(1e-8, 1.0 - b2p_t)

        step_t = z_t * m_t / (math_ops.sqrt(v_t / b2_fix) + epsilon_t)

        # if var.name == self.first_var.name: #'discriminator/final_linear/w:0':
        #     idx = 0
        #     step_t = tf.Print(step_t, [z_t[idx]], 'z_t', summarize=1000)
        #     step_t = tf.Print(step_t, [g[i][idx] for i in range(len(g))], 'g', summarize=1000)
        #     step_t = tf.Print(step_t, [grad[idx]], 'grad', summarize=1000)
        #     step_t = tf.Print(step_t, [b2p_t[idx]], 'b2p_t', summarize=1000)
        #     step_t = tf.Print(step_t, [b2_fix], 'beta2_fix', summarize=1000)
        #     step_t = tf.Print(step_t, [m_t[idx]], 'm_t', summarize=1000)
        #     step_t = tf.Print(step_t, [tf.sqrt(v_t / b2_fix)[idx]], 'v_t', summarize=1000)
        #     step_t = tf.Print(step_t, [step_t], 'step', summarize=1000)

        var_update = state_ops.assign_sub(var, lr_t * step_t, use_locking=self._use_locking)
        return control_flow_ops.group(*([var_update]))

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        raise Exception()

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(x, i, v, use_locking=self._use_locking))

    def _finish(self, update_ops, name_scope):
        with ops.control_dependencies(update_ops):
            return control_flow_ops.group(*update_ops, name=name_scope)