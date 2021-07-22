import torch
from torch.autograd import Variable
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)

###################################################
################## DIVERGENCES ####################
###################################################

def kl_divergence_kumaraswamy(prior_a, a, b):
	"""
	KL divergence for the Kumaraswamy distribution.

	:param prior_a: torch tensor: the prior a concentration
	:param prior_b: torch tensor: the prior b concentration
	:param a: torch tensor: the posterior a concentration
	:param b: torch tensor: the posterior b concentration
	:param sample: a sample from the Kumaraswamy distribution

	:return: scalar: the kumaraswamy kl divergence
	"""

	Euler = torch.tensor(0.577215664901532)
	kl = (1 - prior_a / a) * (-Euler - torch.digamma(b) - 1./b)\
		 + torch.log(a*b /prior_a) - (b-1)/b

	return kl.sum()



def kl_divergence_normal(prior_mean, prior_scale, posterior_mean, posterior_scale):
	"""
	 Compute the KL divergence between two Gaussian distributions.

	:param prior_mean: torch tensor: the mean of the prior Gaussian distribution
	:param prior_scale: torch tensor: the scale of the prior Gaussian distribution
	:param posterior_mean: torch tensor: the mean of the posterior Gaussian distribution
	:param posterior_scale: torch tensor: the scale of the posterior Gaussian distribution

	:return: scalar: the kl divergence between the prior and posterior distributions
	"""

	device = torch.device("cuda" if posterior_mean.is_cuda else "cpu")


	prior_scale_normalized = F.softplus(torch.Tensor([prior_scale],device=device),beta=10)
	posterior_scale_normalized = F.softplus(posterior_scale,beta=10)

	kl_loss = -0.5 + torch.log(prior_scale_normalized) - torch.log(posterior_scale_normalized) \
				+ (posterior_scale_normalized ** 2 + (posterior_mean - prior_mean)**2) / (2 * prior_scale_normalized**2)


	return kl_loss.sum()
# scan for deeper children  
def get_children(model):
	model_children = list(model.children())
	for child in model_children:
		model_children+=get_children(child)
	return model_children

def model_kl_divergence_loss(model, kl_weight = 1.):
	"""
	Compute the KL losses for all the layers of the considered model.

	:param model: nn.Module extension implementing the model with our custom layers
	:param kl_weight: scalar: the weight for the KL divergences

	:return: scalar: the KL divergence for all the layers of the model.
	"""

	device = torch.device("cuda" if next(model.parameters()).is_cuda else "cpu")
	kl_sum = torch.Tensor([0]).to(device)

	# get the layers as a list
	model_children = get_children(model)#list(model.children())
	n=0.000001
	for layer in model_children:
		if hasattr(layer, 'loss'):
			kl_sum += layer.loss
			n += layer.n


	kl_sum=kl_sum/n
	return kl_weight * kl_sum[0]



###########################################
########## DISTRIBUTIONS ##################
###########################################


def kumaraswamy_sample(conc1, conc0, sample_shape):
	"""
	Sample from the Kumaraswamy distribution given the concentrations

	:param conc1: torch tensor: the a concentration of the distribution
	:param conc0: torch tensor: the b concentration of the distribution
	:param batch_shape: scalar: the batch shape for the samples

	:return: torch tensor: a sample from the Kumaraswamy distribution
	"""

	device = torch.device("cuda" if conc1.is_cuda else "cpu")

	x = (1e-6 - (1. - 1e-6)) * torch.rand(sample_shape).to(device) + (1. - 1e-6)
	q_u =  (1. - x **(1. /conc0))**(1. / conc1)

	return q_u

def bin_concrete_sample(a, temperature, hard = False, eps = 1e-8):
	""""
	Sample from the binary concrete distribution
	"""

	device = torch.device("cuda" if a.is_cuda else "cpu")
	U = torch.rand(a.shape)
	L = Variable(torch.log(U + eps) - torch.log(1. - U + eps)).to(device)
	X = torch.sigmoid((L + a) / temperature)
	#X.data = X.data.clamp(1e-4, 1. - 1e-3)
	#print('aaaaaa', X.max(), X.min())

	return X

def concrete_sample(a, temperature, hard = False, eps = 1e-8, axis = -1,rand=True):
	"""
	Sample from the concrete relaxation.

	:param probs: torch tensor: probabilities of the concrete relaxation
	:param temperature: float: the temperature of the relaxation
	:param hard: boolean: flag to draw hard samples from the concrete distribution
	:param eps: float: eps to stabilize the computations
	:param axis: int: axis to perform the softmax of the gumbel-softmax trick

	:return: a sample from the concrete relaxation with given parameters
	"""

	device = torch.device("cuda" if a.is_cuda else "cpu")
	U = torch.rand(a.shape,device=device)
	G = - torch.log(- torch.log(U + eps) + eps)
	if rand==True:
		a=a*1.0

	t = (a + Variable(G)) / temperature

	y_soft = F.softmax(t, axis)

	if hard:
		#_, k = y_soft.data.max(axis)
		_, k = a.data.max(axis)
		shape = a.size()

		if len(probs.shape) == 2:
			y_hard = a.new(*shape).zero_().scatter_(-1, k.view(-1, 1), 1.0)
		else:
			y_hard = a.new(*shape).zero_().scatter_(-1, k.view(-1, probs.size(1), 1), 1.0)


		y = Variable(y_hard - y_soft.data) + y_soft

	else:
		y = y_soft

	return y


###############################################
################ CONSTRAINTS ##################
###############################################
class parameterConstraints(object):
	"""
	A class implementing the constraints for the parameters of the layers.
	"""

	def __init__(self):
		pass

	def __call__(self, module):
		if hasattr(module, 'posterior_un_scale'):
			scale = module.posterior_un_scale
			scale = scale.clamp(-7., 1000.)
			module.posterior_un_scale.data = scale

		if hasattr(module, 'bias_un_scale'):
			scale = module.bias_un_scale
			scale = scale.clamp(-7., 1000.)
			module.bias_un_scale.data = scale

		if hasattr(module, 'conc1') and module.conc1 is not None:
			conc1 = module.conc1
			conc1 = conc1.clamp(-6., 1000.)
			module.conc1.data = conc1

		if hasattr(module, 'conc0') and module.conc0 is not None:
			conc0 = module.conc0
			conc0 = conc0.clamp(-6., 1000.)
			module.conc0.data = conc0

		if hasattr(module, 't_pi') and module.t_pi is not None:
			t_pi = module.t_pi
			t_pi = t_pi.clamp(-7, 600.)
			module.t_pi.data = t_pi

