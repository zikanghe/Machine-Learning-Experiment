# Plot L96d
import numpy as np


# Default value of paramter (overwrited using the configuration files)




# trunc HMM (no param)


import numpy as np

#TODO: allow list of index instead of maxind

from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Layer, Dense
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from  tensorflow.keras import backend as K

def buildmodel(archi, m=36, reg=0.0001, batchlayer=1):
	inputs = Input(shape=(m,1))
	border = int(np.sum(np.array([kern//2 for nfil,kern,activ in archi])))
	x = Periodic1DPadding(padding_size=border)(inputs)
	x = BatchNormalization()(x)
	for i, (nfil, kern, activ) in enumerate(archi):
		if i == batchlayer:
			x = BatchNormalization()(x)
		x = Conv1D(nfil, kern, activation=activ)(x)
		#x = BatchNormalization()(x)
	output= Conv1D(1,1,activation='linear', kernel_regularizer=regularizers.l2(reg))(x)
	return Model(inputs,output)

def buildmodel2(m=36):
	inputs = Input(shape=36)
	x = BatchNormalization()(inputs)
	x = Dense(100,activation="relu")(x)
	x = Dense(50,activation="relu")(x)
	output = Dense(36,activity_regularizer=regularizers.L2(0.0001))(x)
	return Model(inputs,output)

class Periodic1DPadding(Layer):
	"""Add a periodic padding to the output

	# Arguments
		padding_size: tuple giving the padding size (left, right)

	# Output Shape
		input_shape+left+right
	"""


	def __init__ (self, padding_size, **kwargs):
		super(Periodic1DPadding, self).__init__(**kwargs)
		if isinstance(padding_size, int):
			padding_size = (padding_size, padding_size)
		self.padding_size = tuple(padding_size)

	def compute_output_shape( self, input_shape ):
		space = input_shape[1:-1]
		if len(space) != 1:
			raise ValueError ('Input shape should be 1D with channel at last')
		new_dim = space[0] + np.sum(self.padding_size)
		return (input_shape[0],new_dim,input_shape[-1])



	def build( self , input_shape):
		super(Periodic1DPadding,self).build(input_shape)

	def call( self, inputs ):
		vleft, vright = self.padding_size
		leftborder = inputs[:, -vleft:, :]
		rigthborder = inputs[:, :vright, :]
		return K.concatenate([leftborder, inputs, rigthborder], axis=-2)


