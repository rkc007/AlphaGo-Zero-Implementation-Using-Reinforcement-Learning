Our source code consists of the following files inside the utils_6 folder :
	policyValueNet.py : Architecture and training procedure for the neural network
	MCTS.py : Gives policy for a given state after running MCTS for 100 simulations. 
	selfPlay.py : Starts training the model after loading the latest checkpoint
	utils.py : Contains helper functions to be used for MCTS & Policy-Value Network
	config.py : Parameters required for the configuration
	enums.py : Enumeration class for BLACK & WHITE colours

To install the dependencies of this project, run:
pip install -r requirements.txt


To download the model in `model_6` folder:
sh download_model_6.sh


To start the train (best checkpoint will be automatically loaded):
sh train.sh


To start playing with AlphaGoPlayer_6.py against a random/fixed agent, execute:
python tournament.py


To visualise the decrement in the loss of policy & value (inside the utils_6 folder):
tensorboard logdirs='./logs' --port=6006