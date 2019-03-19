# Import Stuff Here

'''
This is a bot object. It's job is to deploy on a stock and train it's model.
Later, it will run on datastreams and continue training it's model.
TODO:
Initialize models, perhaps multiple.
Setup interaction with the environment.

'''

class bot:
    '''This bot is the agent in the environment and runs on profits.
    '''
    def __init__(self, things):
        '''Initialize all things here.
        Inputs:
        Returns:
        '''
        # Data Initialization method on stock

        # Base Initializations

        # Model Initializations

        # Cram Initializations

        # Advance Initializations

        pass

    # Init Methods
    def populate(self, arg):
        '''Call to data object.
        Check for stock existence
        if not:
            Call data engine and wait for stock data.
        '''
        pass

    def _mind(self, things):
        '''Initialize model here.
        may be a few so as to not init multiple for checks.

        Inputs:
        Returns:
        '''
        pass

    # Main Methods
    def cram(self, things, stock):
        '''A method to get up to speed after called on a given stock. Includes
        isntantiating a base model, and training. Then passing over to 'advance'.

        Inputs:
        Returns:
        '''
        pass

    def advance(self, things):
        '''To continue learning as time progresses after initialization.

        Inputs:
        Returns:
        '''
        pass

    # Tertiary Methods
    def act(self, things):
        '''A basic class to force an action given a state.

        Inputs:
        Returns:
        '''
        pass

    def replay(self, things):
        '''A method to use hindsight experience replay for value estimation.

        Inputs:
        Returns:
        '''
        pass
