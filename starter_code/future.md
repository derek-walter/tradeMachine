# tradeMachine
Stock Trading Beginnings

### Thank You to:   
Source|Link|For   
------- |  -------- | -------   
**Galvanize Staff** | [GitHub](https://github.com/gSchool) | Everything
**Siraj Raval** | [GitHub](https://github.com/llSourcell) | ML / Time Series
**Derrick Mwiti** | [Medium](https://heartbeat.fritz.ai/using-a-keras-long-shortterm-memory-lstm-model-to-predict-stock-prices-a08c9f69aa74) | LSTM Stocks

Note - I reused no code, but ideas are just as important.   
I make no guarantees, though you probably don't want to use this model.

#### The goal is to create a stock trading machine.  
![classes](resources/classes.svg)   

This has varying levels of complexity but as an MVP we want the following:  
* Collect Time Based Data: Prices, News, Multiple Stocks  
* Create a Neural Network to predict data through time  
* Extras: Execute trades, manage a portfolio, RL

#### Limited Project Scope for Capstone 2  
![classes_initial](resources/classes_initial.svg)   

With the above structure, we can focus on the "bot".  
The bot will analyze a data stream and make predictions...
Once deployed, the bot will need to:
- [ ] Check data instances or start new data instance (allows multiple bots on same stock)
- [ ] Train up to current time on given stock (cram)
- [ ] Train on new data as time progresses (update)
- [ ] Be controlled easily (reset, change parameters, kill)  
- [ ] And obviously, predict stuff / learn

### Review of Current Methods  
#### Time Series Methods   
- [ ] ARIMA    
- [ ] RNN  
- [x] RNN + LSTM  
- [ ] Gut Feel (Humans)  

#### Stock Trading (IMO)  
- [ ] Intraday/High Frequency (Technicals, Statistics, Some Indicators)  
- [ ] EOD (Technicals, Indicators, News)  
- [x] Swing/Position (Macro News, Specific News, Indicators)   
- [ ] Trend Trading (Buy, Sell, Hold, Ride Trend)  

#### Reinforcement Learning   
- [x] Q Learning    
- [ ] Policy Gradients   
- [ ] Actor-Critic (Deep Deterministic Policy Gradients for continuous actions)   

### Putting it Together  (Thoughts)

The flow of information through the bot will be as follows:  
The handler provides the bot with state information, rewards, and a "done" variable (from environment interaction)
State information is the length of the time window, and is passed element
by element into an activation function to encourage convergence (Not sure about this, see image below)  
State information is passed elementwise into input nodes  
Typical cascade through the network, except output is Buy, Sell, Hold  (May use LinearACTV to elicit confidence in scalar value, something like a sigmoid removes information there)  

![State](resources/state.svg)

##### The rewards are where everything gets interesting:  
The bot needs to learn:
* that a buy now enables a sell later
* A lower buy than sell yields reward  
* That holding is a valid strategy   
Likely, the time it waits between transactions, it's hungriness for profit and other things it learns will
be directly related to our reward scheme. I may train on OpenAI Stock Gym while i get other stuff working.

1. Buy
> The buy needs to include a punishment, in extreme cases it also would include a volume but that may require extensive training  

2. Sell
> The sell will be a function of the bots inventory. If the inventory is empty it cant sell.   
If it can sell, (I may include another set of logic for volume of sale)  It will reap reward. **But** the data we have is differenced prices, hence we need to include raw prices that the bot can easily access in order to calculate size of reward.

3. Hold
> It must learn to be patient if the price isn't moving.   
I may want to include some tiny reward to encourage the swing trading / minimum purchase needs of my account

4. Exploration
> I am thinking of forcing the bot to buy occasionally at early time steps but having epsilon decay with the total number of buys the bot has made. In this way, I'll force some reward back into the expectation.


#####The Data Engine will be made up of:  

Level | Source | Destination  
----- | ------ | ------  
Stock Data | From API | To Clean Method  
Stationary Data | From Method | To Intermediate Data Storage
News | From API | To Clean/NLP Method
Sentiment | From NLP Method | To Intermediate Data
Intermediate Data | From local | To Environment.

You'll notice the modularity of the above. I can easily add or remove methods for NLP, differencing, or whatever.
