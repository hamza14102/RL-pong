---
jupyter:
    jupytext:
        formats: ipynb,md
        text_representation:
            extension: .md
            format_name: markdown
            format_version: "1.3"
            jupytext_version: 1.14.5
    kernelspec:
        display_name: Python 3 (ipykernel)
        language: python
        name: python3
---

# Reinforcement Learning

-   `submitted.py`: Your homework. Edit, and then submit to <a href="https://www.gradescope.com/courses/486387">Gradescope</a>.
-   `mp11_notebook.ipynb`: This is a <a href="https://anaconda.org/anaconda/jupyter">Jupyter</a> notebook to help you debug. You can completely ignore it if you want, although you might find that it gives you useful instructions.
-   `pong.py`: This is a program that plays Pong. If called interactively, it will call the module `pong_display.py` to create a display, so that you can play. If told to use a Q-learner, it will call your `submitted.py` to do Q-learning.
-   `tests/test_visible.py`: This file contains about half of the <a href="https://docs.python.org/3/library/unittest.html">unit tests</a> that Gradescope will run in order to grade your homework. If you can get a perfect score on these tests, then you should also get a perfect score on the additional hidden tests that Gradescope uses.
-   `requirements.txt`: This tells you which python packages you need to have installed, in order to run `grade.py`. You can install all of those packages by typing `pip install -r requirements.txt` or `pip3 install -r requirements.txt`.

### Table of Contents

1. <a href="#section1">Playing Pong</a>
1. <a href="#section2">Creating a Q-Learner Object</a>
1. <a href="#section3">Epsilon-First Exploration</a>
1. <a href="#section4">Q-Learning</a>
1. <a href="#section5">Saving and Loading Your Q and N Tables</a>
1. <a href="#section6">Exploitation</a>
1. <a href="#section7">Acting</a>
1. <a href="#section8">Training</a>
1. <a href="#section9">Extra Credit</a>

<a id='section1'></a>

## Playing Pong

Pong was the <a href="https://en.wikipedia.org/wiki/Pong">first video game produced by Atari.</a> It is a simple game, based on table tennis. Here is a two-person version of the game: https://commons.wikimedia.org/wiki/File:Pong_Game_Test2.gif

We will be playing a one-person version of the game:

-   When the ball hits the top, bottom, or left wall of the playing field, it bounces.
-   The right end of the playing field is open, except for the paddle. If the ball hits the paddle, it bounces, and the player's score increments by one. If the ball hits the open space, the game is over; the score resets to zero, and a new game begins.

The game is pretty simple, but in order to get a better feeling for it, you may want to try playing it yourself. Use the up arrow to move the paddle upward, and the down arrow to move the paddle downward. See how high you can make your score:

```python
!python3 pong.py
```

Once you figure out how to use the arrow keys to control your paddle, we hope you will find that the game is not too hard for a human to play. However, for a computer, it's difficult to know: where should the paddle be moved at each time step? In order to see how difficult it is for a computer to play, let's ask the "random" player to play the game.

**WARNING:** The following line will open a pygame window. The pygame window will be hidden by this window -- in order to see it, you will need to minimize this window. The pygame window will consume a lot of CPU time just waiting for the processor, so in order to kill it, you will need to come back to this window, click on the block below, then click the Jupyter "stop" button (the square button at the top of this window) in order to stop processing.

```python
!python3 pong.py --player random
```

<a id='section2'></a>

## Creating a Q-Learner Object

The first thing you will do is to create a `q_learner` object that can store your learned Q table and your N table (table of exploration counts).

Like any other object-oriented language, python permits you to create new object classes in order to store data that will be needed from time to time. If you are not already very, very familiar with python classes, you might want to study the python class tutorial: https://docs.python.org/3/tutorial/classes.html

Like any other object in python, a `q_learner` object is created by calling its name as a function, e.g., `my_q_learner=submitted.q_learner()`. Doing so calls the function `submitted.q_learner.__init__()`. Let's look at the docstring to see what it should do.

```python
import submitted, importlib
importlib.reload(submitted)
help(submitted.q_learner.__init__)
```

Write your `__init__` function to meet the requirements specified in the docstring. Once you have completed it, the following code should run without errors:

```python
importlib.reload(submitted)

q_learner = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])

print(q_learner)

```

<a id='section3'></a>

## Epsilon-First Exploration

In order to manage the exploration/exploitation tradeoff, we will be using both "epsilon-first" and "epsilon-greedy" (https://en.wikipedia.org/wiki/Multi-armed_bandit#Semi-uniform_strategies).

The epsilon-first strategy explores every state/action pair at least `nfirst` times before it ever starts to exploit any strategy. Your `q_learner` should have a table to keep track of how many times it has explored a state/action pair prior to the start of any exploitation. The method for storing that table is up to you; in order to have some standardized API, therefore, you need to write a method called `report_exploration_counts` that returns a list of the three exploration counts for a given state.

```python
importlib.reload(submitted)
help(submitted.q_learner.report_exploration_counts)
```

Write `report_exploration_counts` so that it returns a list or array for any given state. Test your code with the following:

```python
importlib.reload(submitted)
q_learner = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
print('This is how many times state [0,0,0,0,0] has been explored so far:')
print(q_learner.report_exploration_counts([0,0,0,0,0]))
print('This is how many times state [9,9,1,1,9] has been explored so far:')
print(q_learner.report_exploration_counts([9,9,1,1,9]))
```

When your learner first starts learning, it will call the function `choose_unexplored_action` to choose an unexplored action. This function should choose a function uniformly at random from the set of unexplored actions in a given state, if there are any:

```python
importlib.reload(submitted)
help(submitted.q_learner.choose_unexplored_action)
```

If this has been written correctly, the following block should generate a random sequence of actions. If the next block produces the same action 5 times in a row, that is the wrong result, and the result would be that your code does not pass the autograder.

```python
importlib.reload(submitted)
q_learner = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))

```

After all three actions have been explored `nfirst` times, the function `choose_unexplored_action` should return `None`, as shown here:

```python
importlib.reload(submitted)
q_learner = submitted.q_learner(0.05,0.05,0.99,1,[10,10,2,2,10])
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))
print('Next action:',q_learner.choose_unexplored_action([9,9,1,1,9]))

```

<a id='section4'></a>

## Q-Learning

The reinforcement learning we are implementing is called Q-learning (https://en.wikipedia.org/wiki/Q-learning).

Q-learning keeps a table $Q[s,a]$ that specifies the expected utility of action $a$ in state $s$. The organization of this table is up to you. In order to have a standard API, the first thing you should implement is a function `report_q` with the following docstring:

```python
importlib.reload(submitted)
help(submitted.q_learner.report_q)
```

When your `q_learner` is first initialized, the value of $Q[state,action]$ should be zero for all state/action pairs, thus the `report_q` function should return lists of zeros:

```python
importlib.reload(submitted)
q_learner=submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
print('Q[0,0,0,0,0] is now:',q_learner.report_q([0,0,0,0,0]))
print('Q[9,9,1,1,9] is now:',q_learner.report_q([9,9,1,1,9]))
```

There are actually many different Q-learning algorithms available, but when people refer to Q-learning with no modifier, they usually mean the time-difference (TD) algorithm. For example, this is the algorithm that's described on the wikipedia page (https://en.wikipedia.org/wiki/Q-learning). This is the algorithm you will implement for this MP.

In supervised machine learning, the learner tries to imitate a reference label. In reinforcement learning, there is no reference label. Q-learning replaces the reference label with a "local Q" value, which is the utility that was obtained by performing action $a$ in state $s$ one time. It is usually calculated like this:

$$Q_{local}(s_t,a_t) = r_t + \gamma\max_{a_{t+1}}Q(s_{t+1},a_{t+1})$$

where $r_t$ is the reward that was achieved by performing action $a_t$ in state $s_t$, $s_{t+1}$ is the state into which the game transitioned, and $a_{t+1}$ is one of the actions that could be performed in that state. $Q_{local}$ is computed by your `q_local` function, which has this docstring:

```python
importlib.reload(submitted)
help(submitted.q_learner.q_local)
```

Initially, `q_local` should just return the given reward, because initially, all Q values are 0:

```python
importlib.reload(submitted)
q_learner = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,10,10])
print('Q_local(6.25,[9,9,1,1,9]) is currently:',q_learner.q_local(6.25,[9,9,1,1,9]))
```

Now you can use `q_learner.q_local` as the target for `q_learner.learn`. The basic algorithm is

$$Q(s,a) = Q(s,a) + \alpha (Q_{local}(s,a)-Q(s,a))$$

Here is the docstring:

```python
importlib.reload(submitted)
help(submitted.q_learner.learn)
```

The following block checks a sequence of Q updates:

1. First, $Q([9,9,1,1,9],-1)$ is updated. Since all Q values start at zero, it will be updated to just have a value equal to $\alpha$ (0.05) times the given reward (6.25) for a total value of 0.3125.
1. When we print out $Q([9,9,1,1,9],:)$, we see that one of the elements has been updated.
1. Next, update $Q([9,9,1,1,8],1)$ with a given reward, and with $[9,9,1,1,9]$ as the given next state. Since $Q([9,9,1,1,9],-1)$ is larger than zero, the next-state Q-value should be multiplied by $\gamma$ (0.99) and added to the reward (3.1), then multiplied by $\alpha$, giving a total value of 0.17046875.
1. The resulting Q-value is reported.

```python
importlib.reload(submitted)
q_learner = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
q_learner.learn([9,9,1,1,9],-1,6.25,[0,0,0,0,0])
print('Q[9,9,1,1,9] is now',q_learner.report_q([9,9,1,1,9]))
q_learner.learn([9,9,1,1,8],1,3.1,[9,9,1,1,9])
print('Q[9,9,1,1,8] is now',q_learner.report_q([9,9,1,1,8]))
```

<a id='section5'></a>

## Saving and Loading your Q and N Tables

After you've spent a long time training your `q_learner`, you will want to save your Q and N tables so that you can reload them later. The format of Q and N is up to you, therefore it's also up to you to write the `save` and `load` functions. Here are the docstrings:

```python
importlib.reload(submitted)
help(submitted.q_learner.save)
```

```python
importlib.reload(submitted)
help(submitted.q_learner.load)
```

These functions can be tested by doing one step of training one `q_learner`, then saving its results, then loading them into another `q_learner`:

```python
importlib.reload(submitted)
q_learner1 = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
print('Next action:',q_learner1.choose_unexplored_action([9,9,1,1,9]))
q_learner1.learn([9,9,1,1,9],-1,6.25,[0,0,0,0,0])
print('N1[9,9,1,1,8] is now',q_learner1.report_exploration_counts([9,9,1,1,9]))
print('Q1[9,9,1,1,8] is now',q_learner1.report_q([9,9,1,1,9]))
q_learner1.save('test.npz')

q_learner2 = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
print('N2[9,9,1,1,8] starts out as',q_learner2.report_exploration_counts([9,9,1,1,9]))
print('Q2[9,9,1,1,8] starts out as',q_learner2.report_q([9,9,1,1,9]))
q_learner2.load('test.npz')
print('N2[9,9,1,1,8] is now',q_learner2.report_exploration_counts([9,9,1,1,9]))
print('Q2[9,9,1,1,8] is now',q_learner2.report_q([9,9,1,1,9]))

```

<a id='section6'></a>

## Exploitation

A reinforcement learner always has to trade off between exploration (choosing an action at random) versus exploitation (choosing the action with the maximum expected utility). Before we worry about that tradeoff, though, let's first make sure that exploitation works.

```python
importlib.reload(submitted)
help(submitted.q_learner.exploit)
```

```python
importlib.reload(submitted)
q_learner1 = submitted.q_learner(0.05,0.05,0.99,5,[10,10,2,2,10])
q_learner1.learn([9,9,1,1,9],-1,6.25,[0,0,0,0,0])
print('Q1[9,9,1,1,9] is now',q_learner1.report_q([9,9,1,1,9]))
print('The best action and Q from state [9,9,1,1,9] are',q_learner1.exploit([9,9,1,1,9]))
```

<a id='section7'></a>

## Acting

When your learner decides which action to perform, it should trade off exploration vs. exploitation using both the epsilon-first and the epsilon-greedy strategies:

1. If there is any action that has been explored fewer than `nfirst` times, then choose one of those actions at random. Otherwise...
1. With probability `epsilon`, choose an action at random. Otherwise...
1. Exploit.

```python
importlib.reload(submitted)
help(submitted.q_learner.act)
```

In order to test all three types of action (epsilon-first exploration, epsilon-greedy exploration, and exploitation), let's create a learner with `nfirst=1` and `epsilon=0.25`, and set it so that the best action from state `[9,9,1,1,9]` is `-1`. With these settings, a sequence of calls to `q_learner.act` should produce the following sequence of actions:

1. The first three actions should include each possible action once.
1. After the first three actions, 3/4 of the remaining actions should be `-1`. The remaining 1/4 should be randomly chosen.

```python
importlib.reload(submitted)
q_learner=submitted.q_learner(0.05,0.25,0.99,1,[10,10,2,2,10])
q_learner.learn([9,9,1,1,9],-1,6.25,[0,0,0,0,0])
print('An epsilon-first action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-first action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-first action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
print('An epsilon-greedy explore/exploit action:',q_learner.act([9,9,1,1,9]))
```

<a id='section8'></a>

## Training

Now that all of your components work, you can try training your algorithm. Do this by giving your `q_learner` as a player to a new `pong.PongGame` object. Set `visibility=False` so that the `PongGame` doesn't create a new window.

```python
import pong, importlib, submitted
importlib.reload(pong)
help(pong.PongGame.__init__)
```

As you can see, we should set `visibility=False` so that the `PongGame` doesn't create a new window. We should also make sure that the PongGame uses the same state quantization as the learner.

```python
importlib.reload(pong)
importlib.reload(submitted)
state_quantization = [10,10,2,2,10]
q_learner=submitted.q_learner(0.05,0.05,0.99,5,state_quantization)

pong_game = pong.PongGame(learner=q_learner, visible=False, state_quantization=state_quantization)
print(pong_game)
```

In order to train our learner, we want it to play the game many times. To do that we use the PongGame.run function:

```python
help(pong_game.run)

```

In order to make sure our learner is learning, let's tell `pong_game.run` to output all 3 Q-values of all of the 4000 states in every time step.

To make sure that's not an outrageous amount of data, let's tell it to only output the Q values once/reward, and ask it to only collect 5000 rewards:

```python
states = [[x,y,vx,vy,py] for x in range(10) for y in range(10) for vx in range(2) for vy in range(2) for py in range(10) ]

scores, q_achieved, q_states = pong_game.run(m_rewards=500, states=states)

print('The number of games played was',len(scores))
print('The number of rewards was',len(q_states))
print('The size of each returned Q-matrix was',q_states[0].shape)

```

The returned value of `q_states` is a list of 4000x3 numpy arrays (20 states, 3 actions). The list contains `m_rewards` of these. We want to convert it into something that matplotlib can plot.

```python
import numpy as np

Q = np.array([np.reshape(q,-1) for q in q_states])
print('Q is now of shape',Q.shape)
print('the max absolute value of Q is ',np.amax(abs(Q)))
```

```python
%matplotlib inline

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14,6),layout='tight')
ax = [ fig.add_subplot(2,1,x) for x in range(1,3) ]
ax[0].plot(np.arange(0,len(q_states)),Q)
ax[0].set_title('Q values of all states')
ax[1].plot(np.arange(0,len(q_states)),q_achieved)
ax[1].set_title('Q values of state achieved at each time')
ax[1].set_ylabel('Reward number')
```

OK, now let's try running it for a much longer period -- say, 5000 complete games. We won't ask it to print out any states this time.

```python
scores, q_achieved, q_states = pong_game.run(m_games=5000, states=[])
# scores, q_achieved, q_states = pong_game.run(m_games=10000, states=[])

print('The number of games played was',len(scores))
print('The number of video frames was',len(q_states))
print('The size of each returned Q-matrix was',q_states[0].shape)
```

Now let's plot the score, to see if it improved over time. We will also plot the local average, averaged over 10 consecutive games, to see if that has improved. Notice that we can use `np.convolve` to compute the local average.

These numbers are really noisy, with a really large maximum. We will plot `np.log10(1+x)`, rather than x, so that we can better see the average numbers, and ignore the very large noisy spikes.

```python
%matplotlib inline

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(14,9),layout='tight')
ax = [ fig.add_subplot(3,1,x) for x in range(1,4) ]
ax[0].plot(np.arange(0,len(scores)),np.log10(1+np.array(scores)))
ax[0].plot([0,5000],np.log10([7,7]),'k--')
ax[0].set_title('Game scores')
ax[1].plot(np.arange(4991),np.log10(1+np.convolve(np.ones(10)/10,scores,mode='valid')))
ax[1].plot([0,4991],np.log10([7,7]),'k--')
ax[1].set_title('Game scores, average 10 consecutive games')
ax[2].plot(np.arange(0,len(q_achieved)),q_achieved)
ax[2].set_title('Q values of state achieved at each time')
ax[2].set_ylabel('Game number')
```

Hooray, it has learned! If you are getting a ten-game average score of better than 6, then you are ready to submit your model for grading. In order to do that, you need to save the model:

```python
q_learner.save('trained_model.npz')
```

<a id='section9'></a>
