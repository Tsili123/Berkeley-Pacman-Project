# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        current_food = currentGameState.getFood()
        food_list = current_food.asList()

        for states in newGhostStates:
            if states.getPosition() == newPos:
                return float("-inf") 

        food_distance = float("-inf")
        for food_position in food_list:#evaluation function returns best score(max_value)
            #I used negative manhattan distance to have accurate corresponding
            distance = -manhattanDistance(newPos, food_position)
            if distance > food_distance:
                food_distance = distance

        return food_distance

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def min_agent(gameState, agent_index, depth):
            total_agents = gameState.getNumAgents()

            if EndGame(gameState, depth):
                return self.evaluationFunction(gameState)

            score = float("+inf")
            direction = 0
            for action in gameState.getLegalActions(agent_index):
                game_state = gameState.generateSuccessor(agent_index, action)

                if agent_index  == total_agents-1:
                    direction = max_agent(game_state, 0, depth + 1)
                else:
                    direction = min_agent(game_state, agent_index + 1, depth)

                if direction < score:
                    score = direction

            return score

        def max_agent(gameState, agent_index, depth):
            total_agents = gameState.getNumAgents()
            #total agents is the number of ghosts + 1

            if EndGame(gameState, depth):
                return self.evaluationFunction(gameState)

            score = float("-inf")
            direction = 0
            for action in gameState.getLegalActions(agent_index):
                game_state = gameState.generateSuccessor(agent_index, action)

                # check if agent_index is the last ghost 
                if agent_index == total_agents - 1:
                    direction = max_agent(game_state, 0, depth + 1)#maximizer turn , pacman plays
                else:
                    direction = min_agent(game_state, agent_index + 1, depth)#minimizer turn , ghost play

                if direction > score:
                    score = direction

            return score
        
        def EndGame(gameState, depth):
            return depth == self.depth or gameState.isLose() or gameState.isWin()


        best_action = []
        
        score = float("-inf")
        # self.index is always 0.
        for action in gameState.getLegalActions(self.index):
            #print(self.index)

            # Game state fro the specific action
            game_state = gameState.generateSuccessor(self.index, action)

            direction = min_agent(gameState=game_state, agent_index=1, depth=0)

            if direction > score:
                score = direction
                best_action = action

        return best_action
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def min_agent(gameState, agent_index, depth, alpha, beta):
            total_agents = gameState.getNumAgents()

            if EndGame(gameState, depth):
                return self.evaluationFunction(gameState)

            score = float("+inf")
            direction = 0
            for action in gameState.getLegalActions(agent_index):
                game_state = gameState.generateSuccessor(agent_index, action)

                if agent_index  == total_agents-1:
                    direction = max_agent(game_state, 0, depth + 1, alpha, beta)
                else:
                    direction = min_agent(game_state, agent_index + 1, depth, alpha, beta)

                if direction < score:
                    score = direction
                
                if direction < beta:#update best value for minimizer
                    beta = direction

                if direction < alpha:#prune
                    return score


            return score

        def max_agent(gameState, agent_index, depth , alpha, beta):
            total_agents = gameState.getNumAgents()
            #total agents is the number of ghosts + 1

            if EndGame(gameState, depth):
                return self.evaluationFunction(gameState)

            score = float("-inf")
            direction = 0
            for action in gameState.getLegalActions(agent_index):
                game_state = gameState.generateSuccessor(agent_index, action)

                # check if agent_index is the last ghost
                if agent_index == total_agents - 1:
                    direction = max_agent(game_state, 0, depth + 1, alpha, beta)#maximizer turn , pacman plays
                else:
                    direction = min_agent(game_state, agent_index + 1, depth, alpha, beta)#minimizer turn , ghost play
                
                if direction > score:
                    score = direction
                
                if direction > alpha:#update best value for maximizer
                    alpha = direction

                if direction > beta:#prune,it might get higher, but it can't get any lower for minimizer
                    return score

            return score
        
        def EndGame(gameState, depth):
            return depth == self.depth or gameState.isLose() or gameState.isWin()


        best_action = []
        
        score = float("-inf")
        alpha = float("-inf")
        beta  = float("+inf")
        #alpha:best already explored option along the path to the root for the maximizer
        #beta :best already explored option along the path to the root for the minimizer

        # self.index is always 0.
        for action in gameState.getLegalActions(self.index):
            #print(self.index)

            # Game state fro the specific action
            game_state = gameState.generateSuccessor(self.index, action)

            direction = min_agent(gameState=game_state, agent_index=1, depth=0, alpha=alpha, beta=beta)
            #update score
            if direction > score:
                score = direction
                best_action = action
            #maximizer starts first always
            if direction > alpha:
                    alpha = direction

        return best_action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        #pacman
        def max_agent(gameState, depth , index):
            total_agents = gameState.getNumAgents()
            if EndGame(gameState, depth):
                return self.evaluationFunction(gameState)

            max_val = float("-inf")
            for action in gameState.getLegalActions(index):
                successor_state = gameState.generateSuccessor(index, action)
                if index == total_agents - 1 :
                    expect_val = max_agent(successor_state, depth+1 ,0)
                else:
                    expect_val =  expectimax_agent(successor_state, depth, index + 1)

                if expect_val > max_val:
                    max_val = expect_val

            return max_val

        # every ghost
        #same as above
        def expectimax_agent(gameState, depth, index):
            total_agents = gameState.getNumAgents()
            if EndGame(gameState, depth):
                return self.evaluationFunction(gameState)


            action_count = 0
            for i in gameState.getLegalActions(index):
                action_count += 1

            actions = gameState.getLegalActions(index)

            chance = 1.0/action_count
            expecti_val = 0

            for action in actions:
                successor_state = gameState.generateSuccessor(index, action)
                if index == total_agents - 1 :
                    temp = max_agent(successor_state, depth+1 ,0)
                else:
                    temp =  expectimax_agent(successor_state, depth, index + 1)
                expecti_val += temp*chance #compute propability
            return expecti_val 

        def EndGame(gameState, depth):
            return depth == self.depth or gameState.isLose() or gameState.isWin()

        score = float("-inf")

        for actions in gameState.getLegalActions(0):#get actions for pacman
            successor_state = gameState.generateSuccessor(0, actions)
            direction =  expectimax_agent(successor_state , 0, 1)#ghost turn
            if direction > score:
                score = direction
                best_action = actions
        return best_action


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    
    # Useful info
    position = currentGameState.getPacmanPosition()
    ghost_state = currentGameState.getGhostStates()
    food_state = currentGameState.getFood()
    ret_val  = 0
    
    # find closest food and try avoiding ghosts
    min_food_dist = float("+inf")
    # find closest ghost
    min_ghost_dist = float("+inf")

    for ghosts in ghost_state:  
        distance = manhattanDistance(ghosts.getPosition(),position)
        if distance < min_ghost_dist:
            min_ghost_dist = distance
            min_ghost_state = ghosts
    
    # if ghost is scared then try to chase it
    if(min_ghost_state.scaredTimer > 0):
        ret_val += 1.0/(1+min_ghost_dist)#Taylor series

    # find closest food distance 
    for foods in food_state.asList():
        distance = manhattanDistance(position, foods)
        if distance < min_food_dist:
            min_food_dist = distance

    ret_val +=  currentGameState.getScore() + 1.0/(1+min_food_dist) - 1.0/(1+min_ghost_dist)
    return ret_val

# Abbreviation
better = betterEvaluationFunction
