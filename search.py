# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was Insed by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def GraphSearching(problem, fringe, fringeIns):
    
    # initialize the  set to be empty
    closed = set()   

    # initialize fringe
    startingState = (problem.getStartState(), 0, [])  # node format : (state, priority , direction)

    #print(startingState)

    fringeIns(fringe, startingState, 0)      

    # loop  
    while not fringe.isEmpty():
        
        # pick a leaf node and remove it from the fringe
        (state,priority,direction) = fringe.pop()

        # if the node contains a goal state then return the corresponding solution
        if problem.isGoalState(state):
             return direction

        # Ins the node to the closed set
        if not state in closed:
            closed.add(state)

            # expand the picked node, Insing the nodes to the fringe
            # because we might have many succesors for one state , create variables for each successor
            for succState, succDirection, succPriority in problem.getSuccessors(state):
                varDirection = direction + [succDirection]
                varPriority = priority + succPriority         
                varState = (succState, varPriority , varDirection)
                fringeIns(fringe, varState, varPriority)

    # if the fringe is empty then return failure
    return "Failure!Empty fringe."

def fringeIns(fringe, node, prio):
        fringe.push(node)  # node is like : (state, priority , direction)

def fringeIns2(fringe, node, prio):
        fringe.push(node,prio) 

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    fringe = util.Stack()    # use stack data structure (LIFO)

    return GraphSearching(problem, fringe, fringeIns)

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    
    fringe = util.Queue()    # use queue data structure (FIFO)
    
    return GraphSearching(problem, fringe, fringeIns)

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"

    fringe = util.PriorityQueue()     #priotity queue data structure

    return GraphSearching(problem, fringe, fringeIns2)

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"

    def fringeIns3(fringe, nodes, prio):
        prio = prio + heuristic(nodes[0], problem)   # f(n)=g(n)+h(n), heuristic = nullHeuristic
        fringe.push(nodes,prio)  
        
    fringe = util.PriorityQueue()    #priotity queue data structure

    return GraphSearching(problem, fringe, fringeIns3)



# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
