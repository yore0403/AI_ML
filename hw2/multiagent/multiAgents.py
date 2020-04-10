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
from game import Actions
import random, util, math
import search,searchAgents


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
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #print scores
        #print legalMoves[chosenIndex]

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
        newPosX,newPosY = newPos
        curPos = currentGameState.getPacmanPosition()
        evaluateScoresWeight = [10.0, 50.0,80.0, 100.0,100000.0]#for evaulate weight
        #[food,ghost,capsule,ghostcapsule,ghosteaten]
        expDecayFactors = [0.25, 1, 0.6,0,0.7]
        #[food,ghost,capsule]  

        if currentGameState.isWin() or successorGameState.isWin():
            return MAXNUM
        elif currentGameState.isLose() or successorGameState.isLose():
            return -MAXNUM

        stopPenalty = -100.0

        score = 0.0
        foodScore = 0.0 
        ghostScore = 0.0
        capsuleScore = 0.0

        # food evaluation 
        curFood = currentGameState.getFood()
        newFood = successorGameState.getFood()

        


        foodList = curFood.asList()

        # less food higher score
        foodScore += currentGameState.hasFood(newPosX,newPosY)*evaluateScoresWeight[FOOD]
        #foodScore -= evaluateScoresWeight[FOOD] * len(foodList)

        minFoodDistance = MAXNUM
        #minFoodDistance = min(map(lambda x: util.manhattanDistance(newPos, x), foodList))
        #minFoodDistance = min(map(lambda x: searchAgents.mazeDistance(newPos, x,currentGameState), foodList))
        minFoodDistance = minMazeDistance(successorGameState)
        #print newPos, minFoodDistance

        # same numfood see nearst food distance  
        foodScore -= minFoodDistance*evaluateScoresWeight[FOOD]*0.01


        # ghost evaluation     
        curGhostStates = currentGameState.getGhostStates()
        newGhostStates = successorGameState.getGhostStates()
        
        ghostEvaluation = 0.0
        minActiveGhostDistance = MAXNUM
        minScaredGhostDistance = MAXNUM
        numActiveGhost = 0

        for ghostState in curGhostStates:
            ghostDistance = searchAgents.mazeDistance(newPos, (int(ghostState.getPosition()[0]),int(ghostState.getPosition()[1])),currentGameState)
            if ghostState.scaredTimer < 2: # dont chase for assurance
                numActiveGhost += 1
                if minActiveGhostDistance > ghostDistance:
                    minActiveGhostDistance = ghostDistance
            
                ghostEvaluation -= math.exp(-1.0 * expDecayFactors[GHOST] * ghostDistance) # run
                if ghostDistance < 2:
                    ghostScore -= math.pow(10,6-2*ghostDistance)# ghost too close!!! run !!! 
            else:
                ghostEvaluation += 20*math.exp(-1.0 * expDecayFactors[GHOSTEATEN] * ghostDistance)# chase
                if ghostDistance < 1 and newPos == ghostState.start.pos:
                    ghostEvaluation -= 25*math.exp(-1.0 * expDecayFactors[GHOSTEATEN] * ghostDistance)        
        ghostScore += evaluateScoresWeight[GHOST]*ghostEvaluation 
        ghostScore += numActiveGhost*evaluateScoresWeight[GHOSTEATEN]

        #capsule evaluation

        capsuleList = currentGameState.getCapsules()
        numCapsule = len(capsuleList)
        minCapsuleDistance = MAXNUM
        capsuleEvaluation = 0.0

        for capsule in capsuleList:
            capsuleDistance = searchAgents.mazeDistance(newPos, capsule, currentGameState)
            if minCapsuleDistance > capsuleDistance:
                minCapsuleDistance = capsuleDistance
            capsuleEvaluation += math.exp(-1.0 * expDecayFactors[CAPSULE] * capsuleDistance) # run
        
        # eat capsule when ghost close enough 
        capsuleScore += (minCapsuleDistance < minActiveGhostDistance)*capsuleEvaluation*(1/(ghostDistance*ghostDistance/8+0.875))*evaluateScoresWeight[CAPSULE]
        # compensate the score which minus by activeGhost
        capsuleScore -= numCapsule*evaluateScoresWeight[GHOSTEATEN]*2


        #emergency
        #if ghostDistance <= 2 and minCapsuleDistance < minActiveGhostDistance:
        #    ghostScore -= math.pow(10,7-2*ghostDistance)# ghost too close!!! run !!! 



        
        #print action, foodScore,ghostScore,capsuleScore

        score += foodScore + ghostScore + capsuleScore

        #print score

        if action == Directions.STOP:
            score += stopPenalty
        return score

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
    """
      Your minimax agent (question 2)
    """
    def minimax(self, state, depth, agent = 0):
        actions = state.getLegalActions(agent)
        
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), Directions.STOP
        if (agent + 1)%state.getNumAgents() == 0:depth = depth-1
        scores = [self.minimax(state.generateSuccessor(agent, action), depth, (agent + 1)%state.getNumAgents())[0] for action in actions]
        if agent == 0:
            bestScore = max(scores)
        else:
            bestScore = min(scores)

        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return bestScore, actions[chosenIndex]




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
        """
        "*** YOUR CODE HERE ***"
        mm = self.minimax(gameState, self.depth, 0)
        return mm[1]
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def alphabeta(self, state, depth, agent, alpha = -100000000, beta = 100000000):
        actions = state.getLegalActions(agent)
        
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), Directions.STOP
        """
        if (agent + 1)%state.getNumAgents() == 0:depth = depth-1
        print depth
        scores = [self.alphabeta(state.generateSuccessor(agent, action), depth, (agent + 1)%state.getNumAgents(),alpha, beta)[0] for action in actions]
   
        if agent == 0:
            v = -MAXNUM
            for action in actions:
                v = max(v,self.alphabeta(state.generateSuccessor(agent, action), depth , (agent + 1)%state.getNumAgents(),alpha, beta)[0])
                if v >= beta: return v,action
                alpha = max(alpha,v)
        else:
            v = MAXNUM
            for action in actions:
                v = min(v,self.alphabeta(state.generateSuccessor(agent, action), depth , (agent + 1)%state.getNumAgents(),alpha, beta)[0])
                if alpha >= v: return v,action
                beta = min(beta,v)
        
        bestIndices = [index for index in range(len(scores)) if scores[index] == v]
        chosenIndex = random.choice(bestIndices)
        return v, actions[chosenIndex]
        """
        v = 100000000
        if (agent + 1)%state.getNumAgents() == 0:depth = depth-1
        scores = [self.alphabeta(state.generateSuccessor(agent, action), depth, (agent + 1)%state.getNumAgents(),alpha, beta)[0] for action in actions]

        if agent == 0:
            scores.append(-v)
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)
            if bestScore > beta:
                return bestScore, actions[chosenIndex]
            alpha = max(alpha,bestScore)
        else:
            scores.append(v)
            bestScore = min(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            chosenIndex = random.choice(bestIndices)
            if alpha > bestScore:
                return bestScore, actions[chosenIndex]
            beta = min(beta,bestScore)

        return bestScore, actions[chosenIndex]
        
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ab = self.alphabeta(gameState, self.depth, 0)

        return ab[1]
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """
    def minimax(self, state, depth, agent = 0):

        actions = state.getLegalActions(agent)
        
        if  state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state), Directions.STOP
        
        scores = [self.minimax(state.generateSuccessor(agent, action), depth - 1, (agent + 1)%state.getNumAgents())[0] for action in actions]

        if agent == 0:
            bestScore = max(scores)
        else:
            bestScore = min(scores)
        """
            if not scores:
                bestScore = -1000000.0
            else:
                bestScore = sum(scores) / float(len(scores))
        """
        bestIndices = [index for index in range(len(scores)) if scores[index] >= bestScore]
        """
        print scores
        print bestScore
        print "i",bestIndices
        """
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best
        return bestScore, actions[chosenIndex]

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.minimax(gameState, self.depth * 2, 0)[1]
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>

    """

    "*** YOUR CODE HERE ***"
    pos = currentGameState.getPacmanPosition()
    evaluateScoresWeight = [10.0, 50.0,80.0, 100.0,1000.0]#for evaulate weight
    #[food,ghost,capsule,ghostcapsule,ghosteaten]
    expDecayFactors = [0.25, 0.8, 0.6,0,0.6]
    #[food,ghost,capsule]  

    if currentGameState.isWin():
        return MAXNUM
    elif currentGameState.isLose():
        return -MAXNUM

    score = 0.0
    foodScore = 0.0 
    ghostScore = 0.0
    capsuleScore = 0.0

    # food evaluation 
    food = currentGameState.getFood()
    foodList = food.asList()

    # less food higher score
    foodScore -= evaluateScoresWeight[FOOD] * len(foodList)

    mindistance = MAXNUM
    #mindistance = min(map(lambda x: util.manhattanDistance(pos, x), foodList))
    #mindistance = min(map(lambda x: searchAgents.mazeDistance(pos, x,currentGameState), foodList))
    minFoodDistance = minMazeDistance(currentGameState)

    # same numfood see nearst food distance  
    foodScore -= minFoodDistance*evaluateScoresWeight[FOOD]*0.05


    # ghost evaluation     
    ghostStates = currentGameState.getGhostStates()
    
    ghostEvaluation = 0.0
    minActiveGhostDistance = MAXNUM
    minScaredGhostDistance = MAXNUM
    numActiveGhost = 0

    for ghostState in ghostStates:
        ghostDistance = searchAgents.mazeDistance(pos, (int(ghostState.getPosition()[0]),int(ghostState.getPosition()[1])),currentGameState)
        
        if ghostState.scaredTimer < 2: # dont chase for assurance
            numActiveGhost += 1
            if minActiveGhostDistance > ghostDistance:
                minActiveGhostDistance = ghostDistance
            ghostEvaluation -= math.exp(-1.0 * expDecayFactors[GHOST] * ghostDistance) # run
        else:

            #if minScaredGhostDistance > ghosDistance:
            #    minScaredGhostDistance = ghostDistance
            #if ghostState.scaredTimer < ghostDistance:
            ghostEvaluation += 20*math.exp(-1.0 * expDecayFactors[GHOSTEATEN] * ghostDistance)# chase
            if ghostDistance == 0 and pos == ghostState.start.pos:
                ghostEvaluation -= 25*math.exp(-1.0 * expDecayFactors[GHOSTEATEN] * ghostDistance)



    ghostScore += evaluateScoresWeight[GHOST]*ghostEvaluation 
    ghostScore += numActiveGhost*evaluateScoresWeight[GHOSTEATEN]

    #capsule evaluation

    capsuleList = currentGameState.getCapsules()
    numCapsule = len(capsuleList)
    minCapsuleDistance = MAXNUM
    capsuleEvaluation = 0.0

    for capsule in capsuleList:
        capsuleDistance = searchAgents.mazeDistance(pos, capsule, currentGameState)
        if minCapsuleDistance > capsuleDistance:
            minCapsuleDistance = capsuleDistance
        capsuleEvaluation += math.exp(-1.0 * expDecayFactors[CAPSULE] * capsuleDistance) # run
    
    # eat capsule when ghost close enough 
    capsuleScore += (minCapsuleDistance < minActiveGhostDistance)*capsuleEvaluation*(1/(ghostDistance*ghostDistance/8+0.875))*evaluateScoresWeight[CAPSULE]
    # compensate the score which minus by activeGhost
    capsuleScore -= numCapsule*evaluateScoresWeight[GHOSTEATEN]*2


    #emergency
    #if ghostDistance <= 2 and minCapsuleDistance < minActiveGhostDistance:
    #    ghostScore -= math.pow(10,7-2*ghostDistance)# ghost too close!!! run !!! 

    #print foodScore, "/",ghostScore, "/",capsuleScore + random.random()

    score+= foodScore + ghostScore + capsuleScore

    return score

    """



    pos = currentGameState.getPacmanPosition()



    evaluateScoresWeight = [1000.0, 500.0, 200.0, 5000.0,10000.0]#for evaulate weight

    
    expDecayFactors = [0.25, 0.6, 0.75]
    #food decay normally and will get higher score when near food gathering
    #ghost decay faster but increase rapid when near
    #capsule increase rapidly when ghost is also near     
    stopPenalty = -3000.0

    score = 0.0
    foodScore = 0.0 
    ghostScore = 0.0
    capsuleScore = 0.0
    # food evaluation 
    food = currentGameState.getFood()
    foodScore += evaluateScoresWeight[0] * 80 # basic score less food decease less

    foodList = food.asList()
    # less food higher score
    foodScore -= evaluateScoresWeight[0]*20 * len(foodList)

    if not foodList:
        score += 100000000 #win


    minDistanceFood = (-1,-1)
    mindistance = 10000
    for food in foodList:
        dist = searchAgents.mazeDistance(food, pos,currentGameState)
        if dist < mindistance:
            mindistance = dist
            minDistanceFood =food
    # same food see if nearst food distance  
    print (10 - mindistance)*evaluateScoresWeight[0]*0.2
    foodScore += (10 - mindistance)*evaluateScoresWeight[0]*0.3
    

    # ghost evaluation 
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    ghostPos = []
    for state in ghostStates:
        ghostPos+=state.getPosition()


    ghostEvaluation = 0.0
    minGhostDistance = 100000.0
    if len(ghostPos) < (currentGameState.getNumAgents() - 1):
        ghostScore += evaluateScoresWeight[3]

    for ghostState in ghostStates:
        ghostDistance = searchAgents.mazeDistance(pos, (int(ghostState.getPosition()[0]),int(ghostState.getPosition()[1])),currentGameState)
        if minGhostDistance > ghostDistance:
            minGhostDistance = ghostDistance
        if ghostState.scaredTimer == 0:
            ghostEvaluation+= 1
        elif ghostState.scaredTimer > 39:
            print "eaten"
            ghostEvaluation+= 3*math.exp(-1.0 * expDecayFactors[2] * ghostDistance) #for eat capsule 
        if ghostState.scaredTimer < 2: # dont chase for assurance
            ghostEvaluation -= math.exp(-1.0 * expDecayFactors[2] * ghostDistance) # run
        else:
            ghostEvaluation += 20*math.exp(-1.0 * expDecayFactors[2]*0.8 * (ghostDistance))# chase
    ghostScore += evaluateScoresWeight[3]*ghostEvaluation
  
    #capsule evaluation

    capsuleList = currentGameState.getCapsules()
    minCapsuleDistance = 100000.0
    capsuleEvaluation = 0.0
    


    for capsule in capsuleList:
        capsuleDistance = searchAgents.mazeDistance(pos, capsule, currentGameState)

        if minCapsuleDistance > capsuleDistance:
            minCapsuleDistance = capsuleDistance
        capsuleEvaluation += math.exp(-1.0 * expDecayFactors[1] * capsuleDistance) # run
    
    capsuleScore += (minCapsuleDistance < minGhostDistance)*capsuleEvaluation*(1/(ghostDistance*ghostDistance/8+0.825))*evaluateScoresWeight[4]

    if ghostDistance <= 2 and minCapsuleDistance < minGhostDistance:
        ghostScore -= math.pow(10,7-ghostDistance)# ghost too close!!! run !!! 
    if ghostDistance == 0:
        score -= 100000000


    print foodScore, "/",ghostScore, "/",capsuleScore
    score+= foodScore + ghostScore + capsuleScore

    return score
"""


# Abbreviation
better = betterEvaluationFunction

MAXNUM = 10000000000 #1e8
FOOD = 0
GHOST = 1
CAPSULE = 2
GHOSTCAPSULE = 3
GHOSTEATEN = 4



def mindistance(corners, position):
    if len(corners) == 0:
        return 0
 
    hn = []  
    for location in corners:
        dis = abs(location[0] - position[0]) + abs(location[1] - position[1])
        corners.remove(location)
        dis += mindistance(corners, location)
        hn.append(dis)
    return min(hn)


"""    
class _BreadthFirstSearch(_GraphSearch):
    def __init__(self,start_state):
        _GraphSearch.__init__(self,start_state)
        self.frontier = util.Queue()
        global node
        node = Node(start_state,0,0,0)
        self.frontier.push(node)
    #frontier is queue
    #put node
    def put_frontier(self,node):
        self.frontier.push(node)



def recursive_bfs(problem,_bfs,result):
    
    node = _bfs.pop_frontier()
    node_state = node.get_state()

    node_cost = node.get_cost()
    problem.getSuccessors(node.get_state())
    if problem.isGoalState(node_state):
        _bfs.put_explored(node)
        result = _bfs.search_path(node,result)

        return result
    else:
        for successor in problem.getSuccessors(node_state):
            in_explored = _bfs.in_explored(successor[0])
            in_frontier = _bfs.in_frontier(successor[0])
            if (in_explored == None and in_frontier == None):
                if successor[0] != node.get_predecessor():
                    node2 = Node(successor[0],node_state,successor[1],node_cost+successor[2])
                    _bfs.put_frontier(node2)

        _bfs.put_explored(node)
        recursive_bfs(problem,_bfs,result)

def breadthFirstSearch(problem):
    #Search the shallowest nodes in the search tree first.
    "*** YOUR CODE HERE ***"

    
    result = []
    _bfs = _BreadthFirstSearch(problem.getStartState())
    recursive_bfs(problem,_bfs,result)
    return result
"""

class NearestFoodSearchProblem:
    """
    A search problem associated with finding the a path that  
    nearest food (dots) in a Pacman game.

    A search state in this problem is a tuple ( pacmanPosition, foodGrid ) where
      pacmanPosition: a tuple (x,y) of integers specifying Pacman's position
      foodGrid:       a Grid (see game.py) of either True or False, specifying remaining food
    """
    def __init__(self, currentGameState):
        self.pos = currentGameState.getPacmanPosition()
        self.food = currentGameState.getFood()
        self.walls = currentGameState.getWalls()
        self.currentGameState = currentGameState
        self._expanded = 0 # DO NOT CHANGE
        self.heuristicInfo = {} # A dictionary for the heuristic to store information

    def getStartState(self):
        return (self.pos,self.food)

    def isGoalState(self, state):
        return self.food[state[0][0]][state[0][1]]

    def getSuccessors(self, state):
        "Returns successor states, the actions they require, and a cost of 1."
        successors = []
        self._expanded += 1 # DO NOT CHANGE
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state[0]
            dx, dy = Actions.directionToVector(direction)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()
                nextFood[nextx][nexty] = False
                successors.append( ( ((nextx, nexty), nextFood), direction, 1) )
        return successors
def minMazeDistance(gameState):
    prob = NearestFoodSearchProblem(gameState)
    return len(search.bfs(prob))