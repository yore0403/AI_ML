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
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import sys
sys.setrecursionlimit(100000) 


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

def str2directions(var):
    from game import Directions
    return {
    'East': Directions.EAST,
    'West': Directions.WEST,
    'North': Directions.NORTH, 
    'South': Directions.SOUTH, 
    }[var]




class Node():
    def __init__(self,state = 0,predecesor = 0,action = 0,cost = 0):
        self.state = state
        self.predecesor = predecesor
        self.action = action
        self.cost = cost

    def __call__(self,state,predecesor,direction,cost):
        self.state = state
        self.predecesor = predecesor
        self.direction = direction
        self.cost = cost
    def get(self):
        return [self.state,self.predecesor,self.action,self.cost]
    def get_state(self):
        return self.state
    def get_predecessor(self):
        return self.predecesor
    def get_cost(self):
        return self.cost
    def get_direction(self):
        return self.action
    def compare(self,node2):
        return self.distance>node2.get_distance()
    def print_node(self):
        print [self.state,self.predecesor,self.direction,self.cost]


class _GraphSearch():
    def __init__(self,start_state):
        global node
        node = Node(start_state,0,0,0)
        self.explored = []
        self.goal = None

    def __call__(self,start_state):
        global node
        node = Node(start_state,0,0,0)
        self.frontier = [node]
        self.explored = []
        self.goal = Node()

    def set_goal(self,state):
        self.goal = state
    def get_goal(self):
        return self.goal

    #get property by node
    def get_cost(self,node):
        for node1 in self.explored:
            if node1 == node:
                return node1.get_cost()
    def get_node(self,node):
        for node1 in self.explored:
            if node1.get_state() == node:
                return node1
 
    #get last item with out pop
    def get_explored(self):
        return self.explored[-1]
    def get_frontier(self):
        return self.frontier[-1]
    
    #pop last item
    def pop_explored(self):
        return self.explored.pop()
    def pop_frontier(self):
        return self.frontier.pop()

    #put item
    def put_explored(self,node):
        self.explored.append(node)
    def put_frontier(self,node):
        self.frontier.append(node)

    #when c2>c1 replace n2 to n1 
    def refresh_frontier(self,node,node1):
        self.frontier.remove(node)
        self.frontier.push(node1)
    def refresh_explored(self,node,node1):
        self.explored.remove(node)
        self.frontier.push(node1)

    #when search done, find the min cost path 
    def search_path(self,node,result):

        if self.get_cost(node) == 0:
            result.reverse()
            return result
        else:
            result.append(node.get_direction())
            self.search_path(self.get_node(node.get_predecessor()),result)
   
    #print method
    def print_frontier(self):
        print "frontier: "
        for element in self.frontier.list:
            element.print_node()
    def print_explored(self):
        print "explored: "
        for element in self.explored:
            element.print_node()

    #check wether the node state is in list
    def in_explored(self,node_state):
        
        for node in self.explored:
            if node.get_state() == node_state:
                return node
        return None        
        """
        list_state = [node.get_state() for node in self.explored]
        dict_state = dict.fromkeys(list_state,True)
        if node_state in dict_state:
            return node
        return None
        """
    def in_frontier(self,node_state):
        for node in self.frontier.list:
            if node.get_state() == node_state:
                return node
        return None
    def is_in_dict(self,node_state):
        list_node = [node.get_state() for node in self.frontier.list ]+[node.get_state() for node in self.explored]
        dict_node = dict.fromkeys(list_node,True)
        return  node_state in dict_node
    #check is frontier empty
    def is_frontier_empty(self):
        return self.frontier.isEmpty()






class _depthFirstSearch(_GraphSearch):
    def __init__(self,start_state):
        _GraphSearch.__init__(self,start_state)
        self.frontier = util.Stack()
        global node
        node = Node(start_state,0,0,0)
        self.frontier.push(node)

    def get_frontier(self):
        node = self.frontier.pop()
        self.frontier.push(node)
        return node
    def put_frontier(self,node):
        self.frontier.push(node)


def recursive_dfs(problem,_dfs,result):
    
    node = _dfs.get_frontier()#(state,predecessor,cost)
    state,predecessor,action,cost = node.get()

    
    if _dfs.get_goal() is None:#optimized
        if problem.isGoalState(state):
            _dfs.set_goal(state)

    for successor in problem.getSuccessors(state):
        if successor[0] != predecessor:
            in_explored = _dfs.in_explored(successor[0])
            in_frontier = _dfs.in_frontier(successor[0])
            new_cost = cost+successor[2]
            node2 = Node(successor[0],state,successor[1],new_cost)
            if (in_explored == None and in_frontier == None):
                _dfs.put_frontier(node2)
                recursive_dfs(problem,_dfs,result)
            else:

                if(in_explored is not None):
                    if(in_explored.get_cost() > new_cost):
                        _dfs.refresh_explored(in_explored,node2)
                        recursive_dfs(problem,_dfs,result)
                else:
                    if(in_frontier.get_cost() > new_cost):
                        _dfs.refresh_frontier(in_frontier,node2)
                        recursive_dfs(problem,_dfs,result)
                
    _dfs.put_explored(_dfs.pop_frontier())
    
  
    if _dfs.is_frontier_empty():
        

        result = _dfs.search_path(_dfs.get_node(_dfs.get_goal()),result)
        

        return result

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    problem self.searchType(state)
    """
    "*** YOUR CODE HERE ***"
    result = []
    _dfs = _depthFirstSearch(problem.getStartState())
    recursive_dfs(problem,_dfs,result)
    #cProfile.runctx('recursive_dfs(problem,_dfs,result)',None,locals())
    return result





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
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    
    result = []
    _bfs = _BreadthFirstSearch(problem.getStartState())
    recursive_bfs(problem,_bfs,result)
    return result


def priority_function(node):
    return node.get_cost()
    

class _UniformCostSearch(_GraphSearch):
    def __init__(self,start_state):
        _GraphSearch.__init__(self,start_state)
        self.frontier = util.PriorityQueueWithFunction(priority_function)
        global node
        node = Node(start_state,0,0,0)
        self.frontier.push(node)
        

    def put_frontier(self,node):
        self.frontier.push(node)
    def in_frontier(self,node_state): 
        for node in self.frontier.heap:
            if node[2].get_state() == node_state:
                return node
        return None
    def refresh_frontier(self,node,node1):
        self.frontier.heap.remove(node)
        self.frontier.push(node1)

def recursive_ucs(problem,_ucs,result):
    #_ucs.sort_frontier()
    node = _ucs.pop_frontier()
    node_state = node.get_state()
    node_cost = node.get_cost()

    if problem.isGoalState(node_state):
        _ucs.put_explored(node)
        result = _ucs.search_path(node,result)

        return result
    else:
        for successor in problem.getSuccessors(node_state):
            in_explored = _ucs.in_explored(successor[0])
            in_frontier = _ucs.in_frontier(successor[0])
            if (in_explored == None and in_frontier == None):
                if successor[0] != node.get_predecessor():
                    node2 = Node(successor[0],node_state,successor[1],node_cost+successor[2])
                    _ucs.put_frontier(node2)
            else:
                if(in_explored != None):
                    if(in_explored.get_cost()>node_cost+1):
                        node2 = Node(successor[0],node_state,successor[1],node_cost+successor[2])
                        _ucs.refresh_explored(in_explored,node2)
                elif(in_frontier != None):
                    if(in_frontier[2].get_cost()>node_cost+1):
                        node2 = Node(successor[0],node_state,successor[1],node_cost+successor[2])
                        _ucs.refresh_frontier(in_frontier,node2)

        _ucs.put_explored(node)
        recursive_ucs(problem,_ucs,result)







def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    result = []
    _ucs = _UniformCostSearch(problem.getStartState())


    recursive_ucs(problem,_ucs,result)
    return result

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


class _AStarSearch(_GraphSearch):
    def __init__(self,start_state):
        _GraphSearch.__init__(self,start_state)
        self.frontier = util.PriorityQueueWithFunction(priority_function)
        global node
        node = Node(start_state,0,0,0)
        self.frontier.push(node)
        

    def put_frontier(self,node):
        self.frontier.push(node)
    def in_frontier(self,node_state): 
        for node in self.frontier.heap:
            if node[2].get_state() == node_state:
                return node
        return None


def recursive_ass(problem,_ass,heuristic,result):
    node = _ass.pop_frontier()
    node_state = node.get_state()
    node_cost = node.get_cost()
    if problem.isGoalState(node_state):
        _ass.put_explored(node)
        result = _ass.search_path(node,result)
        return result
    else:

        for successor in problem.getSuccessors(node_state):
            in_explored = _ass.in_explored(successor[0])
            in_frontier = _ass.in_frontier(successor[0])
            next_cost = node_cost + heuristic(successor[0],problem)
            if (in_explored == None and in_frontier == None):
                if successor[0] != node.get_predecessor():
                    node2 = Node(successor[0],node_state,successor[1],next_cost)
                    _ass.put_frontier(node2)
            else:
                if(in_explored != None):
                    if(in_explored.get_cost()>next_cost):
                        node2 = Node(successor[0],node_state,successor[1],next_cost)
                        _ass.refresh_explored(in_explored,node2)
                elif(in_frontier != None):
                    if(in_frontier[2].get_cost()>next_cost):
                        node2 = Node(successor[0],node_state,successor[1],next_cost)
                        _ass.refresh_frontier(in_frontier,node2)        

        _ass.put_explored(node)
        recursive_ass(problem,_ass,heuristic,result)    




def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    
    result = []
    _ass = _AStarSearch(problem.getStartState())
    recursive_ass(problem,_ass,heuristic,result)
    
    return result

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
