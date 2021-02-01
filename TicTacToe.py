import numpy as np
import pickle

class Game:
    def __init__(self, nrow, ncol):
        self.ncol = ncol
        self.nrow = nrow
        self.size = ncol*nrow
        self.nbturn = 0
        self.plturn = 'X'
        self.board = np.full((nrow, ncol), ' ', dtype=str)
        self.boardHash = None
        self.isEnd = False

    def show(self):
        for i in range(2*self.ncol+1):
            print("- ", end='')
        for i in range(self.nrow):
            print("\n| ", end='')
            for j in range(self.ncol):
                print(self.board[i,j],end='')
                print(" | ", end='')
            print("")
            for i in range(2*self.ncol+1):
                print("- ", end='')
        print("")

    def clear(self):
        self.board = np.full((self.nrow, self.ncol), ' ', dtype=str)
        self.nbturn = 0
        self.plturn = 'X'
        self.isEnd = False
        self.boardHash = None

    def availablePositions(self):
        positions = []
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.board[i, j] == ' ':
                    positions.append((i, j))  # need to be tuple
        return positions

    def checkEnd(self):
        for i in range(self.nrow):
            row = self.board[i,:]
            for j in range(self.ncol-3):
                if (all(elem == row[j] for elem in row[j:j+4]) and row[j]!=' '):
                    self.isEnd = True
        for i in range(self.ncol):
            col = self.board[:,i]
            for j in range(self.nrow-3):
                if (all(elem == col[j] for elem in col[j:j+4]) and col[j]!=' '):
                    self.isEnd = True 
        for i in range(self.nrow-3):
            for j in range(self.ncol-3):
                diag = np.diagonal(self.board[i:i+4, j:j+4])
                if (all(elem == diag[0] for elem in diag) and diag[0]!=' '):
                    self.isEnd = True 
                diag = np.diagonal(np.fliplr(self.board[i:i+4, j:j+4]))
                if (all(elem == diag[0] for elem in diag) and diag[0]!=' '):
                    self.isEnd = True
        if (self.isEnd == False):
            if (self.nbturn == self.size):
                self.isEnd = True
                self.plturn = ' '
            else:
                self.plturn = 'X' if self.plturn == 'O' else 'O'

    def checkEnd3(self):
        for i in range(self.nrow):
            row = self.board[i,:]
            if (all(elem == row[0] for elem in row[0:3]) and row[0]!=' '):
                self.isEnd = True
        for i in range(self.ncol):
            col = self.board[:,i]
            if (all(elem == col[0] for elem in col[0:3]) and col[0]!=' '):
                self.isEnd = True 
        diag = np.diagonal(self.board)
        if (all(elem == diag[0] for elem in diag) and diag[0]!=' '):
            self.isEnd = True 
        diag = np.diagonal(np.fliplr(self.board))
        if (all(elem == diag[0] for elem in diag) and diag[0]!=' '):
            self.isEnd = True
        if (self.isEnd == False):
            if (self.nbturn == self.size):
                self.isEnd = True
                self.plturn = ' '
            else:
                self.plturn = 'X' if self.plturn == 'O' else 'O'
        
    def play(self, pos):
        if (0<=int(pos[0])<self.nrow and 0<=int(pos[1])<self.ncol):
            if (self.board[int(pos[0]),int(pos[1])] == ' '):
                self.board[int(pos[0]),int(pos[1])] = self.plturn
                self.nbturn += 1

    def getHash(self):
        self.boardHash = self.board.reshape(self.ncol * self.nrow)
        return self.boardHash
        

class Player:
    def __init__(self, name, exp_rate=0.3):
        self.name = name
        self.states = []  # record all positions taken
        self.lr = 0.2
        self.exp_rate = exp_rate
        self.decay_gamma = 0.9
        self.states_value = {}  # state -> value

    def getHash(self, board):
        boardHash = board.reshape(len(board) * len(board[0]))
        return boardHash

    def chooseAction(self, game):
        positions = game.availablePositions()
        current_board = game.board
        if np.random.uniform(0, 1) <= self.exp_rate:
            # take random action
            idx = np.random.choice(len(positions))
            action = positions[idx]
        else:
            value_max = -999
            for p in positions:
                next_board = current_board.copy()
                next_board[p[0],p[1]] = game.plturn
                next_boardHash = str(self.getHash(next_board))
                if self.states_value.get(next_boardHash) is None:
                    value = 0
                else:
                    value = self.states_value.get(next_boardHash)
                # print("value", value)
                if value >= value_max:
                    value_max = value
                    action = p
        # print("{} takes action {}".format(self.name, action))
        return action

    def feedReward(self, reward):
        for st in reversed(self.states):
            if self.states_value.get(st) is None:
                self.states_value[st] = 0
            self.states_value[st] += self.lr * (self.decay_gamma * reward - self.states_value[st])
            reward = self.states_value[st]

    def reset(self):
        self.states = []
 
    def savePolicy(self):
        fw = open('policy_' + str(self.name), 'wb')
        pickle.dump(self.states_value, fw)
        fw.close()

    def loadPolicy(self, file):
        fr = open(file, 'rb')
        self.states_value = pickle.load(fr)
        fr.close()

def endOfGame(p1, p2, game):
    result = game.plturn
    # backpropagate reward
    if result == 'X':
        p1.feedReward(1)
        p2.feedReward(0)
    elif result == 'O':
        p1.feedReward(0)
        p2.feedReward(1)
    else:
        p1.feedReward(0.1)
        p2.feedReward(0.5)

def main():
    g = Game(3, 3)
    mode = input("1: train\n2: play\n")
    if(mode == '1'):
        p1 = Player("CPUX_33")
        p2 = Player("CPUO_33")
        print('Training...')
        for i in range(50000):
            p1.reset()
            p2.reset()
            g.clear()
            while (not g.isEnd):
                if g.plturn=='X':
                    g.play(p1.chooseAction(g))
                    p1.states.append(str(g.getHash()))
                else:
                    g.play(p2.chooseAction(g))
                    p2.states.append(str(g.getHash()))
                g.checkEnd()
            endOfGame(p1, p2, g)
            print('{}/50000\r'.format(i+1), end='')
        p1.savePolicy()
        p2.savePolicy()
    elif(mode == '2'):
        cpu = Player("CPU")
        player = input("1: First\n2: Second\n")
        if(player == '1'):
            cpu.loadPolicy("policy_CPUO_33")
            g.show()
            while (not g.isEnd):
                if g.plturn=='X':
                    g.play(input().split())
                    g.show()
                else:
                    g.play(cpu.chooseAction(g))
                    g.show()
                g.checkEnd3()
        elif(player == '2'):       
            cpu.loadPolicy("policy_CPUX_33")
            while (not g.isEnd):
                if g.plturn=='X':
                    g.play(cpu.chooseAction(g))
                    g.show()
                else:
                    g.play(input().split())
                    g.show()
                g.checkEnd3()
        print("End of game")
        print("{} winner".format('No' if(g.plturn==' ') else g.plturn))
    elif(mode == '3'):
        cpu = Player("CPU")
        cpu.loadPolicy("policy_CPUO_33")
        print(len(cpu.states_value))
if __name__ == "__main__":
    main()
        