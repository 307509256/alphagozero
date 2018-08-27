'''
The MIT License (MIT)
Copyright (c) 2013 Dave P.
'''
import threading
import multiprocessing
import signal
import sys
import ssl
from SimpleWebSocketServer import WebSocket, SimpleWebSocketServer, SimpleSSLWebSocketServer
from optparse import OptionParser

from conf import conf
from play import game_init
from engine import ModelEngine, COLOR_TO_PLAYER
from model import load_best_model
import string
import os
from __init__ import __version__

SIZE = conf['SIZE']

class Engine(object):
    def __init__(self, model, logfile):
        self.board, self.player = game_init()
        self.start_engine(model)
        self.logfile = logfile

    def start_engine(self, model):
        self.engine = ModelEngine(model, conf['MCTS_SIMULATIONS'], self.board)

    def name(self):
        return "AlphaGoZero Python - {} - {} simulations".format(self.engine.model.name, conf['MCTS_SIMULATIONS']) 

    def version(self):
        return __version__

    def protocol_version(self):
        return "2"

    def list_commands(self):
        return ""

    def boardsize(self, size):
        size = int(size)
        if size != SIZE:
            raise Exception("The board size in configuration is {0}x{0} but GTP asked to play {1}x{1}".format(SIZE, size))
        return ""

    def komi(self, komi):
        # Don't check komi in GTP engine. The algorithm has learned with a specific
        # komi that we don't have any way to influence after learning.
        return ""

    def parse_move(self, move):
        if move.lower() == 'pass':
            x, y, z = 0, 0, SIZE
            return x, y, z
        else:
            number = move[0]
            letter = move[1]
            number2 = move[2:]
            x = int(number) - 1
            y = string.ascii_uppercase.index(letter)
            #if y >= 9:
            #    y -= 1 # I is a skipped letter
            z = int(number2) - 1

        # x, y = x, SIZE - y - 1
        return x, y, z

    # genmove  3G1
    def print_move(self, x, y, z):
        # x, y = x, SIZE - y - 1

        #if y >= 8:
        #    y += 1 # I is a skipped letter

        move = str(x + 1) + string.ascii_uppercase[y] + str(z + 1)
        return move

    def play(self, color, move):
        announced_player = COLOR_TO_PLAYER[color]
        assert announced_player == self.player
        x, y, z = self.parse_move(move)
        self.board, self.player = self.engine.play(color, x, y, z)
        return ""

    def genmove(self, color):
        announced_player = COLOR_TO_PLAYER[color]
        assert announced_player == self.player

        x, y, z, policy_target, value, self.board, self.player, policy = self.engine.genmove(color)
        self.player = self.board[0, 0, 0, 0, -1]  # engine updates self.board already 
        #with open(self.logfile, 'a') as f:
        #    f.write("PLAYER" + str(self.player) + '\n')
        move_string = self.print_move(x, y, z)
        result = move_string
        return result

    def clear_board(self):
        self.board, self.player = game_init()
        return ""

    def parse_command(self, line):
        tokens = line.strip().split(" ")
        command = tokens[0]
        args = tokens[1:]
        method = getattr(self, command)
        result = method(*args)
        if not result.strip():
            return "=\n\n"
        return "= " + result + "\n\n"

model = load_best_model()
engine = Engine(model, 0)  

class SimpleEcho(WebSocket):

   def handleMessage(self):
      print ("self.data: ", self.data)
      result = engine.parse_command(self.data)   
      print ("result: ", result)
      self.sendMessage(result)

   def handleConnected(self):
      pass

   def handleClose(self):
      pass

clients = []
class SimpleChat(WebSocket):

   def handleMessage(self):
      for client in clients:
         if client != self:
            client.sendMessage(self.address[0] + u' - ' + self.data)

   def handleConnected(self):
      print (self.address, 'connected')
      for client in clients:
         client.sendMessage(self.address[0] + u' - connected')
      clients.append(self)

   def handleClose(self):
      clients.remove(self)
      print (self.address, 'closed')
      for client in clients:
         client.sendMessage(self.address[0] + u' - disconnected')

def consumer(pipe):
  model = load_best_model()
  engine = Engine(model, 0)
  output_p , input_p = pipe
  input_p.close() 
  while True:
    try:
      item = output_p.recv()
      result = engine.parse_command(item)    
    except EOFError:
      break
    print ("<<<" + result)
  print ("consumer done")

def test():
    result = engine.parse_command("clear_board")   
    print ("result1: ", result)
    result = engine.parse_command("play B 3A8")   
    print ("result2: ", result)
    result = engine.parse_command("genmove W")   
    print ("result3: ", result)

if __name__ == "__main__":
   parser = OptionParser(usage="usage: %prog [options]", version="%prog 1.0")
   parser.add_option("--host", default='', type='string', action="store", dest="host", help="hostname (localhost)")
   parser.add_option("--port", default=8000, type='int', action="store", dest="port", help="port (8000)")
   parser.add_option("--example", default='echo', type='string', action="store", dest="example", help="echo, chat")
   parser.add_option("--ssl", default=0, type='int', action="store", dest="ssl", help="ssl (1: on, 0: off (default))")
   parser.add_option("--cert", default='./cert.pem', type='string', action="store", dest="cert", help="cert (./cert.pem)")
   parser.add_option("--key", default='./key.pem', type='string', action="store", dest="key", help="key (./key.pem)")
   parser.add_option("--ver", default=ssl.PROTOCOL_TLSv1, type=int, action="store", dest="ver", help="ssl version")

   (options, args) = parser.parse_args()

   cls = SimpleEcho
   if options.example == 'chat':
      cls = SimpleChat

   if options.ssl == 1:
      server = SimpleSSLWebSocketServer(options.host, options.port, cls, options.cert, options.key, version=options.ver)
   else:
      server = SimpleWebSocketServer(options.host, options.port, cls)

   """
   (output_p, input_p) = multiprocessing.Pipe()
   cons_p = multiprocessing.Process(target=consumer, args=((output_p, input_p),))
   cons_p.start()
   output_p.close()
   
   print ("start new thread...\n")
   one_thr = threading.Thread(target=aigo, args=['name'])
   one_thr.start()
   one_thr.join() 
   """ 

   def close_sig_handler(signal, frame):
      server.close()
      sys.exit()

   signal.signal(signal.SIGINT, close_sig_handler)
   # recive message
   server.serveforever()
 
   
