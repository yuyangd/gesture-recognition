"""
Serve webcam images from a Redis store using Tornado.
Usage:
   python server.py
"""

import base64
from io import BytesIO
import time

import coils
import numpy as np
import redis
from tornado import websocket, web, ioloop
import torch
import torchvision

import torchvision.transforms as transforms
import torch.nn.functional as F
import PIL.Image

MAX_FPS = 100

class IndexHandler(web.RequestHandler):
    def get(self):
        self.render('index-video.html')

class SocketHandler(websocket.WebSocketHandler):
    """ Handler for websocket queries. """
    
    def __init__(self, *args, **kwargs):
        """ Initialize the Redis store and framerate monitor. """

        super(SocketHandler, self).__init__(*args, **kwargs)
        self._store = redis.Redis()
        self._fps = coils.RateTicker((1, 5, 10))
        self._prev_image_id = None
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).cpu()
        self.std = torch.Tensor([0.229, 0.224, 0.225]).cpu()

    # def eval_model(self, img):
        
    def on_message(self, message):
        print('receive message' + message)
        """ Retrieve image ID from database until different from last ID,
        then retrieve image, de-serialize, encode and send to client. """

        # while True:
        #     time.sleep(1./MAX_FPS)
        #     image_id = self._store.get('image_id')
        #     if image_id != self._prev_image_id:
        #         break
        # self._prev_image_id = image_id
        # image = self._store.get('image')
        # image = BytesIO(image)
        # image = np.load(image) #
        # image = base64.b64encode(image).decode('utf-8')
        # # pred = self.eval_model(image)
        # # print(pred)


        """ TODO: message is currently a webp image format url, analyse and return result """

        response = {
          "text": "Hello world"
        }

        self.write_message(response)

        # Print object ID and the framerate.
        text = '{} {:.2f}, {:.2f}, {:.2f} fps'.format(id(self), *self._fps.tick())
        print(text)

app = web.Application([
    (r'/', IndexHandler),
    (r'/ws', SocketHandler),
])

if __name__ == '__main__':
    app.listen(9000)
    ioloop.IOLoop.instance().start()
