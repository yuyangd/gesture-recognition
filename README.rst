Gesture Recognition
===============

Webcam over websocket in Python, inspired from https://github.com/vmlaker/hello-websocket

.. image:: https://github.com/vmlaker/hello-websocket/blob/master/diagram.png?raw=true


Installation
------------

Install opencv
::
   apt-get install python-opencv
   brew install opencv

Also install Redis server:
::

   apt-get install redis-server
   brew install redis

Run Redis
::

   redis-server /usr/local/etc/redis.conf

Now create virtualenv
::
   virtualenv venv
   source venv/bin/activate

Usage
-----

First run the *recorder*:
::

   python recorder.py

Now (in a different shell) run the *server*:
::

   python server.py
   
Go to http://localhost:9000 to view the webcam.
