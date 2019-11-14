Gesture Recognition
===============

Webcam over websocket in Python, inspired from https://github.com/vmlaker/hello-websocket
Gesture Recognition for ["cowboy","ninja", "bear"]

Installation
------------

Install opencv
::
   apt-get install python-opencv
   brew install opencv

Now create virtualenv
::
   virtualenv venv
   source venv/bin/activate

Usage
-----

Now (in a different shell) run the *server*:
::

   python server.py
   
Go to http://localhost:9000 to view the webcam.
