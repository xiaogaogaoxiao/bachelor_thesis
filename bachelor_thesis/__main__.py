import skeltorch
from .data import BachelorThesisData
from .runner import BachelorThesisRunner

# Create and run Skeltorch project
skeltorch.Skeltorch(BachelorThesisData(), BachelorThesisRunner()).run()
