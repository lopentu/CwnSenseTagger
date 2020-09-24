import pickle
import os
from .download import get_model_path
from .cwn_graph_utils import CwnGraphUtils

class CwnBase(CwnGraphUtils):
    """The base cwn reference data.
    """

    def __init__(self):        
        model_path = get_model_path()
        cwn_path = os.path.join(model_path, "cwn_graph.pyobj")                   
        if not os.path.exists(cwn_path):
            raise FileNotFoundError("Cannot find model data, have you tried CwnSenseTagger.download() ?")

        with open(cwn_path, "rb") as fin:
            data = pickle.load(fin)
            if len(data) == 2:
                V, E = data
                meta = {}
            else:
                V, E, meta = data
        super(CwnBase, self).__init__(V, E, meta)        
    