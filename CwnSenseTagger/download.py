import os
import zipfile
import tempfile

def get_model_path():
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cwn_graph")
    model_path = os.path.join(cache_dir, "cwn-wsd-model")
    return model_path

def download(upgrade=False):
    if os.path.exists(get_model_path()) and not upgrade:
        print("A copy of DistilTag model already exists. Use upgrade=True to overwrite")
        return 

    import gdown    
    url = "https://drive.google.com/uc?id=1xTdrqOuvLFs2ElHUmCJIcrnO3fKrbH5K"
    with tempfile.TemporaryDirectory("distiltag") as tmpdir:
        outpath = os.path.join(tmpdir, "cwn-wsd-model.zip")
        gdown.download(url, outpath, quiet=False)        
        setup_model(outpath)

def setup_model(zip_path):
    print("setting up model...")
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cwn_graph")
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as ref:
        ref.extractall(cache_dir)

    print("CwnSenseTagger model installed.")