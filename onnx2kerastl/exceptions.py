class UnsupportedLayer(Exception):
    def __init__(self, layer_description: str):
        self.layer_description = layer_description


class OnnxUnsupported(Exception):
    pass
