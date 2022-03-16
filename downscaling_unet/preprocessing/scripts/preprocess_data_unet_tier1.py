__author__ = "Michael Langguth"
__email__ = "m.langguth@fz-juelich.de"
__date__ = "2022-03-16"
__update__ = "2022-03-16"

class Preprocess_Unet_Tier1(Abstract_Preprocessing):

    def __init__(self, source_dir, output_dir):
        """

        """
        super().__init__("preprocess_unet_tier1", source_dir, output_dir)