from customModels.predator import Predator
from database.threeDLoMatch import ThreeDLoMatch

# pytorch with rocm won't run without this
from os import putenv
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
putenv("PYTORCH_ROCM_ARCH", "gfx1031")

def main():
    model = Predator("indoor")
    data = ThreeDLoMatch()
    
    for pair in data:
        pair.transform = model(pair)
        pair.show(apply_transform = True)
    
if __name__ == "__main__":
    main()